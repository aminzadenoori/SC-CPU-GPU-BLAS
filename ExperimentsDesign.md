# Experimental Design — BLAS Backends (CPU) vs a CUDA/C GPU Implementation for Single-Cell RNA-seq Workflows

## 1. Motivation

The single-cell RNA-seq workflows under study (Seurat 5, OSCA/Bioconductor, and scrapper) all reduce, at their computational core, to dense and sparse linear algebra: per-cell QC reductions, log-normalization, variance modelling, scaling, and above all dimensionality reduction (PCA via IRLBA, exact SVD, or randomized SVD). On a CPU these operations are not executed by R or C++ directly — they are dispatched to a **BLAS/LAPACK backend** linked at build or load time. The choice of backend (reference Netlib, OpenBLAS, BLIS, AMD AOCL, Intel oneMKL) changes two things at once:

1. **Runtime**, because architecture-tuned kernels (vectorized for AVX2/AVX-512, blocked for cache, multithreaded) can be one to two orders of magnitude faster than the reference implementation; and
2. **Numerical output**, because each backend sums floating-point reductions in a different order and uses different fused-multiply-add and blocking strategies. These tiny differences propagate through PCA into the neighbor graph and can flip borderline cluster assignments.

We want to quantify *both* effects across CPU backends, and then compare the best CPU configurations against a from-scratch **GPU implementation written in C on top of the CUDA libraries** (cuBLAS, cuSOLVER, cuSPARSE, etc.), which we treat as the performance target — the "gold standard" for speed — while keeping the biological ground-truth labels as the reference for correctness.

The study is staged by hardware: we first characterize the full feasible space of BLAS backends on **AMD (Zen) CPUs**, then port the winning configurations to **Intel** and **Apple Silicon (M series)**.

### Source of the workflows and relationship to prior work

All three R workflow specifications (Seurat 5, OSCA, scrapper) and the four datasets (1.3M, BE1, sc mixology, cord blood) used here are taken from the bioRxiv preprint **doi:10.1101/2025.10.28.681564** (posted 29 Oct 2025, CC-BY 4.0), *<https://www.biorxiv.org/content/10.1101/2025.10.28.681564v1>*. That preprint benchmarks five single-cell workflows — Seurat, OSCA, and scrapper in R, plus Scanpy and **rapids singlecell** in Python — comparing accuracy and scalability, with PCA as the exemplar step, and reports that linking R to an optimized BLAS/LAPACK alone gives roughly a 15× speedup over the reference implementation, while the GPU-aware RAPIDS path is about an order of magnitude faster than the fastest CPU method.

This experiment **extends that work** in two directions it did not pursue: (1) instead of keeping R's default BLAS to mimic a typical user, we run a **systematic sweep across the full feasible BLAS/LAPACK space** on a modern AMD EPYC node (§5), isolating the backend as a controlled factor; and (2) instead of using the existing Python RAPIDS/rapids-singlecell library as the GPU arm, we **design a bespoke CUDA/C implementation** (§7) so that every pipeline step — not just PCA — is GPU-native and directly comparable to the R workflows under matched parameters. The preprint's own infrastructure used older AMD EPYC 7301 / Intel Xeon CPUs and NVIDIA A100/P100 GPUs; this design re-runs on the EPYC 9654 (Zen 4) node specified in §3.1.

## 2. Research questions and hypotheses

**RQ1 (BLAS → runtime).** On a fixed AMD CPU, how much does the BLAS/LAPACK backend change per-step and end-to-end runtime of each R workflow, and which steps absorb the effect?

**RQ2 (BLAS → numerics).** How much does the backend change the actual results — PC embeddings, neighbor graphs, and final cluster labels — relative to a single-threaded reference run and relative to one another?

**RQ3 (CPU vs GPU).** How does the best CPU configuration compare to the CUDA/C implementation in runtime, scalability, memory, energy, and clustering accuracy against ground truth?

**RQ4 (portability).** Do the AMD findings generalize to Intel and Apple Silicon, or are the rankings architecture-specific?

**Hypotheses.**
- **H1.** Architecture-tuned backends (AOCL-BLIS, OpenBLAS with Zen kernels) outperform reference Netlib BLAS by a large margin; oneMKL is competitive but sensitive to its CPU-dispatch behavior on AMD.
- **H2.** The BLAS effect concentrates on scaling, normalization, and PCA; graph-based steps (Louvain/Leiden) and the gradient-descent embeddings (t-SNE/UMAP) are comparatively BLAS-insensitive.
- **H3.** Cross-backend numerical differences are small in subspace norm but flip a non-negligible fraction of *borderline* cells between clusters; the fraction is dataset-dependent.
- **H4.** The GPU dominates at large cell counts but shows measurable numerical divergence from the CPU reference (FMA ordering, atomics, optional TF32). Drop-in NVBLAS offload captures part of that GPU speedup on the GEMM-heavy steps without a rewrite, but less than the bespoke CUDA/C pipeline, which also accelerates the non-BLAS steps.

## 3. Factors (independent variables)

| Factor | Levels | Notes |
|---|---|---|
| Workflow | Seurat 5, OSCA, scrapper, GPU-CUDA/C | First three from the methods section; the fourth is the implementation we design |
| CPU BLAS/LAPACK backend | see §5 | Primary factor for Phase 1 |
| Hardware platform | **AMD EPYC 9654 (Zen 4 / Genoa, 96 cores, 768 GB)** primary → Intel → Apple M → NVIDIA GPU | Staged, §9; primary node specified in §3.1 |
| Dataset | BE1, sc mixology, cord blood, 1.3M | §4 |
| Dataset size | full; plus 100k / 500k / 1M subsamples of 1.3M | Scalability sweep |
| Thread count | 1, 2, 4, 8, 16, 32, 48, 64, 96 (and SMT 192) | Crossed with backend; full sweep to 96 physical cores |
| Repetitions | ≥ 5 timed runs per cell of the matrix | For variance, §11 |

### 3.1 Primary hardware node (AMD phase)

The primary CPU node is a single **AMD EPYC 9654 (Zen 4, "Genoa")**: 96 physical cores / 192 threads, 768 GB RAM. The relevant microarchitectural facts that shape the design:

- **AVX-512 support.** Zen 4 implements AVX-512 (via a double-pumped 256-bit datapath). This matters for two backends: architecture-tuned libraries should be built with the **`zen4` target** (OpenBLAS Zen4 kernels, AOCL-BLIS Zen4, BLIS `zen4` config) to use AVX-512, and the oneMKL dispatch question changes — on Zen 4 the comparison is now **MKL default dispatch vs forced AVX-512 vs forced AVX2**, not just AVX2 (see §5).
- **NUMA / chiplet layout.** Genoa exposes 12 CCDs and is configurable for Nodes-Per-Socket (NPS1/2/4). NUMA placement strongly affects BLAS throughput at high thread counts, so the **NPS setting is recorded and held fixed**, and thread/memory pinning is NUMA-aware (§10). We report results for the chosen NPS and note it as a fixed environmental condition.
- **768 GB RAM.** The full 1.3M-cell dataset fits comfortably in host memory, so on the **CPU side there is no out-of-core requirement** — the in-memory path is used throughout, and only the **GPU** arm faces VRAM limits requiring batching (§7). This keeps the CPU↔GPU comparison clean: any CPU slowdown is compute, not paging.
- **Thread sweep.** Strong-scaling is measured across 1→96 physical cores, with an additional SMT-on point at 192 threads to check whether simultaneous multithreading helps or hurts each BLAS (it often hurts compute-bound GEMM).

## 4. Datasets (held constant across all backends)

| Dataset | Genes × Cells | Ground truth | Role |
|---|---|---|---|
| BE1 | 36,753 × 29,606 | Cell line (7 lung lines) | Accuracy + benchmarking |
| sc mixology | 11,786 × 3,918 | Cell line (5 lines, 10x) | Accuracy + benchmarking |
| Cord blood (CITE-seq) | 20,400 × 7,858 | ADT-gated labels | Accuracy + benchmarking |
| 1.3M | ~genes × 1,308,421 | None | Scalability + cross-backend numerical concordance only |

For the 1.3M dataset, subsampling is at the cellular level (random subsets of 100k, 500k, 1M without replacement; gene count held constant), matching the procedure already used in the methods. Because it lacks ground-truth annotation, it is used only for runtime scaling and for measuring how far backends diverge from each other — not for accuracy scoring.

## 5. CPU BLAS/LAPACK backends — AMD phase (RQ1, RQ2)

The goal of Phase 1 is to cover the **full feasible backend space for AMD Zen**, pairing each BLAS with a defined LAPACK provider since PCA/SVD goes through LAPACK as well as BLAS.

| # | BLAS | LAPACK paired | Why include it |
|---|---|---|---|
| 1 | Reference Netlib BLAS | Reference LAPACK | Unoptimized baseline; also the numerical reference (deterministic, single-threaded) |
| 2 | OpenBLAS (Zen/Zen2/Zen3/Zen4 kernels) | OpenBLAS-bundled LAPACK | Most common default in conda/R; strong AMD support |
| 3 | BLIS (vanilla, `zen*` configs) | libFLAME | Portable, well-documented multithreading model |
| 4 | AMD AOCL-BLIS | AOCL libFLAME | AMD's own optimized fork — expected top performer on Zen |
| 5 | Intel oneMKL — default dispatch | oneMKL LAPACK | Runs on AMD; characterize its native code-path selection on Zen 4 |
| 6 | Intel oneMKL — forced AVX-512 **and** forced AVX2 paths | oneMKL LAPACK | Zen 4 supports AVX-512, so test both forced paths vs the default dispatch (the known AMD dispatch caveat) |
| 7 | ATLAS (optional, tuned build) | ATLAS/LAPACK | Legacy auto-tuned baseline; include only if build is feasible |
| 8 | **NVBLAS** (GPU-offload, hybrid) | host LAPACK + cuBLAS | Drop-in BLAS that offloads Level-3 (GEMM-type) calls to the NVIDIA GPU and **falls back to a host CPU BLAS** for the rest; tests transparent GPU acceleration of the *unmodified* R workflows |

For each backend we additionally vary the **threading configuration** (single-threaded; pinned to N cores) and explicitly control nested parallelism between the workflow's own threads and the BLAS threads (a common confound — see §12). We record the exact version, build flags, and microarchitecture target for every library.

**On NVBLAS (backend 8) specifically.** Unlike backends 1–7, NVBLAS is a *hybrid* CPU/GPU layer, not a pure CPU library. It intercepts Level-3 BLAS calls and dispatches them to the GPU via cuBLAS, while delegating Level-1/2 routines, unsupported calls, and small problems to a **host BLAS that must sit underneath it** (e.g. OpenBLAS or AOCL-BLIS). This has three design implications:

- It only accelerates the GEMM-heavy steps (scaling, PC projection, parts of PCA); graph and gradient-descent steps see no benefit, so we expect speedup only where §6.2 predicts BLAS sensitivity.
- The **host-BLAS fallback is a sub-factor**: NVBLAS-over-OpenBLAS and NVBLAS-over-AOCL are distinct configurations and both are tested.
- It is configured via `nvblas.conf` / `NVBLAS_CONFIG_FILE` (offload threshold, fallback library, device selection), all of which are recorded.

Conceptually NVBLAS gives us a **third rung on a GPU-acceleration spectrum**: (i) CPU-only BLAS → (ii) NVBLAS = drop-in GPU offload of unmodified workflows, no algorithmic control → (iii) the bespoke CUDA/C pipeline = full control of every step. The contrast between (ii) and (iii) directly measures how much is gained by rewriting the whole pipeline for the GPU versus just swapping the BLAS layer.

## 6. The three R workflows — step-by-step specification and differences

This is the crux of a fair comparison: **the three pipelines are not the same algorithm**. They share the same ten-step skeleton but differ in functions, statistical models, and defaults at almost every step. Those differences produce different results *even on identical hardware and BLAS*, so they must be either harmonized or explicitly accounted for before any backend or GPU effect can be attributed.

### 6.1 Per-step comparison

| Step | Seurat 5 | OSCA (Bioconductor) | scrapper | Consequential difference |
|---|---|---|---|---|
| **Find mito genes** | `PercentageFeatureSet`, regex `^MT-`/`^mt-` | Map gene symbols to chromosome via `EnsDb.Hsapiens.v75` / `EnsDb.Mmusculus.v79`; `perCellQCMetrics` (scuttle) | Regex `^mt-`/`^MT-` on row names | **Identification method differs**: regex (Seurat, scrapper) assumes naming convention; OSCA uses annotation, robust to naming but tied to EnsDb version |
| **Filtering / QC** | `subset` (SeuratObject) with **fixed manual thresholds per dataset** (e.g. BE1: 200–5000 features, <25k counts, <5% mito) | Filter on sequencing depth + QC metrics from prior step | `computeRnaQcMetrics` → `suggestRnaQcThresholds` → `filterRnaQcMetrics` (**adaptive/automatic thresholds**) | **Cell survival differs**: Seurat hard-codes ranges; scrapper derives thresholds adaptively → different surviving cell sets unless harmonized |
| **Normalization** | `LogNormalize`: scale by per-cell total, then log | `logNormCounts` (scuttle): size factors + log | `centerSizeFactors` → `normalizeCounts` | All log-based but **size-factor definition differs** |
| **Highly variable genes** | `FindVariableFeatures`, **vst**, top 1,000 | `modelGeneVar` (scran) + `getTopHVGs`, top 1,000 | `modelGeneVariances` + `chooseHighlyVariableGenes`, top 1,000 | Same count (1,000) but **different variance model** → different gene sets |
| **Scaling** | `ScaleData` on **all genes** (center 0, unit variance) | No separate step; `scale=TRUE` inside `runPCA` (HVGs only) | No separate step; `scale=TRUE` inside `runPca` (HVGs only) | **Scope differs**: Seurat scales all genes; OSCA/scrapper scale only HVGs within PCA |
| **PCA** | **IRLBA** (truncated iterative), 50 PCs, on HVGs | **Exact SVD** (`runSVD`/BiocSingular), 50 PCs, scaling inside | `runPca` (libscran C++), **default 25 PCs → we set 50** | **SVD algorithm differs** (approximate iterative vs exact vs C++ randomized) — a direct numerical divergence |
| **t-SNE** | `RunTSNE`, 50 PCs, **perplexity 18** | `runTSNE`, 50 PCs, **perplexity scales with n** | `runTsne`, 50 PCs, **perplexity 30** | **Perplexity differs across all three** — embeddings not directly comparable unless fixed |
| **UMAP** | `RunUMAP`, 50 PCs, uwot, **cosine** metric (default 30 neighbors) | `runUMAP`, 50 PCs (scater defaults) | 50 PCs, **15 neighbors** | **Metric and neighbor count differ** |
| **Louvain** | `FindNeighbors` SNN **k=20** + `FindClusters` algorithm=1; resolution **0.2 / 0.1 / 0.2** (BE1 / scmix / cb) | `clusterCells` + `NNGraphParam` (bluster defaults); resolution **0.5 all datasets** | `buildSnnGraph` + multilevel Louvain; resolution **0.18 / 0.16 / 0.20** | **Graph k and resolution differ substantially** → different cluster counts |
| **Leiden** | `FindClusters` algorithm=4; resolution **0.2 / 0.08 / 0.2** | `clusterCells` + `NNGraphParam` Leiden; resolution **0.5 all datasets** | `buildSnnGraph` + Leiden; resolution **0.18 / 0.16 / 0.20** | Same as Louvain row — resolution/k mismatch |

### 6.2 The differences that actually matter for the benchmark

Most consequential, in rough order:

1. **PCA algorithm (IRLBA vs exact SVD vs libscran randomized).** This is the most BLAS-bound step *and* the biggest source of cross-workflow numerical divergence. Because we are studying BLAS effects on exactly this step, the SVD method must be pinned within any controlled comparison.
2. **Scaling scope (all genes vs HVGs-only).** Seurat scales the full matrix before PCA; OSCA/scrapper scale only the 1,000 HVGs inside PCA. This changes both the runtime profile (Seurat does much more scaling work) and the PCs themselves.
3. **Filtering strategy (manual fixed vs adaptive).** Different surviving cell sets mean the workflows are not even operating on the same input — a silent confound that looks like a method effect.
4. **Perplexity / UMAP neighbors / metric.** t-SNE and UMAP outputs are not comparable across the three as published (perplexity 18 vs n-scaled vs 30; cosine vs default; 30 vs 15 neighbors).
5. **Clustering resolution and graph k.** Resolutions differ by up to ~3× between packages and k differs (Seurat 20 vs bluster default vs scrapper default), producing different numbers of clusters — so ARI vs ground truth is partly a parameter artifact, not a backend effect.

**Design consequence.** We run the benchmark in two modes:
- **(A) As-published mode** — each workflow exactly as in the methods, to reproduce real-world behavior; cross-workflow numerical comparison is interpreted with the §6.1 differences in mind.
- **(B) Harmonized mode** — identical knobs across all four implementations (same filtered matrix, same HVG set, scaling on HVGs only, fixed SVD method + 50 PCs, fixed perplexity, fixed UMAP metric/neighbors, fixed clustering algorithm/resolution/k). Mode B is where BLAS and GPU effects can be cleanly attributed, because everything except the linear-algebra backend is held constant.

## 7. The GPU/CUDA workflow we need to design

To be comparable to the three R workflows, the GPU implementation must (a) follow the same ten-step skeleton, (b) map each step onto CUDA libraries, and (c) be **configurable** so it can replicate either a specific workflow (as-published mode A) or the harmonized parameterization (mode B). It is written in C and treated as the speed gold standard; biological labels remain the accuracy reference.

| Step | What it must reproduce | CUDA library / kernel | Configurable knobs (to match each workflow) |
|---|---|---|---|
| Find mito genes | Mito fraction per cell | Thrust/CUB reductions over a sparse matrix | regex-prefix list **or** annotation-based gene set (to match Seurat/scrapper vs OSCA) |
| Filtering / QC | Drop low-quality cells | cuSPARSE row ops + Thrust masks | fixed thresholds **or** adaptive (MAD) thresholds |
| Normalization | Size factors + log | Thrust/CUB per-cell sums; element-wise CUDA kernel | size-factor definition (total-count vs library-size) |
| HVG selection | Top-1,000 variable genes | Custom CUDA variance kernels (mean-variance trend) | model choice (vst-like vs scran-like decomposition) |
| Scaling | Center/scale features | cuBLAS GEMV / element-wise kernels | scope = all genes **or** HVGs-only |
| **PCA** | 50 PCs from HVGs | **cuSOLVER** (exact SVD/eig) **and** a randomized-SVD path | exact vs randomized; number of PCs (fix at 50); single/double precision; **TF32 on/off** |
| t-SNE | 2-D embedding from 50 PCs | cuML kernels or custom CUDA (Barnes-Hut/FFT) | perplexity (18 / n-scaled / 30) |
| UMAP | 2-D embedding from 50 PCs | cuML / custom CUDA | metric (cosine/euclidean), n-neighbors (15/30) |
| Neighbor graph | SNN graph on 50 PCs | RAFT/cuVS or FAISS-GPU KNN → SNN build | k (20 / package default) |
| Louvain / Leiden | Community detection | cuGraph Louvain / Leiden | resolution per dataset; algorithm (Louvain multilevel / Leiden) |

Key design requirements for comparability:

- **Same algorithmic knobs exposed as the R workflows**, so mode-A replication is exact and mode-B harmonization is possible.
- **PCA returns 50 PCs** (not the libscran default of 25) and offers both an exact (cuSOLVER) and a randomized path, so it can mirror OSCA's exact SVD and Seurat/scrapper's iterative/randomized SVD respectively.
- **Precision controls:** single/double precision toggle and the ability to **disable TF32**, so GPU↔CPU numerical comparison is fair rather than confounded by reduced-precision tensor cores.
- **VRAM strategy for 1.3M:** batched/out-of-core handling is expected and is recorded as a constraint with its overhead, rather than silently worked around.
- The GPU is the **performance reference**, not numerical ground truth — atomics and FMA ordering make exact CPU reproduction impossible by design.

## 8. Response variables (dependent measures)

**Performance** — wall-clock per step and end-to-end (median of repeats); peak memory (host RSS and GPU VRAM); energy (CPU via RAPL, GPU via NVML); throughput (cells/s); speedup vs reference; strong/weak scaling efficiency across threads and across 1.3M subsamples.

**Accuracy vs ground truth** (BE1, sc mixology, cord blood) — ARI and NMI between predicted clusters and ground-truth labels; per-label F1 / confusion matrix for cell-type recovery.

**Cross-backend / cross-workflow numerical concordance** (all datasets) — PC subspace agreement via **principal angles / Grassmann distance**; kNN-graph agreement via mean **Jaccard overlap**; cluster-label agreement via ARI (implementation-vs-implementation, not vs truth); determinism across repeated runs of the same configuration.

## 9. Phased experimental matrix

**Phase 0 — Harness and correctness.** Build orchestration (Snakemake/Nextflow), containerize each BLAS backend, verify each workflow reproduces its single-threaded reference within tolerance, and lock parameters for modes A and B.

**Phase 1 — AMD CPU sweep (primary).** Full factorial: {Seurat, OSCA, scrapper} × {7 BLAS backends} × {thread configs} × {BE1, sc mixology, cord blood} × {≥5 repeats}, in both mode A and mode B; 1.3M subsamples for scalability rather than full factorial. Answers RQ1, RQ2.

**Phase 2 — Port to Intel and Apple M.** Best CPU configs from Phase 1 re-run on Intel (adding oneMKL native path) and Apple Silicon (adding **Apple Accelerate**, plus OpenBLAS/BLIS where buildable). Tests whether the AMD ranking holds. Answers RQ4.

**Phase 3 — GPU comparison.** Best CPU config (per architecture) vs **NVBLAS-offloaded workflows** (over each host-BLAS fallback) vs the bespoke **CUDA/C implementation**, on all datasets, single/double precision, TF32 on/off, in harmonized mode B. This is the three-rung spectrum from §5: CPU-only → drop-in GPU offload → full rewrite. Answers RQ3.

**Phase 4 — Scalability sweep.** 1.3M at 100k / 500k / 1M / full across surviving CPU configs and the GPU, reporting runtime, memory, and cross-backend numerical divergence as a function of n.

## 10. Controls and confound management

Fixed seeds for every stochastic step; identical algorithmic parameters in mode B (HVGs = 1,000; 50 PCs; matched SNN k and resolutions; matched QC thresholds); pinned software versions for R, every package, every BLAS/LAPACK, CUDA toolkit and driver; **NUMA-aware thread and memory pinning with a fixed Nodes-Per-Socket (NPS) setting** recorded for the EPYC 9654 node, plus explicit control of nested parallelism (R threads × BLAS threads); CPU governor `performance` and turbo/boost disabled or reported in both states; warmup runs on an otherwise-idle node; same compiler and flags where the library is the variable; containerized environments per backend; a single-threaded reference run as the numerical anchor; hashed input matrices verified identical across runs.

## 11. Statistical analysis plan

Report **median and a dispersion measure (IQR or MAD)** over repeats rather than single timings. For backend comparisons within a dataset/workflow, use paired non-parametric tests (Wilcoxon signed-rank) with multiple-comparison correction (Holm/BH). Express speedups with bootstrap confidence intervals. For accuracy, report the full ARI/NMI distribution across repeats. For numerical divergence, report distributions of principal angles and Jaccard overlaps. A mixed-effects model (random effect for run, fixed effects for backend, workflow, dataset) can summarize main effects and interactions.

## 12. Known pitfalls and package-level confounds

- **The workflows already differ algorithmically** — see §6. This is the single biggest threat to interpretation; mode B exists to neutralize it.
- **oneMKL on AMD** may pick a non-optimal code path on Zen 4; backends (5) and (6) test default dispatch vs forced AVX-512 and forced AVX2 explicitly (Zen 4 supports AVX-512).
- **Nested parallelism / oversubscription** (R threads × BLAS threads) produces bogus slowdowns; pin both.
- **GPU non-determinism** (atomics, FMA ordering, TF32) makes exact reproduction impossible; the GPU is a *speed* reference, biology is the *accuracy* reference.
- **VRAM limits at 1.3M** force batching/out-of-core **on the GPU only** — the 768 GB host node holds the full dataset in memory, so this is not a CPU-side concern. Record the GPU batching overhead.
- **IRLBA vs exact vs randomized SVD** give different PCs even on the same backend — fix the SVD method within a comparison.

## 13. Deliverables

1. A reproducible benchmark harness (containers + orchestration + locked environments), supporting mode A and mode B.
2. The configurable CUDA/C reference implementation with precision toggles.
3. Raw per-step timing, memory, and energy logs for every matrix cell.
4. Accuracy tables (ARI/NMI/F1) and concordance tables (principal angles, Jaccard, implementation-vs-implementation ARI).
5. A written analysis answering RQ1–RQ4, including AMD → Intel → Apple M portability and CPU-best vs GPU.

---

*Assumptions made (stated rather than guessed silently):* the GPU implementation targets NVIDIA/CUDA specifically (Apple M is treated as a CPU/Accelerate platform here, not as a GPU backend); the "gold standard" label refers to performance, with biological ground truth as the accuracy reference; and algorithmic knobs are harmonized in mode B while mode A preserves each workflow as published. Tell me if any of these should change — e.g. if you also want a Metal/Apple-GPU arm, or want the GPU result treated as the numerical reference.

---

## Reference

The workflows and datasets benchmarked here are drawn from the bioRxiv preprint: *Benchmarking single-cell RNA-seq analysis workflows* (working reference), doi:10.1101/2025.10.28.681564, posted 29 October 2025, available under a CC-BY 4.0 license at <https://www.biorxiv.org/content/10.1101/2025.10.28.681564v1>. Replace the working title with the full citation (authors, exact title) from the published record when finalizing.
