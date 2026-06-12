# Experimental Design — Accelerating Single-Cell RNA-seq Workflows with Different BLAS Backends (CPU), Benchmarked Against the rapids singlecell GPU Gold Standard

## 1. Motivation

The single-cell RNA-seq workflows under study (Seurat 5, OSCA, and scrapper in R; Scanpy in Python) all reduce, at their computational core, to dense and sparse linear algebra: per-cell QC reductions, log-normalization, variance modelling, scaling, and above all dimensionality reduction (PCA via IRLBA, exact SVD, ARPACK, or randomized SVD). On a CPU these operations are not executed by R or Python directly — they are dispatched to a **BLAS/LAPACK backend** linked at build or load time. The choice of backend (reference Netlib, OpenBLAS, BLIS, AMD AOCL, Intel oneMKL) changes two things at once:

1. **Runtime**, because architecture-tuned kernels (vectorized for AVX2/AVX-512, blocked for cache, multithreaded) can be one to two orders of magnitude faster than the reference implementation; and
2. **Numerical output**, because each backend sums floating-point reductions in a different order and uses different fused-multiply-add and blocking strategies. These tiny differences propagate through PCA into the neighbor graph and can flip borderline cluster assignments.

The **central question** is how much the BLAS backend can accelerate the CPU workflows — the three R pipelines and the Python (Scanpy) pipeline — and how close the best CPU configuration can get to a GPU-native pipeline. For the GPU side we do **not** build a new implementation: we take **rapids singlecell** (the scverse GPU library built on the Scanpy API, using CuPy + NVIDIA RAPIDS) as the **speed gold standard**, exactly as in the source preprint, and drive it from R via **reticulate** so it lives in the same harness as the R workflows. The CPU workflows are then measured against this gold standard across the full feasible BLAS space.

The study is staged by hardware: we first characterize the BLAS space on **AMD (Zen 4) CPUs**, then port the winning configurations to **Intel** and **Apple Silicon (M series)**.

### Source of the workflows and relationship to prior work

All workflow specifications (Seurat 5, OSCA, scrapper, Scanpy, rapids singlecell) and the four datasets (1.3M, BE1, sc mixology, cord blood) are taken from the bioRxiv preprint **doi:10.1101/2025.10.28.681564** (posted 29 Oct 2025, CC-BY 4.0), *<https://www.biorxiv.org/content/10.1101/2025.10.28.681564v1>*. That preprint benchmarks the same five workflows for accuracy and scalability, using PCA as the exemplar step, and reports two findings that motivate this design directly: linking R to an optimized BLAS/LAPACK alone gives roughly a **15× speedup** over the reference implementation, and the GPU-aware RAPIDS path is about **an order of magnitude faster** than the fastest CPU method.

This experiment **extends that work** in one specific direction it deliberately did not pursue. The preprint kept R's and Python's *default* BLAS to mimic a typical user; here we instead run a **systematic, controlled sweep across the full feasible BLAS/LAPACK space** (§5) on a modern AMD EPYC node (§3.1), isolating the backend as the experimental factor, and we measure how far that sweep moves each CPU workflow toward the rapids singlecell gold standard. The GPU arm is reused as-is (rapids singlecell via reticulate), not reimplemented.

## 2. Research questions and hypotheses

**RQ1 (BLAS → runtime).** On a fixed AMD CPU, how much does the BLAS/LAPACK backend change per-step and end-to-end runtime of each CPU workflow (Seurat, OSCA, scrapper, Scanpy), and which steps absorb the effect?

**RQ2 (BLAS → numerics).** How much does the backend change the actual results — PC embeddings, neighbor graphs, and final cluster labels — relative to a single-threaded reference run and relative to one another?

**RQ3 (CPU vs GPU gold standard).** With the best BLAS configuration, how close do the CPU workflows (R and Scanpy) get to the rapids singlecell GPU pipeline in runtime, memory, and clustering accuracy against ground truth? Does an optimized BLAS materially shrink the CPU↔GPU gap reported in the preprint?

**RQ4 (portability).** Do the AMD findings generalize to Intel and Apple Silicon (where no CUDA GPU arm is available)?

**Hypotheses.**
- **H1.** Architecture-tuned backends (AOCL-BLIS, OpenBLAS with Zen 4 kernels) outperform reference Netlib BLAS by a large margin; oneMKL is competitive but sensitive to its CPU-dispatch behavior on AMD.
- **H2.** The BLAS effect concentrates on scaling, normalization, and PCA; graph-based steps (Louvain/Leiden) and the gradient-descent embeddings (t-SNE/UMAP) are comparatively BLAS-insensitive.
- **H3.** Cross-backend numerical differences are small in subspace norm but flip a non-negligible fraction of *borderline* cells between clusters; the fraction is dataset-dependent.
- **H4.** rapids singlecell remains the fastest at large cell counts, but an optimized CPU BLAS closes a substantial part of the gap on the BLAS-bound steps (especially PCA), so the CPU↔GPU gap is much smaller than the default-BLAS comparison suggests.

## 3. Factors (independent variables)

| Factor | Levels | Notes |
|---|---|---|
| Workflow | Seurat 5 (R), OSCA (R), scrapper (R), Scanpy (Python), rapids singlecell (Python/GPU) | First four are CPU/BLAS-dependent; rapids singlecell is the GPU gold standard |
| CPU BLAS/LAPACK backend | see §5 | Primary factor for Phase 1; applies to both R and Scanpy |
| Hardware platform | **CPU node: 2× EPYC 9654 (Zen 4, 192 physical / 384 logical cores, 768 GB); GPU node: 2× EPYC 9135 (Zen 5, 1.5 TB) + NVIDIA H200 NVL 141 GB HBM3** → Intel → Apple M | Two separate machines; see §3.1 |
| Dataset | BE1, sc mixology, cord blood, 1.3M | §4 |
| Dataset size | full; plus 100k / 500k / 1M subsamples of 1.3M | Scalability sweep |
| Thread count | 1, 2, 4, 8, 16, 32, 64, 96, 128, 192 (and SMT to 384) | Crossed with backend; sweep to 192 physical / 384 logical on the CPU node |
| Repetitions | ≥ 5 timed runs per cell of the matrix | For variance, §11 |

**Unified harness via reticulate.** The whole benchmark is orchestrated from R. The R workflows run natively; the Python workflows (Scanpy and rapids singlecell) are invoked through the **reticulate** package, as done in the source preprint, so all five pipelines share one driver, one set of input matrices, and one timing/logging path. This keeps the CPU↔GPU and R↔Python comparisons clean and removes harness-level confounds.

### 3.1 Hardware nodes

The CPU workflows and the GPU workflow run on two separate machines, exactly as in the source benchmark.

**CPU node (BLAS sweep — Phase 1).** Dual-socket **AMD EPYC 9654 (Zen 4, "Genoa")** — 96 cores per socket, **192 physical / 384 logical cores total** — with **768 GB RAM and no GPU**. Design implications:

- **AVX-512.** Zen 4 implements AVX-512 (double-pumped 256-bit datapath). Tuned libraries must target **`zen4`** (OpenBLAS Zen4 kernels, AOCL-BLIS Zen4, BLIS `zen4`); the oneMKL question is default vs forced AVX-512 vs forced AVX2 (§5).
- **Dual-socket NUMA.** Two sockets, each with Genoa's 12-CCD layout and configurable NPS (NPS1/2/4), make NUMA placement critical at high thread counts. The **socket count, NPS, and pinning are recorded and held fixed**, with NUMA-aware thread/memory binding (§10); cross-socket memory traffic is a first-order effect to watch when scaling past ~96 threads onto the second socket.
- **768 GB RAM.** The full 1.3M dataset fits in host memory, so the CPU side has **no out-of-core requirement** — any CPU slowdown is compute, not paging.
- **Thread sweep.** 1→192 physical cores, plus an SMT point at 384 logical cores to test whether SMT helps or hurts each BLAS (it usually hurts compute-bound GEMM).
- **No local GPU.** This node has no NVIDIA card, so the GPU-offload backend (NVBLAS, backend 8) and rapids singlecell **cannot run here** — they run on the GPU node below.

**GPU node (rapids singlecell + NVBLAS).** Dual-socket **AMD EPYC 9135 (Zen 5, "Turin")**, **1.5 TB RAM**, and one **NVIDIA H200 NVL with 141 GB HBM3 memory**. Implications:

- **141 GB VRAM removes the out-of-core problem.** All four datasets, including 1.3M, fit comfortably in VRAM (the HVG matrix for 1.3M is only a few GB dense), so rapids singlecell runs **fully in-VRAM with no batching** — unlike the A100/P100 (40/16 GB) used in the source preprint, where VRAM was a binding constraint. This strengthens the gold-standard timing: no host↔device paging overhead.
- **Different host CPU.** The GPU node's host is **Zen 5 EPYC 9135**, not the Zen 4 9654 of the CPU node. Pure GPU-step timings are comparable across nodes, but any **host-side / CPU-fallback work is not apples-to-apples**. NVBLAS in particular runs against the 9135 host BLAS here, so its CPU baseline is **re-measured on this node** for a fair within-node comparison (§5).

## 4. Datasets (held constant across all backends)

| Dataset | Genes × Cells | Ground truth | Role |
|---|---|---|---|
| BE1 | 36,753 × 29,606 | Cell line (7 lung lines) | Accuracy + benchmarking |
| sc mixology | 11,786 × 3,918 | Cell line (5 lines, 10x) | Accuracy + benchmarking |
| Cord blood (CITE-seq) | 20,400 × 7,858 | ADT-gated labels | Accuracy + benchmarking |
| 1.3M | ~genes × 1,308,421 | None | Scalability + cross-backend numerical concordance only |

For the 1.3M dataset, subsampling is at the cellular level (random subsets of 100k, 500k, 1M without replacement; gene count held constant), matching the procedure in the source preprint. Because it lacks ground truth, it is used only for runtime scaling and cross-backend numerical concordance, not accuracy scoring.

## 5. CPU BLAS/LAPACK backends — AMD phase (RQ1, RQ2)

Phase 1 covers the **full feasible backend space for AMD Zen 4**, pairing each BLAS with a defined LAPACK provider since PCA/SVD goes through LAPACK as well as BLAS. **This sweep applies to both the R workflows and the Python (Scanpy) workflow**: R is relinked via `update-alternatives` (libblas/liblapack), and Python's numpy/scipy/scikit-learn are linked against the matching BLAS (conda `libblas`/`liblapack` variants, or built from source), so Scanpy's PCA and scaling steps see the same backends.

| # | BLAS | LAPACK paired | Why include it |
|---|---|---|---|
| 1 | Reference Netlib BLAS | Reference LAPACK | Unoptimized baseline; also the numerical reference (deterministic, single-threaded) |
| 2 | OpenBLAS (Zen4 kernels) | OpenBLAS-bundled LAPACK | Most common default in conda/R; strong AMD support |
| 3 | BLIS (`zen4` config) | libFLAME | Portable, well-documented multithreading model |
| 4 | AMD AOCL-BLIS | AOCL libFLAME | AMD's own optimized fork — expected top performer on Zen 4 |
| 5 | Intel oneMKL — default dispatch | oneMKL LAPACK | Runs on AMD; characterize its native code-path selection on Zen 4 |
| 6 | Intel oneMKL — forced AVX-512 **and** forced AVX2 | oneMKL LAPACK | Zen 4 supports AVX-512, so test both forced paths vs the default (the known AMD dispatch caveat) |
| 7 | ATLAS (optional, tuned build) | ATLAS/LAPACK | Legacy auto-tuned baseline; include only if build is feasible |
| 8 | **NVBLAS** (GPU-offload, hybrid) | host LAPACK + cuBLAS | Drop-in BLAS that offloads Level-3 (GEMM-type) calls to the NVIDIA GPU and **falls back to a host CPU BLAS** for the rest; transparently accelerates the *unmodified* R/Scanpy workflows |

For each backend we additionally vary the **threading configuration** (single-threaded; pinned to N cores) and explicitly control nested parallelism between the workflow's own threads and the BLAS threads (a common confound — see §12). We record the exact version, build flags, and microarchitecture target for every library.

**On NVBLAS (backend 8) specifically.** Unlike backends 1–7, NVBLAS is a *hybrid* CPU/GPU layer. It intercepts Level-3 BLAS calls and dispatches them to the GPU via cuBLAS, while delegating Level-1/2 routines, unsupported calls, and small problems to a **host BLAS that must sit underneath it** (e.g. OpenBLAS or AOCL-BLIS). Because it works at the `libblas` level, it can offload both R's and numpy's Level-3 calls (LD_PRELOAD + `nvblas.conf`). Implications:

- It only accelerates the GEMM-heavy steps (scaling, PC projection, parts of PCA); graph and gradient-descent steps see no benefit, matching the §6.2 sensitivity map.
- The **host-BLAS fallback is a sub-factor**: NVBLAS-over-OpenBLAS and NVBLAS-over-AOCL are distinct configurations, both tested.

Conceptually this gives a **three-rung GPU-acceleration spectrum**: (i) CPU-only BLAS → (ii) NVBLAS = drop-in GPU offload of the unmodified workflows, no algorithmic change → (iii) **rapids singlecell** = a fully GPU-native pipeline (the gold standard). The contrast between (ii) and (iii) shows how much is gained by a GPU-native pipeline versus just offloading the BLAS layer of the existing CPU workflows. Because NVBLAS needs a local NVIDIA GPU, **backend 8 is evaluated on the GPU node (§3.1)**, with its host-BLAS fallback baseline re-measured on that same node (Zen 5 EPYC 9135) rather than compared directly against the EPYC 9654 numbers.

## 6. The CPU workflows — step-by-step specification and differences

The CPU pipelines are **not the same algorithm**. They share the same ten-step skeleton but differ in functions, statistical models, and defaults at almost every step, producing different results *even on identical hardware and BLAS*. These differences must be harmonized or accounted for before any backend effect can be attributed.

### 6.1 Per-step comparison (R workflows)

| Step | Seurat 5 | OSCA (Bioconductor) | scrapper | Consequential difference |
|---|---|---|---|---|
| **Find mito genes** | `PercentageFeatureSet`, regex `^MT-`/`^mt-` | Map gene symbols to chromosome via `EnsDb.Hsapiens.v75` / `EnsDb.Mmusculus.v79`; `perCellQCMetrics` (scuttle) | Regex `^mt-`/`^MT-` on row names | **Identification method differs**: regex (Seurat, scrapper) vs annotation (OSCA, robust to naming but tied to EnsDb version) |
| **Filtering / QC** | `subset` with **fixed manual thresholds per dataset** | Filter on sequencing depth + QC metrics from prior step | `computeRnaQcMetrics` → `suggestRnaQcThresholds` → `filterRnaQcMetrics` (**adaptive thresholds**) | **Cell survival differs** → different surviving cell sets unless harmonized |
| **Normalization** | `LogNormalize` (per-cell total + log) | `logNormCounts` (scuttle): size factors + log | `centerSizeFactors` → `normalizeCounts` | Size-factor definition differs |
| **Highly variable genes** | `FindVariableFeatures`, **vst**, top 1,000 | `modelGeneVar` (scran) + `getTopHVGs`, top 1,000 | `modelGeneVariances` + `chooseHighlyVariableGenes`, top 1,000 | Same count, **different variance model** → different gene sets |
| **Scaling** | `ScaleData` on **all genes** | `scale=TRUE` inside `runPCA` (HVGs only) | `scale=TRUE` inside `runPca` (HVGs only) | **Scope differs**: Seurat scales all genes; others scale only HVGs |
| **PCA** | **IRLBA**, 50 PCs, on HVGs | **Exact SVD** (`runSVD`/BiocSingular), 50 PCs | `runPca` (libscran), **default 25 PCs → set 50** | **SVD algorithm differs** (iterative vs exact vs C++ randomized) |
| **t-SNE** | `RunTSNE`, 50 PCs, **perplexity 18** | `runTSNE`, 50 PCs, **perplexity scales with n** | `runTsne`, 50 PCs, **perplexity 30** | **Perplexity differs across all three** |
| **UMAP** | `RunUMAP`, 50 PCs, uwot, **cosine** (≈30 neighbors) | `runUMAP`, 50 PCs (scater defaults) | 50 PCs, **15 neighbors** | **Metric and neighbor count differ** |
| **Louvain** | `FindNeighbors` SNN **k=20** + `FindClusters` algo=1; res **0.2/0.1/0.2** | `clusterCells` + `NNGraphParam`; res **0.5 all** | `buildSnnGraph` + multilevel; res **0.18/0.16/0.20** | **Graph k and resolution differ** |
| **Leiden** | `FindClusters` algo=4; res **0.2/0.08/0.2** | `clusterCells` + `NNGraphParam` Leiden; res **0.5 all** | `buildSnnGraph` + Leiden; res **0.18/0.16/0.20** | Resolution/k mismatch |

### 6.2 The differences that actually matter

In rough order of impact: **(1) PCA algorithm** (IRLBA vs exact vs randomized) — the most BLAS-bound step and the biggest numerical divergence, so the SVD method must be pinned within a controlled comparison; **(2) scaling scope** (all genes vs HVGs-only); **(3) filtering strategy** (manual fixed vs adaptive → different surviving cells); **(4) t-SNE/UMAP parameters** (perplexity 18 / n-scaled / 30; cosine vs default; 30 vs 15 neighbors); **(5) clustering resolution and graph k**.

### 6.3 Scanpy (Python CPU) and rapids singlecell (GPU)

**Scanpy** follows the same ten-step skeleton with its own functions (`pp.calculate_qc_metrics`, `pp.filter_cells`/`filter_genes`, `pp.normalize_total` + `pp.log1p`, `pp.highly_variable_genes`, `pp.scale`, `tl.pca`, `pp.neighbors`, `tl.leiden`/`louvain`, `tl.umap`/`tl.tsne`), with exact defaults taken from the Scanpy section of the source preprint. Crucially, Scanpy's heavy steps run through **numpy / scipy / scikit-learn**, which link to a CPU BLAS just as R does — its PCA uses ARPACK (scipy) or randomized SVD (scikit-learn). **Scanpy therefore participates in the full §5 BLAS sweep**, and the preprint's observation that stock Python is much faster than stock R is itself a BLAS-linking effect we will reproduce and control.

**rapids singlecell** mirrors the Scanpy API but executes on the GPU via CuPy + NVIDIA RAPIDS (cuML PCA/UMAP/t-SNE, cuGraph Louvain/Leiden, cuVS/cuML neighbors). It is GPU-native end-to-end — not a BLAS-offload of a CPU pipeline — which is exactly why it is the **speed gold standard**. It is run from R through reticulate, configured to the harmonized mode-B parameters (50 PCs, matched neighbors/resolution) so its outputs are comparable to the CPU workflows. Defaults follow the rapids singlecell section of the source preprint.

**Two benchmark modes.** **(A) As-published** — each workflow exactly as in the preprint, to reproduce real behavior. **(B) Harmonized** — identical knobs across all CPU workflows and the GPU reference (same filtered matrix, same HVG set, scaling on HVGs only, fixed SVD method + 50 PCs, fixed perplexity, fixed UMAP metric/neighbors, fixed clustering algorithm/resolution/k). Mode B is where BLAS effects and the CPU↔GPU gap can be cleanly attributed.

## 7. GPU reference workflow: rapids singlecell via reticulate (gold standard)

The GPU arm is **not implemented from scratch**. We reuse **rapids singlecell** as published and treat it as the performance gold standard against which the CPU workflows are measured. Per-step coverage maps onto the same skeleton:

| Step | rapids singlecell backend |
|---|---|
| QC / mito | RAPIDS/CuPy reductions |
| Filtering | GPU mask operations |
| Normalization | CuPy element-wise + reductions |
| HVG | GPU variance modelling (Scanpy-compatible flavor) |
| Scaling | CuPy |
| PCA | cuML PCA (GPU SVD) |
| t-SNE / UMAP | cuML |
| Neighbors | cuVS / cuML KNN → SNN |
| Louvain / Leiden | cuGraph |

Design requirements for a fair comparison:

- **Driven from R via reticulate**, in the same harness as the R workflows (as in the source preprint), so input matrices, parameters, and timing are shared.
- **Parameter-matched to mode B** (50 PCs, matched neighbors/resolution/perplexity) so accuracy and concordance are comparable to the CPU workflows.
- **Treated as the speed reference, not numerical ground truth** — GPU FMA ordering, atomics, and optional reduced precision make exact CPU reproduction impossible; biological labels remain the accuracy reference.
- **VRAM headroom:** the H200 NVL's **141 GB HBM3 holds all four datasets — including 1.3M — entirely in VRAM**, so rapids singlecell runs in-VRAM with no batching or host↔device paging. This is a notable change from the source preprint's A100/P100 GPUs, where VRAM limited the GPU arm; here the gold-standard timing is not penalized by memory transfers.
- **Availability:** the GPU arm runs on the dedicated GPU node (§3.1); it is **unavailable on Apple Silicon**, where the comparison is CPU-only (§9).

## 8. Response variables (dependent measures)

**Performance** — wall-clock per step and end-to-end (median of repeats); peak memory (host RSS and GPU VRAM); energy (CPU via RAPL, GPU via NVML); throughput (cells/s); speedup vs the reference-BLAS baseline and vs the rapids singlecell gold standard; strong/weak scaling across threads and 1.3M subsamples.

**Accuracy vs ground truth** (BE1, sc mixology, cord blood) — ARI and NMI between predicted clusters and ground-truth labels; per-label F1 / confusion matrix for cell-type recovery.

**Cross-backend / cross-workflow numerical concordance** (all datasets) — PC subspace agreement via **principal angles / Grassmann distance**; kNN-graph agreement via mean **Jaccard overlap**; cluster-label agreement via ARI (implementation-vs-implementation); determinism across repeats.

## 9. Phased experimental matrix

**Phase 0 — Harness and correctness.** Build the reticulate-driven orchestration (Snakemake/Nextflow + R driver), containerize each BLAS backend and the rapids environment, verify each workflow reproduces its single-threaded reference within tolerance, lock parameters for modes A and B.

**Phase 1 — AMD CPU BLAS sweep (primary).** Full factorial: {Seurat, OSCA, scrapper, Scanpy} × {8 BLAS backends} × {thread configs} × {BE1, sc mixology, cord blood} × {≥5 repeats}, in modes A and B; 1.3M subsamples for scalability. Answers RQ1, RQ2.

**Phase 2 — Port to Intel and Apple M.** Best CPU configs from Phase 1 re-run on Intel (adding oneMKL native path) and Apple Silicon (adding **Apple Accelerate**, plus OpenBLAS/BLIS where buildable; no GPU arm). Answers RQ4.

**Phase 3 — CPU vs GPU gold standard.** Best CPU BLAS config per workflow (incl. NVBLAS offload) vs **rapids singlecell** on all datasets, in harmonized mode B. This realizes the three-rung spectrum from §5: CPU-only BLAS → NVBLAS offload → rapids singlecell. Answers RQ3.

**Phase 4 — Scalability sweep.** 1.3M at 100k / 500k / 1M / full across surviving CPU configs and rapids singlecell, reporting runtime, memory, and cross-backend numerical divergence as a function of n.

## 10. Controls and confound management

Fixed seeds for every stochastic step; identical algorithmic parameters in mode B (HVGs = 1,000; 50 PCs; matched SNN k and resolutions; matched QC thresholds); pinned software versions for R, every R/Python package, every BLAS/LAPACK, CUDA toolkit, RAPIDS, reticulate, and driver; **NUMA-aware thread and memory pinning with a fixed NPS setting** on the EPYC 9654, plus explicit control of nested parallelism (R/Python threads × BLAS threads); CPU governor `performance` and boost disabled or reported in both states; warmup runs on an otherwise-idle node; same compiler/flags where the library is the variable; containerized environments per backend; a single-threaded reference-BLAS run as the numerical anchor; identical filtered input matrices, hashed and verified across all R and Python workflows; reticulate call overhead measured once and subtracted/annotated so it is not charged to the GPU step.

## 11. Statistical analysis plan

Report **median and a dispersion measure (IQR or MAD)** over repeats. For backend comparisons within a dataset/workflow, use paired non-parametric tests (Wilcoxon signed-rank) with multiple-comparison correction (Holm/BH). Express speedups (vs reference BLAS and vs the GPU gold standard) with bootstrap confidence intervals. For accuracy, report the full ARI/NMI distribution. For numerical divergence, report distributions of principal angles and Jaccard overlaps. A mixed-effects model (random effect for run; fixed effects for backend, workflow, dataset) can summarize main effects and interactions.

## 12. Known pitfalls and confounds

- **The workflows already differ algorithmically** (§6) — the biggest threat to interpretation; mode B neutralizes it.
- **oneMKL on AMD** may pick a non-optimal code path on Zen 4; backends (5)–(6) test default vs forced AVX-512 vs forced AVX2.
- **Python BLAS linking** must be controlled explicitly — numpy/scipy silently pick up whatever BLAS the environment provides, which is exactly the effect under study; pin it per backend.
- **Nested parallelism / oversubscription** (R or Python threads × BLAS threads) on 96 cores produces bogus slowdowns; pin both.
- **reticulate overhead and Python interpreter startup** can contaminate timings if charged to the GPU step; measure and annotate separately.
- **rapids singlecell non-determinism** (GPU atomics, FMA ordering, precision) makes exact CPU reproduction impossible; it is a *speed* reference, biology is the *accuracy* reference.
- **VRAM is not a binding constraint here** — the H200's 141 GB holds 1.3M entirely in VRAM, so unlike the source preprint's smaller GPUs there is no batching overhead to model; the gold standard runs fully in-VRAM.
- **Two separate nodes with different host CPUs** — the CPU sweep runs on the Zen 4 EPYC 9654 node and the GPU arm on the Zen 5 EPYC 9135 + H200 node. Compare GPU-step and end-to-end numbers with that in mind; do **not** attribute host-side differences (data loading, reticulate marshalling, NVBLAS CPU fallback) to the GPU itself.
- **No GPU arm on Apple Silicon** — Phase 2 on Apple M is CPU-only; do not compare its rows to GPU rows.
- **IRLBA vs exact vs randomized SVD** give different PCs even on the same backend — fix the SVD method within a comparison.

## 13. Deliverables

1. A reproducible, reticulate-driven benchmark harness (containers + orchestration + locked R/Python environments), supporting modes A and B, wrapping the R workflows, Scanpy, and rapids singlecell behind one driver.
2. Per-backend BLAS-linked environments for both R and Python, with verification (`sessionInfo()` / `numpy.show_config()`).
3. Raw per-step timing, memory, and energy logs for every matrix cell.
4. Accuracy tables (ARI/NMI/F1) and concordance tables (principal angles, Jaccard, implementation-vs-implementation ARI), plus speedup-to-GPU summaries.
5. A written analysis answering RQ1–RQ4, including AMD → Intel → Apple M portability and how much an optimized BLAS shrinks the CPU↔GPU gap relative to the gold standard.

---

*Assumptions made (stated rather than guessed silently):* the GPU gold standard is rapids singlecell as published, reused via reticulate rather than reimplemented; "gold standard" refers to performance, with biological ground truth as the accuracy reference; Scanpy is included as a Python CPU workflow subject to the same BLAS sweep; algorithmic knobs are harmonized in mode B while mode A preserves each workflow as published; and the GPU arm is unavailable on Apple Silicon. Tell me if any of these should change.

## Reference

The workflows and datasets benchmarked here are drawn from the bioRxiv preprint, doi:10.1101/2025.10.28.681564, posted 29 October 2025, CC-BY 4.0, <https://www.biorxiv.org/content/10.1101/2025.10.28.681564v1>. Insert the full citation (authors, exact title) from the published record when finalizing.
