# Experimental Design — BLAS Backends for scRNA-seq Workflows vs the rapids singlecell GPU Gold Standard


## 1. Motivation

The CPU workflows (Seurat 5, OSCA, scrapper in R; Scanpy in Python) all dispatch their heavy linear algebra to a **BLAS/LAPACK backend** linked at load time. The backend choice changes runtime by 1–2 orders of magnitude *and* shifts FP reduction order enough to flip borderline cluster assignments.

This study extends the source preprint (doi:10.1101/2025.10.28.681564) — which compared only **reference BLAS vs OpenBLAS in R** — to a controlled sweep across **8 BLAS configurations** applied to both R *and* Python (Scanpy). The GPU arm (**rapids singlecell**) is reused as-is via reticulate as the speed gold standard.

The design is organized as a **three-rung GPU-acceleration spectrum**:
1. **CPU-only BLAS** (backends 1–7): zero code change, "free" speed-up from configuration.
2. **NVBLAS offload** (backend 8): unmodified workflows, Level-3 BLAS calls transparently offloaded to GPU.
3. **rapids singlecell**: GPU-native pipeline. The speed reference.

## 2. Research questions

- **RQ1** — BLAS backend effect on per-step and end-to-end runtime.
- **RQ2** — BLAS backend effect on numerical results (PCs, neighbor graphs, cluster labels).
- **RQ3** — How much of the CPU↔GPU gap is closed by (a) optimized host BLAS, (b) NVBLAS offload, (c) residual that only a GPU-native rewrite closes.
- **RQ4** — Portability to Intel and Apple Silicon; on Apple M, perf-per-watt vs raw speed.

## 3. Factors

| Factor | Levels |
|---|---|
| Workflow | Seurat 5, OSCA, scrapper, Scanpy, rapids singlecell |
| BLAS backend | 8 configurations (§5) |
| Hardware | CPU node: 2× EPYC 9654 (Zen 4, 192c/384t, 768 GB). GPU node: 2× EPYC 9135 (Zen 5, 1.5 TB) + NVIDIA H200 NVL 141 GB. Then Intel + Apple M. |
| Dataset | BE1, sc mixology, cord blood, 1.3M (+ 100k/500k/1M subsamples) |
| Threads | 1 → 192 physical, +384 SMT |
| Repeats | ≥ 5 per cell |

Single R-driven harness; Python workflows invoked via **reticulate**.

The H200's 141 GB VRAM holds the full 1.3M dataset in-VRAM with no batching, removing the VRAM bottleneck of the source preprint's A100/P100s.

## 4. Datasets

| Dataset | Genes × Cells | Ground truth | Role |
|---|---|---|---|
| BE1 | 36,753 × 29,606 | 7 lung cell lines | Accuracy |
| sc mixology | 11,786 × 3,918 | 5 cell lines | Accuracy |
| Cord blood (CITE-seq) | 20,400 × 7,858 | ADT-gated labels | Accuracy |
| 1.3M | × 1,308,421 | None | Scalability + numerical concordance |

## 5. BLAS/LAPACK backends (applied to both R and Scanpy)

| # | BLAS | LAPACK | Note |
|---|---|---|---|
| 1 | Reference Netlib | Reference | Baseline + numerical anchor |
| 2 | OpenBLAS (Zen4) | OpenBLAS | Common default; in source preprint |
| 3 | BLIS (`zen4`) | libFLAME | Portable |
| 4 | AMD AOCL-BLIS | AOCL libFLAME | Expected top on Zen 4 |
| 5 | Intel oneMKL — default | oneMKL | Tests AMD dispatch behavior (historical `MKL_DEBUG_CPU_TYPE=5` issue) |
| 6 | oneMKL — forced AVX-512 / AVX2 | oneMKL | Both forced paths |
| 7 | ATLAS (optional) | ATLAS | Legacy baseline if buildable |
| 8 | **NVBLAS** (GPU-offload, hybrid) | host LAPACK + cuBLAS | Intercepts Level-3 only; falls back to host BLAS. **v13.1 (Jan 2026), actively maintained.** GPU node only. Host fallback = OpenBLAS *and* AOCL (sub-factor). |

**NVBLAS caveat (RQ3-critical).** Only Level-3 calls are offloaded. So OSCA's **exact SVD** benefits; Seurat's **IRLBA** and Scanpy's **ARPACK** are iterative (Level-2-heavy) and may not benefit or may regress. `NVBLAS_TILE_DIM` and the dispatch heuristic must be tuned per workflow; report defaults + tuned. Apple M has no NVBLAS arm.

## 6. Workflow harmonization

The five workflows differ algorithmically (HVG model, scaling scope, SVD method, t-SNE perplexity, UMAP metric/neighbors, clustering k and resolution). Two run modes:

- **Mode A — as-published**: each workflow at its source-preprint defaults.
- **Mode B — harmonized**: same filtered matrix, same HVG set, scaling on HVGs only, fixed SVD method + 50 PCs, matched perplexity / UMAP / clustering. **Mode B is where BLAS effects can be cleanly attributed.**

Per-step parameter table per workflow is locked in Phase 0 (omitted here for brevity; matches source preprint exactly for Mode A).

## 7. Response variables

- **Performance**: per-step + end-to-end wall time, peak RSS, VRAM, energy (RAPL / NVML / `powermetrics`), throughput, **perf-per-watt** (headline metric on Apple M), speed-up vs reference BLAS and vs rapids singlecell.
- **Accuracy** (BE1, sc mixology, cord blood): ARI, NMI, per-label F1.
- **Numerical concordance**: PC subspace agreement (principal angles), kNN graph agreement (Jaccard), label agreement (implementation-vs-implementation ARI).

## 8. Phases

- **P0** — Harness, containers, correctness checks.
- **P1** — AMD CPU BLAS sweep (primary): {4 CPU workflows} × {8 backends} × {thread configs} × {3 accuracy datasets} × {≥5 reps}, modes A and B. → RQ1, RQ2.
- **P2** — Intel (native oneMKL) and Apple M (Accelerate + OpenBLAS/BLIS), CPU only. → RQ4.
- **P3** — CPU BLAS vs NVBLAS-offloaded same workflow vs rapids singlecell, all datasets, mode B. Decomposes the CPU↔GPU gap into the three rungs. → RQ3.
- **P4** — Scalability sweep on 1.3M subsamples.

## 9. Controls

Fixed seeds; pinned versions across R/Python/BLAS/CUDA/RAPIDS/driver; NUMA pinning with fixed NPS on Zen 4; explicit control of nested parallelism (workflow threads × BLAS threads); CPU governor `performance` (boost state recorded); warmup runs; containers per backend; reference-BLAS single-threaded run as numerical anchor; identical input matrices hashed across all workflows; reticulate overhead measured separately and not charged to GPU step.

## 10. Statistics

Median + IQR/MAD over repeats. Paired Wilcoxon for backend comparisons (Holm/BH correction). Bootstrap CIs for speed-up. Mixed-effects model with random run effect and fixed effects for backend × workflow × dataset.

## 11. Key pitfalls

- Workflows differ algorithmically — mode B neutralizes this.
- oneMKL dispatch on AMD: backends (5)–(6) explicitly cover default vs forced.
- Python BLAS linking is silent — pin per backend.
- NVBLAS helps only Level-3-dominated workflows (OSCA exact SVD); report per-workflow, never aggregated.
- NVBLAS tuning is non-trivial; report defaults + tuned.
- GPU node ≠ CPU node — never attribute host-side differences to the GPU.
- No GPU arm on Apple M.

## 12. Deliverables

1. Reproducible reticulate-driven harness with per-backend containers.
2. Verified BLAS-linked environments (`sessionInfo()`, `numpy.show_config()`).
3. Raw per-step timing/memory/energy logs.
4. Accuracy + concordance tables with CPU↔GPU gap decomposed by the three rungs.
5. **Recommendations table**: *"hardware X + workflow Y → install backend Z"*, with expected speed-up and accuracy impact. The end-user artifact.
6. Written analysis answering RQ1–RQ4.

---

**Reference.** Source preprint: doi:10.1101/2025.10.28.681564, posted 29 Oct 2025, CC-BY 4.0, <https://www.biorxiv.org/content/10.1101/2025.10.28.681564v1>.
