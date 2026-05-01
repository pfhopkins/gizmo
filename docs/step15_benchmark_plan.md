# Step 15 — Scaling & Benchmark Suite (Plan)

Status: PROPOSED, awaiting sign-off. Owner: Phil + Claude. Branch: `gpu_kokkos_bench`.

This document is the source of truth for the Step-15 benchmarking effort. It is written
to be read cold by anyone (or a future agent) picking up the work. Update in place as
the suite evolves; do not fork copies.

---

## 1. Goals

Two consumer tracks drive every design choice. Same runs, different plots.

1. **Proposal-quality scaling figures** — direct counterparts of
   `~/Documents/proposals/NSF/frontera_2021_renewal/{StrongScaling_M2e6,WeakScalingResolution,Time_vs_CPU_FB}.pdf`
   and `.../main_text_old/figure/scaling_{strong,weak_res,weak_size}.pdf`. These feed
   upcoming computing-time proposals on hybrid CPU-GPU machines.
2. **Bottleneck / regression hunting** — phase-resolved per-step time, % in `misc`,
   peak memory, communication vs. compute, GPU occupancy. Goal: identify any
   unexpected bottleneck, OOM at scale, work that should be on device but isn't,
   and validate (or invalidate) the §6 optimization backlog in
   [`gpu_port_handoff_20260423.md`](gpu_port_handoff_20260423.md).

---

## 2. Reference codes (three)

| Tag | Source | Build (Vista) | Build (Mac) | Purpose |
|----|--------|---------------|-------------|---------|
| **A. GPU-current** | `gpu_kokkos_bench` HEAD | `SYSTYPE=Vista` (Kokkos CUDA, sm_90) | `SYSTYPE=MacBookCellar_Kokkos` (Kokkos OMP) | System under test |
| **B. CPU-Kokkos-current** | `gpu_kokkos_bench` HEAD | `SYSTYPE=Vista_CPU` (Kokkos OMP) | (same as A on Mac) | Isolates GPU-offload value from algorithm changes |
| **C. Legacy pre-GPU** | `gizmo-cpp` HEAD (production) | `SYSTYPE=Vista` (no Kokkos) or `Frontera`-style | `SYSTYPE=MacBookCellar` | Production CPU baseline; ensures port hasn't regressed real science runs |

**Why `gizmo-cpp` HEAD over a frozen 2020 branch:** same variable names, same I/O,
same params files. Analysis pipelines are identical. The historical-fidelity gain of
2020-frozen does not justify the divergence cost.

**Reference-C scope:** all problems including FIRE and STARFORGE (`gizmo-cpp` runs
these in production today). Any params-name drift between branches gets a thin
translation shim under `test/benchmark/step15/refs/legacy_params/<problem>/params.txt`.

---

## 3. Platforms

| Platform | Hardware | Builds run | Notes |
|----------|----------|-----------|-------|
| Mac (laptop) | Apple Silicon, 8–12 perf cores | A, C | Single node; (ranks × threads) ≤ phys_cores |
| Vista CPU | Grace ARM, 72 cores/node | B, C | OMP-only sweep axis |
| Vista GPU | NVIDIA GH200 (1 GPU/node) | A | 1 rank per GPU; OMP only for host-side prep |

Frontier (AMD MI250X / HIP) is **out of scope** for Step 15. Re-evaluate after Step 15
results inform porting priorities.

---

## 4. OpenMP / Kokkos thread control

Most likely source of silent benchmark-invalidation. Rules:

- Always set explicitly per-run, in the SLURM/local launch wrapper:
  ```
  export OMP_NUM_THREADS=<N>
  export OMP_PROC_BIND=close
  export OMP_PLACES=cores
  unset KMP_AFFINITY     # avoid Intel/GNU OMP collisions
  ```
- For Kokkos builds, also pass `--kokkos-num-threads=$OMP_NUM_THREADS` on the GIZMO
  command line. (Kokkos respects this independent of `OMP_NUM_THREADS`.)
- One sweep axis per study. Hold `(ranks × threads_per_rank) = cores_per_node`,
  vary the split:
  - **Vista CPU (72 cores/node):** `{(72,1), (36,2), (18,4), (9,8), (4,18), (1,72)}`
  - **Vista GPU (1 GPU/node):** ranks = nodes; `OMP_NUM_THREADS` set to host CPUs/rank,
    typically 8–18, but expected to be off the hot path
  - **Mac (8 perf cores assumed):** `{(8,1), (4,2), (2,4), (1,8)}`
- **Pinning verifier.** Add a one-line print at startup:
  `[BENCH-PIN] mpi_rank=R/total nthreads_omp=X kokkos_concurrency=Y cores_visible=Z`.
  Driver scans output and aborts the sweep if X != Y or X != intended.

Record `(nranks, nthreads_per_rank, cores_used_per_node, OMP_PROC_BIND)` in every
run's `metadata.json`. No exceptions.

---

## 5. Problem suite

Idealized first (clean log-log plots), real-physics second (corner-case stress).

| Problem | Stresses | Sizes (N or factor) | Builds | Tier |
|---------|----------|---------------------|--------|------|
| `poisson_box` | hydro+gravity, idealized | 64³, 128³, 256³, 512³ | A, B, C | 1 |
| `gravtree_vanilla` | gravity tree only | 64³ → 512³ | A, B, C | 2 |
| `gmc_cooling` | MHD+cooling+SF | 30³, 50³, 80³, 128³, 200³ | A, B, C | 2 |
| `gmc_cooling_pmgrid` | + tree-PM long-range | same | A, B, C | 2 |
| `isodisk_mechfb_sinks` | STARFORGE-realistic | 0.25×, 1×, 4× of stock IC | A, B, C | 3 |
| `fire` (m11i downsample) | full FIRE3 pipeline | 0.25×, 1×, 4×, 16× | A, B, C | 4 |

Sizes are ICs we generate (or upsample/downsample existing ones). Generation script:
extend `test/benchmark/make_benchmark_ics.py`.

---

## 6. Scaling axes

For each (problem × build × platform):

- **Strong scaling.** Fix N (mid-size for that problem). Vary ranks ∈ {1, 2, 4, 8}.
- **Weak scaling — by-rank.** N/rank ≈ const. Ranks ∈ {1, 2, 4, 8}, sizes scale.
- **Weak scaling — by-resolution.** N varies, ranks fixed at 1 (or 8 for the largest).
  This is the "increase resolution at fixed compute" plot.
- **Time-vs-CPU.** Single point per build at proposal-target N — feeds the headline
  "time to ship a STARFORGE/FIRE run" comparison.

Always run ≥10 timesteps. Discard steps 0–1 (domain build, BVH amortization).
Report **median(steps 2–9)** as the headline per-step time. Also report
mean and 90th-percentile.

---

## 7. Diagnostics

### 7.1 Spine: `cpu.txt`
Already produced. Parser converts to `timings.json` with phase totals,
% of total, and `% in misc`. **Watchdog: misc fraction > 5% triggers a flag** in the
aggregator output; > 10% blocks the run from contributing to proposal plots until
investigated.

### 7.2 New diagnostics (compile-flag gated)
New macro: **`GIZMO_BENCHMARK_VERBOSE`**. Default OFF.
- All new prints use `fprintf(stderr, "[BENCH] ...\n")` so they're trivially
  separable from physics stdout (`grep '\[BENCH\]'`) and strippable post-hoc.
- New per-step phase prints (only ones not already covered by `cpu.txt`):
  - Ghost exchange MPI time (separate send/recv)
  - LET pack / unpack / `MPI_Iallgatherv` wait
  - BVH build (gpu_morton + gpu_peano_walk)
  - Density h-convergence iterations + time
  - Kokkos kernel launch counter (cheap atomic)
- One-shot end-of-run prints: peak RSS, peak GPU memory, total kernel launches.

### 7.3 GPU utilization (Vista GPU only)
SLURM script polls `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`
at 1 Hz, dumps to `gpu_util.csv`. Aggregator computes: mean utilization while
GIZMO is running, peak memory, time-to-saturation.

### 7.4 Validation gate
Before contributing to scaling plots, every (problem × build × N) run completes a
1-step parallel reference comparison against the gold snapshot for that problem.
Any drift > tolerance → run is logged as `validation_failed` and excluded.
Carries forward the verification rules in
[`gpu_port_verification_rules.md`](gpu_port_verification_rules.md).

---

## 8. Permanent data layout

Outside the source tree so `git clean` / rebuilds cannot nuke results.

```
/Users/phopkins/Documents/work/code/gizmo/gizmo_vscode/benchmarks_step15/
  runs/
    YYYYMMDD_HHMMSS_<system>_<build>_<problem>_N<size>_R<ranks>_T<threads>/
      params.txt   Config.sh   gizmo.out   gizmo.err   cpu.txt
      timings.json metadata.json
      gpu_util.csv               # GPU runs only
      validation.json            # 1-step ref-comparison result
  aggregated/
    all_runs.csv                 # one row per run, all knobs + headline numbers
    phase_breakdown.csv          # one row per (run, phase)
    manifest.json                # idempotency tracking for re-runs
  figures/
    proposal_strong_scaling_<problem>.pdf
    proposal_weak_scaling_resolution_<problem>.pdf
    proposal_weak_scaling_size_<problem>.pdf
    proposal_time_vs_cpu_<problem>.pdf
    diagnostic_phase_breakdown_<problem>.pdf
    diagnostic_misc_fraction.pdf
    diagnostic_gpu_util_<problem>.pdf
  notebooks/
    01_aggregate.ipynb  02_plots_proposal.ipynb  03_plots_diagnostic.ipynb
  STEP15_RESULTS.md              # narrative + figure index
```

`benchmarks_step15/` mirrored to Dropbox path TBD for laptop-loss survival
(decision: copy at end of each tier).

---

## 9. Suite layout (in-repo)

```
test/benchmark/step15/
  sweeps/                        # config-driven sweep specs
    tier1_infra.yaml
    tier2_idealized.yaml
    tier3_proposal.yaml
    tier4_realphysics.yaml
  drivers/
    build_three_refs.sh          # builds A, B, C → per-config binaries
    run_sweep.sh                 # reads sweep yaml → SLURM submissions
    run_local.sh                 # mac / interactive variant
    submit_with_dependency.sh    # validation runs gate scaling runs
  refs/
    legacy_params/<problem>/     # gizmo-cpp params shims if names drift
  analysis/
    parse_cpu_txt.py             # cpu.txt → timings.json
    parse_gizmo_out.py           # [BENCH] lines + headline numbers
    aggregate.py                 # walks runs/ → all_runs.csv
    plots_proposal.py
    plots_diagnostic.py
  sanity/
    check_omp_pinning.py
    check_validation.py
```

Each `sweep_yaml` declares: builds, problems, sizes, (ranks × threads) grid,
walltime, account, partition, output prefix. Driver materializes
deterministically-named run dirs and writes `manifest.json` so re-runs are
idempotent.

---

## 10. Tiered rollout (budget gating)

Total cap: ~500 node-hr (25% of 2000-hr allocation). Each tier ends in a review
gate: results in hand → fix bottlenecks → unlock next tier.

| Tier | Scope | Est. node-hr | Queue |
|------|-------|--------------|-------|
| 1 | Harness validation: `poisson_box` 64³, all 3 builds, 1 rank, ~10 steps | ~5 | `gh-dev` |
| 2 | Idealized + mid-complexity strong+weak: poisson_box, gravtree_vanilla, gmc_cooling[_pmgrid], up to 256³, 1–4 ranks | ~30 | `gh-dev` + `gh` |
| 3 | Full proposal sweep: above to 512³, 1–8 ranks, all axes | ~100 | `gh` |
| 4 | FIRE + STARFORGE big-N: `isodisk_mechfb_sinks`, `fire`, longer runs | ~300 | `gh` |

Mac runs (free) interleave throughout for sanity-check single-node points and to
de-risk Vista runs.

---

## 11. Implementation order

1. **Infra (Mac smoke test).** `build_three_refs.sh` for Mac only (A, C — B and Mac-A are
   identical). Pinning wrapper. `parse_cpu_txt.py` + `aggregate.py`. Toy plot from a
   1-step `poisson_box` 64³ on Mac.
1a. **Vista bring-up (Tier-2 prep).** Backport a `Vista` and `Vista_CPU` SYSTYPE
    block to a local-only `gizmo-cpp-bench` branch (forked from `gizmo-cpp` HEAD;
    not for upstream merge). Then `build_three_refs.sh` covers A/B/C on Vista.
2. **Tier 1 → review.** Sanity-check headline numbers vs. §5 of handoff
   (poisson_box 125k: CPU 1.92, GPU 0.27 s/step). Recover those numbers within
   ~10% before proceeding.
3. **Tier 2 → review.** First diagnostic phase-breakdown plot. Hunt for
   `misc > 5%`. Cross-check against §6 backlog.
4. **Tier 3 → review.** Lock down proposal figures. Iterate plot styling against
   `~/Documents/proposals/.../scaling_*.pdf` examples.
5. **Tier 4 → review.** FIRE/STARFORGE. Final `STEP15_RESULTS.md` with
   bottleneck list + concrete fix-or-defer recommendations against the
   handoff backlog.

---

## 12. Risks / open issues

- **`gmc_cooling_rt` ~25% divergence** (handoff §6) — exclude this problem from
  Step 15. Use `gmc_cooling` (no RT) instead.
- **Static `MaxPart` / `MaxNodes`** (handoff §6) — large-N runs may hit allocation
  limits. Tier-3 onward includes `PartAllocFactor` sweeps if OOM observed.
- **Wall-time-out runs** must mark `metadata.json:status="walltime_exceeded"` and
  be excluded from medians, not silently averaged.
- **Reference-C param drift** between `gizmo-cpp` and `gpu_kokkos_bench` — if a
  param name moved, the legacy shim handles it; if a *physical* default changed,
  we run BOTH defaults on C and document the choice.
- **Kokkos overhead at small N** — handoff §5 already notes <100k particles is
  overhead-bound on GPU. Sweep includes points below this threshold deliberately,
  to characterize the crossover, not to set the proposal headline.

---

## 13. Hard rules carried forward

From [`gpu_port_verification_rules.md`](gpu_port_verification_rules.md), unchanged:

- Test = comparison against reference, not exit code.
- Always multi-rank.
- Never loosen tolerances to make a test pass.
- No test artifacts in commits.
- Verify `Config.sh` against `GIZMO_config.h` after every build.
- Ask before advancing tiers.
