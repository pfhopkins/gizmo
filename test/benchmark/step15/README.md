# Step-15 Benchmark Suite

See [`docs/step15_benchmark_plan.md`](../../../docs/step15_benchmark_plan.md) for the
full plan. This directory holds the harness.

## Layout

```
step15/
  sweeps/              JSON sweep specs (one per tier)
  drivers/
    build_three_refs.sh    Build A, B, C binaries per problem
    gizmo_launch.sh        OMP/Kokkos pinning launcher (writes pinning.log)
    run_sweep.py           Drive a sweep: materialize run dirs, launch local or SLURM
  refs/legacy_params/  per-problem params shims for the gizmo-cpp legacy build
  analysis/
    parse_cpu_txt.py       cpu.txt -> timings.json
    parse_gizmo_out.py     gizmo.out + gizmo.err -> run_summary.json
    aggregate.py           runs/ -> aggregated/all_runs.csv + phase_breakdown.csv
    plots_diagnostic.py    sanity-check plots from aggregated CSVs
  sanity/
    check_omp_pinning.py   verify pinning.log + [BENCH-PIN] runtime prints
  ic_cache/            Cached generated ICs (gitignored, regenerated on demand)
  build/               Cached binaries from build_three_refs.sh (gitignored)
```

Output (run artifacts and aggregated data) lives **outside** the source tree:
`/Users/phopkins/Documents/work/code/gizmo/gizmo_vscode/benchmarks_step15/`.

## Tier-1 smoke test (Mac)

```bash
# 1. Build A and C for poisson_box
./test/benchmark/step15/drivers/build_three_refs.sh poisson_box --platform mac --builds A,C

# 2. Run the sweep
./test/benchmark/step15/drivers/run_sweep.py \
   test/benchmark/step15/sweeps/tier1_infra.json --dry-run    # preview
./test/benchmark/step15/drivers/run_sweep.py \
   test/benchmark/step15/sweeps/tier1_infra.json              # actually run

# 3. Inspect results
ls /Users/phopkins/Documents/work/code/gizmo/gizmo_vscode/benchmarks_step15/runs/
cat /Users/phopkins/Documents/work/code/gizmo/gizmo_vscode/benchmarks_step15/aggregated/all_runs.csv

# 4. Toy plot
python3 test/benchmark/step15/analysis/plots_diagnostic.py \
   /Users/phopkins/Documents/work/code/gizmo/gizmo_vscode/benchmarks_step15
```

`run_sweep.py` flags:
- `--dry-run` — print the plan, no actions.
- `--no-launch` — materialize run dirs (binaries, params, ICs) but don't launch.
- `--only A,C` — restrict to specific builds.
- `--filter poisson_box` — restrict to problems containing this substring.

## Adding a new problem

`run_sweep.py` currently has IC handling wired only for `poisson_box`. To add
`gmc_cooling` etc., extend `build_run()` with the appropriate IC step and
list the problem in your sweep JSON.
