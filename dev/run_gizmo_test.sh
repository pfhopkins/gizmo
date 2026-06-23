#!/bin/bash
# =============================================================================
# run_gizmo_test.sh  --  build + run GIZMO test problems on TACC, no Python.
#
# Mirrors what the pytest harness (test/harness/gizmo/test.py) does for the
# build+run phase, but as plain bash so you can run it from an idev session
# without the slow conda/pytest startup. The lightweight assert+plot phase can
# then be done separately with:
#       GIZMO_TEST_SKIP_BUILD_RUN=1  python -m pytest test/<name>
# which reuses the snapshots this script produced (no rebuild/rerun).
#
# WHAT IT DOES, per test <name>:
#   1. cp test/<name>/Config.sh -> src/Config.sh   (+ OPENMP=<t> if -t given)
#   2. make clean && make -j     (SYSTYPE via env override; default Vista)
#   3. move src/GIZMO -> test/<name>/GIZMO
#   4. fetch ICs / cooling tables if missing  (LOGIN NODE only -- has internet)
#   5. cd test/<name> && <launcher> ./GIZMO <name>.params 0   (writes output/)
#
# PRESERVE (-p): if test/<name>/output/ already has snapshots, SKIP build+run
#                for that test entirely (no nuking of existing results).
#
# USAGE:
#   bash dev/run_gizmo_test.sh [opts] <test_name> [<test_name> ...]
#
#   -s SYSTYPE   build systype           (default: Vista; e.g. Vista_CPU, Frontera_Kokkos)
#   -n NRANKS    MPI ranks               (default: 1)
#   -t NTHREADS  OpenMP threads/rank     (default: 0 = none; appends OPENMP=N to Config)
#   -l LAUNCHER  MPI launcher            (default: "ibrun -n NRANKS"; override e.g. -l "mpirun -np 4")
#   -j JOBS      make -j parallelism     (default: 8)
#   -c "FLAGS"   VARIANT: extra Config.sh flags for one build+run, space-separated.
#                Repeatable -- each -c is a separate variant with its own output dir.
#                Output is renamed to match the pytest harness variant_output_dir()
#                so GIZMO_TEST_SKIP_BUILD_RUN validation finds it. Use -c '' for the
#                plain baseline. No -c at all => single baseline run into output/.
#   -p           PRESERVE existing output: skip build+run if snapshots present
#   -f           FETCH only: download ICs/cooling tables, then SKIP build+run
#                (run this once on a LOGIN node to prep, before idev)
#   -b           build only (no run)
#   -r           run only  (no build; reuse existing test/<name>/GIZMO)
#   -h           help
#
# EXAMPLES:
#   bash dev/run_gizmo_test.sh -f soundwave evrard       # LOGIN: just fetch ICs, no run
#   bash dev/run_gizmo_test.sh soundwave                 # idev: 1 rank GPU build+run (Vista)
#   bash dev/run_gizmo_test.sh -n 1 -t 8 soundwave       # 1 rank, 8 OMP threads
#   bash dev/run_gizmo_test.sh -p soundwave evrard       # only run ones missing output
#   bash dev/run_gizmo_test.sh -s Frontera_Kokkos -t 8 soundwave   # CPU/Kokkos on Frontera
#   # evrard's two variants (baseline + tidal_adaptive), matching test_evrard.py:
#   bash dev/run_gizmo_test.sh -c '' \
#        -c 'TIDAL_TIMESTEP_CRITERION ADAPTIVE_TREEFORCE_UPDATE=0.06' evrard
#
# NOTE: -c reproduces the GENERIC extra_config_flags mechanism (evrard, etc.).
#       field_loop uses a bespoke per-variant convention (its own Config, patched
#       params, output_Base/CG/MG dirs) that lives only in test_field_loop.py --
#       run that one through pytest, not this script's -c.
# =============================================================================
set -euo pipefail

SYSTYPE="${SYSTYPE:-Vista}"
NRANKS=1
NTHREADS=0
MAKE_JOBS=8
LAUNCHER=""
PRESERVE=0
BUILD_ONLY=0
RUN_ONLY=0
FETCH_ONLY=0
VARIANTS=()

usage() { sed -n '2,/^# ===/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while getopts "s:n:t:l:j:c:pfbrh" opt; do
  case "$opt" in
    s) SYSTYPE="$OPTARG" ;;
    n) NRANKS="$OPTARG" ;;
    t) NTHREADS="$OPTARG" ;;
    l) LAUNCHER="$OPTARG" ;;
    j) MAKE_JOBS="$OPTARG" ;;
    c) VARIANTS+=("$OPTARG") ;;
    p) PRESERVE=1 ;;
    f) FETCH_ONLY=1 ;;
    b) BUILD_ONLY=1 ;;
    r) RUN_ONLY=1 ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done
shift $((OPTIND - 1))
[ "$#" -ge 1 ] || { echo "ERROR: give at least one test name"; usage 1; }

# Default launcher: ibrun on TACC, with explicit rank count.
[ -z "$LAUNCHER" ] && LAUNCHER="ibrun -n ${NRANKS}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
cd "$REPO_ROOT"
[ -f src/Makefile ] || { echo "ERROR: src/Makefile not found in $REPO_ROOT"; exit 1; }

export SYSTYPE   # Makefile honors the SYSTYPE env var (overrides Makefile.systype)
echo "=== repo:    $REPO_ROOT"
echo "=== SYSTYPE: $SYSTYPE   ranks=$NRANKS  omp=$NTHREADS  launcher='$LAUNCHER'"

# --- helper: fetch ICs / exact-solution files into a test dir (login node) ---
fetch_ics() {
  local name="$1" dir="$2" f
  local url1="http://www.tapir.caltech.edu/~phopkins/sims"
  local url2="https://users.flatironinstitute.org/~mgrudic/gizmo_tests/${name}"
  for f in "${name}_ics.hdf5" "${name}_exact.txt" "${name}_exact.hdf5"; do
    [ -f "$dir/$f" ] && continue
    wget -q "$url1/$f" -O "$dir/$f" 2>/dev/null || \
    wget -q "$url2/$f" -O "$dir/$f" 2>/dev/null || rm -f "$dir/$f"
  done
  if [ ! -f "$dir/${name}_ics.hdf5" ]; then
    echo "ERROR: missing $dir/${name}_ics.hdf5 and could not download it."
    echo "       Run this script once on a LOGIN node (compute nodes have no internet)."
    return 1
  fi
}

# --- helper: cooling tables if the test needs them ---------------------------
fetch_cooling() {
  local dir="$1"
  if [ ! -e "$dir/spcool_tables" ]; then
    wget -q "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/spcool_tables.tgz" \
         -O "$dir/spcool_tables.tgz" 2>/dev/null \
      && tar -xf "$dir/spcool_tables.tgz" -C "$dir/" && rm -f "$dir/spcool_tables.tgz" \
      || echo "WARN: could not fetch spcool_tables (need internet / login node)"
  fi
  [ -e "$dir/TREECOOL" ] || cp "$REPO_ROOT/data/cooling/TREECOOL" "$dir/" 2>/dev/null || true
}

# No -c given => one plain baseline run.
[ "${#VARIANTS[@]}" -eq 0 ] && VARIANTS=("")

# Replicate the harness variant_suffix(): "_" + "__".join(sanitized flags), where
# each non-alphanumeric char becomes "_". Empty flags (baseline) -> "".
variant_suffix() {
  local v="$1" out="" f san
  [ -z "$v" ] && { printf ''; return; }
  for f in $v; do
    san="$(printf '%s' "$f" | sed 's/[^[:alnum:]]/_/g')"
    [ -z "$out" ] && out="$san" || out="${out}__${san}"
  done
  printf '_%s' "$out"
}
STASH="__output_baseline_stash__"

for name in "$@"; do
  testdir="test/$name"
  echo
  echo "########## $name ##########"
  [ -d "$testdir" ] || { echo "ERROR: $testdir does not exist"; exit 1; }

  # FETCH only: download ICs/cooling once per test, then skip build+run.
  if [ "$FETCH_ONLY" -eq 1 ]; then
    echo ">>> FETCH only: downloading ICs/cooling for $name (no build, no run)."
    fetch_ics "$name" "$testdir"
    grep -q 'get_cooling_tables' "$testdir/test_${name}.py" 2>/dev/null && fetch_cooling "$testdir"
    continue
  fi

  for vflags in "${VARIANTS[@]}"; do
    suffix="$(variant_suffix "$vflags")"
    target="$testdir/output${suffix}"
    vlabel="${vflags:-baseline}"
    echo
    echo "----- variant: $vlabel  ->  $target -----"

    # PRESERVE: skip this variant if its output dir already has snapshots
    if [ "$PRESERVE" -eq 1 ] && compgen -G "$target/snapshot_*.hdf5" >/dev/null; then
      echo ">>> PRESERVE: $target already has snapshots -- skipping build+run."
      continue
    fi

    # ---- BUILD (base Config.sh + OPENMP + this variant's extra flags) ----
    if [ "$RUN_ONLY" -eq 0 ]; then
      echo ">>> building ($name, variant: $vlabel)"
      rm -f src/GIZMO 
      cp "$testdir/Config.sh" src/Config.sh
      [ "$NTHREADS" -gt 0 ] && printf '\nOPENMP=%s\n' "$NTHREADS" >> src/Config.sh
      for f in $vflags; do printf '\n%s\n' "$f" >> src/Config.sh; done
      make -C src clean
      make -C src -j"$MAKE_JOBS"
      [ -f src/GIZMO ] || { echo "ERROR: build failed (no src/GIZMO)"; exit 1; }
      mv src/GIZMO "$testdir/GIZMO"
      chmod +x "$testdir/GIZMO"
      echo ">>> built -> $testdir/GIZMO"
    fi

    # ---- RUN ----
    if [ "$BUILD_ONLY" -eq 0 ]; then
      [ -f "$testdir/GIZMO" ] || { echo "ERROR: $testdir/GIZMO not found (build first, or drop -r)"; exit 1; }
      fetch_ics "$name" "$testdir"
      grep -q 'get_cooling_tables' "$testdir/test_${name}.py" 2>/dev/null && fetch_cooling "$testdir"

      # Reproducibility / threading env (matches the pytest harness run_test()).
      export OPENBLAS_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      export OMP_PROC_BIND=false
      export OMP_PLACES=threads
      if [ "$NTHREADS" -gt 0 ]; then export OMP_NUM_THREADS="$NTHREADS"; else unset OMP_NUM_THREADS || true; fi

      # GIZMO always writes output/ (params OutputDir). For a variant, stash any
      # existing baseline output/ aside, run, then rename output/ -> target and
      # restore the baseline. Matches the harness stash/finalize logic.
      rm -rf "$target"
      stashed=0
      if [ -n "$suffix" ] && [ -d "$testdir/output" ]; then
        rm -rf "$testdir/$STASH"; mv "$testdir/output" "$testdir/$STASH"; stashed=1
      fi
      rm -rf "$testdir/output"

      echo ">>> running: $LAUNCHER ./GIZMO ${name}.params 0   (in $testdir; stdout->gizmo.out stderr->gizmo.err)"
      ( cd "$testdir" && $LAUNCHER ./GIZMO "${name}.params" 0 1>gizmo.out 2>gizmo.err )

      if [ -n "$suffix" ]; then
        [ -d "$testdir/output" ] && mv "$testdir/output" "$target"
        [ "$stashed" -eq 1 ] && mv "$testdir/$STASH" "$testdir/output"
      fi
      echo ">>> done. snapshots:"
      ls -1 "$target"/snapshot_*.hdf5 2>/dev/null || echo "   (none written -- check $testdir/gizmo.out and $testdir/gizmo.err)"
    fi
  done
done

echo
echo "=== all requested tests processed."
echo "=== to validate + plot without rerunning:"
echo "===   GIZMO_TEST_SKIP_BUILD_RUN=1  python -m pytest test/<name>"
