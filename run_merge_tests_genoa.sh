#!/bin/bash
# Acceptance tests for the starforge_dev_integration -> gpu_kokkos_unified port.
#
#   cd /mnt/home/mgrudic/code/gizmo_starforge_kokkos_merge
#   sbatch run_merge_tests_genoa.sh              # the curated acceptance subset (DEFAULT_K)
#   sbatch run_merge_tests_genoa.sh -k "hernquist or shu_M120"   # subset
#   sbatch run_merge_tests_genoa.sh -b           # build only, no tests
#   sbatch --time=24:00:00 run_merge_tests_genoa.sh -A -t 7200   # EVERY test, no -k filter
#
# -A exists because the default is a curated slice: init.cc / forcetree.cc sit in every
# test's path, so a subset cannot show collateral damage from changes there.
#
# Runs against THIS working tree, uncommitted changes included.
#SBATCH --job-name=gizmo_kokkos_merge
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=genoa
#SBATCH --time=12:00:00
#SBATCH --output=merge_tests-%j.out
#SBATCH --error=merge_tests-%j.err
set -euo pipefail

KEXPR=""
BUILD_ONLY=0
RUN_ALL=0
TIMEOUT="${GIZMO_TEST_TIMEOUT:-3600}"
while getopts "k:t:bA" opt; do
    case $opt in
        k) KEXPR="$OPTARG" ;;
        t) TIMEOUT="$OPTARG" ;;
        b) BUILD_ONLY=1 ;;
        A) RUN_ALL=1 ;;
        *) echo "usage: sbatch $0 [-k EXPR] [-t TIMEOUT] [-b] [-A]"; exit 1 ;;
    esac
done

TREE="${GIZMO_TREE:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$TREE" || { echo "FATAL: cannot cd to '$TREE'"; exit 1; }
if [ ! -e .git ] || ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "FATAL: '$TREE' is not a usable git checkout."; exit 1
fi

# ---- Kokkos -----------------------------------------------------------------
# No kokkos module exists on Rusty (see scripts/build_kokkos_rusty.sh for the evidence);
# build a local one once. Override KOKKOS_PATH if you have your own install.
export KOKKOS_PATH="${KOKKOS_PATH:-$HOME/opt/kokkos-openmp}"
if [ ! -f "$KOKKOS_PATH/include/Kokkos_Core.hpp" ]; then
    echo "=== no Kokkos at $KOKKOS_PATH; building it now (one-off, ~5 min)"
    ./scripts/build_kokkos_rusty.sh "$KOKKOS_PATH"
fi

# ---- environment ------------------------------------------------------------
module --force purge
# texlive: several tests render labels with matplotlib text.usetex, which 'module --force
# purge' strips. gsl/hdf5/fftw come from the module set the RUSTY_GENOA block expects.
module load modules/2.4-20250724 gcc openmpi gsl hdf5/mpi-1.12.3 fftw texlive

export SYSTYPE=RUSTY_GENOA
# Without a per-task cpuset an inherited OMP_PROC_BIND=close/OMP_PLACES=cores stacks every
# rank onto the same cores, which is catastrophic rather than merely suboptimal.
export OMP_PROC_BIND=false
unset OMP_PLACES
export GIZMO_TEST_TIMEOUT="${TIMEOUT}"

echo "=== node $(hostname)  cores=$(nproc)  SYSTYPE=$SYSTYPE  timeout=${GIZMO_TEST_TIMEOUT}s"
echo "=== tree $(pwd)  HEAD $(git rev-parse --short HEAD)  (uncommitted changes INCLUDED)"
echo "=== KOKKOS_PATH=$KOKKOS_PATH"
git log --oneline -8
git status --porcelain | head -20

# ---- build smoke test -------------------------------------------------------
# Build once by hand before pytest, so a compile break is reported as a compile break
# rather than as 20 mysterious "GIZMO did not run" failures.
echo "=== build smoke test (wind_singlestar Config: sinks + cooling, exercises the ported paths)"
cp test/wind_singlestar/Config.sh src/Config.sh
echo "OPENMP=2" >> src/Config.sh
make -C src clean >/dev/null
if make -C src -j"$(nproc)" KOKKOS_PATH="$KOKKOS_PATH" 2>&1 | tail -40; then
    if [ -x src/GIZMO ]; then echo "=== BUILD OK"; else echo "=== BUILD FAILED (no src/GIZMO)"; exit 1; fi
else
    echo "=== BUILD FAILED"; exit 1
fi
[ "$BUILD_ONLY" = "1" ] && { echo "=== -b given, stopping after build"; exit 0; }

# ---- tests ------------------------------------------------------------------
# Acceptance set for what has been ported so far, per the merge plan:
#   hernquist      -> Father[] guards + the dt_tidal 0.5 factor (tidal variant)
#   shu_M120       -> sinks + RT
#   wind_singlestar / shu_jets -> sink seed-mass fixes, merge criteria
#   gmc_cooling    -> cooling chain incl. the bounded h2 partition loop
#   evrard, sedov, soundwave -> baseline hydro/gravity, should be untouched
DEFAULT_K="hernquist or shu_M120 or wind_singlestar or gmc_cooling or evrard or sedov or soundwave"
if [ "$RUN_ALL" = "1" ]; then
    KEXPR=""
    echo "=== pytest over the WHOLE suite (no -k filter)"
else
    [ -z "$KEXPR" ] && KEXPR="$DEFAULT_K"
    echo "=== pytest -k '${KEXPR}'"
fi

PY=/mnt/home/mgrudic/python_work/bin/python
# -u: stdout is the Slurm output file, not a tty, so Python block-buffers by default and
# results would appear in multi-KB bursts. -v prints one line per test as it finishes.
# --continue-on-collection-errors: one unimportable test module must not abort a 279-case run.
#
# BOTH paths run one pytest per test directory rather than a single pass over test/. pytest
# only prints tracebacks in its end-of-run FAILURES section, so a single pass killed by the
# wall clock yields PASSED/FAILED verdicts and no reasons at all -- job 6883048 timed out at
# 24h with 42 failures and zero diagnosis, and 6892638 came within two hours of losing a
# targeted subset the same way. Per directory, a timeout loses only the directory in flight.
# Costs one pytest startup per directory (~1s) and gives up cross-directory ordering.
rc_any=0
for d in test/*/; do
    # only directories holding a test module; skip harness/, __pycache__/, data dirs
    compgen -G "${d}test_*.py" > /dev/null || continue
    if [ "$RUN_ALL" = "1" ]; then
        echo "=== pytest ${d}"
        # -m 'not slow' drops test/fewbody, which is ~192 GIZMO runs on its own. Run it directly
        # (`pytest test/fewbody`) when you want it; -k selects it explicitly on the other path.
        "$PY" -u -m pytest "$d" -ra -s -v --continue-on-collection-errors --tb=short -m 'not slow' || rc_any=1
    else
        # exit 5 is "no test matched -k in this directory", which is the normal case for most
        # of them under a subset filter -- not a failure. Anything else nonzero is.
        rc=0
        "$PY" -u -m pytest "$d" -k "$KEXPR" -ra -s -v --continue-on-collection-errors --tb=short || rc=$?
        [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || rc_any=1
    fi
done
exit $rc_any
