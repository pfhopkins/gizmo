#!/bin/bash
# Submit the N-body validation tests as INDEPENDENT Slurm jobs, one per test, run in parallel.
#
#   ./run_nbody_validation.sh              # write the job scripts and submit
#   ./run_nbody_validation.sh -n           # write them, don't submit
#   ./run_nbody_validation.sh -k binary    # just one
#
# HOW PARALLEL BUILDS ARE MADE SAFE. Every test builds in the repo root -- Config.sh,
# GIZMO_config.h, all object files, the GIZMO binary -- so concurrent jobs would corrupt each
# other. Two changes make this work (python_src/gizmo/test.py):
#   * build_gizmo_for_test holds an exclusive flock on .gizmo_build.lock for the build only,
#     not the run. Builds are minutes and serialise; runs are tens of minutes and overlap, so
#     nearly all the parallelism survives. The kernel drops the lock if a job is killed, so a
#     dead job cannot wedge the others.
#   * the build used to "rm -f test/*/GIZMO", deleting every OTHER test's binary. That is fatal
#     when a sibling job has already built and is about to run. It now removes only its own.
# flock is verified working on GPFS, so the lock holds across nodes.
#
# RESOURCES ARE PER TEST, not a full node each. These are correctness runs, not benchmarks, so
# the timing isolation an exclusive node buys is not worth idling 63 cores for a two-particle
# problem. Slurm gives each job its requested CPUs exclusively even on a shared node, so a
# co-tenant cannot steal them.
#
# The floor is the BUILD, not the run: build_gizmo_for_test runs "make -j8" regardless of how
# small the test is, so anything asking for fewer than 8 cores just makes its own build slower.
# Hence 8 for the small tests. fewbody is the exception -- it sizes its own concurrency from the
# cores it can see (cores/2 problems, 2 ranks each), so it genuinely scales with the allocation.
set -uo pipefail

SUBMIT=1
KEXPR=
WALLTIME=8:00:00
while getopts "nk:w:h" o; do case $o in
  n) SUBMIT=0 ;; k) KEXPR=$OPTARG ;; w) WALLTIME=$OPTARG ;;
  h) sed -n '2,22p' "$0"; exit 0 ;; *) exit 2 ;;
esac; done

REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

# test | cores | note        (cores = max(what the run uses, 8 for make -j8))
TESTS=(
  "binary|8|1 rank x 1 thread; build-bound. 1000 orbits + order sweep; guards the timestep normalization"
  "triple|8|1 rank x 1 thread; build-bound. 6-bin hierarchy + order sweep; guards the source prediction"
  "plummer_binaries|8|2 ranks x 4 threads (PB_MAX_CORES=8 caps it). Synced-state energy, direct potential"
  "plummer_binaries_realistic|8|2 ranks x 1 thread; build-bound. Realistic population; NEVER RUN BEFORE"
  "fewbody|64|self-sizes to cores/2 concurrent problems; the only test that uses a whole node"
)

echo "=== $(date) N-body validation ==="
echo "    repo:   $REPO"
echo "    branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "    dirty:  $(git status --porcelain | grep -vc '^??') tracked file(s) modified"
echo

mkdir -p nbody_val
JOBIDS=()
for entry in "${TESTS[@]}"; do
    IFS='|' read -r name cores note <<< "$entry"
    [ -n "$KEXPR" ] && [[ "$name" != *"$KEXPR"* ]] && continue
    f="test/$name/test_$name.py"
    [ -f "$f" ] || { echo "  SKIP $name: no $f"; continue; }

    cat > "nbody_val/$name.sbatch" <<SLURM
#!/bin/bash
#SBATCH --job-name=nb_$name
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cores
#SBATCH --time=$WALLTIME
#SBATCH --output=$REPO/nbody_val/${name}-%j.out
#SBATCH --error=$REPO/nbody_val/${name}-%j.err
set -uo pipefail
cd $REPO

echo "=== \$(date) $name on \$(hostname) ==="
echo "    branch \$(git rev-parse --abbrev-ref HEAD) @ \$(git rev-parse --short HEAD)"
echo "    $note"
echo

module --force purge
# texlive is required, not optional: some plots use matplotlib's usetex path and shell out to
# 'latex'. Without it a test FAILS after GIZMO ran correctly and every assertion passed.
module load modules openmpi gsl hdf5/mpi-1.12.3 texlive

export SYSTYPE=RUSTY_ICELAKE
# Must not be inherited: an inherited binding pins every rank to the same low-numbered cores,
# which is silent and costs more than an order of magnitude.
export OMP_PROC_BIND=false
unset OMP_PLACES
export GIZMO_TEST_TIMEOUT=\${GIZMO_TEST_TIMEOUT:-7200}

# The build takes .gizmo_build.lock; sibling jobs wait there and then run in parallel.
\$HOME/python_work/bin/pytest $f -v -s -ra
rc=\$?

echo
echo "=== \$(date) $name done (exit \$rc) ==="
echo "  NOTE binary and triple rest at 'xfailed', not 'passed': both still record the"
echo "  4th-order secular residual as an expected failure. FAILED is a real regression."
exit \$rc
SLURM
    chmod +x "nbody_val/$name.sbatch"

    if [ "$SUBMIT" -eq 1 ]; then
        jid=$(sbatch --parsable "nbody_val/$name.sbatch" 2>&1)
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
            JOBIDS+=("$jid"); printf "  submitted %-28s job %-9s %s cores\n" "$name" "$jid" "$cores"
        else
            echo "  FAILED to submit $name: $jid"
        fi
    else
        printf "  wrote     %-28s nbody_val/%-28s %s cores\n" "$name" "$name.sbatch" "$cores"
    fi
done

echo
if [ "$SUBMIT" -eq 1 ] && [ ${#JOBIDS[@]} -gt 0 ]; then
    echo "  watch:   squeue -u \$USER -n $(IFS=,; echo "${JOBIDS[*]}" | sed 's/[0-9]*/nb_*/' )"
    echo "  logs:    nbody_val/<test>-<jobid>.out"
    echo "  results: for f in nbody_val/*.out; do echo \"== \$f\"; grep -aE '^(PASSED|FAILED|XFAIL|XPASS)' \$f; done"
else
    echo "  not submitted. submit with:  for f in nbody_val/*.sbatch; do sbatch \$f; done"
fi
echo "=== $(date) done ==="
