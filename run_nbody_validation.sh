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
# Each job asks for a whole exclusive node. That is not about core count -- these are small
# problems -- it is so no job's timing or memory traffic depends on a co-tenant, and so a long
# test cannot be slowed by a sibling that happens to land on the same node.
set -uo pipefail

SUBMIT=1
KEXPR=
WALLTIME=8:00:00
while getopts "nk:w:h" o; do case $o in
  n) SUBMIT=0 ;; k) KEXPR=$OPTARG ;; w) WALLTIME=$OPTARG ;;
  h) sed -n '2,22p' "$0"; exit 0 ;; *) exit 2 ;;
esac; done

REPO=$(cd "$(dirname "$0")" && pwd); cd "$REPO"

# test | ranks | omp | note
TESTS=(
  "binary|1|1|2 sinks e=0.9 q=0.1, 1000 orbits + order sweep; guards the timestep normalization"
  "triple|1|1|6-bin hierarchy, 50 outer orbits + order sweep; guards the source prediction"
  "plummer_binaries|4|2|512 sinks in a cluster; synced-state energy, now with a direct potential"
  "plummer_binaries_realistic|4|2|realistic IMF/period/eccentricity population; NEVER RUN BEFORE"
  "fewbody|2|1|48 problems x 4 variants; read the direct_* variants, tree_* fail for tree-force reasons"
)

echo "=== $(date) N-body validation ==="
echo "    repo:   $REPO"
echo "    branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "    dirty:  $(git status --porcelain | grep -vc '^??') tracked file(s) modified"
echo

mkdir -p nbody_val
JOBIDS=()
for entry in "${TESTS[@]}"; do
    IFS='|' read -r name ranks omp note <<< "$entry"
    [ -n "$KEXPR" ] && [[ "$name" != *"$KEXPR"* ]] && continue
    f="test/$name/test_$name.py"
    [ -f "$f" ] || { echo "  SKIP $name: no $f"; continue; }

    cat > "nbody_val/$name.sbatch" <<SLURM
#!/bin/bash
#SBATCH --job-name=nb_$name
#SBATCH --partition=cca
#SBATCH --constraint=icelake
#SBATCH --nodes=1
#SBATCH --exclusive
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
            JOBIDS+=("$jid"); printf "  submitted %-28s job %s  (%s ranks x %s omp)\n" "$name" "$jid" "$ranks" "$omp"
        else
            echo "  FAILED to submit $name: $jid"
        fi
    else
        printf "  wrote     %-28s nbody_val/%s.sbatch\n" "$name" "$name"
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
