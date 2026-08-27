#!/bin/bash
# plummer_binaries + plummer_binaries_realistic against the fixed tree.
#
#   sbatch run_plummer_genoa.sh
#
# Why now: 256 hard binaries at np=2 is the most detachment-exposed production-shaped workload
# we have (constant boundary traffic from close pairs), and the reconciled upstream test finally
# measures what matters -- SYNCED-state energy (hard 1% assert, no xfail) plus the momentum-drift
# panel that actually discriminates tree from direct. plummer_binaries_realistic (sampled IMF,
# eccentric orbits, 10 crossings) has never run on kokkos at all.
#
# Both tests were reconciled to origin/starforge_dev 93e62a83 wholesale; the kokkos half-mass-only
# Lagrange policy and the 1000 AU geometry had already converged upstream, so nothing local was
# kept back. The old snapshot-energy xfail is gone -- if energy fails now it fails loudly, which
# post-detachment-fix is the point.
#
#SBATCH --job-name=plummer_pair
#SBATCH --partition=cca
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=genoa
#SBATCH --time=8:00:00
#SBATCH --output=plummer_pair-%j.out
#SBATCH --error=plummer_pair-%j.err
set -uo pipefail

# Runs against the tree it was SUBMITTED from (sbatch copies the script into spool, so $0
# is useless); GIZMO_TREE overrides. Enables one-job-per-worktree concurrency.
TREE="${GIZMO_TREE:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=/mnt/home/mgrudic/python_work/bin/python
export GIZMO_TEST_TIMEOUT="${GIZMO_TEST_TIMEOUT:-14400}"
export OMP_PROC_BIND=false
unset OMP_PLACES

module --force purge
module load modules/2.4-20250724 gcc openmpi gsl hdf5/mpi-1.12.3 fftw texlive
export SYSTYPE=RUSTY_GENOA
export KOKKOS_PATH="${KOKKOS_PATH:-$HOME/opt/kokkos-openmp}"

cd "$TREE" || exit 1
echo "=== node $(hostname)  cores=$(nproc)  HEAD $(git rev-parse --short HEAD)  (uncommitted changes INCLUDED)"
if [ ! -f "$KOKKOS_PATH/include/Kokkos_Core.hpp" ]; then
    ./scripts/build_kokkos_rusty.sh "$KOKKOS_PATH" || exit 1
fi

rc=0
"$PY" -u -m pytest test/plummer_binaries -ra -s -v --tb=short || rc=$?
"$PY" -u -m pytest test/plummer_binaries_realistic -ra -s -v --tb=short || rc=$?
exit $rc
