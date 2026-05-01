#!/usr/bin/env bash
# gizmo_launch.sh — Launch a GIZMO benchmark run with deterministic OMP/Kokkos pinning.
#
# Usage (from inside the per-run directory):
#   gizmo_launch.sh <binary> <params> <nranks> <nthreads_per_rank> [is_kokkos=1|0]
#
# Writes pinning.log next to the run, with the chosen env and verification info.
# Aborts if (ranks * threads) exceeds available cores, or if intent disagrees
# with what the OS reports.

set -euo pipefail

BIN="$1"
PARAMS="$2"
NRANKS="$3"
NTHREADS="$4"
IS_KOKKOS="${5:-1}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not executable: ${BIN}" >&2; exit 2; }
[[ -f "${PARAMS}" ]] || { echo "ERROR: params not found: ${PARAMS}" >&2; exit 2; }

# Detect available cores
case "$(uname -s)" in
    Darwin) AVAIL_CORES="$(sysctl -n hw.activecpu)" ;;
    Linux)  AVAIL_CORES="$(nproc)" ;;
    *)      AVAIL_CORES=0 ;;
esac

REQUESTED=$(( NRANKS * NTHREADS ))

# Hard fail on oversubscription (would invalidate timings)
if (( AVAIL_CORES > 0 && REQUESTED > AVAIL_CORES )); then
    echo "ERROR: ranks*threads = ${REQUESTED} > available cores = ${AVAIL_CORES}" >&2
    echo "       refusing to oversubscribe a benchmark run." >&2
    exit 3
fi

export OMP_NUM_THREADS="${NTHREADS}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset KMP_AFFINITY 2>/dev/null || true

# Kokkos OpenMP backend reads OMP_NUM_THREADS automatically, and GIZMO's main
# does not strip Kokkos args before its own atoi(argv[2]). To keep argv clean,
# we do NOT pass --kokkos-num-threads here. IS_KOKKOS retained for the log.
KOKKOS_ARGS=""

# Detect launcher: prefer ibrun on TACC, else mpirun
if command -v ibrun >/dev/null 2>&1; then
    LAUNCHER=(ibrun)
else
    LAUNCHER=(mpirun -np "${NRANKS}")
fi

# Write pinning log BEFORE launch so we have it even if GIZMO crashes
cat > pinning.log <<LOG
[BENCH-PIN] $(date -u +%Y-%m-%dT%H:%M:%SZ)
binary           = ${BIN}
params           = ${PARAMS}
nranks_intended  = ${NRANKS}
nthreads_intended= ${NTHREADS}
ranks_x_threads  = ${REQUESTED}
avail_cores_os   = ${AVAIL_CORES}
OMP_NUM_THREADS  = ${OMP_NUM_THREADS}
OMP_PROC_BIND    = ${OMP_PROC_BIND}
OMP_PLACES       = ${OMP_PLACES}
is_kokkos        = ${IS_KOKKOS}
launcher         = ${LAUNCHER[*]}
kokkos_args      = ${KOKKOS_ARGS}
hostname         = $(hostname)
uname            = $(uname -a)
LOG

echo "[gizmo_launch] OMP_NUM_THREADS=${OMP_NUM_THREADS} ranks=${NRANKS} cores_avail=${AVAIL_CORES}"
echo "[gizmo_launch] launching: ${LAUNCHER[*]} ${BIN} ${PARAMS} ${KOKKOS_ARGS}"

# Run, capture stdout/stderr separately so [BENCH] stderr lines can be greped clean
"${LAUNCHER[@]}" "${BIN}" "${PARAMS}" ${KOKKOS_ARGS} >gizmo.out 2>gizmo.err
RC=$?

echo "exit_code        = ${RC}" >> pinning.log
echo "wallclock_end    = $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> pinning.log
exit "${RC}"
