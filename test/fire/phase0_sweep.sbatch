#!/bin/bash
#SBATCH -J phase0_fire_sweep
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 9
#SBATCH -t 00:45:00
#SBATCH -A TG-NAIRR260139
#SBATCH -o phase0_sweep_%j.out
#SBATCH -e phase0_sweep_%j.err

set -u
source $HOME/.bashrc

# Phase 0 sweep: -np 2, -np 4, -np 8 on a single 1-node allocation.
# Captures Nactive histogram (rows 1+2) AND rank-scaling tiny-N (row 5).
# Each ibrun runs ~10 min — enough to clear init + tens of NGL calls
# at varying Nactive.

export GIZMO_PHASE0_DIAG=1
unset  GIZMO_NGL_DUMP_DIR     # forensic dumps off; PHASE0 lines only

cd "$SLURM_SUBMIT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
RUNDIR="${SLURM_SUBMIT_DIR}/phase0_${STAMP}_jid${SLURM_JOB_ID}"
mkdir -p "$RUNDIR"
echo "[phase0] RUNDIR=$RUNDIR"

# Stage inputs into per-rank-count subdirs so output_local_smoke don't collide
for NRANKS in 2 4 8; do
    SUB="$RUNDIR/np${NRANKS}"
    mkdir -p "$SUB/output_phase0"
    cp fire.params Config.sh fire_ics.hdf5 TREECOOL ewald_spc_table_64_dbl.dat "$SUB/"
    cp -r spcool_tables "$SUB/"
    # Patch params to write into the per-rank output dir
    sed -i "s|output_local_smoke|output_phase0|g" "$SUB/fire.params"

    # OMP threads = ceil(72 / NRANKS) — keep core utilization roughly constant
    case $NRANKS in
        2) OMPT=36 ;;
        4) OMPT=18 ;;
        8) OMPT=9  ;;
    esac
    export OMP_NUM_THREADS=$OMPT

    LOG="$SUB/run_np${NRANKS}.log"
    echo "[phase0] launching np=${NRANKS} omp=${OMPT}  log=$LOG"
    cd "$SUB"
    ibrun -n $NRANKS ../../GIZMO fire.params 2 > "$LOG" 2>&1
    RC=$?
    echo "[phase0] np=${NRANKS} rc=$RC"
    cd "$SLURM_SUBMIT_DIR"

    # Quick post-summary so we don't have to wait for harvest
    PHASE0_LINES=$(grep -cE "^PHASE0_(NGL|GHOST)" "$LOG" 2>/dev/null || echo 0)
    echo "[phase0] np=${NRANKS} PHASE0 lines: $PHASE0_LINES"
done

echo "[phase0] sweep complete; data in $RUNDIR"
