#!/bin/bash
# Phase 0 single-rank-count sbatch template. Generate concrete sbatch via
# scripts/gen_phase0_sbatch.sh which substitutes NRANKS and OMP_NUM_THREADS.
# (Or copy this and hand-edit the four 2 / 36 slots.)
#
# Vista note: ibrun uses the SLURM allocation directly. Do NOT pass `-n`
# to ibrun on Vista — prterun interprets it as a separate flag and the
# job fails with rc=66 ("prterun was unable to find the specified
# executable file"). Set rank count via #SBATCH --ntasks-per-node.

#SBATCH -J phase0_np2
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 36
#SBATCH -t 00:20:00
#SBATCH -A TG-NAIRR260139
#SBATCH -o phase0_np2_%j.out
#SBATCH -e phase0_np2_%j.err

set -u
source $HOME/.bashrc

export GIZMO_PHASE0_DIAG=1
unset  GIZMO_NGL_DUMP_DIR
export OMP_NUM_THREADS=36

cd "$SLURM_SUBMIT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="${SLURM_SUBMIT_DIR}/phase0_${STAMP}_jid${SLURM_JOB_ID}_np2"
mkdir -p "$RUN/output_phase0"
cp fire.params Config.sh fire_ics.hdf5 TREECOOL ewald_spc_table_64_dbl.dat "$RUN/"
cp -r spcool_tables "$RUN/"
sed -i "s|output_local_smoke|output_phase0|g" "$RUN/fire.params"
cp ../../GIZMO "$RUN/GIZMO"

cd "$RUN"
ibrun ./GIZMO fire.params 2 > run.log 2>&1
RC=$?
echo "[phase0] rc=$RC  PHASE0 lines: $(grep -cE '^PHASE0_(NGL|GHOST)' run.log || echo 0)"
echo "[phase0] DIAG_NGL lines : $(grep -c 'DIAG_NGL' run.log || echo 0)"
echo "[phase0] data in $RUN"
