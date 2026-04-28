#!/bin/bash
# benchmark_setup.sh — Build GPU + tree-walk binaries, create run directories,
# generate SLURM submission scripts, and optionally submit them.
#
# Usage (on Vista login node):
#   ./test/benchmark/benchmark_setup.sh [nranks] [walltime]
#
# Defaults: nranks=1, walltime=00:15:00
# Creates: bench_gpu/ and bench_tree/ directories ready to submit.

set -eu

GIZMO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GIZMO_ROOT"

NRANKS="${1:-1}"
WALLTIME="${2:-00:15:00}"
NJOBS=8
PARAMS_FILE="test/gmc_cooling/gmc_cooling_quick.params"
IC_FILE="test/gmc_cooling/gmc_cooling_ics.hdf5"
BENCH_DIR="$GIZMO_ROOT/test/benchmark"
ACCOUNT="TG-NAIRR260139"
PARTITION="gh-dev"

# Number of nodes (1 rank per node for GPU, can pack more for CPU)
NNODES=$(( (NRANKS + 0) ))  # 1 rank per node for simplicity

echo "============================================================"
echo "GIZMO Benchmark Setup — $(date)"
echo "Ranks: $NRANKS  Walltime: $WALLTIME"
echo "============================================================"

# Save original Config.sh
cp -f Config.sh Config.sh.bench_backup 2>/dev/null || true

# --- Determine base config from the test ---
TEST_DIR="$(dirname "$PARAMS_FILE")"
if [[ -f "$TEST_DIR/Config.sh" ]]; then
    BASE_CONFIG=$(cat "$TEST_DIR/Config.sh" | grep -v "^#" | grep -v "^$")
else
    BASE_CONFIG="SINGLE_STAR_STARFORGE_DEFAULTS
MAGNETIC
COOLING
METALS
COOL_METAL_LINES_BY_SPECIES
BOX_PERIODIC
GRAVITY_NOT_PERIODIC
ADAPTIVE_TREEFORCE_UPDATE=0.0625"
fi

# ================================================================
# BUILD 1: GPU path
# ================================================================
echo ""
echo ">>> Building GPU path..."
{
    echo "$BASE_CONFIG"
} > Config.sh
make clean > /dev/null 2>&1
make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "ERROR: GPU build failed"; exit 1; fi

# Create GPU run directory
GPU_DIR="$BENCH_DIR/bench_gpu"
rm -rf "$GPU_DIR"
mkdir -p "$GPU_DIR/output"
cp GIZMO "$GPU_DIR/"
cp "$PARAMS_FILE" "$GPU_DIR/params.txt"
cp "$IC_FILE" "$GPU_DIR/"
# Runtime data files
for f in TREECOOL spcool_tables ewald_spc_table_64_dbl.dat; do
    if [[ -e "$BENCH_DIR/$f" ]]; then cp -r "$BENCH_DIR/$f" "$GPU_DIR/";
    elif [[ -e "$GIZMO_ROOT/$f" ]]; then cp -r "$GIZMO_ROOT/$f" "$GPU_DIR/"; fi
done

# Generate SLURM script
cat > "$GPU_DIR/run.bsub" << SLURM_EOF
#!/bin/bash
#SBATCH -J bench_gpu
#SBATCH -p $PARTITION -N $NNODES --ntasks-per-node $NRANKS -t $WALLTIME -A $ACCOUNT
#SBATCH -o bench_gpu.out -e bench_gpu.err
source \$HOME/.bashrc
ibrun ./GIZMO ./params.txt 1>gizmo.out 2>gizmo.err
SLURM_EOF

echo "  GPU binary + run directory: $GPU_DIR"

# ================================================================
# BUILD 2: Tree-walk path
# ================================================================
echo ">>> Building tree-walk path..."
echo "$BASE_CONFIG" > Config.sh
make clean > /dev/null 2>&1
make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "ERROR: Tree-walk build failed"; exit 1; fi

# Create tree-walk run directory
TREE_DIR="$BENCH_DIR/bench_tree"
rm -rf "$TREE_DIR"
mkdir -p "$TREE_DIR/output"
cp GIZMO "$TREE_DIR/"
cp "$PARAMS_FILE" "$TREE_DIR/params.txt"
cp "$IC_FILE" "$TREE_DIR/"
for f in TREECOOL spcool_tables ewald_spc_table_64_dbl.dat; do
    if [[ -e "$BENCH_DIR/$f" ]]; then cp -r "$BENCH_DIR/$f" "$TREE_DIR/";
    elif [[ -e "$GIZMO_ROOT/$f" ]]; then cp -r "$GIZMO_ROOT/$f" "$TREE_DIR/"; fi
done

cat > "$TREE_DIR/run.bsub" << SLURM_EOF
#!/bin/bash
#SBATCH -J bench_tree
#SBATCH -p $PARTITION -N $NNODES --ntasks-per-node $NRANKS -t $WALLTIME -A $ACCOUNT
#SBATCH -o bench_tree.out -e bench_tree.err
source \$HOME/.bashrc
ibrun ./GIZMO ./params.txt 1>gizmo.out 2>gizmo.err
SLURM_EOF

echo "  Tree binary + run directory: $TREE_DIR"

# Restore original Config.sh
cp -f Config.sh.bench_backup Config.sh 2>/dev/null || true

# ================================================================
# SUBMIT
# ================================================================
echo ""
echo ">>> Submitting jobs..."
cd "$GPU_DIR" && sbatch run.bsub && cd "$GIZMO_ROOT"
cd "$TREE_DIR" && sbatch run.bsub && cd "$GIZMO_ROOT"

echo ""
echo "============================================================"
echo "Jobs submitted. Check status with: squeue -u \$USER"
echo "When complete, run:  ./test/benchmark/benchmark_analyze.sh"
echo "============================================================"
