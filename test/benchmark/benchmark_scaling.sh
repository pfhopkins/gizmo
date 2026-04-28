#!/bin/bash
# benchmark_scaling.sh — Run GPU vs tree-walk benchmarks at multiple particle counts.
#
# Generates ICs, builds both binaries, creates per-resolution run directories,
# submits all SLURM jobs. Run benchmark_analyze_scaling.sh after jobs complete.
#
# Usage (on Vista login node):
#   ./test/benchmark/benchmark_scaling.sh [walltime]
#
# Default walltime: 00:30:00
# Creates: bench_gpu_NNN/ and bench_tree_NNN/ for each resolution NNN

set -eu

GIZMO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GIZMO_ROOT"

WALLTIME="${1:-00:30:00}"
NJOBS=8
BENCH_DIR="$GIZMO_ROOT/test/benchmark"
ACCOUNT="TG-NAIRR260139"
PARTITION="gh-dev"

# Resolutions to test (N_per_side: 30^3=27k, 50^3=125k, 80^3=512k)
SIZES="30 50 80"

echo "============================================================"
echo "GIZMO Scaling Benchmark — $(date)"
echo "Resolutions: $SIZES"
echo "============================================================"

# --- Generate ICs ---
echo ">>> Generating benchmark ICs..."
cd "$BENCH_DIR"
python3 make_benchmark_ics.py $SIZES
cd "$GIZMO_ROOT"

# --- Create params file template ---
# Based on gmc_cooling but with generic IC filename placeholder
PARAMS_TEMPLATE="$BENCH_DIR/bench_params.txt"
cat > "$PARAMS_TEMPLATE" << 'PARAMS_EOF'
InitCondFile  ICFILE_PLACEHOLDER
OutputDir     output
ICFormat    3
SnapFormat  3
RestartFile                 restart
SnapshotFileBase            snapshot
OutputListOn                0
OutputListFilename          output_times.txt
NumFilesPerSnapshot         1
NumFilesWrittenInParallel   1
TimeOfFirstSnapshot     0.
TimeBetSnapshot         10.
TimeLimitCPU            172800
CpuTimeBetRestartFile   7200
TimeBegin   0.0
TimeMax 0.001
MaxSizeTimestep         0.001
MinSizeTimestep         1.0e-15
UnitLength_in_cm            3.09e+18
UnitMass_in_g               1.99e+33
UnitVelocity_in_cm_per_s    1.00e+05
UnitMagneticField_in_gauss  1.00e+00
GravityConstantInternal     1e-100
ComovingIntegrationOn   0
BoxSize                 1.00e+02
Omega0                  0.
OmegaLambda             0.
OmegaBaryon             0.
HubbleParam             1.
MaxMemSize          16000
PartAllocFactor     5.0
BufferSize          500
TreeDomainUpdateFrequency   0.005
InitGasTemp     1e4
MinGasTemp      2.73
DesNumNgb               32
SofteningGas    1e-10
SofteningHalo   0.020
SofteningDisk   0.150
SofteningBulge  0.500
SofteningStars  1.00e-01
SofteningBndry  1.00e-01
SofteningGasMaxPhys     0.0005
SofteningHaloMaxPhys    0.01
SofteningDiskMaxPhys    0.075
SofteningBulgeMaxPhys   0.25
SofteningStarsMaxPhys   0.0005
SofteningBndryMaxPhys   0.0005
ConductionCoeff  1.0
ShearViscosityCoeff     1.0
BulkViscosityCoeff      1.0
CritPhysDensity     1.00e+02
SfEffPerFreeFall    1.0
InitMetallicity             1.00e+00
InitStellarAge              0.0
SNeIIEnergyFrac             1.0
GasReturnFraction           1.0
GasReturnEnergy             1.0
InterstellarRadiationFieldStrength  1.00e+00
DivBcleaningParabolicSigma   1.0
DivBcleaningHyperbolicSigma  1.0
PARAMS_EOF

# Save original Config.sh
cp -f Config.sh Config.sh.bench_backup 2>/dev/null || true

# --- Determine base config ---
TEST_DIR="test/gmc_cooling"
BASE_CONFIG=$(cat "$TEST_DIR/Config.sh" | grep -v "^#" | grep -v "^$")

# ================================================================
# BUILD GPU binary
# ================================================================
echo ""
echo ">>> Building GPU binary..."
{
    echo "$BASE_CONFIG"
} > Config.sh
make clean > /dev/null 2>&1 && make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "ERROR: GPU build failed"; exit 1; fi
cp GIZMO "$BENCH_DIR/GIZMO_gpu"

# ================================================================
# BUILD tree-walk binary
# ================================================================
echo ">>> Building tree-walk binary..."
echo "$BASE_CONFIG" > Config.sh
make clean > /dev/null 2>&1 && make -j${NJOBS} > /dev/null 2>&1
if [[ $? -ne 0 ]]; then echo "ERROR: Tree-walk build failed"; exit 1; fi
cp GIZMO "$BENCH_DIR/GIZMO_tree"

# Restore
cp -f Config.sh.bench_backup Config.sh 2>/dev/null || true

# ================================================================
# CREATE RUN DIRECTORIES + SUBMIT
# ================================================================
echo ""
echo ">>> Creating run directories and submitting jobs..."

for N in $SIZES; do
    IC_FILE="bench_${N}_ics.hdf5"
    NPART=$(python3 -c "print(${N}**3)")

    for MODE in gpu tree; do
        DIR="$BENCH_DIR/bench_${MODE}_${N}"
        rm -rf "$DIR"
        mkdir -p "$DIR/output"

        # Copy binary and IC
        cp "$BENCH_DIR/GIZMO_${MODE}" "$DIR/GIZMO"
        cp "$BENCH_DIR/$IC_FILE" "$DIR/"

        # Runtime data files
        for f in TREECOOL spcool_tables ewald_spc_table_64_dbl.dat; do
            if [[ -e "$BENCH_DIR/$f" ]]; then cp -r "$BENCH_DIR/$f" "$DIR/";
            elif [[ -e "$GIZMO_ROOT/$f" ]]; then cp -r "$GIZMO_ROOT/$f" "$DIR/"; fi
        done

        # Params file with correct IC filename
        sed "s|ICFILE_PLACEHOLDER|$IC_FILE|" "$PARAMS_TEMPLATE" > "$DIR/params.txt"

        # SLURM script
        cat > "$DIR/run.bsub" << SLURM_EOF
#!/bin/bash
#SBATCH -J b_${MODE}_${N}
#SBATCH -p $PARTITION -N 1 --ntasks-per-node 1 -t $WALLTIME -A $ACCOUNT
#SBATCH -o slurm.out -e slurm.err
source \$HOME/.bashrc
ibrun ./GIZMO ./params.txt 1>gizmo.out 2>gizmo.err
SLURM_EOF

        echo "  $DIR: N=$NPART ($MODE)"
        cd "$DIR" && sbatch run.bsub && cd "$GIZMO_ROOT"
    done
done

# Cleanup temp binaries
rm -f "$BENCH_DIR/GIZMO_gpu" "$BENCH_DIR/GIZMO_tree"

echo ""
echo "============================================================"
echo "All jobs submitted. Check: squeue -u \$USER"
echo "After completion: ./test/benchmark/benchmark_analyze_scaling.sh"
echo "============================================================"
