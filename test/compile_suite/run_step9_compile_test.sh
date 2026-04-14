#!/bin/bash
# run_step9_compile_test.sh — Focused compilation tests for Step 9 GPU loop ports.
#
# Tests the specific code paths added in Step 9:
#   - DiffFilter + DynamicDiff (TURB_DIFF_DYNAMIC)
#   - Gradient symlist fix (TURB_DIFF_DYNAMIC + GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)
#   - Grain drag (GRAIN_FLUID, PIC_MHD)
#   - Transport flux subcycling (TRANSPORT_SUBCYCLE + RT)
#
# Each config is tested both WITH and WITHOUT GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
# to verify both GPU and tree-walk paths compile.
#
# Usage:
#   ./test/compile_suite/run_step9_compile_test.sh          # local (Kokkos OpenMP)
#   ./test/compile_suite/run_step9_compile_test.sh --gpu     # Vista (CUDA)
#
# After all configs pass compilation, run these test problems for validation:
#   1. test/gmc_cooling         — base hydro+cooling (density/gradient/hydro/cooling GPU)
#   2. test/gmc_cooling_rt      — RT transport subcycling validation
#   3. test/dustywave           — grain drag force validation

set -u

GIZMO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$GIZMO_ROOT"

NJOBS=8
GPU_MODE=0
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU_MODE=1 ;;
    esac
done

# Save originals
cp -f Config.sh Config.sh.step9_backup 2>/dev/null || true
cp -f Makefile.systype Makefile.systype.step9_backup 2>/dev/null || true

if [[ $GPU_MODE -eq 1 ]]; then
    echo 'SYSTYPE="Vista"' > Makefile.systype
    echo "[step9] GPU mode: SYSTYPE=Vista"
else
    grep -q 'MacBookCellar_Kokkos' Makefile.systype.step9_backup 2>/dev/null && \
        cp -f Makefile.systype.step9_backup Makefile.systype || \
        echo 'SYSTYPE="MacBookCellar_Kokkos"' > Makefile.systype
    echo "[step9] CPU/OpenMP mode: $(grep SYSTYPE Makefile.systype)"
fi

PASS=0
FAIL=0
TOTAL=0

# ============================================================
# Step 9 focused configs — GPU path (with GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)
# and tree-walk path (without) for each new feature
# ============================================================
CONFIGS=(
    # --- Base regression: STARFORGE (GPU path) ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY SINGLE_STAR_STARFORGE_DEFAULTS MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES BOX_PERIODIC GRAVITY_NOT_PERIODIC ADAPTIVE_TREEFORCE_UPDATE=0.0625"

    # --- TURB_DIFF_DYNAMIC: DiffFilter + DynamicDiff + gradient symlist fix ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY METALS TURB_DIFF_METALS TURB_DIFF_ENERGY TURB_DIFF_VELOCITY TURB_DIFF_DYNAMIC OUTPUT_TURB_DIFF_DYNAMIC_ERROR"
    "METALS TURB_DIFF_METALS TURB_DIFF_ENERGY TURB_DIFF_VELOCITY TURB_DIFF_DYNAMIC OUTPUT_TURB_DIFF_DYNAMIC_ERROR"
    # TURB_DIFF without DYNAMIC (should still compile, regression)
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY METALS TURB_DIFF_METALS TURB_DIFF_ENERGY TURB_DIFF_VELOCITY"

    # --- Grain drag: basic ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY GRAIN_FLUID"
    "GRAIN_FLUID"
    # Grain drag: with Lorentz + Epstein-Stokes + COOLING (ThermalProperties path)
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES GRAIN_FLUID GRAIN_BACKREACTION GRAIN_LORENTZFORCE GRAIN_EPSTEIN_STOKES=1"
    "MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES GRAIN_FLUID GRAIN_BACKREACTION GRAIN_LORENTZFORCE GRAIN_EPSTEIN_STOKES=1"
    # Grain drag: with collisions
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC GRAIN_FLUID GRAIN_BACKREACTION GRAIN_LORENTZFORCE GRAIN_COLLISIONS"
    "MAGNETIC GRAIN_FLUID GRAIN_BACKREACTION GRAIN_LORENTZFORCE GRAIN_COLLISIONS"
    # PIC_MHD (defines GRAIN_FLUID + PIC physics)
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC PIC_MHD PIC_SPEEDOFLIGHT_REDUCTION=0.01"
    "MAGNETIC PIC_MHD PIC_SPEEDOFLIGHT_REDUCTION=0.01"

    # --- Transport flux subcycling ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES RADTRANSFER RT_SOLVER_EXPLICIT RT_EVOLVE_ENERGY RT_EVOLVE_FLUX RT_INFRARED TRANSPORT_SUBCYCLE=10"
    "MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES RADTRANSFER RT_SOLVER_EXPLICIT RT_EVOLVE_ENERGY RT_EVOLVE_FLUX RT_INFRARED TRANSPORT_SUBCYCLE=10"
    # Transport flux + cooling subcycling
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES RADTRANSFER RT_SOLVER_EXPLICIT RT_EVOLVE_ENERGY RT_EVOLVE_FLUX RT_INFRARED TRANSPORT_SUBCYCLE=10 TRANSPORT_SUBCYCLE_COOLING"
    # RT without TRANSPORT_SUBCYCLE (regression)
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY MAGNETIC COOLING METALS COOL_METAL_LINES_BY_SPECIES RADTRANSFER RT_SOLVER_EXPLICIT RT_EVOLVE_ENERGY RT_EVOLVE_FLUX RT_INFRARED"

    # --- RT M1 (tests RT with non-FLD solver) ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY GALSF METALS RT_M1 RT_SOURCES=1+16+32 RT_XRAY=3 RT_LYMAN_WERNER RT_PHOTOELECTRIC RT_NUV RT_FREEFREE RT_RAD_PRESSURE_OUTPUT"

    # --- STARFORGE + RT (combined, real-world config) ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY SINGLE_STAR_STARFORGE_DEFAULTS COOLING MAGNETIC RT_M1 RT_INFRARED RT_SOURCES=1+16+32 RT_LYMAN_WERNER RT_PHOTOELECTRIC RT_NUV RT_FREEFREE RT_CHEM_PHOTOION=2 EOS_SUBSTELLAR_ISM"

    # --- Dustywave test problem config (exact match) ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY HYDRO_MESHLESS_FINITE_MASS BOX_PERIODIC BOX_SPATIAL_DIMENSION=1 EOS_GAMMA=(5./3.) EOS_ENFORCE_ADIABAT=(3./5.) SELFGRAVITY_OFF GRAIN_FLUID GRAIN_BACKREACTION OUTPUT_IN_DOUBLEPRECISION DEVELOPER_MODE"

    # --- gmc_cooling_rt test problem config (exact match) ---
    "GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY SINGLE_STAR_STARFORGE_DEFAULTS MAGNETIC COOLING SINGLE_STAR_FB_RAD METALS COOL_METAL_LINES_BY_SPECIES BOX_PERIODIC GRAVITY_NOT_PERIODIC ADAPTIVE_TREEFORCE_UPDATE=0.0625 RT_SPEEDOFLIGHT_REDUCTION=1e-4"
)

echo "============================================================"
echo "Step 9 GPU Port Compile Test — $(date)"
echo "Configs: ${#CONFIGS[@]}"
echo "GPU mode: $GPU_MODE"
echo "============================================================"

for i in "${!CONFIGS[@]}"; do
    flags="${CONFIGS[$i]}"
    TOTAL=$((TOTAL + 1))
    label="$(echo "$flags" | tr ' ' '\n' | head -3 | tr '\n' '+' | sed 's/+$//')"

    printf "[%2d/%d] %-70s " "$((i+1))" "${#CONFIGS[@]}" "$label..."

    # Write Config.sh
    {
        for flag in $flags; do
            echo "$flag"
        done
    } > Config.sh

    # Clean and build
    make clean > /dev/null 2>&1
    errfile="/tmp/step9_compile_$((i+1)).stderr"
    if make -j${NJOBS} > /dev/null 2> "$errfile"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        echo "  Errors:"
        grep -iE 'error:' "$errfile" 2>/dev/null | head -5 | sed 's/^/    /'
    fi
done

# Restore originals
cp -f Config.sh.step9_backup Config.sh 2>/dev/null || true
cp -f Makefile.systype.step9_backup Makefile.systype 2>/dev/null || true
make clean > /dev/null 2>&1
rm -f Config.sh.step9_backup Makefile.systype.step9_backup

echo ""
echo "============================================================"
echo "RESULTS: $PASS pass, $FAIL fail out of $TOTAL total"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi

echo ""
echo "All compilation tests passed. Suggested validation test problems:"
echo ""
echo "1. gmc_cooling (base hydro+cooling regression):"
echo "   cd test/gmc_cooling && mpirun -np 2 ../../GIZMO gmc_cooling_quick.params"
echo ""
echo "2. gmc_cooling_rt (RT + transport subcycling):"
echo "   cd test/gmc_cooling_rt && mpirun -np 2 ../../GIZMO gmc_cooling_rt.params"
echo "   python3 test_gmc_cooling_rt.py  # checks nH-T, nH-urad, nH-Tdust curves"
echo ""
echo "3. dustywave (grain drag force):"
echo "   cd test/dustywave && mpirun -np 1 ../../GIZMO dustywave.params"
echo "   python3 test_dustywave.py  # checks v(x) against analytic solution"
echo ""
echo "For multi-rank GPU validation on Vista, use --gpu flag and run each"
echo "test problem with 4-8 ranks to stress ghost exchange + CSR paths."
exit 0
