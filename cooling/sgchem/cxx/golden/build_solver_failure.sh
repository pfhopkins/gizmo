#!/bin/bash
# Solver-failure test (WLM runs 2026-11-17; D-58, D-59): golden/test_solver_failure.cc, nets 5 and 17, two builds:
#   default   phases A + H (logged floor cells, TNOCHEM hand-off grid), C++ vs Fortran DVODE; CSV at %.17g
#   forced    phase R: -DTEST_SGCHEM_FORCED_SOLVER_FAILURE on the driver and the test (retry, TNOCHEM finish,
#             abort, dump replay)
#
# usage: build_solver_failure.sh TREE OUT [SUNDIALS_DIR]
#   TREE  a COPY of the gizmo tree: net 5 or 17 GIZMO_config.h at its root and the SGChem tables
#         (eh2-table.txt, energy-table-x.txt, H2-cooling-ratios.dat, wss_z_collis.txt) in cooling/sgchem
#   OUT   objects, binaries and results (OUT/<build>/result.txt, OUT/default/AH.csv)
#   BUILDS="default forced" (default), EXTRA_CXXFLAGS for both builds (e.g. -DTEST_PARENT_DRIVER, default only)
#   REFERENCE=1: default build only, both backends at rtol 1e-8 (C++ CHEM_RTOL, Fortran RTOL), writes the converged
#   A/H reference to OUT/default/AH.csv (golden/make_converged_references.sh)
# Run inside a batch job (never on the login node).
set -e
TREE=$(readlink -f "$1"); OUT=$(readlink -f -m "$2")
SUN=${3:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d "$SUN/lib64" "$SUN/lib" 2>/dev/null | head -1)
CC_DIR=$TREE/cooling/sgchem; CX=$CC_DIR/cxx
J=${JOBS:-2}
NET=$(sed -n 's/^#define CHEMISTRYNETWORK \([0-9]*\).*/\1/p' "$TREE/GIZMO_config.h")
[ "$NET" = 5 ] || [ "$NET" = 17 ] || { echo "TREE is net $NET, not 5 or 17"; exit 1; }
SUNLIBS="-L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic"
STATUS=0
XF=""; BL=${BUILDS:-default forced}
if [ "${REFERENCE:-0}" = 1 ]; then BL=default; XF="-DSGCHEM_FORTRAN_RTOL=1d-8"; fi
for B in $BL; do
    X="${EXTRA_CXXFLAGS:-}"; [ $B = forced ] && X="$X -DTEST_SGCHEM_FORCED_SOLVER_FAILURE"
    [ "${REFERENCE:-0}" = 1 ] && X="$X -DCHEM_RTOL=1e-8 -DTEST_WRITE_CONVERGED_REFERENCE"
    O=$OUT/$B; mkdir -p $O/obj $O/rundir; ln -sfn "$TREE" $O/rundir/gizmo
    FO=""; CO=""
    for f in coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances rate_eq_simple rate_eq_ulli \
             rate_eq_mol rate_eq_nl99 rate_eq_gong17 jac cool_util const_rates validate_rates calc_photo calc_temp \
             compute_gamma wss_z_collis; do
        gfortran -O2 $XF -I"$TREE" -I"$CC_DIR" -c "$CC_DIR/$f.F" -o $O/obj/$f.o 2>> $O/fortran_warnings.txt &
        FO="$FO $O/obj/$f.o"
        while [ "$(jobs -r | wc -l)" -ge "$J" ]; do wait -n; done
    done
    for s in $(cd "$CX" && ls *.cc | sed 's/\.cc$//'); do
        [ "$s" = chem_gizmo_backend ] && continue
        g++ -O2 -std=c++14 $X -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/$s.cc" -o $O/obj/$s.o &
        CO="$CO $O/obj/$s.o"
        while [ "$(jobs -r | wc -l)" -ge "$J" ]; do wait -n; done
    done
    wait
    g++ -O2 -std=c++14 -Wall $X -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/golden/test_solver_failure.cc" -o $O/obj/test.o
    gfortran $O/obj/test.o $CO $FO $SUNLIBS -lstdc++ -lm -o $O/test_solver_failure
    ( cd $O/rundir && timeout --kill-after=60 3600 $O/test_solver_failure $O/AH.csv ) > $O/result.txt 2> $O/result.err || STATUS=1
    echo "== net $NET $B: $(grep -E 'GATE|===.*:' $O/result.txt | tr '\n' ' ')"
done
exit $STATUS
