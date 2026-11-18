#!/bin/bash
# Net 17 solver-failure test (WLM run 2026-11-17): golden/test_net17_solver_failure.cc, two builds:
#   default   phases A + H (66 logged floor cells, TNOCHEM hand-off grid), C++ vs Fortran DVODE
#   recovery  phase R: -DCHEM_NSTEP=1 -DSGCHEM_SOLVER_FAILURE_ENERGY_ONLY_COOLING on the driver and the test
#
# usage: build_net17_solver_failure.sh TREE OUT [SUNDIALS_DIR]
#   TREE  a COPY of the gizmo tree: net 17 GIZMO_config.h at its root and the SGChem tables
#         (eh2-table.txt, energy-table-x.txt, H2-cooling-ratios.dat, wss_z_collis.txt) in cooling/sgchem
#   OUT   objects, binaries and results
# Run inside a batch job (never on the login node).
set -e
TREE=$(readlink -f "$1"); OUT=$(readlink -f -m "$2")
SUN=${3:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d "$SUN/lib64" "$SUN/lib" 2>/dev/null | head -1)
CC_DIR=$TREE/cooling/sgchem; CX=$CC_DIR/cxx
J=${JOBS:-2}
grep -q "^#define CHEMISTRYNETWORK 17" "$TREE/GIZMO_config.h" || { echo "TREE is not a net 17 tree"; exit 1; }
SUNLIBS="-L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic"
STATUS=0
for B in default recovery; do
    X=""; [ $B = recovery ] && X="-DCHEM_NSTEP=1 -DSGCHEM_SOLVER_FAILURE_ENERGY_ONLY_COOLING"
    O=$OUT/$B; mkdir -p $O/obj $O/rundir; ln -sfn "$TREE" $O/rundir/gizmo
    FO=""; CO=""
    for f in coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances rate_eq_simple rate_eq_ulli \
             rate_eq_mol rate_eq_nl99 rate_eq_gong17 jac cool_util const_rates validate_rates calc_photo calc_temp \
             compute_gamma wss_z_collis; do
        gfortran -O2 -I"$TREE" -I"$CC_DIR" -c "$CC_DIR/$f.F" -o $O/obj/$f.o 2>> $O/fortran_warnings.txt &
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
    g++ -O2 -std=c++14 -Wall $X -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/golden/test_net17_solver_failure.cc" -o $O/obj/test.o
    gfortran $O/obj/test.o $CO $FO $SUNLIBS -lstdc++ -lm -o $O/test_net17_solver_failure
    ( cd $O/rundir && $O/test_net17_solver_failure ) > $O/result.txt 2> $O/result.err || STATUS=1
    echo "== $B: $(grep -E 'GATE|===.*:' $O/result.txt | tr '\n' ' ')"
done
exit $STATUS
