#!/bin/bash
# Fortran-vs-C++ goldens for CHEMISTRYNETWORK 15/16, built from ONE tree copy:
#   test_gate3       tables + leaf kernels, tol 1e-12 (GATE3_ACCEPTANCE.md)
#   test_onezone_ab  RHS (rate_eq vs chem_rhs_netN) at 1e-12 + one-zone
#                    evolutions, final T within 1% (GATE2/3 criterion (a)/(c))
#
# usage: build_onezone_ab.sh TREE OUT [SUNDIALS_DIR]
#   TREE  a COPY of the gizmo tree with GIZMO_config.h at its root and the
#         generated rates files in place (never the worktree or test/gizmo)
#   OUT   objects, binaries, logs and the trajectory CSV
# Run inside a batch job (never on the login node).
set -e
TREE=$(readlink -f "$1"); OUT=$(readlink -f "$2")
SUN=${3:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d "$SUN/lib64" "$SUN/lib" 2>/dev/null | head -1)
CC_DIR=$TREE/cooling/chemcool
CX=$CC_DIR/cxx
FC=gfortran; CXX=g++
NET=$(sed -n 's/^#define CHEMISTRYNETWORK \([0-9]*\).*/\1/p' "$TREE/GIZMO_config.h")
echo "== net $NET, $($FC --version | head -1) / $($CXX --version | head -1)"

mkdir -p "$OUT/obj" "$OUT/rundir"
ln -sfn "$TREE" "$OUT/rundir/gizmo"

FSRC="coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances \
      rate_eq_simple rate_eq_ulli rate_eq_mol rate_eq_nl99 rate_eq_gong17 jac \
      cool_util const_rates validate_rates calc_photo calc_temp compute_gamma \
      wss_z_collis"
FO=""
for f in $FSRC; do
    $FC -O2 -I"$TREE" -I"$CC_DIR" -c "$CC_DIR/$f.F" -o "$OUT/obj/$f.o"
    FO="$FO $OUT/obj/$f.o"
done

# every C++ TU GIZMO compiles under CHEMCOOL_CXX, minus the GIZMO glue
CO=""
for s in $(cd "$CX" && ls *.cc | sed 's/\.cc$//'); do
    [ "$s" = chem_gizmo_backend ] && continue
    $CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/$s.cc" -o "$OUT/obj/$s.o"
    CO="$CO $OUT/obj/$s.o"
done

SUNLIBS="-L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic"
for t in test_gate3 test_onezone_ab; do
    $CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/golden/$t.cc" -o "$OUT/obj/$t.o"
    $FC "$OUT/obj/$t.o" $CO $FO $SUNLIBS -lstdc++ -lm -o "$OUT/${t}_net$NET"
done

STATUS=0
( cd "$OUT/rundir" && "$OUT/test_gate3_net$NET" ) > "$OUT/gate3_net$NET.txt" 2>&1 || STATUS=1
grep -E "FAIL|GOLDEN" "$OUT/gate3_net$NET.txt" | head -40
( cd "$OUT/rundir" && "$OUT/test_onezone_ab_net$NET" "$OUT/onezone_traj_net$NET.csv" ) \
    > "$OUT/onezone_ab_net$NET.txt" 2>"$OUT/onezone_ab_net$NET.err" || STATUS=1
cat "$OUT/onezone_ab_net$NET.txt"
exit $STATUS
