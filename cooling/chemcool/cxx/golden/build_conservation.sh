#!/bin/bash
# Conservation audit + Fortran/C++ goldens for CHEMISTRYNETWORK 3/5/15/16/17.
#
# usage: build_conservation.sh SRC OUT NET [SUNDIALS_DIR]
#   SRC  gizmo source tree (worktree or a frozen copy); it is COPIED, never built in
#   OUT  build/run directory for this net
# Builds, in OUT:
#   test_conservation_netN        probes on: the C++ RHS / driver are compiled from
#                                 instrumented COPIES (derived-species floor counter,
#                                 check_abundances clip ledger, H3+ stock); values
#                                 are untouched, so PHASE R parity is unaffected
#   test_gate3_netN               table/leaf-kernel golden (tol 1e-12)
#   test_onezone_ab_netN          nets 3/15/16 only: existing RHS + one-zone A/B golden
set -e
SRC=$(readlink -f "$1"); OUT=$(readlink -f -m "$2"); NET=$3
SUN=${4:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d "$SUN/lib64" "$SUN/lib" 2>/dev/null | head -1)
CANON_CONFIG=/orion/ptmp/uli/orion/test/gizmo/Config.sh
FC=gfortran; CXX=g++
J=${JOBS:-2}

TREE=$OUT/tree
mkdir -p "$OUT/obj" "$OUT/instr" "$OUT/rundir"
rsync -a --delete --exclude .git --exclude '*.o' --exclude GIZMO --exclude .stella-tables \
      --exclude 'ketju-integrator.localbuild' "$SRC/" "$TREE/"
cp "$CANON_CONFIG" "$TREE/Config.sh"
sed -i "s/^CHEMISTRYNETWORK=.*/CHEMISTRYNETWORK=$NET/" "$TREE/Config.sh"
( cd "$TREE" && rm -f GIZMO_config.h && CONFIG=Config.sh PERL=/usr/bin/perl make -f config-makefile > /dev/null 2>&1 )
grep -q "^#define CHEMISTRYNETWORK $NET" "$TREE/GIZMO_config.h" || { echo "config generation failed"; exit 1; }
ln -sfn "$TREE" "$OUT/rundir/gizmo"
CC_DIR=$TREE/cooling/chemcool
CX=$CC_DIR/cxx
echo "== net $NET  $($FC --version | head -1) / $($CXX --version | head -1)"

# ---- instrumented copies (line numbers preserved in the RHS files) ----
fid=1
for n in 5 15 16 17 3; do
    f=chem_rhs_net$n.cc
    { echo 'extern "C" void probe_floor(int, int, const char*, double); extern "C" void probe_h3p(double, double);'
      echo '#line 1'
      sed -e "s/if (\(ab[A-Za-z0-9_]*\) *< 0\.0) *\1 *= 0\.0;/if (\1 < 0.0) { probe_floor($fid, __LINE__, \"\1\", \1); \1 = 0.0; }/g" \
          -e 's/^\(    abe = abe + abh3p;\)/\1 probe_h3p(abh3p, abHI);/' "$CX/$f"
    } > "$OUT/instr/$f"
    fid=$((fid + 1))
done
sed -e 's/^static int chem_check_abundances(double\* y, ChemCell& c, const double\* atol)$/static int chem_check_abundances_orig(double* y, ChemCell\& c, const double* atol)/' \
    -e 's/fail = chem_check_abundances(y, c, atol);/fail = probe_check_wrap(y, c, atol);/' \
    -e 's/^\(    if (c\.net == 3)  return chem_rhs_net3(t, y, ydot, c);\)/    probe_rhs();\n\1/' \
    "$CX/cvode_driver.cc" > "$OUT/instr/cvode_driver.cc"
python3 - "$OUT/instr/cvode_driver.cc" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
hook = '''extern "C" void probe_check(const double*, const double*, int);
extern "C" void probe_rhs();
static int chem_check_abundances_orig(double* y, ChemCell& c, const double* atol);
static int probe_check_wrap(double* y, ChemCell& c, const double* atol)
{
    double yin[CHEM_NSPEC_MAX];
    for (int j = 0; j < c.nspec; j++) yin[j] = y[j];
    int rc = chem_check_abundances_orig(y, c, atol);
    probe_check(yin, y, rc);
    return rc;
}
'''
s = s.replace('static inline int rhs_dispatch(', hook + 'static inline int rhs_dispatch(', 1)
open(p, 'w').write(s)
EOF
for s in probe_check_wrap probe_rhs; do grep -q "$s" "$OUT/instr/cvode_driver.cc" || { echo "instrumentation $s missing"; exit 1; }; done
for n in 5 15 16 17 3; do grep -c probe_floor "$OUT/instr/chem_rhs_net$n.cc" | sed "s/^/  floors instrumented net$n: /"; done

# ---- Fortran ----
FSRC="coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances \
      rate_eq_simple rate_eq_ulli rate_eq_mol rate_eq_nl99 rate_eq_gong17 jac \
      cool_util const_rates validate_rates calc_photo calc_temp compute_gamma \
      wss_z_collis"
FO=""
for f in $FSRC; do
    $FC -O2 -I"$TREE" -I"$CC_DIR" -c "$CC_DIR/$f.F" -o "$OUT/obj/$f.o" &
    FO="$FO $OUT/obj/$f.o"
    while [ "$(jobs -r | wc -l)" -ge "$J" ]; do wait -n; done
done
wait

# ---- C++: plain and instrumented object sets ----
CO=""; CI=""
for s in $(cd "$CX" && ls *.cc | sed 's/\.cc$//'); do
    [ "$s" = chem_gizmo_backend ] && continue
    $CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/$s.cc" -o "$OUT/obj/$s.o" &
    CO="$CO $OUT/obj/$s.o"
    if [ -f "$OUT/instr/$s.cc" ]; then
        $CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$OUT/instr/$s.cc" -o "$OUT/obj/${s}_instr.o" &
        CI="$CI $OUT/obj/${s}_instr.o"
    else
        CI="$CI $OUT/obj/$s.o"
    fi
    while [ "$(jobs -r | wc -l)" -ge "$J" ]; do wait -n; done
done
wait

SUNLIBS="-L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense \
         -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic"
$CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/golden/test_conservation.cc" -o "$OUT/obj/test_conservation.o"
$FC "$OUT/obj/test_conservation.o" $CI $FO $SUNLIBS -lstdc++ -lm -o "$OUT/test_conservation_net$NET"
TESTS="test_gate3"
[ "$NET" = 3 -o "$NET" = 15 -o "$NET" = 16 ] && TESTS="$TESTS test_onezone_ab"
for t in $TESTS; do
    $CXX -O2 -std=c++14 -I"$CX" -I"$TREE" -I"$SUN/include" -c "$CX/golden/$t.cc" -o "$OUT/obj/$t.o"
    $FC "$OUT/obj/$t.o" $CO $FO $SUNLIBS -lstdc++ -lm -o "$OUT/${t}_net$NET"
done
echo "== built: test_conservation_net$NET $TESTS"
[ "${BUILD_ONLY:-0}" = 1 ] && exit 0

STATUS=0
( cd "$OUT/rundir" && "$OUT/test_conservation_net$NET" "$OUT/conservation_net$NET.csv" ) \
    > "$OUT/conservation_net$NET.txt" 2> "$OUT/conservation_net$NET.err" || STATUS=1
( cd "$OUT/rundir" && "$OUT/test_gate3_net$NET" ) > "$OUT/gate3_net$NET.txt" 2>&1 || STATUS=1
if [ "$NET" = 3 -o "$NET" = 15 -o "$NET" = 16 ]; then
    ( cd "$OUT/rundir" && "$OUT/test_onezone_ab_net$NET" "$OUT/onezone_traj_net$NET.csv" ) \
        > "$OUT/onezone_ab_net$NET.txt" 2> "$OUT/onezone_ab_net$NET.err" || STATUS=1
fi
echo "== net $NET done, status $STATUS"
exit $STATUS
