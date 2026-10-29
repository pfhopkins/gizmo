#!/bin/bash
# Gate-3 golden: leaf kernels (cool_func, cool_util, calc_photo, calc_temp,
# compute_gamma) and the C++ table builder, driven side by side with the
# canonical Fortran over dense parameter grids incl. every guard boundary.
#
# LOCATION RULES (Uli, 2026-07-29):
#   * source lives in the gizmo tree: cooling/sgchem/cxx/
#   * NEVER compile inside test/gizmo -- rsync to fb_tests/_sgchem_cxx_build
#   * stella is figures-only: nothing is written there
set -e

CANON=/orion/ptmp/uli/orion/test/gizmo
BLD=/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build
CX=$BLD/gizmo/cooling/sgchem/cxx
OBJS="coolinmo cheminmo spline photoinit_ism const_rates cool_util wss_z_collis \
      cool_func calc_photo calc_temp compute_gamma"

rsync -a --exclude='.git' --exclude='*.o' "$CANON/" "$BLD/gizmo/"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi 2>/dev/null || true
FC=gfortran; CXX=g++
echo "== compilers: $($FC --version | head -1) / $($CXX --version | head -1)"

python3 "$CX/tables_extract.py"
# re-sync so the regenerated header is the one we compile
rsync -a --exclude='.git' --exclude='*.o' "$CANON/" "$BLD/gizmo/"

OUT=$BLD/gate3
mkdir -p "$OUT/obj" "$OUT/rundir"
ln -sfn "$CANON" "$OUT/rundir/gizmo"

CXXSRC="chem_spline chem_cool_util chem_cool_func chem_calc_photo \
        chem_calc_temp chem_tables_build chem_backend_fortran"

STATUS=0
for net in 5 17; do
    SNAP=$BLD/golden/obj_net$net
    O=""
    for o in $OBJS; do
        if [ ! -f "$SNAP/$o.o" ]; then
            echo "FATAL: $SNAP/$o.o missing -- refresh the snapshot with"
            echo "  cp <onezone>/build_net$net/cooling/sgchem/$o.o $SNAP/"
            exit 1
        fi
        O="$O $SNAP/$o.o"
    done
    tags=$(for o in $OBJS; do readelf -p .comment "$SNAP/$o.o" | grep -o "GCC:.*"; done | sort -u)
    if [ "$(echo "$tags" | wc -l)" != 1 ]; then
        echo "FATAL: mixed compiler tags in obj_net$net snapshot:"; echo "$tags"; exit 1
    fi
    echo "== obj_net$net compiler: $tags"

    # The C++ kernels use GIZMO's own convention, #include "../../../GIZMO_config.h",
    # so the config must sit at the ROOT of the tree being compiled.  Without
    # this the net-5 build silently picked up the canonical (net-17) config --
    # caught by the Gate-3 calc_photo golden, 19872 phrates(10) mismatches.
    cp "$SNAP/GIZMO_config.h" "$BLD/gizmo/GIZMO_config.h"

    for s in $CXXSRC; do
        $CXX -O2 -std=c++14 -I"$CX" -I"$SNAP" -I"$BLD/gizmo" \
             -c "$CX/$s.cc" -o "$OUT/obj/${s}_net$net.o"
    done
    $CXX -O2 -std=c++11 -I"$CX" -c "$CX/rates_gen.cc" -o "$OUT/obj/rates_gen_net$net.o"
    $CXX -O2 -std=c++14 -I"$CX" -I"$SNAP" -I"$BLD/gizmo" \
         -c "$CX/golden/test_gate3.cc" -o "$OUT/obj/test_gate3_net$net.o"

    COBJ="$OUT/obj/test_gate3_net$net.o $OUT/obj/rates_gen_net$net.o"
    for s in $CXXSRC; do COBJ="$COBJ $OUT/obj/${s}_net$net.o"; done

    $FC $COBJ $O -lstdc++ -lm -o "$OUT/test_gate3_net$net"
    echo "===== net $net ====="
    ( cd "$OUT/rundir" && "$OUT/test_gate3_net$net" ) \
        | tee "$OUT/gate3_golden_net$net.txt" || STATUS=1
done
exit $STATUS
