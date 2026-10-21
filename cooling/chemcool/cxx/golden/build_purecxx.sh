#!/bin/bash
# Gate-3 item 6: build the CHEMCOOL_CXX path with NO Fortran at all and prove it.
#
#   * links with g++, not gfortran
#   * links no .F object and no -lgfortran
#   * `nm` asserts zero libgfortran / Fortran-mangled symbols in the binary
#   * runs a one-zone sweep, and an OpenMP repeat of the same cases that must
#     come out bit-identical to the serial run (audit R2)
set -e

CANON=/orion/ptmp/uli/orion/test/gizmo
BLD=/orion/ptmp/uli/orion/fb_tests/_chemcool_cxx_build
CX=$BLD/gizmo/cooling/chemcool/cxx
SUN=$BLD/sundials/install
SUNLIB=$(ls -d $SUN/lib64 $SUN/lib 2>/dev/null | head -1)

rsync -a --exclude='.git' --exclude='*.o' "$CANON/" "$BLD/gizmo/"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi 2>/dev/null || true
CXX=g++
echo "== compiler: $($CXX --version | head -1)"

OUT=$BLD/purecxx
mkdir -p "$OUT/obj" "$OUT/rundir"
ln -sfn "$CANON" "$OUT/rundir/gizmo"

# The pure path: every C++ TU of the port EXCEPT chem_backend_fortran.cc
SRC="rates_gen chem_spline chem_cool_util chem_cool_func chem_calc_photo \
     chem_calc_temp chem_tables_build chem_rhs_net5 chem_rhs_net15 chem_rhs_net16 chem_rhs_net17 \
     cvode_driver chem_backend_cxx"

STATUS=0
for net in 5 17; do
    SNAP=$BLD/golden/obj_net$net
    cp "$SNAP/GIZMO_config.h" "$BLD/gizmo/GIZMO_config.h"
    O=""
    for s in $SRC; do
        $CXX -O2 -std=c++14 -fopenmp -I"$CX" -I"$SNAP" -I"$SUN/include" \
             -c "$CX/$s.cc" -o "$OUT/obj/${s}_net$net.o"
        O="$O $OUT/obj/${s}_net$net.o"
    done
    $CXX -O2 -std=c++14 -fopenmp -DPURE_NET=$net -I"$CX" -I"$SNAP" \
         -I"$SUN/include" -c "$CX/golden/test_purecxx.cc" \
         -o "$OUT/obj/test_purecxx_net$net.o"

    # NB: g++ as the linker, and NO -lgfortran
    $CXX -fopenmp "$OUT/obj/test_purecxx_net$net.o" $O \
         -L"$SUNLIB" -lsundials_cvode -lsundials_nvecserial \
         -lsundials_sunmatrixdense -lsundials_sunlinsoldense \
         -lsundials_sunnonlinsolnewton -lsundials_generic \
         -lm -o "$OUT/purecxx_net$net"

    echo "===== net $net : Fortran-symbol audit ====="
    UNDEF=$(nm -u "$OUT/purecxx_net$net" | grep -iE "gfortran|_gfortran|^ *U [a-z_0-9]+_$" || true)
    DEFINED=$(nm "$OUT/purecxx_net$net" | grep -iE " [TDBC] .*(coolinmo|cheminmo|cool_func_|calc_photo_|calc_temp_|spline_eval_|const_rates_|photoinit_|wss_z|evolve_abundances_|dvode_|coolr_|cooli_|co_data_|h2o_data_)" || true)
    LIBS=$(ldd "$OUT/purecxx_net$net" | grep -i fortran || true)
    if [ -n "$UNDEF" ] || [ -n "$DEFINED" ] || [ -n "$LIBS" ]; then
        echo "FAIL -- Fortran residue found:"
        [ -n "$UNDEF" ]   && { echo "  undefined:"; echo "$UNDEF"; }
        [ -n "$DEFINED" ] && { echo "  defined:";   echo "$DEFINED"; }
        [ -n "$LIBS" ]    && { echo "  shared libs:"; echo "$LIBS"; }
        STATUS=1
    else
        echo "PASS -- zero Fortran symbols, zero libgfortran linkage"
    fi
    ldd "$OUT/purecxx_net$net" | sed 's/^/    /'

    echo "===== net $net : run ====="
    ( cd "$OUT/rundir" && OMP_NUM_THREADS=8 "$OUT/purecxx_net$net" ) \
        | tee "$OUT/purecxx_net$net.txt" || STATUS=1
done
exit $STATUS
