#!/bin/bash
# Gate-3 item 7: rebuild the dual-backend one-zone driver (onezone3) with the
# NOW-PURE-C++ backend, so the Gate-2 acceptance benchmark can be re-run.
#
# Differences from the Gate-2 build3.sh (which lives in the figures-only stella
# repo and is NOT modified):
#   * the C++ source list gains the Gate-3 leaf kernels and the table builder
#   * $BLD/gizmo/GIZMO_config.h is set per net BEFORE compiling, because the
#     C++ TUs use GIZMO's own #include "../../../GIZMO_config.h" convention.
#     Without this the net-5 objects silently pick up the canonical net-17
#     config (caught by the Gate-3 calc_photo golden).
# onezone3.c itself is compiled READ-ONLY out of the stella tree.
set -e

CANON=/orion/ptmp/uli/orion/test/gizmo
BLD=/orion/ptmp/uli/orion/fb_tests/_sgchem_cxx_build
CX=$BLD/gizmo/cooling/sgchem/cxx
OZ=/orion/ptmp/uli/orion/stella/tests/onezone
DRV=/orion/ptmp/uli/orion/stella/tests/validation/chemistry_onezone/src
SUN=$BLD/sundials/install
SUNLIB=$(ls -d $SUN/lib64 $SUN/lib 2>/dev/null | head -1)

rsync -a --exclude='.git' --exclude='*.o' "$CANON/" "$BLD/gizmo/"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi 2>/dev/null || true
gfortran --version | head -1
g++ --version | head -1

# canonical FSRCS list from tests/onezone/Makefile
OBJS="coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances \
      rate_eq_simple rate_eq_ulli jac cool_util const_rates validate_rates \
      calc_photo calc_temp compute_gamma wss_z_collis"

CXXSRC="chem_rhs_net5 chem_rhs_net17 cvode_driver chem_spline \
        chem_cool_util chem_cool_func chem_calc_photo chem_calc_temp \
        chem_tables_build chem_backend_cxx \
        chem_backend_fortran chem_backend_c_api"

mkdir -p "$BLD/bin" "$BLD/obj"

for net in 5 17; do
    B=$OZ/build_net$net
    cp "$B/GIZMO_config.h" "$BLD/gizmo/GIZMO_config.h"
    O=""
    for o in $OBJS; do O="$O $B/cooling/sgchem/$o.o"; done

    gcc -O2 -std=c99 -D_POSIX_C_SOURCE=200809L -I"$B" -c "$DRV/onezone3.c" \
        -o "$BLD/obj/onezone3_net$net.o"

    CXXOBJ=""
    for s in $CXXSRC; do
        g++ -O2 -std=c++14 -I"$CX" -I"$B" -I"$SUN/include" \
            -c "$CX/$s.cc" -o "$BLD/obj/${s}_net$net.o"
        CXXOBJ="$CXXOBJ $BLD/obj/${s}_net$net.o"
    done
    g++ -O2 -std=c++11 -I"$CX" -c "$CX/rates_gen.cc" -o "$BLD/obj/rates_gen_net$net.o"
    CXXOBJ="$CXXOBJ $BLD/obj/rates_gen_net$net.o"

    gfortran -static-libgfortran -static-libgcc \
        "$BLD/obj/onezone3_net$net.o" $CXXOBJ $O \
        -L"$SUNLIB" -lsundials_cvode -lsundials_nvecserial \
        -lsundials_sunmatrixdense -lsundials_sunlinsoldense \
        -lsundials_sunnonlinsolnewton -lsundials_generic \
        -lstdc++ -lm -o "$BLD/bin/onezone3_net$net"
    echo "BUILT $BLD/bin/onezone3_net$net"
done
