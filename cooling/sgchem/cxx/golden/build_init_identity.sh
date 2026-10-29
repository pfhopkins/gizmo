#!/bin/bash
# Build + run test_init_identity: Fortran chemistry init vs the Fortran-free C++ init.
# Usage: build_init_identity.sh <gizmo tree with GIZMO_config.h from a default SGCHEM build> <out dir>
# Never point <out dir> inside the gizmo tree.
set -e
TREE=$(cd "${1:?gizmo tree}" && pwd)
OUT=${2:?out dir}
mkdir -p "$OUT/obj"; OUT=$(cd "$OUT" && pwd)
grep -q "^#define SGCHEM$" "$TREE/GIZMO_config.h" && ! grep -q "^#define SGCHEM_FORTRAN" "$TREE/GIZMO_config.h" || { echo "GIZMO_config.h needs SGCHEM without SGCHEM_FORTRAN"; exit 1; }

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi 2>/dev/null || true
SUN=${SUNDIALS_DIR:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d $SUN/lib64 $SUN/lib 2>/dev/null | head -1)
OPT="-g -O3 -march=native ${EXTRA_OPT:-}"
INC="-I$TREE -I$TREE/cooling/sgchem -I$TREE/cooling/sgchem/cxx -I$GSL_HOME/include -I$HDF5_HOME/include -DH5_USE_16_API -I$SUN/include"
CC=$TREE/cooling/sgchem; CX=$CC/cxx

FSRC="coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances rate_eq_simple rate_eq_ulli rate_eq_mol \
      rate_eq_nl99 rate_eq_gong17 jac cool_util const_rates validate_rates calc_photo calc_temp compute_gamma wss_z_collis"
O=""
for f in $FSRC; do mpifort $OPT ${EXTRA_FOPT:-} -I$TREE -I$CC -c $CC/$f.F -o $OUT/obj/$f.o; O="$O $OUT/obj/$f.o"; done
# every backend TU GIZMO links, minus chem_gizmo_backend.cc (included by the test), plus the Fortran view shim
for f in $(ls $CX/*.cc | xargs -n1 basename | sed 's/\.cc$//'); do
    case $f in chem_gizmo_backend|chem_backend_c_api) continue;; esac
    mpicxx $OPT -std=c++14 $INC -c $CX/$f.cc -o $OUT/obj/$f.o; O="$O $OUT/obj/$f.o"
done
mpicxx $OPT -std=c++14 $INC -c $CX/golden/test_init_identity.cc -o $OUT/obj/test_init_identity.o
mpifort $OUT/obj/test_init_identity.o $O -L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense \
    -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic -lstdc++ -lm -o $OUT/test_init_identity
ln -sfn "$TREE" "$OUT/gizmo"
cd "$OUT" && ./test_init_identity
