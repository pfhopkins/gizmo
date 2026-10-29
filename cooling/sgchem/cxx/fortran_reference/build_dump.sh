#!/bin/bash
# Build and run reference_dump for one network, Fortran (frozen reference) or C++ (Fortran-free) backend.
# Usage: build_dump.sh fortran|cxx <net> <gizmo tree> <out dir>
# The tree needs Config_stella_default.sh and the sgchem data files; it is copied, never written.
set -e
MODE=${1:?fortran|cxx}; NET=${2:?net}; TREE=$(cd "${3:?tree}" && pwd); OUT=${4:?out dir}
mkdir -p "$OUT/obj"; OUT=$(cd "$OUT" && pwd)

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc openmpi gsl hwloc fftw-mpi hdf5-mpi 2>/dev/null || true
SUN=${SUNDIALS_DIR:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d $SUN/lib64 $SUN/lib 2>/dev/null | head -1)

# private tree copy: sources include ../../GIZMO_config.h relative to themselves, so the config must sit at its root
SRC=$TREE; TREE=$OUT/tree
rsync -a --delete --exclude=.git --exclude='*.o' --exclude=GIZMO --exclude='*.hdf5' --exclude='/ketju*' --exclude=.stella-tables \
      --exclude='output*' --exclude=/test "$SRC/" "$TREE/"
sed "s/^CHEMISTRYNETWORK=[0-9]*/CHEMISTRYNETWORK=$NET/" "$SRC/Config_stella_default.sh" > "$TREE/Config.sh"
(cd "$TREE" && perl file_io/prepare-config.perl Config.sh >/dev/null)
grep -q "^#define CHEMISTRYNETWORK $NET" "$TREE/GIZMO_config.h" || { echo "config generation failed"; exit 1; }

OPT="-g -O3 -march=native"
INC="-I$TREE -I$TREE/cooling/sgchem -I$TREE/cooling/sgchem/cxx -I$GSL_HOME/include -I$HDF5_HOME/include -DH5_USE_16_API -I$SUN/include"
CC=$TREE/cooling/sgchem; CX=$CC/cxx
O=""
for f in $(ls $CX/*.cc | xargs -n1 basename | sed 's/\.cc$//'); do
    case $f in chem_gizmo_backend|chem_backend_c_api) continue;; esac
    [ $MODE = cxx ] && [ $f = chem_backend_fortran ] && continue
    mpicxx $OPT -std=c++14 $INC -c $CX/$f.cc -o $OUT/obj/$f.o; O="$O $OUT/obj/$f.o"
done
SUNLIBS="-L$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic"
if [ $MODE = fortran ]; then
    for f in coolinmo cheminmo spline cool_func photoinit_ism dvode evolve_abundances rate_eq_simple rate_eq_ulli rate_eq_mol \
             rate_eq_nl99 rate_eq_gong17 jac cool_util const_rates validate_rates calc_photo calc_temp compute_gamma wss_z_collis; do
        mpifort $OPT -I$TREE -I$CC -c $CC/$f.F -o $OUT/obj/$f.o; O="$O $OUT/obj/$f.o"
    done
    mpicxx $OPT -std=c++14 -DREF_FORTRAN $INC -c $CX/fortran_reference/reference_dump.cc -o $OUT/obj/reference_dump.o
    mpifort $OUT/obj/reference_dump.o $O $SUNLIBS -lstdc++ -lm -o $OUT/reference_dump
else
    mpicxx $OPT -std=c++14 $INC -c $CX/fortran_reference/reference_dump.cc -o $OUT/obj/reference_dump.o
    mpicxx $OUT/obj/reference_dump.o $O $SUNLIBS -lm -o $OUT/reference_dump
fi
ln -sfn "$TREE" "$OUT/gizmo"
mkdir -p "$OUT/data"
cd "$OUT" && ./reference_dump "$OUT/data"
