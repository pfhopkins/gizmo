#!/bin/bash
# Build cooling/sgchem/cxx/fortran_reference/reference_dump and threads_identity for one network; run reference_dump
# (tables, cool_func rates, one-zone runs) into <out dir>/data.
# Usage: build_onezone_dump.sh <net> <out dir>
# Needs: mpicxx, SUNDIALS_DIR (SUNDIALS 6 or 7), KOKKOS_PATH (Kokkos install; the GIZMO headers include it),
#        CHEMCXX_DATADIR (the four chemistry tables, installed by scripts/get_stella_tables.sh).
set -euo pipefail
NET=${1:?net}; OUT=${2:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$(cd "$HERE/../../src" && pwd)
: "${SUNDIALS_DIR:?set SUNDIALS_DIR}" "${KOKKOS_PATH:?set KOKKOS_PATH}" "${CHEMCXX_DATADIR:?set CHEMCXX_DATADIR}"
mkdir -p "$OUT/obj"; OUT=$(cd "$OUT" && pwd); TREE=$OUT/src
rsync -a --delete --exclude='*.o' --exclude=GIZMO --exclude=Config.sh --exclude=GIZMO_config.h "$SRC/" "$TREE/"
sed "s/^CHEMISTRYNETWORK=.*/CHEMISTRYNETWORK=$NET/" "$HERE/Config_onezone.sh" > "$TREE/Config.sh"
(cd "$TREE" && perl file_io/prepare-config.perl Config.sh > /dev/null)
grep -q "^#define CHEMISTRYNETWORK $NET" "$TREE/GIZMO_config.h" || { echo "config generation failed"; exit 1; }
libdir() { if [ -d "$1/lib64" ]; then echo "$1/lib64"; else echo "$1/lib"; fi; }
SUNLIB=$(libdir "$SUNDIALS_DIR"); KKLIB=$(libdir "$KOKKOS_PATH")
OPT="-O3 -march=native -std=c++20"
INC="-I$TREE -I$TREE/cooling/sgchem -I$TREE/cooling/sgchem/cxx -I$SUNDIALS_DIR/include -I$KOKKOS_PATH/include"
[ -n "${HDF5_HOME:-}" ] && INC="$INC -I$HDF5_HOME/include -DH5_USE_16_API"
[ -n "${GSL_HOME:-}" ] && INC="$INC -I$GSL_HOME/include"
CX=$TREE/cooling/sgchem/cxx; O=""
for f in "$CX"/*.cc; do
    b=$(basename "$f" .cc); [ "$b" = chem_gizmo_backend ] && continue   # included by reference_dump.cc
    mpicxx $OPT $INC -c "$f" -o "$OUT/obj/$b.o"; O="$O $OUT/obj/$b.o"
done
mpicxx $OPT $INC -c "$CX/fortran_reference/reference_dump.cc" -o "$OUT/obj/reference_dump.o"
mpicxx "$OUT/obj/reference_dump.o" $O -L"$SUNLIB" -Wl,-rpath,"$SUNLIB" -lsundials_cvode -lsundials_nvecserial \
    -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic \
    -L"$KKLIB" -Wl,-rpath,"$KKLIB" -lkokkoscore -ldl -lm -o "$OUT/reference_dump"
if [ "$NET" = 5 ] || [ "$NET" = 17 ]; then   # the driver's species layout covers nets 5 and 17
mpicxx $OPT -fopenmp -DPURE_NET=$NET $INC -c "$HERE/threads_identity.cc" -o "$OUT/obj/threads_identity.o"
mpicxx -fopenmp "$OUT/obj/threads_identity.o" $O -L"$SUNLIB" -Wl,-rpath,"$SUNLIB" -lsundials_cvode -lsundials_nvecserial \
    -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic -lm -o "$OUT/threads_identity"
fi
mkdir -p "$OUT/data"
cd "$OUT" && ./reference_dump "$OUT/data"
