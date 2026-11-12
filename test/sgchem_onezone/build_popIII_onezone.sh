#!/bin/bash
# Build the Pop III one-zone collapse driver (popIII_onezone.cc) on network 17 with H- from the stellar bands.
# Usage: build_popIII_onezone.sh <out dir>. Needs mpicxx, SUNDIALS_DIR, KOKKOS_PATH.
set -euo pipefail
OUT=${1:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$(cd "$HERE/../../src" && pwd)
: "${SUNDIALS_DIR:?set SUNDIALS_DIR}" "${KOKKOS_PATH:?set KOKKOS_PATH}"
mkdir -p "$OUT/obj"; OUT=$(cd "$OUT" && pwd); TREE=$OUT/src
rsync -a --delete --exclude='*.o' --exclude=GIZMO --exclude=Config.sh --exclude=GIZMO_config.h "$SRC/" "$TREE/"
sed "s/^CHEMISTRYNETWORK=.*/CHEMISTRYNETWORK=17/" "$HERE/Config_onezone.sh" > "$TREE/Config.sh"
printf "GALSF_RESOLVEDISM_TREECOL_PIXELS=12\nGALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES\nSGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS\n" >> "$TREE/Config.sh"
(cd "$TREE" && perl file_io/prepare-config.perl Config.sh > /dev/null)
libdir() { if [ -d "$1/lib64" ]; then echo "$1/lib64"; else echo "$1/lib"; fi; }
SUNLIB=$(libdir "$SUNDIALS_DIR"); KKLIB=$(libdir "$KOKKOS_PATH")
OPT="-O3 -march=native -std=c++20"
INC="-I$TREE -I$TREE/cooling/sgchem -I$TREE/cooling/sgchem/cxx -I$SUNDIALS_DIR/include -I$KOKKOS_PATH/include"
[ -n "${HDF5_HOME:-}" ] && INC="$INC -I$HDF5_HOME/include -DH5_USE_16_API"
[ -n "${GSL_HOME:-}" ] && INC="$INC -I$GSL_HOME/include"
CX=$TREE/cooling/sgchem/cxx; O=""
for f in "$CX"/*.cc; do
    b=$(basename "$f" .cc); [ "$b" = chem_gizmo_backend ] && continue
    mpicxx $OPT $INC -c "$f" -o "$OUT/obj/$b.o"; O="$O $OUT/obj/$b.o"
done
mpicxx $OPT $INC -c "$HERE/popIII_onezone.cc" -o "$OUT/obj/popIII_onezone.o"
mpicxx "$OUT/obj/popIII_onezone.o" $O -L"$SUNLIB" -Wl,-rpath,"$SUNLIB" -lsundials_cvode -lsundials_nvecserial \
    -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_sunnonlinsolnewton -lsundials_generic \
    -L"$KKLIB" -Wl,-rpath,"$KKLIB" -lkokkoscore -ldl -lm -o "$OUT/popIII_onezone"
