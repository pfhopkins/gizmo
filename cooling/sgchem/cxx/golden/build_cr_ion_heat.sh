#!/bin/bash
# CR ionisation heating one-zone test. build_cr_ion_heat.sh <gizmo source with GIZMO_config.h> <outdir>
SRC=$(readlink -f "$1") || exit 1; OUT=$2
[ -f "$SRC/GIZMO_config.h" ] || { echo "no $SRC/GIZMO_config.h (run make GIZMO_config.h with that net's Config.sh)"; exit 1; }
SUN=${SUNDIALS_DIR:-/orion/ptmp/uli/orion/fb_tests/_sundials/install}
SUNLIB=$(ls -d $SUN/lib64 $SUN/lib 2>/dev/null | head -1)
CX=$SRC/cooling/sgchem/cxx
mkdir -p "$OUT/obj" || exit 1
O=""
for f in $CX/*.cc; do
    b=$(basename $f .cc)
    case $b in chem_backend_fortran|chem_backend_c_api|chem_gizmo_backend) continue;; esac
    g++ -O2 -std=c++14 -I"$CX" -I"$SRC" -I"$SUN/include" -c "$f" -o "$OUT/obj/$b.o" || exit 1
    O="$O $OUT/obj/$b.o"
done
g++ -O2 -std=c++14 -I"$CX" -I"$SRC" -I"$SUN/include" "$CX/golden/test_cr_ion_heat.cc" $O \
    -L"$SUNLIB" -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsoldense \
    -lsundials_sunnonlinsolnewton -lsundials_generic -lm -o "$OUT/test_cr_ion_heat" || exit 1
echo "built $OUT/test_cr_ion_heat"
