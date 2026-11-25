#!/bin/bash
# Build and run the SGChem solver-failure and dust-temperature drivers for one network (5 or 17):
#   default  solver_failure (phases A, H) and dust_temperature (F1-F4)
#   forced   solver_failure with -DTEST_SGCHEM_FORCED_SOLVER_FAILURE on the chemistry and the driver (phase R)
# Usage: build_and_run.sh <net> <out dir>. Needs mpicxx, SUNDIALS_DIR, CHEMCXX_DATADIR (the four chemistry tables).
# Results: <out>/<build>/*.txt; exit status 1 if any driver fails. Run inside a batch job.
set -uo pipefail
NET=${1:?net}; OUT=${2:?out dir}
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$(cd "$HERE/../../src" && pwd)
: "${SUNDIALS_DIR:?set SUNDIALS_DIR}" "${CHEMCXX_DATADIR:?set CHEMCXX_DATADIR}"
[ "$NET" = 5 ] || [ "$NET" = 17 ] || { echo "net $NET: only 5 and 17"; exit 2; }
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); TREE=$OUT/src
rsync -a --delete --exclude='*.o' --exclude=GIZMO --exclude=Config.sh --exclude=GIZMO_config.h "$SRC/" "$TREE/"
sed "s/^CHEMISTRYNETWORK=.*/CHEMISTRYNETWORK=$NET/" "$HERE/../sgchem_onezone/Config_onezone.sh" > "$TREE/Config.sh"
(cd "$TREE" && perl file_io/prepare-config.perl Config.sh > /dev/null)
grep -q "^#define CHEMISTRYNETWORK $NET" "$TREE/GIZMO_config.h" || { echo "config generation failed"; exit 1; }
libdir() { if [ -d "$1/lib64" ]; then echo "$1/lib64"; else echo "$1/lib"; fi; }
SUNLIB=$(libdir "$SUNDIALS_DIR")
LIBS="-L$SUNLIB -Wl,-rpath,$SUNLIB -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsoldense \
      -lsundials_sunnonlinsolnewton -lsundials_generic -lm"
OPT="-O2 -std=c++20"
INC="-I$TREE -I$TREE/cooling/sgchem -I$TREE/cooling/sgchem/cxx -I$SUNDIALS_DIR/include"
[ -n "${KOKKOS_PATH:-}" ] && INC="$INC -I$KOKKOS_PATH/include"
CX=$TREE/cooling/sgchem/cxx
STATUS=0
for B in default forced; do
    X=""; [ $B = forced ] && X="-DTEST_SGCHEM_FORCED_SOLVER_FAILURE"
    O=$OUT/$B; mkdir -p $O/obj; OBJ=""
    for f in "$CX"/*.cc; do
        b=$(basename "$f" .cc); [ "$b" = chem_gizmo_backend ] && continue
        mpicxx $OPT $X $INC -c "$f" -o $O/obj/$b.o || { echo "compile $b ($B) failed"; exit 1; }
        OBJ="$OBJ $O/obj/$b.o"
    done
    DRIVERS="solver_failure"; [ $B = default ] && DRIVERS="solver_failure dust_temperature"
    for d in $DRIVERS; do
        mpicxx $OPT -Wall $X $INC -c "$HERE/$d.cc" -o $O/obj/$d.o || { echo "compile $d ($B) failed"; exit 1; }
        mpicxx $O/obj/$d.o $OBJ $LIBS -o $O/$d || { echo "link $d ($B) failed"; exit 1; }
    done
    timeout --kill-after=60 3600 $O/solver_failure "$CHEMCXX_DATADIR" "$HERE" $O/AH.csv > $O/solver_failure.txt 2>&1 || STATUS=1
    echo "== net $NET $B solver_failure: $(grep -E 'GATE|=== SOLVER' $O/solver_failure.txt | tr '\n' ' ')"
    if [ $B = default ]; then
        timeout --kill-after=60 3600 $O/dust_temperature "$CHEMCXX_DATADIR" > $O/dust_temperature.txt 2>&1 || STATUS=1
        echo "== net $NET dust_temperature: $(grep -E 'GATE|^F[234] |=== DUST' $O/dust_temperature.txt | tr '\n' ' ')"
    fi
done
exit $STATUS
