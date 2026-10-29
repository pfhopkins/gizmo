#!/bin/bash
# Compare the current C++ chemistry against the frozen reference in data/ (no Fortran needed).
# Usage: check_reference.sh <gizmo tree> <scratch dir> [nets...]   (default nets: 3 5 15 16 17 18 20)
# data/fortran_netN: Fortran init/kernels/DVODE at 63cfeb88 (nets 3 5 15 16 17).
# data/cxx_netN:     C++ backend at the Fortran-free switch (nets 18, 20 exist only in C++).
# Regenerate a Fortran reference (needs the .F sources): build_dump.sh fortran N <tree> <dir>; reference.py freeze <dir>/data data/fortran_netN
HERE=$(cd "$(dirname "$0")" && pwd)
TREE=${1:?gizmo tree}; OUT=${2:?scratch dir}; shift 2
NETS=${*:-3 5 15 16 17 18 20}
STATUS=0
for n in $NETS; do
    REF=$HERE/data/fortran_net$n; [ -d "$REF" ] || REF=$HERE/data/cxx_net$n
    bash "$HERE/build_dump.sh" cxx $n "$TREE" "$OUT/cxx_net$n" > "$OUT/cxx_net$n.log" 2>&1 || { echo "net $n: dump FAILED, see $OUT/cxx_net$n.log"; STATUS=1; continue; }
    python3 "$HERE/reference.py" compare "$REF" "$OUT/cxx_net$n/data" > "$OUT/compare_net$n.txt"; rc=$?
    echo "net $n vs $(basename $REF): $(grep -E '^cool_func|^onezone' "$OUT/compare_net$n.txt" | tr '\n' ' ') $(tail -1 "$OUT/compare_net$n.txt")"
    [ $rc = 0 ] || STATUS=1
done
exit $STATUS
