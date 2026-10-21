#!/bin/bash
# Build and install the Stella stellar + CHEMCOOL tables from the stella-tables repo when they are missing or were not
# built from the pinned commit (per stella_tables.stamp); tables replaced that way are kept with a .replaced-<date> suffix.
# Called by the Makefile (order-only prerequisite of the executable); safe to run by hand from the GIZMO root.
#   STELLA_TABLES_REPO  git URL or local path of stella-tables (default: the public GitHub repo)
#   STELLA_TABLES_REF   commit/tag to build (default: the version pinned below)
#   STELLA_TABLES_DIR   where to clone and build (default: .stella-tables in the GIZMO root)
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
ST="$GIZMO/galaxy_sf/stellar_tables"; CC="$GIZMO/cooling/chemcool"
OUTS=("$ST/stellar_tables_unified.hdf5" "$ST/stellar_wind_cumulative.hdf5" "$CC/energy-table-x.txt")

REPO="${STELLA_TABLES_REPO:-https://github.com/Ulli1991/stella-tables.git}"
REF="${STELLA_TABLES_REF:-a4830ab18252c8786539e95598fc789f3b6d5142}"   # stella-tables version these GIZMO sources expect
DIR="${STELLA_TABLES_DIR:-$GIZMO/.stella-tables}"
STAMP="$ST/stella_tables.stamp"

missing=0; for f in "${OUTS[@]}"; do [ -e "$f" ] || missing=1; done
if [ $missing = 0 ]; then
    grep -qx "commit: $REF" "$STAMP" 2>/dev/null && exit 0
    echo "get_stella_tables: installed tables are not stella-tables $REF ($(grep -s '^commit:' "$STAMP" || echo 'no stamp')); rebuilding" >&2
fi

if [ -e "$DIR" ] && [ ! -d "$DIR/.git" ]; then mv "$DIR" "$DIR.incomplete-$(date +%Y%m%d%H%M%S)"; fi   # e.g. copied without .git
[ -d "$DIR/.git" ] || git clone -q "$REPO" "$DIR"
git -C "$DIR" fetch -q origin
git -C "$DIR" checkout -q "$REF"
SRC="$DIR"
echo "get_stella_tables: building tables from $REPO @ $REF in $SRC"

if ! (cd "$SRC" && ./build.sh); then echo "get_stella_tables: stella-tables build or tests FAILED; nothing installed" >&2; exit 1; fi

B="$SRC/build"
for f in eh2-table.txt H2-cooling-ratios.dat wss_z_collis.txt; do  # GIZMO tracks these; they must be the tables the energy table was built from
    cmp -s "$B/chemistry/$f" "$CC/$f" || { echo "get_stella_tables: $CC/$f differs from stella-tables; not installing" >&2; exit 1; }
done
old="replaced-$(date +%Y%m%d%H%M%S)"; for f in "${OUTS[@]}"; do [ -e "$f" ] && mv "$f" "$f.$old"; done
cp "$B/stellar_tables_unified.hdf5" "$B/stellar_wind_cumulative.hdf5" "$ST/"
cp "$B/chemistry/energy-table-x.txt" "$CC/"
{ echo "source: $REPO"; echo "commit: $(git -C "$SRC" rev-parse -q --verify HEAD || echo uncommitted)"; date; } > "$ST/stella_tables.stamp"
echo "get_stella_tables: installed ${OUTS[*]}"
