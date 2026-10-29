#!/bin/bash
# Install the Stella stellar + SGCHEM tables these GIZMO sources expect into PREFIX, and check them against the hashes
# recorded in galaxy_sf/stella_tables_check.h (GIZMO checks the same hashes at startup). Run once per machine/version,
# before any run: GIZMO itself never builds or fetches tables, and needs no network.
#   get_stella_tables.sh [--into-tree GIZMO_DIR] [PREFIX]
#   PREFIX               install directory (default: $STELLA_TABLES_PREFIX, else ~/.stella-tables/<ref>)
#   STELLA_TABLES_FROM   take already-built tables from this directory (an install, or a stella-tables build/) instead of building
#   STELLA_TABLES_REPO   git URL or local path of stella-tables to build from (default: the public GitHub repo)
#   STELLA_TABLES_DIR    where to clone and build (default: PREFIX/src)
#   PYTHON               python with numpy and h5py, for the hash check (default: python3)
#   --into-tree DIR      also link the tables into DIR's in-tree default locations (galaxy_sf/stellar_tables/, cooling/sgchem/),
#                        for parameter files with the old default paths and for the SGCHEM_FORTRAN backend (hard-wired path)
# Then set in the parameter file: StellarTablesFile PREFIX/stellar_tables_unified.hdf5, SGCHEMTablesDir PREFIX
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
HDR="$GIZMO/galaxy_sf/stella_tables_check.h"
TREE=""; if [ "${1:-}" = "--into-tree" ]; then TREE="$(cd "$2" && pwd)"; shift 2; fi
REF="$(sed -n 's/^#define STELLA_TABLES_REF "\([0-9a-f]*\)".*/\1/p' "$HDR")"
PREFIX="${1:-${STELLA_TABLES_PREFIX:-$HOME/.stella-tables/$REF}}"
REPO="${STELLA_TABLES_REPO:-https://github.com/Ulli1991/stella-tables.git}"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"; else
    PY=python3
    if ! python3 -c "import h5py, numpy" 2>/dev/null; then
        [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
        if type module >/dev/null 2>&1 && module load python-waterboa >/dev/null 2>&1; then PY="$(command -v python3)"; fi
    fi
fi
FILES=(stellar_tables_unified.hdf5 stellar_wind_cumulative.hdf5 energy-table-x.txt eh2-table.txt H2-cooling-ratios.dat wss_z_collis.txt)

verify() { # verify DIR: every table present with its expected hash
    local d="$1" f want got
    for f in "${FILES[@]}"; do
        [ -e "$d/$f" ] || { echo "get_stella_tables: $d/$f missing" >&2; return 1; }
        want="$(sed -n "s/.*{\"$f\", *\"\([0-9a-f]*\)\"}.*/\1/p" "$HDR")"
        got="$("$PY" -I "$GIZMO/scripts/stella_tables_hash.py" "$d/$f" | cut -d' ' -f1)"
        [ "$got" = "$want" ] || { echo "get_stella_tables: $d/$f has hash $got, expected $want (stella-tables $REF)" >&2; return 1; }
    done
}

mkdir -p "$PREFIX"; PREFIX="$(cd "$PREFIX" && pwd)"
if ! verify "$PREFIX" 2>/dev/null; then
    if [ -n "${STELLA_TABLES_FROM:-}" ]; then
        SRC="$STELLA_TABLES_FROM"; echo "get_stella_tables: copying tables from $SRC"
    else
        DIR="${STELLA_TABLES_DIR:-$PREFIX/src}"
        [ -d "$DIR/.git" ] || git clone -q "$REPO" "$DIR"
        git -C "$DIR" fetch -q origin || true
        git -C "$DIR" checkout -q "$REF"
        echo "get_stella_tables: building tables from $REPO @ $REF in $DIR"
        (cd "$DIR" && ./build.sh) || { echo "get_stella_tables: stella-tables build or tests FAILED; nothing installed" >&2; exit 1; }
        SRC="$DIR/build"
    fi
    for f in "${FILES[@]}"; do
        if [ -e "$SRC/$f" ]; then cp "$SRC/$f" "$PREFIX/"; elif [ -e "$SRC/chemistry/$f" ]; then cp "$SRC/chemistry/$f" "$PREFIX/";
        else echo "get_stella_tables: $f not found in $SRC" >&2; exit 1; fi
    done
    verify "$PREFIX" || { echo "get_stella_tables: installed tables in $PREFIX do not match these sources" >&2; exit 1; }
    { echo "stella-tables: $REF"; echo "source: ${STELLA_TABLES_FROM:-$REPO}"; date; } > "$PREFIX/stella_tables.stamp"
fi
echo "get_stella_tables: $PREFIX holds stella-tables $REF (hashes checked)"
if [ -n "$TREE" ]; then
    mkdir -p "$TREE/galaxy_sf/stellar_tables" "$TREE/cooling/sgchem"
    for f in stellar_tables_unified.hdf5 stellar_wind_cumulative.hdf5; do ln -sfn "$PREFIX/$f" "$TREE/galaxy_sf/stellar_tables/$f"; done
    for f in energy-table-x.txt eh2-table.txt H2-cooling-ratios.dat wss_z_collis.txt; do ln -sfn "$PREFIX/$f" "$TREE/cooling/sgchem/$f"; done
    echo "get_stella_tables: linked into $TREE"
fi
echo "parameter file: StellarTablesFile $PREFIX/stellar_tables_unified.hdf5   SGCHEMTablesDir $PREFIX"
