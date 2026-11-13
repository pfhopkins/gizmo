#!/bin/bash
# One-time setup of what Stella needs outside this repository, into one cache directory; afterwards make and GIZMO find it
# by themselves (GIZMO never fetches anything). Run it where you build (same modules and SYSTYPE). Rerun = check only.
#   get_stella_deps.sh [--prefix DIR] [--ketju | --tables]
#   --prefix DIR  cache directory (default: $STELLA_DEPS_PREFIX, else ~/.stella-deps); for another one, export
#                 STELLA_DEPS_PREFIX=DIR for make and the runs too
#   --ketju       also install KETJU (automatic when src/Config.sh sets KETJU_REGULARIZATION)
#   --tables      only the tables
# Tables: the stella-tables release of STELLA_TABLES_REF (src/galaxy_sf/stella_tables_check.h), installed and checked by
# scripts/get_stella_tables.sh. KETJU: MSTAR and the ketju-integrator-c headers at the pins in src/gravity/ketju/Makefile.ketju,
# built against the Kokkos of this SYSTYPE, at the KETJU_PATH make reports. Needs curl, git, cmake, python3 with h5py.
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${STELLA_DEPS_PREFIX:-$HOME/.stella-deps}"; KETJU=0
grep -qsE '^[[:space:]]*KETJU_REGULARIZATION([[:space:]#]|$)' "$GIZMO/src/Config.sh" && KETJU=1
while [ $# -gt 0 ]; do case "$1" in
    --prefix) PREFIX="$2"; shift;; --ketju) KETJU=1;; --tables) KETJU=0;;
    *) echo "usage: get_stella_deps.sh [--prefix DIR] [--ketju | --tables]" >&2; exit 2;; esac; shift; done
mkdir -p "$PREFIX"; PREFIX="$(cd "$PREFIX" && pwd)"; export STELLA_DEPS_PREFIX="$PREFIX"
die() { echo "get_stella_deps: $*" >&2; exit 1; }
sumcheck() { if command -v sha256sum > /dev/null; then sha256sum --status -c SHA256SUMS; else shasum -a 256 --status -c SHA256SUMS; fi; }

REF="$(sed -n 's/^#define STELLA_TABLES_REF "\([0-9a-f]*\)".*/\1/p' "$GIZMO/src/galaxy_sf/stella_tables_check.h")"
TAG="tables-${REF:0:7}"; TABLES="$PREFIX/$TAG"
if (cd "$TABLES" 2> /dev/null && sumcheck); then T="already installed"
else
    URL="https://github.com/Ulli1991/stella-tables/releases/download/$TAG"; DL="$PREFIX/.download-$TAG"
    echo "get_stella_deps: downloading stella-tables release $TAG"
    rm -rf "$DL"; mkdir -p "$DL"
    curl -fsSL --retry 3 -o "$DL/SHA256SUMS" "$URL/SHA256SUMS" || die "cannot download $URL/SHA256SUMS"
    for f in $(awk '{print $2}' "$DL/SHA256SUMS"); do curl -fsSL --retry 3 -o "$DL/$f" "$URL/$f" || die "cannot download $URL/$f"; done
    (cd "$DL" && sumcheck) || die "release $TAG does not match its SHA256SUMS"
    STELLA_TABLES_FROM="$DL" STELLA_TABLES_SOURCE="release $TAG" bash "$GIZMO/scripts/get_stella_tables.sh" "$TABLES" > /dev/null
    cp "$DL/SHA256SUMS" "$TABLES/"; rm -rf "$DL"; T="installed from release $TAG"
fi
SUMMARY="get_stella_deps: tables  $TABLES (stella-tables $REF, $T, hashes checked)"

if [ $KETJU = 1 ]; then
    ask() { make -s --no-print-directory -C "$GIZMO/src" CONFIGVARS=KETJU_REGULARIZATION ketju-path 2> /dev/null | grep "^KETJU_[A-Z_]*='"; }
    eval "$(ask)" || true
    [ -n "${KETJU_PATH:-}" ] || die "make did not report the KETJU install path; run 'make -C src CONFIGVARS=KETJU_REGULARIZATION ketju-path' to see why"
    [ -d "$KETJU_KOKKOS/include" ] || die "no Kokkos install at '$KETJU_KOKKOS' (from this SYSTYPE); set SYSTYPE and its Kokkos first"
    STAMP="mstar $KETJU_REF, ketju-integrator-c $KETJU_C_REF, Kokkos $KETJU_KOKKOS, $KETJU_CXX"
    if [ "$KETJU_OK" = 1 ] && [ "$(cat "$KETJU_PATH/ketju.stamp" 2> /dev/null)" = "$STAMP" ]; then K="already installed"
    elif [ "$KETJU_OK" = 1 ] && [ "${KETJU_PATH#$PREFIX/}" = "$KETJU_PATH" ]; then K="your KETJU_PATH, not managed here"
    else
        SRC="$PREFIX/.build-${KETJU_PATH##*/}"; rm -rf "$SRC"; mkdir -p "$SRC"
        echo "get_stella_deps: building MSTAR ${KETJU_REF:0:7} with $KETJU_CXX against Kokkos $KETJU_KOKKOS"
        git clone -q "$KETJU_REPO" "$SRC/src" && git -C "$SRC/src" checkout -q "$KETJU_REF" || die "cannot get $KETJU_REPO @ $KETJU_REF"
        { cmake -S "$SRC/src" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$KETJU_CXX" -DKokkos_ROOT="$KETJU_KOKKOS" \
              -DMSTAR_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$KETJU_PATH" -DCMAKE_INSTALL_LIBDIR=lib \
          && cmake --build "$SRC/build" -j 8 && cmake --install "$SRC/build"; } > "$SRC/build.log" 2>&1 || die "MSTAR build failed, see $SRC/build.log"
        git -C "$SRC/src" archive "$KETJU_C_REF" include/ketju_integrator | tar -x -C "$KETJU_PATH" || die "cannot get the ketju-integrator-c headers"
        echo "$STAMP" > "$KETJU_PATH/ketju.stamp"; rm -rf "$SRC"
        eval "$(ask)"; [ "$KETJU_OK" = 1 ] || die "make does not find the install at $KETJU_PATH"
        K="built"
    fi
    SUMMARY="$SUMMARY
get_stella_deps: KETJU   $KETJU_PATH (MSTAR ${KETJU_REF:0:7}, C headers ${KETJU_C_REF:0:7}, Kokkos $KETJU_KOKKOS, $K)"
fi
echo "$SUMMARY"
[ "$PREFIX" = "$HOME/.stella-deps" ] || echo "get_stella_deps: non-default cache: export STELLA_DEPS_PREFIX=$PREFIX for make and the runs"
echo "get_stella_deps: all checks passed; make and GIZMO find these by themselves"
