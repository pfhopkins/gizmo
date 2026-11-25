#!/bin/bash
# One-time install of KETJU, which GIZMO does not carry or fetch (https://github.com/pfhopkins/gizmo/issues/112): MSTAR (the
# C++/Kokkos ketju-integrator) and the ketju-integrator-c headers (the C library is not built or used), at the pins in
# the Makefile, built with this SYSTYPE's compiler against its Kokkos (KOKKOS_ROOT), into the KETJU_PATH make reports.
# Afterwards make finds it by itself. Run it where you build (same modules and SYSTYPE). Rerun = check only.
# The Stella tables are installed by scripts/get_stella_tables.sh.
#   get_stella_deps.sh [--prefix DIR]
#   --prefix DIR  cache directory (default: $STELLA_DEPS_PREFIX, else ~/.stella-deps); for another one, export
#                 STELLA_DEPS_PREFIX=DIR for make too
# Needs git and cmake.
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${STELLA_DEPS_PREFIX:-$HOME/.stella-deps}"
while [ $# -gt 0 ]; do case "$1" in
    --prefix) PREFIX="$2"; shift;;
    *) echo "usage: get_stella_deps.sh [--prefix DIR]" >&2; exit 2;; esac; shift; done
mkdir -p "$PREFIX"; PREFIX="$(cd "$PREFIX" && pwd)"; export STELLA_DEPS_PREFIX="$PREFIX"
die() { echo "get_stella_deps: $*" >&2; exit 1; }

ask() { make -s --no-print-directory -C "$GIZMO" CONFIGVARS=KETJU_REGULARIZATION ketju-path 2> /dev/null | grep "^KETJU_[A-Z_]*='"; }
eval "$(ask)" || true
[ -n "${KETJU_PATH:-}" ] || die "make did not report the KETJU install path; run 'make CONFIGVARS=KETJU_REGULARIZATION ketju-path' to see why"
[ -d "$KETJU_KOKKOS/include" ] || die "no Kokkos install at KOKKOS_ROOT='$KETJU_KOKKOS'; set KOKKOS_ROOT (SYSTYPE block, environment or make line) first"
STAMP="mstar $KETJU_REF, ketju-integrator-c $KETJU_C_REF, Kokkos $KETJU_KOKKOS, $KETJU_CXX"
if [ "$KETJU_OK" = 1 ] && [ "$(cat "$KETJU_PATH/ketju.stamp" 2> /dev/null)" = "$STAMP" ]; then K="already installed"
elif [ "$KETJU_OK" = 1 ] && [ "${KETJU_PATH#$PREFIX/}" = "$KETJU_PATH" ]; then K="your KETJU_PATH, not managed here"
else
    SRC="$PREFIX/.build-${KETJU_PATH##*/}"; rm -rf "$SRC"; mkdir -p "$SRC"
    echo "get_stella_deps: building MSTAR ${KETJU_REF:0:7} (C headers ${KETJU_C_REF:0:7}) with $KETJU_CXX against Kokkos $KETJU_KOKKOS"
    git clone -q "$KETJU_REPO" "$SRC/src" && git -C "$SRC/src" checkout -q "$KETJU_REF" || die "cannot get $KETJU_REPO @ $KETJU_REF"
    { cmake -S "$SRC/src" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$KETJU_CXX" -DKokkos_ROOT="$KETJU_KOKKOS" \
          -DMSTAR_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$KETJU_PATH" -DCMAKE_INSTALL_LIBDIR=lib \
      && cmake --build "$SRC/build" -j 8 && cmake --install "$SRC/build"; } > "$SRC/build.log" 2>&1 || die "MSTAR build failed, see $SRC/build.log"
    git clone -q "$KETJU_REPO" "$SRC/src_c" && git -C "$SRC/src_c" checkout -q "$KETJU_C_REF" || die "cannot get $KETJU_REPO @ $KETJU_C_REF"
    # headers only (GIZMO's data model): the C library is not built
    mkdir -p "$KETJU_PATH/include" && cp -r "$SRC/src_c/include/ketju_integrator" "$KETJU_PATH/include/" || die "cannot copy the ketju-integrator-c headers"
    echo "$STAMP" > "$KETJU_PATH/ketju.stamp"; rm -rf "$SRC"
    eval "$(ask)"; [ "$KETJU_OK" = 1 ] || die "make does not find the install at $KETJU_PATH"
    K="built"
fi
echo "get_stella_deps: KETJU $KETJU_PATH (MSTAR ${KETJU_REF:0:7}, C headers ${KETJU_C_REF:0:7}, Kokkos $KETJU_KOKKOS, $K)"
[ "$PREFIX" = "$HOME/.stella-deps" ] || echo "get_stella_deps: non-default cache: export STELLA_DEPS_PREFIX=$PREFIX for make"
echo "get_stella_deps: all checks passed; make finds KETJU by itself"
