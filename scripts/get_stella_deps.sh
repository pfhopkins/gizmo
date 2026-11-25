#!/bin/bash
# Install of KETJU, which GIZMO does not carry or fetch (https://github.com/pfhopkins/gizmo/issues/112): MSTAR (the
# C++/Kokkos ketju-integrator) with its C interface header, from the LATEST commit of the branch the Makefile names
# (main; there is no pinned commit), built with this SYSTYPE's compiler against its Kokkos (KOKKOS_ROOT) and with MPI
# sharing (MSTAR_ENABLE_MPI), into the KETJU_PATH make reports. The install directory does not carry a commit in its
# name; ketju.stamp in it records the commit that was built, and make and the run log print it.
# Run it where you build (same modules and SYSTYPE). The Stella tables are installed by scripts/get_stella_tables.sh.
#   get_stella_deps.sh [--prefix DIR] [--update]
#   (no option)   install if there is none; otherwise report which commit is installed and whether main has moved on
#   --update      rebuild from the newest main unless the install is already at it (the old install is kept as <dir>.prev)
#   --prefix DIR  cache directory (default: $STELLA_DEPS_PREFIX, else ~/.stella-deps); for another one, export
#                 STELLA_DEPS_PREFIX=DIR for make too
# KETJU_REPO in the environment replaces the repository (e.g. a local mirror for nodes without network).
# Needs git and cmake.
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${STELLA_DEPS_PREFIX:-$HOME/.stella-deps}"; UPDATE=0
while [ $# -gt 0 ]; do case "$1" in
    --prefix) PREFIX="$2"; shift;;
    --update) UPDATE=1;;
    *) echo "usage: get_stella_deps.sh [--prefix DIR] [--update]" >&2; exit 2;; esac; shift; done
mkdir -p "$PREFIX"; PREFIX="$(cd "$PREFIX" && pwd)"; export STELLA_DEPS_PREFIX="$PREFIX"
die() { echo "get_stella_deps: $*" >&2; exit 1; }

ask() { make -s --no-print-directory -C "$GIZMO" CONFIGVARS=KETJU_REGULARIZATION ketju-path 2> /dev/null | grep "^KETJU_[A-Z_]*='"; }
eval "$(ask)" || true
[ -n "${KETJU_PATH:-}" ] || die "make did not report the KETJU install path; run 'make CONFIGVARS=KETJU_REGULARIZATION ketju-path' to see why"
[ -d "$KETJU_KOKKOS/include" ] || die "no Kokkos install at KOKKOS_ROOT='$KETJU_KOKKOS'; set KOKKOS_ROOT (SYSTYPE block, environment or make line) first"
stamp() { [ -f "$KETJU_PATH/ketju.stamp" ] || return 0; sed -n "s/^$1: //p" "$KETJU_PATH/ketju.stamp" | head -1; }
HAVE="$(stamp commit)"
if [ "$KETJU_OK" = 1 ] && [ -z "$HAVE" ]; then
    echo "get_stella_deps: KETJU $KETJU_PATH: an install without ketju.stamp (yours, not managed here); its MSTAR commit is unknown"
    exit 0
fi
# newest commit of the branch (needs the network or a local KETJU_REPO; empty if it cannot be asked)
HEAD_NOW="$(timeout 30 git ls-remote "$KETJU_REPO" "refs/heads/$KETJU_BRANCH" 2> /dev/null | cut -f1 || true)"
if [ "$KETJU_OK" = 1 ] && [ "$UPDATE" = 0 ]; then
    if [ -z "$HEAD_NOW" ]; then S="$KETJU_BRANCH could not be asked for its newest commit"
    elif [ "$HEAD_NOW" = "$HAVE" ]; then S="this is the newest commit of $KETJU_BRANCH"
    else S="$KETJU_BRANCH is now at ${HEAD_NOW:0:7}: rerun with --update to install it"; fi
    echo "get_stella_deps: KETJU $KETJU_PATH: MSTAR ${HAVE:0:7} (built $(stamp date)), $S"
    exit 0
fi
if [ "$KETJU_OK" = 1 ] && [ -n "$HEAD_NOW" ] && [ "$HEAD_NOW" = "$HAVE" ]; then
    echo "get_stella_deps: KETJU $KETJU_PATH: MSTAR ${HAVE:0:7} is the newest commit of $KETJU_BRANCH, nothing to update"
    exit 0
fi
[ "${KETJU_PATH#$PREFIX/}" != "$KETJU_PATH" ] || die "KETJU_PATH=$KETJU_PATH is not under $PREFIX: not installing into a directory this script does not manage"
SRC="$PREFIX/.build-${KETJU_PATH##*/}"; rm -rf "$SRC"; mkdir -p "$SRC"
git clone -q --depth 1 --branch "$KETJU_BRANCH" "$KETJU_REPO" "$SRC/src" 2> "$SRC/clone.log" || die "cannot clone $KETJU_REPO branch $KETJU_BRANCH, see $SRC/clone.log"
COMMIT="$(git -C "$SRC/src" rev-parse HEAD)"
[ -f "$SRC/src/include/ketju_integrator/ketju_integrator.h" ] \
    || die "$KETJU_REPO $KETJU_BRANCH (${COMMIT:0:7}) has no include/ketju_integrator/ketju_integrator.h (the C interface header GIZMO uses as its data model)"
echo "get_stella_deps: building MSTAR ${COMMIT:0:7} ($KETJU_BRANCH of $KETJU_REPO) with $KETJU_CXX against Kokkos $KETJU_KOKKOS, MPI sharing on"
NEW="$KETJU_PATH.new"; rm -rf "$NEW"
{ cmake -S "$SRC/src" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$KETJU_CXX" -DKokkos_ROOT="$KETJU_KOKKOS" \
      -DMSTAR_ENABLE_MPI=ON -DMSTAR_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$NEW" -DCMAKE_INSTALL_LIBDIR=lib \
  && cmake --build "$SRC/build" -j 8 && cmake --install "$SRC/build"; } > "$SRC/build.log" 2>&1 || die "MSTAR build failed, see $SRC/build.log"
{ echo "commit: $COMMIT"; echo "branch: $KETJU_BRANCH"; echo "repository: $KETJU_REPO"; echo "mpi: MSTAR_ENABLE_MPI=ON"
  echo "kokkos: $KETJU_KOKKOS"; echo "compiler: $KETJU_CXX"; echo "date: $(date +%Y-%m-%dT%H:%M:%S)"; } > "$NEW/ketju.stamp"
if [ -e "$KETJU_PATH" ]; then rm -rf "$KETJU_PATH.prev"; mv "$KETJU_PATH" "$KETJU_PATH.prev"; fi
mv "$NEW" "$KETJU_PATH"; rm -rf "$SRC"
eval "$(ask)"; [ "$KETJU_OK" = 1 ] || die "make does not find the install at $KETJU_PATH"
echo "get_stella_deps: KETJU $KETJU_PATH: MSTAR ${COMMIT:0:7} installed${HAVE:+ (was ${HAVE:0:7}, kept as ${KETJU_PATH##*/}.prev)}"
[ "$PREFIX" = "$HOME/.stella-deps" ] || echo "get_stella_deps: non-default cache: export STELLA_DEPS_PREFIX=$PREFIX for make"
