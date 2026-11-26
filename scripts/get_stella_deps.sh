#!/bin/bash
# One-time setup of what Stella needs outside this repository, into one cache directory; afterwards make and GIZMO find it
# by themselves (GIZMO never fetches anything). Run it where you build (same modules and SYSTYPE). Rerun = check only.
#   get_stella_deps.sh [--prefix DIR] [--ketju | --tables] [--update]
#   --prefix DIR  cache directory (default: $STELLA_DEPS_PREFIX, else ~/.stella-deps); for another one, export
#                 STELLA_DEPS_PREFIX=DIR for make and the runs too
#   --ketju       also install KETJU (automatic when src/Config.sh sets KETJU_REGULARIZATION)
#   --tables      only the tables (no SUNDIALS, no KETJU)
#   --update      KETJU: rebuild from the newest commit unless the install is already at it (the old install is kept as
#                 <dir>.prev); implies --ketju. Without it an existing install is only reported, with whether the branch
#                 has moved on
# Tables: the stella-tables release of STELLA_TABLES_REF (src/galaxy_sf/stella_tables_check.h), installed and checked by
# scripts/get_stella_tables.sh. SUNDIALS (CVODE, for SGChem): the release pinned in src/Makefile. KETJU: MSTAR (the C++/Kokkos
# ketju-integrator) with its C interface header, from the LATEST commit of the branch src/gravity/ketju/Makefile.ketju names
# (main; there is no pinned commit), built with this SYSTYPE's compiler against its Kokkos and with MPI sharing
# (MSTAR_ENABLE_MPI), at the KETJU_PATH make reports. The install directory carries no commit in its name; ketju.stamp in
# it records the commit that was built, and make and the run log print it. KETJU_REPO in the environment replaces the
# repository (e.g. a local mirror for nodes without network). Needs curl, git, cmake, python3 with h5py.
set -euo pipefail
GIZMO="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${STELLA_DEPS_PREFIX:-$HOME/.stella-deps}"; KETJU=0; ONLY_TABLES=0; UPDATE=0
grep -qsE '^[[:space:]]*KETJU_REGULARIZATION([[:space:]#]|$)' "$GIZMO/src/Config.sh" && KETJU=1
while [ $# -gt 0 ]; do case "$1" in
    --prefix) PREFIX="$2"; shift;; --ketju) KETJU=1;; --tables) ONLY_TABLES=1;; --update) UPDATE=1; KETJU=1;;
    *) echo "usage: get_stella_deps.sh [--prefix DIR] [--ketju | --tables] [--update]" >&2; exit 2;; esac; shift; done
[ $ONLY_TABLES = 0 ] || KETJU=0
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
    while read -r h f; do [ -e "$TABLES/$f" ] && echo "$h  $f"; done < "$DL/SHA256SUMS" > "$TABLES/SHA256SUMS"   # the files installed
    rm -rf "$DL"; T="installed from release $TAG"
fi
SUMMARY="get_stella_deps: tables  $TABLES (stella-tables $REF, $T, hashes checked)"

if [ $ONLY_TABLES = 0 ]; then
    SV="$(sed -n 's/^SUNDIALS_VERSION *= *//p' "$GIZMO/src/Makefile")"; SH="$(sed -n 's/^SUNDIALS_SHA256 *= *//p' "$GIZMO/src/Makefile")"
    SUN="$PREFIX/sundials-$SV"
    if [ -e "$SUN/include/cvode/cvode.h" ] && [ "$(cat "$SUN/sundials.stamp" 2> /dev/null)" = "$SH" ]; then S="already installed"
    else
        B="$PREFIX/.build-sundials-$SV"; rm -rf "$B" "$SUN"; mkdir -p "$B"
        echo "get_stella_deps: building SUNDIALS $SV"
        curl -fsSL --retry 3 -o "$B/sundials.tar.gz" "https://github.com/LLNL/sundials/releases/download/v$SV/sundials-$SV.tar.gz" || die "cannot download SUNDIALS $SV"
        (cd "$B" && echo "$SH  sundials.tar.gz" > SHA256SUMS && sumcheck) || die "SUNDIALS $SV tarball does not match SUNDIALS_SHA256 in src/Makefile"
        tar -xzf "$B/sundials.tar.gz" -C "$B"
        { cmake -S "$B/sundials-$SV" -B "$B/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SUN" -DCMAKE_INSTALL_LIBDIR=lib \
              -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_INSTALL=OFF \
              -DBUILD_ARKODE=OFF -DBUILD_CVODES=OFF -DBUILD_IDA=OFF -DBUILD_IDAS=OFF -DBUILD_KINSOL=OFF \
          && cmake --build "$B/build" -j 8 && cmake --install "$B/build"; } > "$B/build.log" 2>&1 || die "SUNDIALS build failed, see $B/build.log"
        echo "$SH" > "$SUN/sundials.stamp"; rm -rf "$B"; S="built"
    fi
    SUMMARY="$SUMMARY
get_stella_deps: SUNDIALS $SUN ($SV, $S)"
fi

if [ $KETJU = 1 ]; then
    ask() { make -s --no-print-directory -C "$GIZMO/src" CONFIGVARS=KETJU_REGULARIZATION ketju-path 2> /dev/null | grep "^KETJU_[A-Z_]*='"; }
    eval "$(ask)" || true
    [ -n "${KETJU_PATH:-}" ] || die "make did not report the KETJU install path; run 'make -C src CONFIGVARS=KETJU_REGULARIZATION ketju-path' to see why"
    [ -d "$KETJU_KOKKOS/include" ] || die "no Kokkos install at '$KETJU_KOKKOS' (from this SYSTYPE); set SYSTYPE and its Kokkos first"
    stamp() { [ -f "$KETJU_PATH/ketju.stamp" ] || return 0; sed -n "s/^$1: //p" "$KETJU_PATH/ketju.stamp" | head -1; }
    HAVE="$(stamp commit)"
    # newest commit of the branch (needs the network or a local KETJU_REPO; empty if it cannot be asked)
    HEAD_NOW="$(timeout 30 git ls-remote "$KETJU_REPO" "refs/heads/$KETJU_BRANCH" 2> /dev/null | cut -f1 || true)"
    if [ "$KETJU_OK" = 1 ] && [ -z "$HAVE" ]; then K="an install without ketju.stamp (yours, not managed here); its MSTAR commit is unknown"
    elif [ "$KETJU_OK" = 1 ] && [ $UPDATE = 0 ]; then
        if [ -z "$HEAD_NOW" ]; then S="$KETJU_BRANCH could not be asked for its newest commit"
        elif [ "$HEAD_NOW" = "$HAVE" ]; then S="this is the newest commit of $KETJU_BRANCH"
        else S="$KETJU_BRANCH is now at ${HEAD_NOW:0:7}: rerun with --update to install it"; fi
        K="MSTAR ${HAVE:0:7} (built $(stamp date)), already installed, $S"
    elif [ "$KETJU_OK" = 1 ] && [ -n "$HEAD_NOW" ] && [ "$HEAD_NOW" = "$HAVE" ]; then K="MSTAR ${HAVE:0:7} is the newest commit of $KETJU_BRANCH, nothing to update"
    else
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
        K="MSTAR ${COMMIT:0:7} installed${HAVE:+ (was ${HAVE:0:7}, kept as ${KETJU_PATH##*/}.prev)}"
    fi
    SUMMARY="$SUMMARY
get_stella_deps: KETJU   $KETJU_PATH: $K (Kokkos $KETJU_KOKKOS)"
fi
echo "$SUMMARY"
[ "$PREFIX" = "$HOME/.stella-deps" ] || echo "get_stella_deps: non-default cache: export STELLA_DEPS_PREFIX=$PREFIX for make and the runs"
echo "get_stella_deps: all checks passed; make and GIZMO find these by themselves"
