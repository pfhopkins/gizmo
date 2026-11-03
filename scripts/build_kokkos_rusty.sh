#!/bin/bash
# Build a standalone Kokkos (OpenMP backend, Zen4) for GIZMO on Rusty.
#
#   ./scripts/build_kokkos_rusty.sh [install_prefix]     # default $HOME/opt/kokkos-openmp
#
# Idempotent: exits immediately if <prefix>/include/Kokkos_Core.hpp already exists, so it is
# safe to call unconditionally from a batch script.
#
# Why this exists: there is no kokkos module on Rusty as of 2026-08. `module keyword kokkos`
# matches only trilinos, and trilinos/mpi-16.0.0 installs the Tpetra/Sacado adapter headers
# (KokkosCompat_*, Kokkos_DynRankView_Fad, ...) but NOT Kokkos_Core.hpp, and ships
# libkokkostsqr without libkokkoscore/libkokkoscontainers. So it cannot satisfy this tree,
# which needs standalone Kokkos for every GPU_OBJS translation unit even in a CPU-only build.
set -euo pipefail

PREFIX="${1:-$HOME/opt/kokkos-openmp}"
KOKKOS_VER="${KOKKOS_VER:-4.4.01}"

if [ -f "$PREFIX/include/Kokkos_Core.hpp" ]; then
    echo "kokkos already installed at $PREFIX -- nothing to do"
    exit 0
fi

module --force purge
module load modules/2.4-20250724 gcc openmpi cmake
echo "=== building Kokkos $KOKKOS_VER -> $PREFIX"
echo "=== $(gcc --version | head -1);  $(cmake --version | head -1)"

SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

# KOKKOS_SRC: an unpacked Kokkos source tree, for nodes without outbound https.
# Otherwise fetch with curl, NOT git clone: a global git config with
#   url.git@github.com:.insteadof https://github.com/
# rewrites GitHub https URLs to ssh, which has no key inside a batch job.
if [ -n "${KOKKOS_SRC:-}" ]; then
    KSRC="$KOKKOS_SRC"
else
    TARBALL="https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VER}.tar.gz"
    echo "=== fetching $TARBALL"
    if ! curl -fsSL "$TARBALL" -o "$SRC/kokkos.tar.gz"; then
        echo "FATAL: could not download Kokkos $KOKKOS_VER. Without outbound https, fetch and unpack"
        echo "       the tarball elsewhere and re-run with KOKKOS_SRC=<unpacked dir>."
        exit 1
    fi
    KSRC="$SRC/kokkos"
    mkdir -p "$KSRC"
    tar -xzf "$SRC/kokkos.tar.gz" -C "$KSRC" --strip-components=1
fi

cmake -S "$KSRC" -B "$SRC/build" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ARCH_ZEN4=ON
# Capped: this often runs on a shared login node.
cmake --build "$SRC/build" -j "${KOKKOS_BUILD_JOBS:-16}"
cmake --install "$SRC/build"

echo "=== installed:"
ls "$PREFIX/include/Kokkos_Core.hpp"
ls "$PREFIX"/lib64/libkokkoscore.* 2>/dev/null || ls "$PREFIX"/lib/libkokkoscore.*
echo "=== build GIZMO with SYSTYPE=RUSTY_GENOA (add KOKKOS_PATH=$PREFIX if not the default prefix)"
