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

# Fetch with curl, NOT git clone. The global git config here carries
#   url.git@github.com:.insteadof https://github.com/
# so git silently rewrites any GitHub https URL to ssh, which has no key inside a batch job
# ("Permission denied (publickey)"). curl is not subject to git's URL rewriting, and a release
# tarball is smaller and faster than a clone anyway.
TARBALL="https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VER}.tar.gz"
echo "=== fetching $TARBALL"
if ! curl -fsSL "$TARBALL" -o "$SRC/kokkos.tar.gz"; then
    echo "FATAL: could not download Kokkos $KOKKOS_VER."
    echo "       If this node has no outbound https, fetch the tarball on a login node and untar it to"
    echo "       a directory, then re-run:  $0 $PREFIX  with KOKKOS_SRC=<that dir> set."
    exit 1
fi
mkdir -p "$SRC/kokkos"
tar -xzf "$SRC/kokkos.tar.gz" -C "$SRC/kokkos" --strip-components=1

cmake -S "$SRC/kokkos" -B "$SRC/build" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DKokkos_ENABLE_OPENMP=ON \
      -DKokkos_ENABLE_SERIAL=ON \
      -DKokkos_ARCH_ZEN4=ON
# Cap parallelism: this is often run on a shared login node, where -j$(nproc) is antisocial.
cmake --build "$SRC/build" -j "${KOKKOS_BUILD_JOBS:-16}"
cmake --install "$SRC/build"

echo "=== installed:"
ls "$PREFIX/include/Kokkos_Core.hpp"
ls "$PREFIX"/lib64/libkokkoscore.* 2>/dev/null || ls "$PREFIX"/lib/libkokkoscore.*
echo "=== now build GIZMO with:  make KOKKOS_PATH=$PREFIX"
