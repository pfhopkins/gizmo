#!/bin/bash
# Build hypre for GIZMO on Rusty, for MHD_MODIFIED_GRADIENT's AMG-preconditioned solver.
#
#   ./scripts/build_hypre_rusty.sh [install_prefix]      # default $HOME/opt/hypre
#
# Idempotent: exits immediately if <prefix>/include/HYPRE.h already exists, so it is safe to
# call unconditionally from a batch script.
#
# Why this exists: there is no hypre on Rusty as of 2026-09. `module spider hypre` finds
# nothing, the nix store has no hypre derivation, and neither petsc/3.22.4 nor petsc/3.25.2
# bundles it (checked both prefixes: no HYPRE.h, no libHYPRE). petsc/3.25.2 also lives in the
# modules/2.5 tree, so linking it against this tree's modules/2.4 toolchain would be an ABI
# mismatch regardless.
#
# Without it, MHD_MODIFIED_GRADIENT cannot compile -- mg_gradient_correction.cc includes
# HYPRE.h unless MHD_MODIFIED_GRADIENT_CG_ONLY or _USE_PARDISO is set -- which fails
# test/field_loop's MG variant and the one compile_suite entry that uses the flag.
#
# If the node has no outbound https, fetch the tarball elsewhere, untar it, and re-run with
# HYPRE_SRC=<that dir> set.
set -euo pipefail

PREFIX="${1:-$HOME/opt/hypre}"
HYPRE_VER="${HYPRE_VER:-2.32.0}"

if [ -f "$PREFIX/include/HYPRE.h" ]; then
    echo "hypre already installed at $PREFIX -- nothing to do"
    exit 0
fi

module --force purge
# The SAME toolchain GIZMO builds against; a hypre built elsewhere would ABI-clash.
module load modules/2.4-20250724 gcc openmpi
echo "=== building hypre $HYPRE_VER -> $PREFIX"
echo "=== $(gcc --version | head -1);  $(mpicc --version | head -1)"

SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT

if [ -n "${HYPRE_SRC:-}" ]; then
    echo "=== using pre-fetched source at $HYPRE_SRC"
    cp -a "$HYPRE_SRC" "$SRC/hypre"
else
    # curl, NOT git clone: the global git config here rewrites GitHub https URLs to ssh
    # (url.git@github.com:.insteadof https://github.com/), which has no key in a batch job.
    TARBALL="https://github.com/hypre-space/hypre/archive/refs/tags/v${HYPRE_VER}.tar.gz"
    echo "=== fetching $TARBALL"
    if ! curl -fsSL "$TARBALL" -o "$SRC/hypre.tar.gz"; then
        echo "FATAL: could not download hypre $HYPRE_VER."
        echo "       Fetch it on a machine with outbound https, untar it, and re-run:"
        echo "         HYPRE_SRC=<dir> $0 $PREFIX"
        exit 1
    fi
    mkdir -p "$SRC/hypre"
    tar -xzf "$SRC/hypre.tar.gz" -C "$SRC/hypre" --strip-components=1
fi

cd "$SRC/hypre/src"
./configure --prefix="$PREFIX" --enable-shared --without-superlu --without-openmp
# Cap parallelism: often run on a shared login node, where -j$(nproc) is antisocial.
make -j "${HYPRE_BUILD_JOBS:-8}"
make install

echo
if [ -f "$PREFIX/include/HYPRE.h" ]; then
    echo "=== installed: $PREFIX/include/HYPRE.h"
    ls "$PREFIX"/lib*/libHYPRE* 2>/dev/null | head -3
else
    echo "FATAL: build finished but $PREFIX/include/HYPRE.h is missing"
    exit 1
fi
