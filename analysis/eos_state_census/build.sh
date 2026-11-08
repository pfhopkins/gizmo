#!/bin/bash
# Builds libcensus.so (used by census.py) from census_lib.cc and this GIZMO tree's own src/eos/aneos.cc, so the census evaluates states
# exactly as the run does, and records what it was built from in libcensus.build.txt.
#   build.sh [gizmo_tree]   (default: the repository this directory belongs to)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="${1:-$(cd "$HERE/../.." && pwd)}"
test -f "$TREE/src/eos/aneos.cc" || { echo "no src/eos/aneos.cc under $TREE"; exit 1; }
rm -f "$HERE/libcensus.so" "$HERE/libcensus.build.txt"
mpicxx -std=c++17 -O2 -shared -fPIC -DEOS_ANEOS -DGIZMO_config_H -DANEOS_MAX_MATERIALS=64 -I "$TREE" \
    "$HERE/census_lib.cc" "$TREE/src/eos/aneos.cc" -o "$HERE/libcensus.so" -lm
{
  echo "tree $TREE"
  echo "tree_head $(git -C "$TREE" rev-parse HEAD 2>/dev/null)"
  echo "aneos_h_blob $(git -C "$TREE" hash-object src/eos/aneos.h 2>/dev/null)"
  echo "aneos_cc_blob $(git -C "$TREE" hash-object src/eos/aneos.cc 2>/dev/null)"
  echo "census_lib_sha256 $(shasum -a 256 "$HERE/census_lib.cc" | cut -c1-64)"
} > "$HERE/libcensus.build.txt"
cat "$HERE/libcensus.build.txt"
