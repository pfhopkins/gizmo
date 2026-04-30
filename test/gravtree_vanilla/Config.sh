#!/bin/bash
# Phase 4 Tier 1a GPU gravity walk validation config.
# Stripped evrard config (no ADAPTIVE_GRAVSOFT_FORGAS) so all gating in
# gpu_gravtree.cc is satisfied. GPU gravity tree is always active on Kokkos builds.
HYDRO_MESHLESS_FINITE_MASS
EOS_GAMMA=(5.0/3.0)
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
