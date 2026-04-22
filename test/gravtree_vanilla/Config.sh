#!/bin/bash
# Phase 4 Tier 1a GPU gravity walk validation config.
# Stripped evrard config (no ADAPTIVE_GRAVSOFT_FORGAS) so all gating in
# gpu_gravtree.cc is satisfied. Validation pattern: build twice, once with
# GIZMO_GPU_GRAVTREE and once without; bit-compare snapshot output.
HYDRO_MESHLESS_FINITE_MASS
EOS_GAMMA=(5.0/3.0)
GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
GIZMO_GPU_GRAVTREE
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
