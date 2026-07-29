#!/bin/bash
# Gravity-walk validation config.
# Stripped evrard config (no ADAPTIVE_GRAVSOFT_FORGAS) so all gating in
# gpu_gravtree.cc is satisfied. The GPU gravity tree is always active on Kokkos builds.
HYDRO_MESHLESS_FINITE_MASS
EOS_GAMMA=(5.0/3.0)
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
