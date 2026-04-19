########################################
# hernquist_sidm test — DM_SIDM activating test for the B2 AGSForce GPU port.
# Kokkos/GPU neighbor-list variant (default). For the CPU tree-walk reference
# used to validate the GPU output, comment the two GIZMO_USE_NEIGHBOR_LIST_*
# and OPENMP_GPU_OFFLOAD-related flags by switching SYSTYPE to MacBookCellar
# (non-Kokkos) or by using Config_cpu_reference.sh.
########################################

GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY

# Gravity + box setup (pure DM, no gas, no PM)
BOX_PERIODIC
BOX_SPATIAL_DIMENSION=3

# Adaptive softening for all particle types (required for AGSForce to
# actually dispatch on Type=1 DM — without this, bitmask=0 and the
# partition yields no groups, so the GPU kernel never fires).
ADAPTIVE_GRAVSOFT_FORALL=2

# SIDM: self-interacting dark matter — Type=1 scatters against Type=1
# (the DM_SIDM bitmask selects which types participate).
DM_SIDM=2
