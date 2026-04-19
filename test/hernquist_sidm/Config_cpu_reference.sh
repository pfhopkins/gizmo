########################################
# hernquist_sidm — CPU tree-walk REFERENCE Config (no Kokkos, no GPU).
#
# Used to build the reference solution that the GPU path is validated
# against. Identical to Config.sh except GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
# is OFF so the code follows the legacy ngb-tree-walk path. Build with
# SYSTYPE="MacBookCellar" (non-Kokkos variant) to make sure OPENMP_GPU_OFFLOAD
# is not defined either.
########################################

BOX_PERIODIC
BOX_SPATIAL_DIMENSION=3
ADAPTIVE_GRAVSOFT_FORALL=2
DM_SIDM=2
