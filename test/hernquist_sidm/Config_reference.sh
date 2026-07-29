########################################
# hernquist_sidm — CPU reference Config (no Kokkos, no GPU).
#
# Used to build the reference solution that the GPU path is validated
# against. Build with SYSTYPE="MacBookCellar" (non-Kokkos variant) for
# the CPU-only code path.
########################################

BOX_SPATIAL_DIMENSION=3
ADAPTIVE_GRAVSOFT_FORALL=2
DM_SIDM=2
