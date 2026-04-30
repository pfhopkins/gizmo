# CPU reference build — use SYSTYPE=MacBookCellar (non-Kokkos) for a CPU-only
# tree-walk reference. All Kokkos-enabled SYSTYPEs now always activate GPU
# neighbor-list and dispatch; there is no CPU-fallback mode within Kokkos builds.

OUTPUT_ADDITIONAL_RUNINFO
FIRE_PHYSICS_DEFAULTS=3
FIRE_BHS
STOP_WHEN_BELOW_MINTIMESTEP
OPENMP=2
