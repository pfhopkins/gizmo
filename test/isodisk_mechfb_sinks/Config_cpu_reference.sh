# CPU tree-walk reference — same physics flags as Config.sh but NO
# GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY, so the density + gradient + hydro +
# mech_fb + sink_* dispatchers fall through to legacy CPU tree-walk.
# Built on MacBookCellar_Kokkos systype (MacBookCellar alone is broken per
# feedback_legacy_cpu_tree_path.md) but without the neighbor-list flag.

OUTPUT_ADDITIONAL_RUNINFO
FIRE_PHYSICS_DEFAULTS=3
FIRE_BHS
STOP_WHEN_BELOW_MINTIMESTEP
OPENMP=2
