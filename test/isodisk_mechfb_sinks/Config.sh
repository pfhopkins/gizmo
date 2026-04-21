# Isolated disk + FIRE BH sinks (D1 activation test)
# Exercises: sink_environment_gpu (B4), sink_feed_gpu (C3),
# sink_swallow_and_kick_gpu (D1). SINK_GRAVACCRETION=1 is the default
# under FIRE_BHS, so sink_environment_second_gpu (E2, gated on ==0) does NOT fire here.
# NO COSMIC RAYS - D1 Phase 1 #error-guards COSMIC_RAY_FLUID + SINK_COSMIC_RAYS,
# and FIRE_CRS=0 does NOT disable CRs - must omit FIRE_CRS entirely.

OUTPUT_ADDITIONAL_RUNINFO
FIRE_PHYSICS_DEFAULTS=3
FIRE_BHS
STOP_WHEN_BELOW_MINTIMESTEP
GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
