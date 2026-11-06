########################################
# single_stars_cr_speed: a CR slab free-streaming along B, no CR speed or exchange flag set (variant: speed of light)
########################################
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
MAGNETIC
METALS
GALSF
GALSF_FB_MECHANICAL
GALSF_SINGLE_STARS_FROM_IMF
GALSF_SINGLE_STARS_EVOLUTION_TABLES
OUTPUT_IN_DOUBLEPRECISION
STOP_WHEN_BELOW_MINTIMESTEP
COSMIC_RAY_FLUID
OUTPUT_ADDITIONAL_RUNINFO
