########################################
# single_stars_stochastic_imf -- GALSF_SINGLE_STARS_STOCHASTIC_IMF: standard star formation converts whole 1 Msun cells; each new
# star particle carries one drawn massive star with probability m/MassPerStarIMF (M^-2.3 in [MinMassIMF, MaxMassIMF]).
########################################
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
METALS
GALSF
GALSF_FB_MECHANICAL
GALSF_SINGLE_STARS_FROM_IMF
GALSF_SINGLE_STARS_EVOLUTION_TABLES
GALSF_SINGLE_STARS_STOCHASTIC_IMF
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
STOP_WHEN_BELOW_MINTIMESTEP
GALSF_SINGLE_STARS_WINDS
