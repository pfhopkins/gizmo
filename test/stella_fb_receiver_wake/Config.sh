# the build of test/single_stars_supernovae (no gravity, no cooling: adiabatic) with the deposit before the hydro forces (the order of
# the Stella configurations), the total-energy-exact kick (its u limiter is what the test measures) and the test dumps
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
METALS
GALSF
GALSF_FB_MECHANICAL
GALSF_SINGLE_STARS_FROM_IMF
GALSF_SINGLE_STARS_EVOLUTION_TABLES
GALSF_FB_MECHANICAL_BEFORE_HYDRO_FORCES
HYDRO_KICK_TOTAL_ENERGY_EXACT
OUTPUT_IN_DOUBLEPRECISION
STOP_WHEN_BELOW_MINTIMESTEP
TEST_KICK_FLOOR_DUMP=1e-14
TEST_WAKEUP_DUMP
GALSF_SINGLE_STARS_WINDS
