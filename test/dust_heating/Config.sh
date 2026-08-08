#-------------------------------------------------------------------------------------------
# Protostellar dust heating test. One radiating sink, gravity off, background ISRF off, so the
# only thing under test is the radiation scheme.
#
# SELFGRAVITY_OFF is safe here: gravtree.cc keeps walking the tree when RT_USE_GRAVTREE is set,
# so the luminosity sum still happens while no forces are applied. It also stops the gas moving,
# which matters because the dust temperature is coupled to the gas and would otherwise be
# contaminated by dynamics.
#-------------------------------------------------------------------------------------------

SINGLE_STAR_STARFORGE_DEFAULTS
COOLING
SINGLE_STAR_FB_RT_HEATING        # the approximate (non-RHD) protostellar heating path
SINK_DUST_HEATING_PLANCKMEAN     # the scheme under test
BOX_PERIODIC
SELFGRAVITY_OFF
OPENMP=2

OUTPUT_TEMPERATURE
OUTPUT_DUST_TEMPERATURE          # the solved dust temperature; without it the test can only see
                                 # T_gas, which is a biased proxy (it lags T_dust by ~12%)

#SINK_DUST_HEATING_VERBOSE       # uncomment to log the emitter's internals per sink: rho0, r0,
                                 # tau(r0), the measured slope n_env, R_in/R_max/R_raw, T_phot and
                                 # which clamp fired. Essential when debugging the emitter -- the
                                 # alternative is inferring it from downstream temperatures, which
                                 # is how three wrong diagnoses got made.
