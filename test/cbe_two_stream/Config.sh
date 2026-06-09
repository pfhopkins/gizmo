########################################
# cbe_two_stream — 1D counter-streaming CBE test (mirrors python_harness
# tests/test_two_stream_boosted + test_two_stream_density_wave rest-frame).
#
# Every Type=1 particle carries an internal 2-basis velocity distribution
# (+v_stream / -v_stream) supplied via the IC's VlasovMoments dataset (C7
# reader). No gravity, no hydro, no gas — pure CBE moment advection on a
# uniform 1D periodic mesh. NBASIS=2, NMOMENTS=3 (m, p_x, T_xx) in 1D
# (SECONDMOMENT + WITHGRADIENTS = the production CBE default).
########################################

BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF

# CBE collisionless-Boltzmann integrator; value = number of basis functions.
# SECONDMOMENT (per-basis stress) + WITHGRADIENTS (MFM face reconstruction)
# are the production default.
CBE_INTEGRATOR=2
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
