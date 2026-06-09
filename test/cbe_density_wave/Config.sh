########################################
# cbe_density_wave — 1D density-modulated counter-streaming CBE test
# (mirrors python_harness tests/test_two_stream_density_wave).
#
# Same 2-basis (+v/-v) internal distribution as cbe_two_stream, but the
# particle SPACING is inverse-CDF sampled from rho(x) = 1 + eps cos(2 pi x/L)
# so the total density advects as a wave while each stream advects at +/-v.
# Type=1, no gravity/hydro/gas. NBASIS=2, NMOMENTS=3 (m, p_x, T_xx) in 1D
# (SECONDMOMENT + WITHGRADIENTS = the production CBE default).
########################################

BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF

CBE_INTEGRATOR=2
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
