########################################
# cbe_harmonic_1d — 1D hot Gaussian in a fixed analytic harmonic potential.
#
# Type=1 particles carry a 2-basis internal velocity distribution; the external
# potential a = -Omega^2 (x - x_c) (Omega=1, x_c = BoxSize/2) makes a Gaussian
# f ~ exp[-(v^2 + Omega^2 (x-x_c)^2)/(2 sigma^2)] an exact stationary CBE
# equilibrium. The IC seeds a width mismatch (xwidth_factor=0.7) so the analytic
# response is an UNDAMPED breathing oscillation at 2*Omega; the test checks the
# breathing envelope is preserved (no numerical damping/collapse) and that
# mass / second-moment stay bounded. NBASIS=2, NMOMENTS=3 (m, p_x, T_xx) in 1D
# (SECONDMOMENT + WITHGRADIENTS = the production CBE default).
########################################

BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF
GRAVITY_ANALYTIC
GRAVITY_ANALYTIC_HARMONIC

CBE_INTEGRATOR=2
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_RP_GAUSSIAN
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
