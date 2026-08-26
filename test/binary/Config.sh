########################################
# binary — a single e=0.9, q=0.1 Kepler pair of Type 5 sinks, in physical units
# (pc - km/s - Msun). Two particles, no gas: the smallest configuration that
# exercises the sink integration end to end.
#
# SINGLE_STAR_STARFORGE_DEFAULTS (without the HYBRID_MODEL flag) is just the
# gravity / integration / sink glue -- HERMITE_INTEGRATION,
# GRAVITY_ACCURATE_FEWBODY_INTEGRATION, SINGLE_STAR_TIMESTEPPING,
# ADAPTIVE_TREEFORCE_UPDATE -- with no COOLING / MHD / RT, so anything this test
# measures is attributable to the integrator rather than to the physics modules.
#
# The point of the configuration: e=0.9 swings the required timestep by
# ((1+e)/(1-e))^3 ~ 6900 across the orbit, and q=0.1 means the two components
# want different bins, since each one's timestep responds to its own
# acceleration. One component changing timebin while the other does not is the
# configuration under which a hierarchical scheme stops conserving momentum.
#
# Softening is set 500x below pericentre in binary.params, so the force is
# Newtonian throughout and the kernel cannot mask an integration error.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
# Two particles: per-step timebin chatter would swamp the log and slow the run
# far more than it costs in any other test. TimeMax/2^15 ~ 1.9e-6, roughly
# P_orb/512, so sync-point lines still appear a few hundred times per orbit.
IO_SUPPRESS_TIMEBIN_STDOUT=15
# Energy and momentum are the quantities under test and both are differences of
# nearly-cancelling terms; single precision in the snapshots would put a floor
# around 1e-7 relative, above the effects we care about.
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
