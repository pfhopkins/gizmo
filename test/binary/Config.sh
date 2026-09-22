########################################
# binary — a single e=0.9, q=0.1 Kepler pair of Type 5 sinks in physical units
# (pc - km/s - Msun); two particles, no gas. See test_binary.py for why this
# configuration.
#
# SINGLE_STAR_STARFORGE_DEFAULTS (without the HYBRID_MODEL flag) is just the
# gravity / integration / sink glue -- no COOLING / MHD / RT -- so anything this
# test measures is attributable to the integrator. Softening (binary.params) is
# 500x below pericentre, so the force is Newtonian throughout.
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
