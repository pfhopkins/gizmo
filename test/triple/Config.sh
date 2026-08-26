########################################
# triple — a hierarchical triple of Type 5 sinks in physical units (pc - km/s -
# Msun): an equal-mass 4 AU inner binary under a 1 Msun tertiary at 100 AU.
#
# This is the many-timebin companion to test/binary. A bound PAIR can only ever
# split by ~1 bin (the symmetric 2-body criterion binds both members), so the
# binary test cannot exercise a deep hierarchy. Here the period ratio is 88, the
# tertiary sits ~6 timebins coarser than the inner stars, and the inner stars
# must evaluate it mid-step at drift depths spanning up to 2^6 of their own
# steps. That is the configuration in which the Hermite gravity passes see
# inactive companions far from a step boundary -- the regime the Old*-based
# source prediction in forcetree.cc exists for, and the configuration of the
# production momentum-conservation violation (a hardening triple).
#
# The equal-mass inner pair shares a timebin by symmetry, so it cannot leak
# against itself: any secular momentum drift is the inner<->outer channel.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
# Three particles: per-step timebin chatter would swamp the log. The occupancy
# table still prints at the coarse (tertiary) sync points, which is where the
# bin GAP is measurable -- all three particles are active there.
IO_SUPPRESS_TIMEBIN_STDOUT=15
# Both diagnostics are differences of nearly-cancelling terms; single precision
# in the snapshots would floor them around 1e-7 relative.
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
