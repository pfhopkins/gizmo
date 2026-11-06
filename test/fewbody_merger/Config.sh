########################################
# fewbody_merger — a cross-rank sink-sink merger in a gas-free run, then a Hermite pass (test_merger.py).
# test/fewbody's build, kept separate so that its mergers stay compiled in whatever test/fewbody's
# Config does about them. Mergers are permitted by SinkMergerMaxMass in merger.params: the IC sinks of a
# gas-free run have no gas-cell mass to scale the default ceiling.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
IO_SUPPRESS_TIMEBIN_STDOUT=15
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
