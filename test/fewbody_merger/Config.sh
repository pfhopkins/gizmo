########################################
# fewbody_merger — a sink-sink merger in a gas-free run, on one rank and across two (test_fewbody_merger.py).
# Mergers are permitted by SinkMergerMaxMass in fewbody_merger.params: the IC sinks of a gas-free run have
# no gas-cell mass to scale the default ceiling.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
IO_SUPPRESS_TIMEBIN_STDOUT=15
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
