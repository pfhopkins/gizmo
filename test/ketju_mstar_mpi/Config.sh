########################################
# ketju_mstar_mpi — the plummer_binaries star setup (Type 5 stars, pc - km/s - Msun,
# SINGLE_STAR_STARFORGE_DEFAULTS) with KETJU: every binary is a chain region, spread over
# several MPI ranks. The test adds KETJU_MSTAR_MPI (regions integrated by all their compute tasks) as a variant.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
IO_SUPPRESS_TIMEBIN_STDOUT=15
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
KETJU_REGULARIZATION
KETJU_VERBOSE_INTEGRATION         # region sizes and compute-task ranges in stdout, asserted by the test
