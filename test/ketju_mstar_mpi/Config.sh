########################################
# ketju_mstar_mpi — a Plummer sphere of binaries (Type 5 stars, pc - km/s - Msun,
# SINGLE_STAR_STARFORGE_DEFAULTS) with KETJU: every binary is a chain region, spread over
# several MPI ranks. The test compares the default (regions shared by their compute tasks) with KETJU_MSTAR_SINGLE_TASK.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
IO_SUPPRESS_TIMEBIN_STDOUT=15
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
KETJU_REGULARIZATION
KETJU_VERBOSE_INTEGRATION         # region sizes and compute-task ranges in stdout, asserted by the test
