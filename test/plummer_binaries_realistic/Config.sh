########################################
# plummer_binaries_realistic -- a Plummer cluster whose binaries are drawn from the
# observed distributions (Kroupa IMF, mass-dependent binary fraction, log-normal
# periods, thermal eccentricities) rather than being identical 1000 AU equal-mass
# circular pairs.
#
# Same physics flags as test/plummer_binaries, so the two differ ONLY in their initial
# conditions and run length; any difference in conservation behaviour is attributable
# to the population, which is the point.
########################################
SINGLE_STAR_STARFORGE_DEFAULTS
# Print timebin / sync-point info only when active dt >= bin 45's cadence
# (dt = TimeMax/2^15 = 32.4/32768 ~= 9.9e-4 code units), which is roughly
# t_dyn(a)/680, i.e. just past 1/1000 of the cluster dynamical time.
# Cheaper than the default 16 (~ 1/1300 t_dyn); avoids the per-step IO
# bottleneck that 60 (= print every step) introduced.
IO_SUPPRESS_TIMEBIN_STDOUT=15
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
IO_HERMITE_SYNC
