########################################
# fewbody — suite of independent cold-collapse 3-10 body problems (Type 5
# stars) in physical units (pc - km/s - Msun). Identical to plummer_binaries'
# Config by design: the two tests differ only in their initial condition, so
# anything they disagree on is the IC and not the build. Uses SINGLE_STAR_STARFORGE_DEFAULTS
# which (without the HYBRID_MODEL flag) is just the gravity / integration /
# sink-machinery glue: HERMITE_INTEGRATION, GRAVITY_ACCURATE_FEWBODY_INTEGRATION,
# SINGLE_STAR_TIMESTEPPING, ADAPTIVE_TREEFORCE_UPDATE, etc. — no COOLING / MHD /
# RT. Pulls in sink/SF param defaults so the params file stays minimal.
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
# Energy is read from this in-code diagnostic (measured at full synchronization, independent of
# the output layer) rather than from snapshots.
ENERGY_BUDGET_DIAGNOSTIC
# Snapshots otherwise pair kick-time velocities with drift-time positions. With 3-10 stars that
# error does not average away, and sum(m*v) over stars last kicked at different instants is not
# the momentum at any one time; these datasets give a consistent state.
IO_HERMITE_SYNC
