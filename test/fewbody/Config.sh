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
# The energy budget is read from this, not from snapshots: io.cc writes Velocities at kick-time
# against drift-time positions, an O(dt/2t_dyn) ~ percent error per particle. With 512 stars those
# average away (plummer_binaries measures 5e-4 that way); with 3-10, where one star can hold a
# third of the binding energy, they do not, and no snapshot estimate can resolve a 1% budget.
ENERGY_BUDGET_DIAGNOSTIC
