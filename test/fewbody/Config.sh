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
# The energy budget is read from this, not from snapshots: it is measured in-code at full
# synchronization, which is denser than the snapshot cadence and independent of what the output
# layer writes. Kept as the ground truth even though IO_HERMITE_SYNC (below) now makes a
# snapshot-based estimate viable -- two independent measurements of the same quantity is worth
# more here than one, and this one has no output convention to get wrong.
ENERGY_BUDGET_DIAGNOSTIC
# Without this, snapshots carry Velocities at kick-time against drift-time positions -- an
# O(dt/2 t_dyn) ~ percent error per particle. With 512 stars those average away
# (plummer_binaries measured 5e-4 that way); with 3-10, where one star can hold a third of the
# binding energy, they do not. That is why the energy budget above is read in-code instead.
#
# It matters for MOMENTUM independently of energy: sum(m*v) over stars last kicked at different
# instants is not the momentum at any single time, and momentum is the quantity that responds
# most sharply to the Hermite source-prediction fix -- measured across these 48 problems at 22x
# (tree) and 48x (direct_gravity) against the pre-fix baseline, versus ~2x on energy. With the
# mixed state that floor is partly sampling rather than integration error; these datasets remove
# the ambiguity.
IO_HERMITE_SYNC
