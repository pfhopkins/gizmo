########################################
# plummer_binaries — Plummer sphere of equal-mass circular binaries (Type 5
# stars) in physical units (pc - km/s - Msun). Exercises the gravity- and
# integration-relevant settings from SINGLE_STAR_STARFORGE_DEFAULTS.
########################################
SINGLE_STAR_SINK_DYNAMICS
HERMITE_INTEGRATION=32
ADAPTIVE_GRAVSOFT_FORALL=32
GRAVITY_ACCURATE_FEWBODY_INTEGRATION
SINGLE_STAR_TIMESTEPPING=0
SINGLE_STAR_DIRECT_GRAVITY_RADIUS=1000.
ADAPTIVE_TREEFORCE_UPDATE=0.0625
INPUT_POSITIONS_IN_DOUBLE
OUTPUT_POTENTIAL
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
