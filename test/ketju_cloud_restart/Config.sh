########################################
# ketju_cloud_restart -- gmc_cooling cloud with gravity on and 16 embedded stars (4 tight binaries) under KETJU;
# a run split by a restart must continue the straight run.
########################################
MAGNETIC
COOLING
METALS
COOL_METAL_LINES_BY_SPECIES
BOX_PERIODIC
GRAVITY_NOT_PERIODIC
ADAPTIVE_GRAVSOFT_FORGAS
ADAPTIVE_TREEFORCE_UPDATE=0.0625
EVALPOTENTIAL
OUTPUT_IN_DOUBLEPRECISION
KETJU_REGULARIZATION
KETJU_MERGE_STARS
STOP_WHEN_BELOW_MINTIMESTEP
