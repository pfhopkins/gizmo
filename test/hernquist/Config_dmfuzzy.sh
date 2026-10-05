########################################
# hernquist + fuzzy (scalar-field) dark matter (see hernquist_dmfuzzy.params).
# DM_FUZZY=0 is the fully-conservative Madelung form; 1 and 2 are the direct
# Schroedinger-Poisson integrators (see the User Guide).
########################################
ADAPTIVE_GRAVSOFT_FORALL=2
DM_FUZZY=0
OUTPUT_POTENTIAL
OUTPUT_IN_DOUBLEPRECISION
STOP_WHEN_BELOW_MINTIMESTEP
