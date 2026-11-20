########################################
# hernquist + continuum Vlasov (CBE) integrator (see hernquist_cbe.params).
# 8 velocity-space bases per particle with per-basis second moments and
# gradient (face) reconstruction, with the default compact-support (top-hat) flux.
# Do not add CBE_INTEGRATOR_RP_GAUSSIAN: that flux is a method-comparison option
# and is not suitable for this problem.
########################################
ADAPTIVE_GRAVSOFT_FORALL=2
CBE_INTEGRATOR=8
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_OUTPUT_MOREINFO
OUTPUT_POTENTIAL
OUTPUT_IN_DOUBLEPRECISION
STOP_WHEN_BELOW_MINTIMESTEP
