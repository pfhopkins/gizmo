########################################
# hernquist_sidm CBE variant — adds CBE_INTEGRATOR to the standard SIDM config
# to exercise the C1 do_cbe_drift_kick GPU EP port.
########################################


BOX_SPATIAL_DIMENSION=3
ADAPTIVE_GRAVSOFT_FORALL=2
DM_SIDM=2

# C1: collisionless Boltzmann equation integrator (2 = number of basis functions)
CBE_INTEGRATOR=2
