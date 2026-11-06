########################################
# polytrope_n1 -- hydrostatic equilibrium of a self-gravitating n=1 (gamma=2) polytrope.
# Gravity must balance the MFM pressure gradient; variants (in the test module) add
# ADAPTIVE_GRAVSOFT_FORGAS and TIDAL_TIMESTEP_CRITERION + ADAPTIVE_TREEFORCE_UPDATE.
########################################
HYDRO_MESHLESS_FINITE_MASS
EOS_GAMMA=(2.0)              # P = K rho^2: the n=1 polytrope, whose Lane-Emden solution is analytic
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
OUTPUT_POTENTIAL             # gravitational term of the energy budget, and W(t=0) vs -3/4
OUTPUT_ACCELERATION          # with OUTPUT_HYDROACCELERATION set, "Acceleration" is gravity only
OUTPUT_HYDROACCELERATION     # "HydroAcceleration" = -grad(P)/rho; their sum is the net acceleration
STOP_WHEN_BELOW_MINTIMESTEP
