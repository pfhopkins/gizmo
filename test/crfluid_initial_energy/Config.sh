########################################
# crfluid_initial_energy: fresh start of a two-density lattice with the CR fluid
########################################
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
COSMIC_RAY_FLUID
CRFLUID_SPEEDOFLIGHT_REDUCTION=(3.e7/C_LIGHT_CGS)
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
