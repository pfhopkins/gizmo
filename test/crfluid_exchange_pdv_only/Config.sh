########################################
# crfluid_exchange_pdv_only: periodic lattice blast with the CR fluid on the production path (MAGNETIC, operator-split CR-gas exchange); variant adds CRFLUID_EXCHANGE_PDV_ONLY
########################################
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
MAGNETIC
COOLING_OPERATOR_SPLIT
COSMIC_RAY_FLUID
CRFLUID_SPEEDOFLIGHT_REDUCTION=(3.e7/C_LIGHT_CGS)
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
