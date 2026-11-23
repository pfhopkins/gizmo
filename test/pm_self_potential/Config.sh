########################################
# pm_self_potential -- one particle in a periodic box with PMGRID and the energy step trace
########################################
BOX_PERIODIC
PMGRID=32
EVALPOTENTIAL
OUTPUT_IN_DOUBLEPRECISION
OUTPUT_ENERGY_STEP_TRACE                 % energy step trace (gizmo.conservation_trace)
