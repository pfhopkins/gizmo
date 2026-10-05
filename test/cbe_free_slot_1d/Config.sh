########################################
# cbe_free_slot_1d — 1D free-slot injection test.
#
# Background 4-basis distribution v = (+1, 0, -1, -2) with mass fractions
# ~(0.49, 0.02, 0.49, ~0). A Gaussian-localized perturbation at x=0.5
# flips basis-3's velocity to v=+2 — a stream the neighbours have NO slot
# for. Exercises the free-slot pairing fallback (CBE_PAIRING_USE_FREE_SLOT,
# on by default). Type=1, no gravity/hydro/gas. NBASIS=4,
# NMOMENTS=3 (m, p_x, T_xx) in 1D (SECONDMOMENT + WITHGRADIENTS =
# the production CBE default).
########################################

BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF

CBE_INTEGRATOR=4
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
