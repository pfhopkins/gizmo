########################################
# cbe_free_slot_1d — 1D free-slot injection test (mirrors python_harness
# tests/test_free_slot, the headline scientific result of the harness).
#
# Background 4-basis distribution v = (+1, 0, -1, -2) with mass fractions
# ~(0.49, 0.02, 0.49, ~0). A Gaussian-localized perturbation at x=0.5
# flips basis-3's velocity to v=+2 — a stream the neighbours have NO slot
# for. Exercises the free-slot pairing fallback (CBE_PAIRING_USE_FREE_SLOT,
# default on in C6c). Type=1, no gravity/hydro/gas. NBASIS=4,
# NMOMENTS=2 (m, p_x) in 1D.
########################################

BOX_SPATIAL_DIMENSION=1
BOX_PERIODIC
SELFGRAVITY_OFF

CBE_INTEGRATOR=4
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
