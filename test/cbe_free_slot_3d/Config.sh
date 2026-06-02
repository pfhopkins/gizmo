########################################
# cbe_free_slot_3d — 3D version of the free-slot injection test.
#
# Same physics as cbe_free_slot_1d (4 streams +1/0/-1/-2 along x, a
# Gaussian +2 perturbation at x=0.5), but on a 3D slab: long in x, thin in
# y,z (BOX_LONG_X stretches x; BoxSize sets the transverse extent) to keep
# particle count modest while resolving propagation along x. All velocities
# lie along the x axis, so per basis p_y=p_z=0. Type=1, no gravity/hydro/gas.
# NBASIS=4, NMOMENTS=4 (m, p_x, p_y, p_z) in 3D (no SECONDMOMENT).
#
# BOX_LONG_X=16 with BoxSize=0.0625 => box = [1.0 x 0.0625 x 0.0625].
# BOX_LONG_X + BOX_PERIODIC requires SELFGRAVITY_OFF (begrun.cc:3133) — set.
########################################

BOX_SPATIAL_DIMENSION=3
BOX_PERIODIC
BOX_LONG_X=16
SELFGRAVITY_OFF

CBE_INTEGRATOR=4
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
