########################################
# hernquist CBE production grad+SM test (the final local rung after the
# 1D/free-slot ladder).  Self-gravitating Hernquist DM halo evolved with the
# CBE integrator in its PRODUCTION mode: SECONDMOMENT (per-basis stress) +
# WITHGRADIENTS (MFM face reconstruction).
#
# In 3D, CBE_INTEGRATOR_SECONDMOMENT => NMOMENTS=10 [m, p_xyz, R_xx..R_zz,
# R_xy,R_xz,R_yz] (precompiler_logic.h:154).  ("NMOMENTS=7" in older notes was
# the diagonal-only layout the precompiler no longer emits; full tensor = 10.)
#
# Pure CBE: NO DM_SIDM here -- the Monte-Carlo SIDM scatter contaminates the CBE
# evolution (see STATE 2026-05-30).  The OLD Config_cbe.sh (CBE=2 + DM_SIDM, no
# stress) is the AGS+SIDM smoke the roadmap explicitly says NOT to use for the
# CBE physics analysis; it stays for the C1 EP-port smoke only.
#
# The IC (hernquist_sidm_ics.hdf5, made by make_ic.py) is pure Type=1 DM with NO
# VlasovMoments dataset; the per-particle CBE basis distribution is INITIALIZED
# ON START (synthesized from the local state), no VlasovMoments IC needed
# (Phil 2026-06-08).  NBASIS=8 (=CBE_INTEGRATOR) per Phil -- enough bases to
# represent the halo's isotropic velocity distribution.  DM_SIDM intentionally
# OFF (SIDM Monte-Carlo scatter would contaminate the CBE evolution).
########################################

BOX_SPATIAL_DIMENSION=3
ADAPTIVE_GRAVSOFT_FORALL=2

CBE_INTEGRATOR=8
CBE_INTEGRATOR_SECONDMOMENT
CBE_INTEGRATOR_WITHGRADIENTS
CBE_INTEGRATOR_OUTPUT_MOREINFO

OUTPUT_ADDITIONAL_RUNINFO
STOP_WHEN_BELOW_MINTIMESTEP
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
