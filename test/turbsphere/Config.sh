########################################
# turbsphere -- the STARFORGE stirring setup (MakeCloud defaults): a uniform sphere of radius R = 5 pc
# in a periodic box of 10 R, with ambient gas at 1e-3 of the cloud density, held by the analytic
# confining potential of STARFORGE_GMC_TURBINIT with self-gravity off. Isothermal (c_s = 0.2 km/s),
# stirred to Mach 10 inside R by solenoidal driving with a parabolic spectrum on wavelengths
# 2R/3 - 2R (peak at R). Units pc - km/s - Msun.
#
# The cloud mass gives alpha_vir = 2 at Mach 10; the potential is scaled by the same value.
# GravAccel_GMCTurbInit hard-codes R = BoxSize/10 and a cloud mass fraction of 0.808, which the ICs match.
########################################
BOX_PERIODIC
SELFGRAVITY_OFF
GRAVITY_ANALYTIC
STARFORGE_GMC_TURBINIT=2
HYDRO_MESHLESS_FINITE_MASS
# P = 0.04 rho^gamma: isothermal sound speed 0.2 km/s
EOS_GAMMA=1.001
EOS_ENFORCE_ADIABAT=0.04
TURB_DRIVING
# force recomputed only on driving updates: the only way turbsphere stirring runs are made
TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE
# the loop measures only gas within R = 5 pc of the box centre
TURB_DRIVING_TARGET_SIGMA=5
