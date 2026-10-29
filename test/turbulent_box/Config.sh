########################################
# turbulent_box -- 32^3 periodic box, 10 pc on a side, isothermal (c_s = 0.2 km/s), stirred to
# Mach 10 by solenoidal driving with a parabolic spectrum on 1 <= kL/2pi <= 3 (peak at 2). The
# driving amplitude is set by the TURB_DRIVING_TARGET_SIGMA control loop. Units pc - km/s - Msun.
#
# Stirring phase: no self-gravity. The ICs already carry the mass for alpha_vir = 2 at Mach 10;
# drop SELFGRAVITY_OFF to turn gravity on, but without sinks nothing stops isothermal collapse.
########################################
BOX_PERIODIC
SELFGRAVITY_OFF
HYDRO_MESHLESS_FINITE_MASS
# P = 0.04 rho^gamma: isothermal sound speed 0.2 km/s
EOS_GAMMA=1.001
EOS_ENFORCE_ADIABAT=0.04
TURB_DRIVING
TURB_DRIVING_TARGET_SIGMA=0
