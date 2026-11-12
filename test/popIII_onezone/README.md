# popIII_onezone

One-zone free-fall collapse of pristine gas with the net-17 C++ SGChem chemistry (Omukai 2005 style): each step evolves
chemistry and cooling at fixed density over 0.01 t_ff, then compresses n -> n e^(dt/t_ff) adiabatically. Redshift 17
(T_CMB = 49 K), G0 = 0, zeta = 1e-21 s^-1, He/H = 0.079 by number.

Along the track the driver evaluates the code's own Pop III criterion (`galaxy_sf/resolvedism_popIII.h`, default
parameters: n_H > 1e6, converging flow, n(H2)/n_H > 1e-3, M_J < 1000 Msun) and the Stella Jeans mass
M_J = 2.9156 c_s^3 / (G^1.5 rho^0.5).

Checks:
- Z = 0 reaches the loitering minimum of 150-250 K between n = 1e2 and 1e4 (Bromm, Coppi & Larson 2002).
- The Pop III criterion first passes at n_H = 1e6-3e6. It is the H2 condition that binds: x_H2 = 9.8e-4 at 1e6.
- The Stella Jeans criterion with 1 Msun cells (M_J < 32 Msun) passes only above 1e9 in pristine gas, which is why
  pristine gas needs its own channel.
- Gas at 1e-3 Zsun is not pristine (Z_crit = 1e-4 Zsun).

Needs `SUNDIALS_DIR` (CVODE >= 6) and `CHEMCXX_DATADIR` (the SGChem tables); the test is skipped without them. It builds
its own copy of `cooling/sgchem` with a minimal `GIZMO_config.h` (net 17, CO shielding, WSS CIE cooling).
