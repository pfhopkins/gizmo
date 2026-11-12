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
- H- photodetachment from the stellar bands (`SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS`, env `HM_FIELD_T`,
  `HM_FIELD_G0`, `HM_MODE`): for a 3e4 K field with G0 = 1 and no LW the applied rate equals the direct integral of the
  Shapiro & Kang cross-section over the spectrum (3.7e-9 s^-1, against 4.2e-7 from the ISRF calibration), and Pop III
  forms at 3.5e6 instead of 2.4e7.

Needs `SUNDIALS_DIR` (CVODE >= 6) and `CHEMCXX_DATADIR` (the SGChem tables); the test is skipped without them. It builds
its own copy of `cooling/sgchem` with a minimal `GIZMO_config.h` (net 17, CO shielding, WSS CIE cooling).

Dust-to-gas ratio (`declarations/resolvedism_dgr_rr14.h`, `SGCHEM_DGR_REMY_RUYER14`): at 1e-3 Zsun the Remy-Ruyer+14 law
lets the gas fragment at the CMB floor (M_J < 32 Msun below n = 1e6), while the linear law, with 1e5 times more dust,
heats the gas through H2 formation on grains and delays it beyond 1e7.
