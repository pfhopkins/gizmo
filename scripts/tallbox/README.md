# Stratified shearing box: solar-neighbourhood ISM with FIRE-3

A 1 kpc x 1 kpc x 8 kpc local patch of a galactic disk, in the spirit of TIGRESS (Kim & Ostriker 2017; Kim et al.
2023, TIGRESS-NCR) and GHOSDT (Gurman et al.).

| ingredient | implementation |
|---|---|
| shear, Coriolis, tidal | `BOX_SHEARING=3`, q = 1, Omega = 28 km/s/kpc (`BOX_SHEARING_OMEGA_BOX_CENTER`, now overridable) |
| vertical boundaries | `BOX_OUTFLOW_Z`, mid-plane at z = 4 kpc |
| self-gravity | `GRAVITY_PERIODIC_XY`: tree + 2D Ewald correction, shear-periodic in x/y, open in z |
| stars + DM | `GRAVITY_ANALYTIC_STRATIFIED_DISK`: Phi_* = 2 pi G Sigma_* (sqrt(z^2+z_*^2) - z_*), Phi_DM = 2 pi G rho_DM R0^2 ln(1+z^2/R0^2); R8 values Sigma_* = 42 Msun/pc^2, z_* = 245 pc, rho_DM = 0.0064 Msun/pc^3, R0 = 8 kpc |
| gas | Sigma_gas = 10 Msun/pc^2, solar metallicity, isothermal HSE slab (T = 1e4 K plus turbulent support) and a 1e6 K, 1e-4 cm^-3 floor |
| initial turbulence | solenoidal, E(k) ~ k^4 exp(-2k^2/k_p^2) peaked at 100 pc, sigma_1D = 7 km/s |
| physics | `FIRE_PHYSICS_DEFAULTS=3` |

## Files
- `Config_tallbox.sh`, `tallbox.params`: the parameters are set for 1e3 Msun cells. For 10 Msun cells, change the entries marked `[res]`: gas softening floor 0.25 pc, stars 3 pc (gas softening is adaptive; at 1e3 Msun the 1 pc floor is never reached and stars use 15 pc, about the gas kernel radius where they form).
- `make_tallbox_ics.py --mgas <Msun>`: writes the ICs (1e4 cells at 1e3 Msun, 1e6 at 10 Msun). `--help` lists the knobs.
- `tallbox_diagnostics.py <run>/output`: time series of Sigma_gas, Sigma_*, Sigma_SFR, scale height, mid-plane P/k, sigma_1D, phase fractions, and the residual from the background shear. Plots to `diagnostics.png`.

## Notes
- Gravity: the Ewald tables (49 x 49 x 97, per shear slot) are refilled collectively as the shear offset advances, about every 0.55 Myr at Omega = 28 and q = 1. Each refill is about 1 s of CPU split over the ranks. Tests: `test/slab_ewald` (forces against an independent lattice sum, with and without shear) and `test/strat_hse` (HSE slab).
- LEBRON radiation, including FUV, uses only the nearest x/y image. Each receiver therefore sees sources within a ±L/2 square: a horizontal cutoff of about 500 pc set by the box size, rather than the log-divergent infinite-plane sum. In the halo (|z| >~ L) the field falls off like 1/z^2. This is a known box-size dependence of G0.
- The shear velocity jump across the x-boundary (q Omega L = 28 km/s) is corrected in hydro, gradients, the gravity jerk, FIRE mechanical feedback and gas-cell merging.
