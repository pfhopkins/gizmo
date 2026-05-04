# biermann_growth — sign-anchor test for MHD_BATTERY_MECHANISMS

## Status (2026-05-01)

**Scaffold complete, IC tuning needed.** The test files are wired up (Config,
params, IC generator, pytest scaffold) but the simulation has not yet been
made to run cleanly to completion in either unit system tried so far.

## Physics (verified analytically)

2D periodic box, B = 0 initially, with imposed orthogonal sinusoidal
density and internal-energy perturbations:
  rho(x,y) = rho_0 (1 + a sin(2 pi x / L))     => grad n_e parallel x-hat
  u(x,y)   = u_0   (1 + b sin(2 pi y / L))     => grad T_e parallel y-hat

Without COOLING (`feedback_kernel_test_must_fire.md`):
  T_e_cell = T_gas = u (gamma-1) m_p / k_B
  n_e_cell = nHcgs() (fully ionized fallback)

Biermann induction (Kulsrud 1997, also see hydro/battery_functions.h):
  dB_z/dt = -(c k_B / (e n_e)) (grad n_e) x (grad T_e)
At (x = y = 0):
  dB_z/dt = -(c k_B / e) a b T_0 k_phys^2  [G/s]
n_e cancels — the prediction is independent of the mean density.

This is the **sign-anchor** test. The Squire-Hopkins 2018 streaming-instability
test gives the gold-standard signed growth rate; for commit 7/N we only need
to confirm sign + order-of-magnitude.

## Open issue: numerical runaway at IC

Both unit systems tried so far (pure-cgs; tuned cgs-with-m_p mass unit and
sound-speed velocity unit) produce a NaN cascade after a handful of sync
points: `divV` saturates to +/-1e+120, `InternalEnergy` blows up to ~1e+190.

Diagnosis: the runaway starts after roughly Sync-Point 24 (Time ~ 0.002 of a
0.5 target), with `Systemstep` collapsing to 0 — the integrator is stuck on
some particle whose acceleration cannot be resolved. Density/pressure look
sane until the catastrophic step.

Likely culprits to investigate next session:
- Pressure-gradient amplitude vs Courant: with a = b = 1e-2, pressure
  perturbation is O(2%), which should be benign. Verify by running the SAME
  IC with `MHD_BATTERY_MECHANISMS` *off* — if it still blows up, the issue is
  our IC (not the battery code).
- `MaxSizeTimestep` may be too coarse for the chosen unit system. Try
  reducing to `1e-4` (relative to TimeMax = 0.5) or letting GIZMO autoselect.
- Tilted-grid IC (irregular particle placement) may help avoid grid-aligned
  instabilities at the corners (cos(kx) cos(ky) = +1 modes).
- Confirm `nH_cgs` and `T_K` reach the cooling-side accessors as expected
  by adding a one-time debug print in `eos.cc::set_eos_pressure` — if the
  per-cell `T_e_cell` values have unexpected magnitudes, that points to a
  unit-conversion bug rather than a hydro instability.

## Files

- `Config.sh` — minimal MHD + battery + nlist build flags
- `biermann_growth.params` — runtime params (currently with cgs-tuned units)
- `make_biermann_growth_ics.py` — IC generator (grid, periodic, B=0)
- `test_biermann_growth.py` — pytest scaffold with sign + order-of-magnitude
  assertions against the analytic prediction

## Validation gate

Per `feedback_kernel_test_must_fire.md`: kernel must actually fire, not just
compile. The `mhd_wave` smoke test (commit `36659f3c`) confirms the per-pair
flux call is reached — but that 1D problem produces ~0 Biermann EMF, so the
*physics* is still unvalidated. This test, once running, is what closes that
gap.
