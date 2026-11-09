# freefall: pressureless collapse of a uniform sphere

A cold, static, uniform gas sphere collapses under its own gravity (MFM hydro, tree gravity,
non-periodic). Modelled on OpenGadget3's `FreefallCollapse` CI test (Dolag et al. 2026,
arXiv:2609.17658, sec 2.5).

## Analytic solution

Code units G = M = R = 1. Before shells cross, a shell that starts at rest at radius r0 around mass
M(<r0) follows the radial Kepler orbit

    r(t) = r0 cos^2(beta),     t = t_ff (2/pi) (beta + sin(2 beta)/2),
    t_ff = (pi/2) sqrt(r0^3 / (2 G M(<r0))).

For a uniform sphere t_ff = pi/(2 sqrt 2) = 1.1107 for every shell, so the collapse is homologous
and the sphere stays uniform. The test applies this per Lagrange radius, taking r0 and M(<r0) = f M
from the t=0 snapshot, so the lattice's surface rounding does not enter the reference. At t=0 the
field is a = -G M r/R^3 inside the sphere and -G M r/r^3 outside.

## Setup

| | | why |
|---|---|---|
| ICs | `make_freefall_ics.py`, built by the test on first use (`_ensure_ic`) | |
| N | 2^15 gas cells | |
| positions | cubic lattice, spacing d = 0.0504, per-axis jitter uniform in ±0.1 d, the N cells nearest the origin kept | random sampling would bury the force error in Poisson noise; the jitter keeps the tree off a degenerate configuration |
| density | lattice density set equal to 3M/(4 pi R^3), so the interior field is exactly -G M r/R^3; M(<r)/r^3 is flat to 0.3% for r > 0.5 | |
| u0 | 1e-5 (c_s0 = 3.3e-3) | sound-crossing time at the end of the run is ~37 >> t_ff; edge pressure force ~2e-4 of gravity |
| softening | SofteningGas = 0.025 = 0.5 d (Plummer-equivalent; kernel support 0.07) | ags variant: the gas kernel radius (~0.1 at t=0), with SofteningGas overridden to 1e-4 so the tree's softening floor (2.8 SofteningGas) does not bind |
| TimeMax | 1.0 = 0.90 t_ff | radius down 2.85x, density up 23x; the fixed softening kernel is still ~1/5 of the sphere radius. OG3 runs to 0.95 t_ff |
| snapshots | every 0.05 | |

Variants: `baseline` (fixed softening), `ags` (`ADAPTIVE_GRAVSOFT_FORGAS`), `tidal_adaptive`
(`TIDAL_TIMESTEP_CRITERION` + `ADAPTIVE_TREEFORCE_UPDATE=0.06`).

With both `OUTPUT_ACCELERATION` and `OUTPUT_HYDROACCELERATION` set, the snapshot's `Acceleration` is
the gravitational acceleration only and `HydroAcceleration` the hydro part (`file_io/io.cc`,
`IO_ACCEL`); with no comoving integration both are plain code units. `OUTPUT_SOFTENING` writes
`Softening_KernelRadius`, the kernel support radius the gravity solver used for each cell.

## Checks

Every tolerance in `test_freefall.py` is marked PROVISIONAL: none has been calibrated against a
run yet. Calibrate them from the first runs, and check each one fails when the physics it targets
is broken.

- **Pressure is negligible** (setup check): rms |HydroAcceleration| / rms |Acceleration| at every
  snapshot. If this fails, the gas is not cold enough for the pressureless solution and the checks
  below are not testing what they claim.
- **(a) t=0 force.** From `snapshot_000`:
  - radial acceleration against -G M r/R^3, as the mean over 14 radial bins of
    |<a_r / a_analytic> - 1|, over 0.2 < r/R < 0.9 (inside the edge kernel, and away from the
    centre where a -> 0);
  - rms tangential / rms radial acceleration over the same range (should be ~0; the softened
    direct sum on this IC gives 7e-3 from the lattice's granularity alone);
  - particle by particle against a softened direct sum (`pytreegrav`, `method="bruteforce"`) with
    softening = `Softening_KernelRadius`: rms |a - a_direct| / |a_direct| for r/R > 0.2. The two
    use the same convention (compact-support radius of the M4 cubic spline, larger softening for
    unequal pairs), and the t=0 force has no AGS zeta term (zeta is first set by the hydro
    gradient pass, after the t=0 gravity call), so this applies to the ags variant too. It
    isolates tree error from sampling noise: the direct sum itself matches the analytic field to
    5e-4 (binned) on this IC. Sensitive to the opening criterion (`ErrTolForceAcc`), node
    moments, and AGS softening handling in the walk. Skipped, after all other checks have run, if
    `pytreegrav` is not installed (it is not among the package dependencies).
- **(b) Lagrange radii** r_10, r_50, r_90 (centred on the centre of mass) against the analytic
  solution at every snapshot; asserted at the final one. Sensitive to anything that changes the
  integrated force over the collapse: force errors that grow as the density rises, forces left
  stale by `ADAPTIVE_TREEFORCE_UPDATE` while the dynamical time shrinks by ~5x, and timestep-
  hierarchy desynchronisation as the edge cells (largest |a|) move to shorter steps than the
  centre. r_90 lies within one softening kernel of the surface, where the softened field is
  weaker than the point-mass one, so it lags by construction and has a looser tolerance.
- **(c) Energy**: thermal + kinetic + 0.5 sum(m phi), first vs last snapshot
  (`assert_energy_conserved(include_potential=True)`). With `ADAPTIVE_GRAVSOFT_FORGAS` the
  softened potential changes as the softening shrinks, so the budget closes only as well as the
  zeta correction makes it (the same caveat as `evrard`).
- **(d) Centre-of-mass drift** (`momentum_drift_common`): the sphere starts at rest, so any COM
  velocity is correlated force error.

OG3's `verify_freefall.py` sums the absolute r_50 error over snapshots and then divides by the
particle count, so its tolerance tightens or loosens with N and with the snapshot count rather than
with the error; this test asserts the plain relative error of each Lagrange radius.

## Outputs

`freefall_lagrange.png` (Lagrange radii against the analytic curves, and their relative error),
`freefall_energy.png`, `freefall_accel.png` (t=0 force against the analytic field and the direct
sum), `momentum_drift.png`, one `summary_<variant>.npz` per variant; all rebuilt from whichever
variants have run.
