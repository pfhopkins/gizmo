# Brio-Wu MHD shock tube

The Brio & Wu (1988) MHD Riemann problem, as in Hopkins & Raives (2016). It exercises the MHD
Riemann solver on every wave family at once: a left fast rarefaction, a compound wave (an
intermediate slow shock attached to a slow rarefaction, across which By changes sign), a
contact, a slow shock and a right fast rarefaction.

## Setup

- 2D periodic box, `BOX_LONG_X=16`, BoxSize 0.25: x in [0, 4), y in [0, 0.25); 896 x 56 particles
  on a lattice (spacing 1/224), equal spacing on both sides, so the particle mass jumps by 8.
- gamma = 2. Left (x < 2): rho = 1, P = 1, v = 0, B = (0.75, 1, 0). Right: rho = 0.125, P = 0.1,
  v = 0, B = (0.75, -1, 0). The IC nudges u in the two particle columns either side of the jump
  by up to 4% (negligible: <0.03% change in the reference).
- TimeMax 0.2. This is the classic [0,1], t = 0.1 problem stretched by 2 in x: the waves span
  x = 1.64 - 2.74.
- B is in GIZMO's internal units (MU_0 = 1, magnetic pressure B^2/2) since the params leave the
  unit parameters at their defaults; the test asserts this from the snapshot header.
- The box is periodic, so there is a second, inverted jump at x = 0 = 4. Its waves reach
  x = 0.39 and x = 3.23 by t = 0.2, well clear of the main structure.

## Reference solution

`test/mhd_1d_reference.py`: an independent 1D finite-volume solver (HLLD fluxes, minmod-limited
primitive reconstruction, RK2), run with 4096 cells on the same periodic domain. It is computed
on first use and cached as `briowu_reference_4096.npz` (gitignored; ~15 s). Brio-Wu has no
unique exact solution (the compound wave is not admissible in the classical sense), so a
converged numerical solution is the standard reference.

Plateau values (t = 0.2): rho = 0.676 behind the left rarefaction, 0.695 between the compound
wave and the contact (vy = -1.584, By = -0.534), 0.235 between the contact and the slow shock,
0.117 behind the slow shock (By = -0.903), consistent with the published solution (e.g. Ryu &
Jones 1995).
Convergence: the L1 difference between the 2048- and 4096-cell solutions, over the comparison
window and relative to each field's range there, is 0.19% (rho), 0.37% (vx), 0.27% (vy), 0.10%
(By); the 1024-cell solution differs by 0.5-1%.

## Comparison

The particles are binned onto x (mean over y) with bin width the mean interparticle spacing.
The comparison window is the reference's disturbed region around x = 2 plus 5 bins each side
(x = 1.59 - 2.80); `clean_window` asserts that waves from the periodic image have not reached it.

| Assertion | Bound | Sensitive to |
|---|---|---|
| L1 of rho, vx, vy, By over the window, / reference range | 0.03, 0.04, 0.04, 0.03 | Riemann-solver errors (wave speeds, the wrong state between waves), slope-limiter over-diffusion smearing the compound wave and the contact, misplaced waves |
| Binned mean on 7 plateaus (rho on all four; vy, By between compound wave and contact; By behind the slow shock) | 2% relative | the intermediate states the Riemann solver must produce, independent of resolution: a solver that mishandles the rotational/compound structure gets vy and By wrong between the compound wave and the contact |
| rms and 99th percentile of \|divB\| h/\|B\| in the window, h = (m/rho)^(1/2) | 0.05, 0.25 | Dedner cleaning strength, face-closure (gradient) errors feeding divB |
| Total energy (thermal + kinetic + B^2/2 V) between first and last snapshot | 1% | non-conservative divB source terms, Dedner damping, flux bookkeeping |

All bounds are PROVISIONAL (not yet tuned against a run). For scale, the reference solver at the
run's own resolution (896 cells) scores L1 = 0.006 / 0.012 / 0.008 / 0.003 and is within 0.9% on
every plateau; the same solver with 4x fewer cells (224) fails the vx bound (0.047). h is the
cell size, not SmoothingLength: the kernel radius is ~2.5x larger at DesNumNgb = 20 and would
inflate the metric without measuring anything different. OpenGadget3 asserts rms < 0.05 on this
problem.
