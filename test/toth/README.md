# Toth MHD shock tube

The colliding-flow MHD shock tube of Toth (2000; also Ryu & Jones 1995), as in
Hopkins & Raives (2016). Two supersonic streams collide in an oblique field, producing two fast
shocks, two weak slow shocks and a contact. It tests strong-shock capturing and the small
rotational/slow-shock structure riding on it.

## Setup

- 2D periodic box, `BOX_LONG_X=16`, BoxSize 0.25: x in [0, 4); 896 x 56 equal-mass particles.
- gamma = 5/3, rho = 1 everywhere, Bx = By = 5/sqrt(4 pi) = 1.4105. Left (x < 2): P = 20, vx = 10.
  Right: P = 1, vx = -10. The IC raises u in the two particle columns either side of the jump
  (to 47.7 and 40.1, from 30 and 1.5); in the reference this changes the L1 by up to 0.6% (vy).
- TimeMax 0.08. The main structure spans x = 1.61 - 2.37.
- B in GIZMO's internal units (MU_0 = 1, magnetic pressure B^2/2); the test asserts this from
  the snapshot header.
- The periodic image of the jump at x = 0 = 4 is a diverging flow, a strong double rarefaction
  (rho down to ~0.003) whose heads reach x = 1.32 and 2.98 by t = 0.08, well clear of the main
  structure. The rarefaction is not compared.

## Reference solution

`test/mhd_1d_reference.py` (HLLD, minmod-limited primitive reconstruction, RK2), 4096 cells on
the same periodic domain, cached as `toth_reference_4096.npz` (gitignored; ~30 s on first use).

Plateau values (t = 0.08), consistent with the published solution (e.g. Ryu & Jones 1995):
rho = 2.680, By = 3.839, P = 150.98 behind the left fast shock; vy = 0.357, By = 4.038 between
the two slow shocks (straddling the contact, rho = 2.671 | 3.850); rho = 3.748, vy = -0.388,
By = 5.427, P = 143.57 behind the right fast shock. Convergence: the L1 difference between the 2048- and 4096-cell solutions over the
window, relative to each field's range, is 0.22% (rho), 0.08% (vx), 0.38% (vy), 0.23% (By).

## Comparison

Same procedure as briowu: particles binned onto x with bin width the mean interparticle spacing;
the window is the reference's disturbed region plus 5 bins each side (x = 1.59 - 2.39), checked
to be untouched by the periodic image.

| Assertion | Bound | Sensitive to |
|---|---|---|
| L1 of rho, vx, vy, By over the window, / reference range | 0.03, 0.02, 0.05, 0.03 | shock positions (wrong post-shock states move the shocks), over-diffusion of the narrow slow-shock/contact complex at x = 2.0 - 2.1, Riemann-solver errors |
| Binned mean on 7 plateaus (rho, By behind the left fast shock; vy, By between the slow shocks; rho, vy, By behind the right fast shock) | 2% relative | the Riemann-solver intermediate states; vy and By between the slow shocks test the rotational structure directly |
| rms and 99th percentile of \|divB\| h/\|B\| in the window, h = (m/rho)^(1/2) | 0.05, 0.25 | Dedner cleaning strength, face-closure errors feeding divB at the strong shocks |
| Total energy (thermal + kinetic + B^2/2 V) between first and last snapshot | 1% | non-conservative divB source terms, Dedner damping, flux bookkeeping |

All bounds are PROVISIONAL (not yet tuned against a run). For scale, the reference solver at 896
cells scores L1 = 0.006 / 0.002 / 0.011 / 0.006 and is within 0.3% on every plateau; with 224
cells it fails the rho bound (0.030). vx has the smallest L1 because its range (20) is set by the
fast shocks; vy carries the delicate structure. The narrow slow-shock plateau leaves ~15 bins
after trimming 3 bins at each end; if GIZMO smears the slow shocks more than that, the plateau
assertion will say so before the value is meaningful.
