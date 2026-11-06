# Toth MHD shock tube

The colliding-flow MHD shock tube of Toth (2000; also Ryu & Jones 1995), as in
Hopkins & Raives (2016). Two supersonic streams collide in an oblique field, producing two fast
shocks, two weak slow shocks and a contact. It tests strong-shock capturing and the small
rotational/slow-shock structure riding on it.

## Setup

- 2D periodic box, `BOX_LONG_X=16`, BoxSize 0.25: x in [0, 4); 896 x 56 equal-mass particles.
- gamma = 5/3, rho = 1 everywhere, Bx = By = 5/sqrt(4 pi) = 1.4105. Left (x < 2): P = 20, vx = 10.
  Right: P = 1, vx = -10. The IC raises u in the two particle columns either side of the jump
  (to 47.7 and 40.1, from 30 and 1.5); in a 1D grid solution this changes the L1 by up to 0.6% (vy).
- TimeMax 0.08. The main structure spans x = 1.61 - 2.37.
- B in GIZMO's internal units (MU_0 = 1, magnetic pressure B^2/2); the test asserts this from
  the snapshot header.
- The periodic image of the jump at x = 0 = 4 is a diverging flow, a strong double rarefaction
  (rho down to ~0.002) whose heads reach x = 1.28 and 3.01 by t = 0.08, well clear of the main
  structure. The rarefaction is not compared.

## Reference solution

`toth_exact.txt`: Athena++ (Stone et al. 2020; HLLD, PLM, VL2, CFL 0.4) with 8192 cells on the
same periodic domain, columns `x rho vx vy vz By Bz P`, provenance in the header. Read by
`load_reference` in `test/mhd_1d_reference.py`; `test/mhd_reference_athena/` has the input file
and the commands to regenerate it.

Plateau values (t = 0.08) match Ryu & Jones (1995) table 1a to the 5 digits tabulated:
rho = 2.6797, vx = 0.72113, vy = 0.23138, By = 3.8388, P = 150.98 behind the left fast shock;
vy = 0.357, By = 4.038 between the two slow shocks (straddling the contact); rho = 3.7481,
vx = 0.70505, vy = -0.38804, By = 5.4271, P = 143.57 behind the right fast shock. PPM (xorder 3,
with VL2 or RK3) gives the same plateaus to <0.1% at 8192 cells. Convergence: the L1 difference
between the 4096- and 8192-cell solutions over the window, relative to each field's range, is
0.09% (rho), 0.03% (vx), 0.17% (vy), 0.09% (By).

## Comparison

Same procedure as briowu: particles binned onto x with bin width twice the mean interparticle
spacing; the window is the reference's disturbed region plus 5 bins each side (x = 1.57 - 2.41), checked
to be untouched by the periodic image.

| Assertion | Bound | Sensitive to |
|---|---|---|
| L1 of rho, vx, vy, By over the window, / reference range | 0.03, 0.02, 0.05, 0.03 | shock positions (wrong post-shock states move the shocks), over-diffusion of the narrow slow-shock/contact complex at x = 2.0 - 2.1, Riemann-solver errors |
| Binned mean on 7 plateaus (rho, By behind the left fast shock; vy, By between the slow shocks; rho, vy, By behind the right fast shock) | 2% relative | the Riemann-solver intermediate states; vy and By between the slow shocks test the rotational structure directly |
| rms and 99th percentile of \|divB\| h/\|B\| in the window, h = (m/rho)^(1/2) | 0.05, 0.25 | Dedner cleaning strength, face-closure errors feeding divB at the strong shocks |
| Total energy (thermal + kinetic + B^2/2 V) between first and last snapshot | 1% | non-conservative divB source terms, Dedner damping, flux bookkeeping |

All bounds are PROVISIONAL (not yet tuned against a run). For scale, Athena++ at 896 cells
scores L1 = 0.007 / 0.003 / 0.011 / 0.008 and is within 0.3% on every plateau; with 224 cells it
scores rho = 0.024, By = 0.024 against the 0.03 bounds. vx has the smallest L1 because its
range (20) is set by the fast shocks; vy carries the delicate structure. The narrow slow-shock
plateau (~0.11 wide, ~12 bins) leaves ~6 bins after trimming 3 bins at each end; if GIZMO smears the slow shocks more than that, the plateau
assertion will say so before the value is meaningful.

## Known failure: the Powell induction source at the fast shocks (2026-11-05)

Against the Athena++ reference, rho and v_x match on every plateau to 0.06-0.25%, but B_y is
+1.4% (left) and +4.5% (right) behind the fast shocks, +2.5% between the slow shocks, and the v_y
plateaus are +11-12%; |divB| h/|B| has rms 0.013 (briowu: 0.003) and peaks at the two fast shocks,
where v_y shows particle-scale spikes of +-1. The excess is present as soon as the shocks form
(t=0.02) and is then constant; a 1792x56 run gives the same numbers while the rho and v_x errors
halve, so it is systematic, not resolution.

Cause: the Powell (8-wave) source in the induction equation, `DtB -= divB * v` in
hydro/hydro_toplevel.cc, is not conservative. Integrating -divB v_y along the gas path through each
fast shock from the snapshots predicts +0.049 (left) and +0.23 (right) against the observed +0.055 and
+0.245, in the baseline, constrained-gradient, MFV and 2x runs alike; about 95% of the source is the
particle-scale correlation of divB with the v_y noise. Toth (2000) designed this problem to show that
8-wave schemes get the jump conditions at strong shocks wrong, and that is what this is. The scheme is
Galilean invariant (the problem boosted by v_x+10 or v_y+5 reproduces every plateau to 4 digits), so
the term is not a frame error. Dropping it fixes the lab-frame problem and breaks the boosted ones;
a Galilean-invariant smoothed-velocity form fixes B_y but leaves v_y at +-6% and worsens B_x; dropping
the momentum/energy Powell terms instead makes B_y worse; dropping all of them blows up (energy +11%).
MHD_CONSTRAINED_GRADIENT halves divB (rms 0.005) and brings B_y to +0.9/+1.3/+1.7% but the v_y
plateaus stay +5-6%; Dedner cleaning strength (0.2-2x) barely matters; MFV is worse (+9%).

The baseline is therefore a strict xfail with this reason. Do not loosen the tolerances on resolution
grounds: the error does not converge. Diagnostic runs, patches and plots:
/mnt/ceph/users/mgrudic/og3_tests_2026-11-05/toth_diag/ (summary_table.txt, profiles_base.png,
variants_By_vy.png, powell_est.py, smoothpowell.patch, nopowell_ind.patch).
