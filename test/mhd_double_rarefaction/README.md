# mhd_double_rarefaction

Two symmetric MHD rarefactions receding from x = 2 with a uniform normal field, on the same periodic
[0, 4] x [0, 0.25] strip as briowu and toth (896 x 56 particles, square lattice, mass proportional to
density). gamma = 5/3, rho = 1, P = 0.45, Bx = 0.75, By = 0.5, vx = -/+v0:

| case | v0 | t_end | centre of the reference solution |
|---|---|---|---|
| near_vacuum | 2 | 0.3 | rho_min = 0.011, P_min = 4.8e-4 |
| vacuum | 4 | 0.2 | rho_min = 2.7e-4, P_min = 3.2e-7 |

These are the MHD analogue of Einfeldt's 1-2-0-3 problem (parameters chosen here), with a normal
field so that the magnetic terms act at the vacuum. The periodic image of the jump at
x = 0 = 4 is a pair of colliding flows; the comparison window (mhd_1d_reference.clean_window)
excludes everything its waves reach.

## What it checks

At the centre the Riemann solver meets vacuum or near-vacuum states on ~1e4 faces per run. The test
compares the final state with a converged Athena++ solution: L1 over the window, normalised by each
field's range (rho, vx, vy, By), the mean drift of Bx from its initial value (Bx is exactly constant
in 1D), the rms of |divB| h/|B| over the window, and positivity of rho and P.

Measured at 896 cells, MHD_CONSTRAINED_GRADIENT=1; tolerances are ~10-20% above the largest value:

| near_vacuum | 2 / 4 / 16 ranks | tolerance |
|---|---|---|
| L1 vx | 0.0093 / 0.0096 / 0.0095 | 0.0106 |
| divB rms | 0.0036 / 0.0035 / 0.0036 | 0.0039 |
| L1 vy | 0.034 / 0.035 / 0.035 | 0.041 |
| L1 By | 0.041 / 0.043 / 0.043 | 0.050 |
| L1 rho | 0.018 / 0.018 / 0.018 | 0.021 |

The vacuum case (a true vacuum at the centre) is mainly a positivity and robustness check; its
tolerances are ~15% above the measured values.

## Reference solution

`mhd_double_rarefaction_exact.hdf5` (downloaded on demand from https://users.flatironinstitute.org/~mgrudic/gizmo_tests/mhd_double_rarefaction/), one group per case (x, rho, vx, vy, vz, By, Bz, P; Bx and gamma
as attributes): Athena++ commit 053bc738d5c85dac717326b4338b34c7db4e715d, `configure.py -b
--prob=shock_tube --flux=hlld`, HLLD + PLM + VL2, CFL 0.4, 8192 cells on the same periodic [0, 4]
domain. Inputs: `athinput.near_vacuum`, `athinput.vacuum` (`bin/athena -i athinput.<case>`, under
a minute on one core). 4096 vs 8192 cells differ by <= 3.9e-4 (near_vacuum) and <= 1.2e-3 (vacuum)
in L1/range over x in [1, 3]. The density and pressure floors (1e-10, 1e-12) are never reached.
