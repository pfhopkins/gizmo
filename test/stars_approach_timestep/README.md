# stars_approach_timestep

`STARS_APPROACH_TIMESTEP` (stella_dev 64dd07bf7, a694e2761; on by default for Stella single stars, opt out with
`STARS_APPROACH_TIMESTEP_OFF`): a Type-4 star's step is limited by STARFORGE's two-body step,
dt = sqrt(2 eta) S / (1/t_approach + 1/t_freefall), to the nearest star found in the gravity walk (the sink proximity
machinery with Type 4 as the node-distance type); KETJU chain members skip their partners, which MSTAR integrates.

Problem: Stella cloud production flags; two 2 Msun main-sequence stars 1 pc apart approaching at 50 km/s, impact parameter
0.05 pc (outside the softened kernel, so the pair potential is Newtonian), thin gas (n_H = 1e-3), 40 kyr, 2 ranks.
Without the approach step the stars step with the acceleration criterion, about 7 kyr at closest approach, against an
encounter time of 1 kyr.

- `test_encounter_kick`: the perpendicular change of the pair's relative velocity equals the exact two-body orbit's to 1 %;
  the pair energy never moves by more than 5 % of G m1 m2 / b; the closest sampled separation is below 0.1 pc.
- `test_control_without_the_step_fails_the_bar`: built with `STARS_APPROACH_TIMESTEP_OFF` the kick error exceeds 1 % (negative control).
- `test_mass`: gas plus star mass constant to 1e-12 (double-precision snapshots).

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
