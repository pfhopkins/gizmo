# stars_approach_timestep

`STARS_APPROACH_TIMESTEP` (stella_dev 64dd07bf7, a694e2761; on by default for Stella single stars, opt out with
`STARS_APPROACH_TIMESTEP_OFF`): a Type-4 star's step is limited by STARFORGE's two-body step,
dt = sqrt(2 eta) S / (1/t_approach + 1/t_freefall), to the nearest star found in the gravity walk (the sink proximity
machinery with Type 4 as the node-distance type); KETJU chain members skip their partners, which MSTAR integrates.

Problem: Stella cloud production flags; two 2 Msun main-sequence stars 1 pc apart approaching at 50 km/s, impact parameter
0.05 pc (outside the softened kernel, so the pair potential is Newtonian), thin gas (n_H = 1e-3), 40 kyr, 2 ranks.
Without the approach step the stars step with the acceleration criterion, about 7 kyr at closest approach, against an
encounter time of 1 kyr.

- `test_pair_energy_through_the_encounter`: pair energy (kinetic minus G m1 m2 / r) changes by less than 1e-3 of the initial
  kinetic energy; the closest sampled separation is below 0.1 pc.
- `test_mass`: gas plus star mass constant to 1e-12.

Inputs: `SUNDIALS_DIR`; tables from `GIZMO_STELLA_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
