# KETJU integrator unit tests

Tests of the ketju-integrator repository, separate from this repository, from a source checkout given by the environment
variable `KETJU_SOURCE_DIR` (branch `main`, the C++/Kokkos MSTAR library). Without it the test is skipped. GIZMO itself builds
against the install at `KETJU_PATH`. No GIZMO build or simulation is involved.

KETJU in GIZMO uses the C++/Kokkos MSTAR only. The former C library (`KETJU_LEGACY_C_LIBRARY`, removed) is not built or
tested here; GIZMO still reads its header `ketju_integrator/ketju_integrator.h` (pin `KETJU_C_REF`) as its data model.

## What runs

`test_ketju_kokkos_ctests` builds `KETJU_SOURCE_DIR` (Kokkos from `KOKKOS_ROOT` or the systype Kokkos (HOME_GIZMO); build directory
in a pytest temporary directory) and runs its 5 ctests: `mstar_unit_tests` (KETJU's unit tests and the GIZMO-addition
and softening-kink cases below on `mstar::mstar`), `mstar_compare_c` (against stored results of the former C library: Kepler,
Pythagorean, Plummer cores, PN binary and cusp, BH mergers, contact collisions, tide, softening), `mstar_extra_force`,
`mstar_embedded_tide`, `mstar_driver`. See `ketju-integrator/README.md`.

## Cases of `mstar_unit_tests`

- `Integration_*`: binary integration (Newtonian, PN, spin, three-body PN terms, unit systems)
  against stored results of the old integrator.
- `Integration_with_mergers`, `Instant_mergers`: BH mergers (head-on, from an orbit, multiple
  mergers with per-particle extra data).
- `MST`: tree construction on random particles, host and device tree.

`Gizmo_*`: GIZMO's additions, each with an analytic or conserved-quantity check and,
where feasible, a check that fails without the feature or fix:
- `Gizmo_collision_radius.null_after_create`: creating a system sets `collision_radius` on a garbage-filled struct.
- `Gizmo_star_collisions.head_on_contact`: radial infall, contact time vs the Kepler solution, product at the CoM with
  summed mass and momentum, E_end = E0 + dE_collision; no merger without radii.
- `Gizmo_star_collisions.graze_and_miss`: flyby with two-body pericentre 0.8 (R1 + R2) collides, 1.25 misses; a bound
  third star; energy balance.
- `Gizmo_collision_energy.includes_other_members_potential`: dE_collision = -E_pair + the pair's potential-energy change
  with the other members = the exact total-energy change (the pair term alone misses it).
- `Gizmo_pericentre_guard.osculating_pericentre_time`: an r.v sign change counts only if the osculating pericentre was
  passed since the last check (and not near apocentre).
- `Gizmo_pericentre_guard.k3_triple_no_false_collision`: the K3 triple of `test/ketju_star_collision` in host-sized calls,
  no collision, energy conserved (positive control: MSTAR alone does not reproduce the GIZMO false collision).
- `Gizmo_external_tide.harmonic_com_and_energy`: tide T = -w^2 I, CoM moves as X0 cos(w t), energy incl. the tide
  potential conserved, no step-count blow-up; no CoM motion with the tide off.
- `Gizmo_collision_energy.tide_and_softening`: collision energy with the tide and softening.

`Softening_kink.*`: GBS steps landed on the softening kernel's break points (two-body
orbit, collision with tide and softening converge; off without softening).

Negative checks run by hand (reverting each fix): init, pericentre guard, dE_others and the tide's work each make
their test fail.

## Running

    KETJU_SOURCE_DIR=<checkout> pytest test/ketju_unit/test_ketju_unit.py

Build ~1 min with 4 make jobs; the runs take seconds.
