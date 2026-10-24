# KETJU integrator unit tests

Runs the original googletest suite of the KETJU integrator (MSTAR; `ketju-integrator/unit-tests/`,
GPLv3, Mannerkoski et al.) against GIZMO's vendored, modified `ketju-integrator/` sources.
No GIZMO build or simulation is involved.

## What is tested

33 cases, each on 1, 2 and 4 MPI tasks. The original 26:
- `Integration_*`: binary integration (Newtonian, PN, spin, three-body PN terms, unit systems)
  against stored results of the old integrator.
- `Integration_with_mergers`, `Instant_mergers`: BH mergers (head-on, from an orbit, multiple
  mergers with per-particle extra data).
- `MST/ConstructionTest`: Prim and divide-and-conquer MST construction, 2 to 1000 random particles.

`Gizmo_*` (`unit-tests/gizmo_additions.cpp`): GIZMO's additions, each with an analytic or conserved-quantity check and,
where feasible, a check that fails without the feature or fix:
- `Gizmo_collision_radius.null_after_create`: `ketju_create_system` sets `collision_radius` on a garbage-filled struct.
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

Negative checks run by hand (reverting each fix): init, pericentre guard, dE_others and the tide's work in B each make
their test fail.

## Running

    module load gcc/13; module load openmpi; module load cmake/3.30
    pytest test/ketju_unit/test_ketju_unit.py

The test copies `ketju-integrator/` to a pytest temporary directory (its CMake writes `bin/` and
`lib/` into the source tree), builds with `-DUNIT_TESTS=ON -DBUILD_GMOCK=OFF` and the vendored
googletest subset in `ketju-integrator/thirdparty/googletest/`, and runs with
`mpiexec --oversubscribe`. Build ~1 min with 4 make jobs; the runs take seconds.
