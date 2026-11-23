# single_stars_invariant_checks

`TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS` (test only, off by default): run-time invariant checks of the single-star model.
Each check prints `SINGLE_STAR_CHECK <tag> FAIL Task=<rank> ...` when its invariant breaks, and
`SINGLE_STAR_CHECK <tag> RAN Task=<rank> n=<cases so far>` at the first and every 1000th call of its site. The checks only
print; the run goes on. All of them run on the host (no check inside a device kernel). Without the flag nothing is compiled.

| tag | where | invariant |
|---|---|---|
| `fire_slot_deposit` | feedback deposit applied to a cell (`mechanical_fb.cc`) | each metal slot of the cell gains exactly the metal mass sent to it (8 ulp of the terms) |
| `fire_slot_invariant_cell` | same | a cell whose C..Fe slots summed stay below total Z keeps that |
| `fire_slot_invariant` | after the feedback, all gas | no metal slot negative; C..Fe summed below total Z |
| `mass_sync` | same | `CellP.Mass` equals `P.Mass` (1e-10) |
| `fb_booked_deposited` | end of a single-star event (`single_star_feedback.cc`) | the star lost the ejecta it booked (1e-6); after a death it is below its trigger mass |
| `lifetime_consistent` | band luminosities (`single_star_luminosity.cc`) | a star that radiates has not died |
| `donor_mass_floor` | gas collection (`sfr_collect_gas.cc`) | every donor keeps 0.01 `MinMassForParticleMerger` |
| `donor_cooldown` | gas collection (`sfr_collect_gas.cc`) | every cell that gave mass is blocked until its density is recomputed and cools down for `SFCooldownFreeFallFraction` of its free-fall time, as the parent |
| `tree_insert` | new single star (`single_star_attach_to_new_star`, `sfr_collect_gas.cc`) | the new star sits at its parent's position, in its parent's tree node |
| `dgr_consistent` | SGChem cell step (`sgchem_cell.cc`) | hot gas (above 1e5 K in the chemistry and stored): no dust in the chemistry nor in the TreeCol columns |

With the flag every single-star event also prints `SINGLE_STAR_YIELDS ch=<SN|AGB|wind|Ia> ID=... t=... table <27 table
element ejecta [Msun]> slots <ejecta in every metal slot [Msun]>`.

Problem: the Stella cloud flags (`scripts/stella/Config_stella_cloud.sh`) plus the switch, `TEST_SINGLE_STARS_BIRTH_AT_GIVEN_CELLS`
and `TEST_SINGLE_STARS_TYPE_IA_RATE_MULTIPLIER`; the lattice and stars of `single_stars_metals` (16^3 cells of 1 Msun at
n_H = 10 cm^-3, 100 K, the tables' solar composition in all 27 slots; a 20 Msun core collapse, a 3 Msun AGB death whose white
dwarf then explodes, a 1 Gyr white dwarf exploding at once, a 35 Msun star 1 Myr old); two stars born by the test hook at gas
cells with IDs 1101 (20 Msun) and 3277 (2 Msun), in octants without stars, which collect their gas from neighbours. 5e-3 time units
(4.9 kyr), 2 ranks.

Checks (bars written before the first run):
1. `test_checks_pass_in_stella_run`: no line prints FAIL; all tags above except `fire_slot_invariant_cell` print RAN; each
   source (SN, AGB, Type Ia, winds) prints its yield trace with 27 table elements. Measured: SN 1, AGB 1, Type Ia 2, wind 32
   trace lines.
2. `test_broken_invariant_is_reported`: the same run with total Z set to half the sum of the 25 metals in every cell, 2.5e-4
   time units: `fire_slot_invariant` prints FAIL at the first feedback step (for example "2047 of 2048 cells"),
   `fire_slot_invariant_cell` stays silent (it tests only cells that held the invariant before), and the run reaches its end.
3. `test_identities_unique_past_roll_over_and_restart`: 40 stars from one cell in one call (the parent's generation counter
   rolls over after 31): every star's (ID, child, generation, `ParticleIDWraps`, `ParticleIDLineage`) in the snapshot is
   distinct; a snapshot restart from it (stars set apart by small offsets, GKS-TI18) brings every born star back, and every
   star present in both snapshots keeps its identity.

Reference implementation: stella_dev `STELLA_SELFCHECK` (`galaxy_sf/resolvedism_selfcheck.h`); the same tags, log prefix
`STELLA_SELFCHECK`.
