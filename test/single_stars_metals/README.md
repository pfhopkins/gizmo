# single_stars_metals

Metals of single-star builds in the FIRE-3 layout (slot 0 total Z, 1 He, 2-10 C, N, O, Ne, Mg, Si, S, Ca, Fe; H
implicit), with the production chemistry (`SGCHEM`, network 5). Build: `GALSF_SINGLE_STARS_FROM_IMF`,
`GALSF_SINGLE_STARS_EVOLUTION_TABLES`, `GALSF_FB_MECHANICAL`, `GALSF_SINGLE_STARS_TYPE_IA` (with the test-only Type Ia
rate multiplier), `COOL_METAL_LINES_BY_SPECIES`. Problem: a 16^3 lattice of 1 Msun cells at n_H = 10 cm^-3, 100 K, no
gravity, 2 ranks (`make_single_stars_metals_ics.py`).

Checks:
1. `test_solar_abundances`: a fresh start (start flag 0) with `InitMetallicity` 1 and 0.3 gives total Z = 0.014 x
   `InitMetallicity` and slots 2-10 = the tables' solar composition (their initial composition at the Z = 0.014 node,
   Asplund+09 present day) x `InitMetallicity`, He = primordial + scaled excess, to 1e-10 (the code's constants carry 10
   digits). This is stella_dev's start (`SolarAbundances[0]` = 0.014 and the tables' element pattern). Before: total
   Z = 0.02 and GIZMO's older pattern (C, N, O 32 %, 82 %, 44 % above the tables' values). With
   `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS` the 16 other elements start at the tables' solar values too.
2. `test_27_element_closure` (build with `GALSF_SINGLE_STARS_27_ELEMENT_YIELDS`, `METALS_EXTRA_PASSIVE_SPECIES=16`):
   four stars at Z = 0.014, each alone in its region, 5e-3 time units: a 20 Msun core collapse (winds counted as
   returned), a 3 Msun AGB death, a 1 Gyr white dwarf (Type Ia at once) and a 35 Msun star 1 Myr old (winds); the AGB
   white dwarf explodes too. Initial metals: the tables' solar composition in all 27 slots, total Z = the sum of the 25
   metals. In every gas cell, split cells included, total Z equals the sum of the 25 metal slots to 1e-12 (measured
   1.4e-15; start 1.2e-16); every slot gains. Before: 11.9 % off in the cells around the Type Ia, 0.38 % over the box
   (the 16 elements were only diluted).
3. `test_27_element_shared_slots_identical`: the same run with the default build (11 slots): event log, slots 0-10,
   masses, energies, velocities, positions and star masses bitwise identical.
4. `test_27_element_supernova_yields`: one 20 Msun core collapse: each of the 26 element slots gains the table SN
   element ejecta scaled to the ejecta mass (measured 5.5e-13, bar 1e-11; Ni: none in the table, none in the gas); the
   total Z gain equals the event's metals to 1.3e-14 of the gas metal mass (bar 1e-12).
5. `test_dust_above_metals_stops` (build with `SGCHEM_DGR_INDEPENDENT_OF_METALLICITY`): a fresh start with the
   constant dust-to-gas ratio 0.01 (`SGChemDustToGasRatio` 1) and `InitMetallicity` 0.3 (Z = 0.0042, dust above the
   metals) stops at start-up with a message, as stella_dev; with `InitMetallicity` 1 it runs.

Reference: the stellar tables (stella-tables a4830ab) and stella_dev `core/init.cc`.
