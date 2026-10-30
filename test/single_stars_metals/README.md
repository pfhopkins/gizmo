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
   Z = 0.02 and GIZMO's older pattern (C, N, O 32 %, 82 %, 44 % above the tables' values).

Reference: the stellar tables (stella-tables a4830ab) and stella_dev `core/init.cc`.
