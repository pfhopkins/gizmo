# stella_dense_sf

A dense cold clump forms Stella stars inside diffuse warm gas, with the production model. One short run serves
several checks.

## Setup

- Stella production model (`Config_stella_default.sh`) in a 16 pc periodic box, gravity not periodic.
- 500 cells of 1 Msun at n_H = 1e6 cm^-3 and 10 K at the centre, inside diffuse gas at n_H = 1 cm^-3 and 1e4 K (1 Msun
  cells), solar metallicity. The diffuse gas sits on the long top-level step (`MaxSizeTimestep` = `TimeMax`/4), the clump
  forms stars on short steps. Runs to 39 kyr on 4 ranks (well under a minute).

## What is tested

- Start-up table self-tests: the ejecta checks (metallicity, mass budget, sign, AGB carbon) run on the tables the
  feedback injects (`elem_ej_SN_mass`, `elem_ej_AGB_mass`, cumulative winds) and pass.
