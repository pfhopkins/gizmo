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
- Gas smoothing lengths are uncapped (`MinGasKernelRadiusFractional 0`): the median kernel radius of the n_H > 1e5 cells
  stays below half of 2.8 x `SofteningGas`, the floor the default fraction 1 would impose.
- `sfr.txt` books Stella star formation: summed over its rows (written on full-sync steps), column 5 equals the drawn
  masses of all stars in `SFinfo.txt` (the last row is at `TimeMax`), and column 4 times the
  interval since the previous row gives the same mass.
