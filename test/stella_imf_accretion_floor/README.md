# Star formation in gas drained to the IMF-accretion floor creates no mass

A Stella star collects its drawn IMF mass from gas cells above the accretion floor (1% of the merge mass). A cell at the
floor cannot donate, so it must not form a star: before the fix such a draw found no donor and was given 1e-10 code mass
from nowhere (zoom run full_1e8_c_L12_ketju1: 991 draws, +1460 Msun).

## Setup

- 12^3 cells of 1 Msun (float32, as in Gadget ICs) at n_H = 200 cm^-3, 4 ranks, Stella production model with the
  forced-SF density cutoff; `nHcutoffSF` and `CritPhysDensity` 1e-4, no SF cooldown: every cell forms one star per step
- `TreeDomainUpdateFrequency 2`: no decomposition, so no merging, on whole-box steps (as in a deep zoom sub-step)

## What is tested

- the box drains: every gas cell ends at the floor
- no star is drawn without donor mass
- total baryonic mass conserved to 1e-14 (mass_balance_steps.txt)
- no star heavier than its drawn mass (SFinfo.txt)
