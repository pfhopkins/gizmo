# Star formation in gas drained to the IMF-accretion floor creates no mass

A Stella star collects its drawn IMF mass from gas cells above the accretion floor (1% of the merge mass). In gas drained
to the floor a draw cannot collect its mass; it must be undone (gas back to the donor cells), never filled from nowhere
(before the fix: 1e-10 code mass per such star; zoom run full_1e8_c_L12_ketju1: 991 stars, +1460 Msun).

## Setup

- 12^3 cells of 1 Msun (float32, as in Gadget ICs) at n_H = 200 cm^-3, B = 2 uG along z (set in the ICs: a snapshot start
  reads B from there), 4 ranks, Stella production model with the forced-SF density cutoff; `nHcutoffSF` and
  `CritPhysDensity` 1e-4, SF cooldown 1e-4 t_ff (~1.6 steps): cells form stars every other step and the box drains
- `TreeDomainUpdateFrequency 2`: no decomposition, so no merging, on whole-box steps (as in a deep zoom sub-step)
- second run: one 64 Msun cell and SF efficiency 1e8 per free-fall time, so that cell forms ~100 stars per accretion call

## What is tested (every accretion call, test flag TEST_IMF_FAILED_DRAW_RESTORE_CHECK)

- each gas cell ends with its mass less what it gave to kept stars, its field less the matching share, u and v unchanged
  (1e-12): undone draws restore their donors also in calls with kept stars; B is non-zero, so the field restore is tested
- the grants keyed to a star add up to the mass the star collected (1e-12); star keys are unique, also past 31 stars from one
  cell in one call
- every donor is blocked for SF and cooled down
- gas plus new stars keep mass and momentum (1e-13); no star gets mass without donors
- total baryonic mass conserved to 1e-14, every mass-trace checkpoint to 1e-12 (gizmo.conservation_trace)
- every star holds its drawn mass to within 1% (SFinfo.txt)
