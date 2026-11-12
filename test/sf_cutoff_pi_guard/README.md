# Forced-SF density cutoff respects the HII-region veto

`GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF` forces star formation above `nHcutoffSF`. It must keep the vetoes that
`get_starformation_rate` applies before its SFR criteria: zero-mass cells and ionised (HII-region) cells.

## Setup

- 30 Msun Stella star (in the IC, age 0.1 Myr) at the centre of a uniform n_H = 200 cm^-3 box, 24^3 cells of 1 Msun, 4 ranks
- `nHcutoffSF 150`: every cell is above the cutoff; `FacSfThreshMJ 1e-6`: the Jeans test fails everywhere, so every
  new star comes from the forced path
- photoionisation runs before star formation in a step, so the HII region is marked before the first SF pass

## What is tested

- an HII region forms (>= 50 ionised cells; radius R from the ionised volume)
- no new star inside R - 1.5 dx; without the guard the cutoff converts the HII region too
- new stars outside R + 1.5 dx (the cutoff works in neutral gas)
