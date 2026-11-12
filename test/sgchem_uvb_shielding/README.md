# sgchem_uvb_shielding

`SGCHEM_UVB_TREECOL_SHIELDING`: in comoving runs the TREECOOL (FG20) HI, HeI, HeII photoionisation and heating rates of the
SGChem UV-background floor are attenuated per HEALPix pixel in 10 FG20 bands by exp(-sum_s sigma_s,band N_s), with N_s the
TreeCol gas column times the cell's neutral H, He0, He+ fractions, and averaged over the pixels. The G0 floor from the
background stays unshielded.

Checks here (standalone, no GIZMO build): the factors against the exact integral over the FG20 z = 6.5 spectrum with Verner+96
cross sections (10 %, 15 % below 0.05, where above 0.01), exactly 1 at zero column, heating per ionisation hardening behind
the column. The stored per-pixel gas column: `treecol_pixels::test_gas_column_for_uvb_shielding`.
