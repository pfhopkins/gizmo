# sgchem_uvb_shielding

Shielding of the cosmological UV background (TREECOOL, FG20) by the TreeCol columns, `SGCHEM_UVB_TREECOL_SHIELDING`, in the
zoom configuration (net 17, HEALPix columns, no periodic box). A one-zone driver (`sgchem_uvb_shielding_driver.cc`) runs `begrun()`
on a small comoving box at z = 6.5, sets a cell's density, initial state (100 K, x_HII = 1e-4) and TreeCol gas columns, and evolves it
with `do_the_cooling_for_particle` for 30 Myr. The test builds GIZMO with and without the flag.

Checks:

1. Zero columns: the result is bit-for-bit that of the build without shielding (n_H = 0.1, 1, 10, 100).
2. Band attenuation factors (10 bands, `chem_uvb_shield_factors`) against the FG20 z = 6.5 spectrum (`fg20_spectrum_z6_7.txt`)
   integrated with Verner+96 cross sections: within 10% where the exact factor is > 0.05, 15% where it is > 0.01, for the HI and HeI
   photoionisation and heating rates, with and without a neutral He column. Exactly 1 at zero column. HI heating per ionisation
   rises monotonically behind the column (to 10.6 times the thin value deep inside).
3. A uniform sphere (centre column n_H R = 1e20 cm^-2, the 12 HEALPix pixel columns are its chords): cells whose smallest pixel
   column is >= 1e19 cm^-2 stay neutral (x_HII < 1e-2) and cold (< 1000 K); the unshielded build ionises the gas at n_H <= 1. At
   n_H = 0.1 (fully ionised when thin) the x_HII = 0.5 front lies within 0.25 dex in column of where the exact-spectrum, pixel-averaged,
   shielded HI rate balances case-B recombination.

`make_uvb_shielding_bands.py` regenerates the band table in `cooling/sgchem/chem_interface.cc` from the full FG20 spectrum file.
