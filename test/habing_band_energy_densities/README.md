# habing_band_energy_densities

`scripts/habing_band_energy_densities.py` integrates the Habing (1968) and Draine (1978) interstellar radiation field in the G0 bands
of the chemistry (6-13.6, 8-13.6, 11.2-13.6 and 6-8 eV; spectral forms of Draine 2011, eqs. 12.6-12.7) and prints the Draine / 1.7
values that `cooling/sgchem/chem_interface.h` carries as `HABING_EGY_DENSITY_*`, and the Mathis, Mezger & Panagia (1983) absorbed-power
shares used for the grain heating. No GIZMO build.

Checks: the Habing field in 6-13.6 eV is 5.29e-14 erg cm^-3 to 1% (Draine 2011 sec. 12.5); each constant in `chem_interface.h` equals
the script's Draine / 1.7 to its four digits (5e-4).
