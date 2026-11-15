# ism_dust_five_element_kernels

Unit test of `src/solids/ism_dust_five_element_kernels.h`, the dust processes of the five-element ISM dust model
(`GALSF_ISMDUSTCHEM_FIVE_ELEMENT_MODEL`; stella_dev `GALSF_RESOLVEDISM_DUST`). No GIZMO build or simulation is needed.

The reference is the stella_dev code (`galaxy_sf/resolvedism_dust.cc`) transcribed line by line in the test, with the rates in 1/yr
and dt in yr (stella_dev multiplied the 1/yr rates by yr / time unit, the inverse of the conversion; see the series text):
- the rate rows of the stella_dev dust self-test: the Nozawa+06 thermal sputtering fits of carbon and silicate and the sticking
  (Hirashita 2000) at T = 1e4 ... 1e8 K;
- its destruction rows: the McKee 1989 swept-mass fraction for n0 = 0.1, 1, 100, 1e4 cm^-3 and cells of 1, 10, 100 Msun;
- one dust step (sputtering, growth, clamp to the element pools) on 70 cells from hot diffuse to cold dense gas, at 0.1, 1, 3 solar
  and off-solar mixtures where O, Mg or Fe limit the silicate;
- the shock-tracked destruction factors (option `GALSF_ISMDUSTCHEM_FIVE_ELEMENT_SHOCK_DESTRUCTION`) at 30 to 1000 km/s;
- the fresh-start seed at Z = 0, 0.01, 1, 3.6 solar and for an O-poor cell.

Pass: relative difference below 1e-9 in every number, with plain -O2 and with the production flags. The condensation rows of the
self-test (dust in the ejecta of real table stars) are checked by `single_stars_metals` (`test_single_star_dust_yields`).
