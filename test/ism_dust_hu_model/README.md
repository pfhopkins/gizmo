# ism_dust_hu_model

Dust processes of Hu et al. 2019, 2023, 2026 (`GALSF_ISMDUSTCHEM_HU_MODEL`) in place of the accretion and sputtering of the live ISM dust
model, on its fields (element dust, species, sources), which keep their transport, snapshots and restart. Chemistry is SGChem network 5.

Problems: 8^3 lattices of 1 Msun cells at solar metallicity, half the carbon and 82% of the silicon in dust: hot gas (n_H = 1, u of 1e7 K for
a neutral gas, 5.2e6 K after ionisation) for 20 kyr, cold gas (n_H = 1e3, 39 K) for 150 kyr.

Checks:
1. hot gas at rest: ln(D1/D0) = -3 n_H Y(T) dt / a_eff for carbon and silicate, with the Nozawa+06 fits of Hu+19 (1%);
2. cold gas: the logistic growth of Hu+23/26 of carbon and silicate, rate Z' n_H sqrt(T/100) / 1.5 Gyr, pools without the CO-bound C and O (2%);
   the gas-dust accretion is booked as a dust source;
3. the metals do not change, slot by slot (1e-12, no diffusion); dust stays inside its elements; element dust, species and sources add up;
4. the same dust mass on 2 and 4 ranks (1e-8).
5. a fresh start seeds half of each cell's carbon and 82% of its silicon in FeMgSiO4 (capped at 90% of O, Mg, Fe), from supernovae;
6. with single stars (a 20 Msun supernova and a 3 Msun AGB death, 16^3 lattice, 0.5 kyr): by default the supernova makes no dust and destroys
   none by swept mass, the AGB death condenses dust; with `GALSF_ISMDUSTCHEM_HU_SN_EJECTA_CONDENSATION` the supernova adds carbon dust (> 0.01 Msun),
   with `GALSF_ISMDUSTCHEM_HU_SN_SWEPT_MASS_DESTRUCTION` the gas loses, 50 yr after the event, over twice what it loses by sputtering without it (measured 5.4 times; the destruction saturates in the kernel of 1 Msun cells).

Reference: analytic rates; comparison with the reference implementation on the same lattices in the PR text.
