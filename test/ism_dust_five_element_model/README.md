# ism_dust_five_element_model

The five-element ISM dust model (`GALSF_ISMDUSTCHEM_FIVE_ELEMENT_MODEL`; stella_dev `GALSF_RESOLVEDISM_DUST`) in place of the
accretion and sputtering of the live ISM dust model, on its fields (element dust, species, sources), which keep their transport,
snapshots and restart. Dust lives in C (carbon) and O, Mg, Si, Fe (FeMgSiO4). Chemistry is SGChem network 5.

Problems: 8^3 lattices of 1 Msun cells at solar metallicity, 30% of the carbon and of the silicon in dust: hot gas (n_H = 1,
5.2e6 K after ionisation) for 20 kyr, cold gas (n_H = 1e3, 35 K) for 150 kyr; a fresh start; a 20 Msun supernova and a 3 Msun AGB
death in a 16^3 lattice with half the carbon and silicon in dust, 50 yr.

Checks (the reference is the stella_dev code, transcribed in `test/ism_dust_five_element_kernels`):
1. hot gas: ln(D1/D0) = -3 n Y(T) dt / 0.1 micron for carbon and silicate, Nozawa+06 fits, n = rho / m_p as stella_dev (1%);
2. cold gas: the growth of the stella_dev step (Hirashita sticking with SGChem's dust temperature, Dwek 1998 rates), integrated
   cell by cell in 200 steps at the mean temperatures (3%); silicate stoichiometric; booked as the gas-dust accretion source;
3. the metals do not change, slot by slot (1e-12); dust only in C, O, Mg, Si, Fe and inside its elements; element dust, species
   and sources add up;
4. the same dust mass on 2 and 4 ranks (1e-8);
5. a fresh start seeds dust-to-gas 0.01 Z/0.014, 30% carbon (at most half the C), 70% silicate (at most 90% of O, Mg, Si, Fe),
   from supernovae (1e-6);
6. a supernova in 1 Msun cells: every cell it enriches by over half loses all its old dust (McKee 1989 swept mass with efficiency
   0.4 inside it, up to all of a cell's dust, as stella_dev; the shared loop's own rule would leave 60%); cells at the kernel edge,
   with a small share of the energy, lose part.
