# sgchem_comoving_divergence

In a comoving run the chemistry must get the physical velocity divergence, Hubble flow included (`3 H(a)` on top of the peculiar
`div v / a^2`; at network 20 `H(a)` per axis in the LVG `dv/dr`), as stella_dev (`chem_interface.cc`) and the port's star
formation and dust do. The divergence sets the LVG escape columns of the H2, CII and CO cooling.

Problem: the `sgchem_uv_background` lattice (n_H = 100 cm^-3 physical, 8000 K, solar metals) at a = 0.25, at rest in comoving
coordinates (pure Hubble flow), periodic so no edge drives a peculiar flow, Omega_M the box's own matter (`check_omega`),
Omega_L 0, Delta ln a = 5.7e-5, built at networks 5 and 20 with `TEST_SGCHEM_PRINT_CELL_VELOCITY_DIVERGENCE` (prints
`SGCHEM_DIVV` with H(a), div v, dv/dr and the peculiar dv_k/dx_k in s^-1 for each cell step).

- `test_hubble_rate`: the printed H(a) is H0 sqrt(Omega_M / a^3 + (1 - Omega_M) / a^2) (with the curvature term, as `hubble_function`).
- `test_divergence_is_hubble_flow`: every cell step: div v = sum_k g_k + 3 H, and at network 20 dv/dr = mean_k |g_k + H|, to 1e-6 of H.
- `test_mass_and_energy`: gas mass constant to 1e-12; E_th + E_kin changes by the ledger's chemistry and floor rows plus the
  adiabatic Hubble term -2 int E_th dln a (trapezoid over 19 snapshots), to 2e-5 of the initial E_th.

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else `scripts/get_stella_tables.sh`).
