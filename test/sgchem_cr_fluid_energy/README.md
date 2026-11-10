# sgchem_cr_fluid_energy

Energy exchange between the CR fluid (`COSMIC_RAY_FLUID`) and the SGChem chemistry: the chemistry's CR
ionisation heating (the fluid's share of zeta) is taken out of the CR energy, and the fluid's own loss is hadronic plus
Coulomb on free electrons only. A uniform periodic box of 8^3 cells of 1 Msun (n_H ~ 30 cm^-3, x_H2 = 0.4) with
2 eV cm^-3 of CRs, no sources; 2 ranks, 0.29 Myr; ledger rows (`OUTPUT_MASS_METAL_ENERGY_LEDGER`) at 7 snapshots.

| test | checks |
|---|---|
| `test_cr_energy_closes` | the ionisation energy taken from the fluid is the fluid's share (> 0.5, <= 1) of the chemistry's CR ionisation heating and >= 1e-3 E_CR,0; E_CR - E_CR,0 = -(fluid losses + taken ionisation energy) up to the unbooked CR-gas exchange at the kicks (reported, < 1e-3 E_CR,0) |
| `test_cr_loss_rate` | fluid losses between rows = the rate (0.87 x 6.37e-16 + 0.53 x 3.09e-16 n_e X_H)(rho/m_p) E_CR, trapezoid with snapshot n_e (2 %) |

Inputs: `SUNDIALS_DIR`; chemistry tables from `SGCHEM_TABLES_DIR` (else installed by `scripts/get_stella_tables.sh`).
