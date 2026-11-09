# sgchem_onezone

One-zone check of the SGChem chemistry networks (`src/cooling/sgchem/`, `SGCHEM` + `CHEMISTRYNETWORK=N`), without a GIZMO run.

For each network (3, 5, 15, 16, 17, 18, 20) `build_onezone_dump.sh` compiles `cooling/sgchem/cxx/fortran_reference/reference_dump.cc`
against this tree and writes the chemistry tables, the cooling-function rates on a (T, n) grid and 18 one-zone runs
(n_H = 1/100/1e4 cm^-3, T0 = 50/8000/1e6 K, Z = 0.1/1 solar; 12-pixel columns of 10 pc, G0 = 1.7). `reference.py compare`
checks them against the frozen reference: the SGChem Fortran networks for 3/5/15/16/17, the C++ backend at its introduction
for 18/20. Bars: tables 1e-6, rates 1e-10, final one-zone temperature 1e-2 (the Fortran DVODE and CVODE step differently).
For nets 5 and 17, `threads_identity.cc` solves eight cells serially and 64 times concurrently on OpenMP threads; the
temperatures must be bit-identical.

`test_sgchem_net5_euv_h2[10, 100]` (`euv_h2.cc`, from stella_dev `golden/test_euv_h2.cc`): network 5 with ionising photons
(Gamma_HI = 1e-9 s^-1, <h nu> = 18.6 eV, sigma_H2/sigma_HI = 1.34). The H2 channel adds exactly its rate and heating to
the right-hand side. A molecular cell (x_H2 = 0.4999) loses its H2 and ionises at G0 = 1.7 and at G0 = 1e-4 (shielded).
Without the H2 channel the shielded cell stays molecular (x_H2 = 0.45 after 1 Myr). G0 = 0 cannot stand for shielding:
the backend then skips every photo-rate, the ionising ones too, as the SGChem Fortran does.

Inputs (not in the repository):
- `SUNDIALS_DIR` (6.x or 7.x), `KOKKOS_PATH`, `mpicxx`;
- the chemistry tables: `SGCHEM_TABLES_DIR`, else installed by `scripts/get_stella_tables.sh`; content hashes as in
  `src/galaxy_sf/stella_tables_check.h`;
- the frozen reference data: `SGCHEM_REFERENCE_DATA`, checked against `fortran_reference/data.sha256`.

The test is skipped, with the reason, when an input is not set.
