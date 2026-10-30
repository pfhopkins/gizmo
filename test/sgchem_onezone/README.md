# sgchem_onezone

One-zone check of the SGChem chemistry networks (`src/cooling/sgchem/`, `SGCHEM` + `CHEMISTRYNETWORK=N`), without a GIZMO run.

For each network (3, 5, 15, 16, 17) `build_onezone_dump.sh` compiles `cooling/sgchem/cxx/fortran_reference/reference_dump.cc`
against this tree and writes the chemistry tables, the cooling-function rates on a (T, n) grid and 18 one-zone runs
(n_H = 1/100/1e4 cm^-3, T0 = 50/8000/1e6 K, Z = 0.1/1 solar; 12-pixel columns of 10 pc, G0 = 1.7). `reference.py compare`
checks them against the frozen reference: the SGChem Fortran networks.
Bars: tables 1e-6, rates 1e-10, final one-zone temperature 1e-2 (the Fortran DVODE and CVODE step differently).
For nets 5 and 17, `threads_identity.cc` solves eight cells serially and 64 times concurrently on OpenMP threads; the
temperatures must be bit-identical.

Inputs (not in the repository):
- `SUNDIALS_DIR` (6.x or 7.x), `KOKKOS_PATH`, `mpicxx`;
- the chemistry tables: `SGCHEM_TABLES_DIR`, else installed by `scripts/get_stella_tables.sh`; content hashes as in
  `src/galaxy_sf/stella_tables_check.h`;
- the frozen reference data: `SGCHEM_REFERENCE_DATA`, checked against `fortran_reference/data.sha256`.

The test is skipped, with the reason, when an input is not set.
