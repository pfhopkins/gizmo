# stellar_evolution_tables

Start-up check of the single-star stellar-evolution tables (`GALSF_SINGLE_STARS_EVOLUTION_TABLES`).

The tables (stella-tables, https://github.com/Ulli1991/stella-tables: lifetimes, tracks, band luminosities, winds,
yields, remnants) are not in the repository and are never fetched by GIZMO. `scripts/get_stella_tables.sh` builds or
copies them once, at the stella-tables commit these sources expect, and checks their content hashes; the harness helper
`get_stellar_evolution_tables()` calls it and returns the `StellarTablesFile` value. At start-up rank 0 recomputes the
content hash of the main table and of the wind table next to it and stops if either differs.

Problem: a uniform periodic box of 16^3 gas cells, run for a few steps (2 ranks).

Checks:
1. the installed tables: both files are reported ok and the run reaches `TimeMax`;
2. a path without tables: the run stops at start-up and names the file and the install script;
3. a file with the right name and other content: the run stops at start-up and names its hash.

Reference: the hashes compiled into `src/galaxy_sf/stellar_tables_check.cc`, recomputed by `scripts/stella_tables_hash.py`
(same definition: HDF5 datasets sorted by path, native byte order, attributes excluded).
