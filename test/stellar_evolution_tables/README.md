# stellar_evolution_tables

Start-up check and lookups of the single-star stellar-evolution tables (`GALSF_SINGLE_STARS_EVOLUTION_TABLES`).

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

The loader (one copy of the age-resolved tables per node in MPI shared memory) and its lookups are checked by a
standalone build of `src/galaxy_sf/stellar_evolution_tables.cc` (`unit/`, like the helmholtz unit test), which evaluates
every lookup on a fixed grid of 64 masses x 8 metallicities x 7 ages (226,843 values) on 2 ranks:
4. at the grid nodes the 2D tables come back exactly (compared with the HDF5 datasets); the pre-SN mass (dataset) equals the
   remnant plus the SN element ejecta (exploding stars) or the AGB element ejecta (white dwarfs) to 1e-12; the lifetime
   falls with mass above 2 Msun; every value of a stored sample (`unit/lookup_reference.txt`, every 61st value)
   agrees with the reference implementation (stella_dev `resolvedism_stellar_tables.cc`, built the same way) to 1e-12
   relative; both ranks give the same values.
The tables themselves are not monotone in lifetime below 2 Msun (around 0.09, 0.35 and 1.8-1.9 Msun); that is a
property of the stella-tables build, not of the lookups.
