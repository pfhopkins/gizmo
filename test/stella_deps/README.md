# stella_deps

`scripts/get_stella_deps.sh`, the one setup command for what Stella needs outside this repository (stella-tables release,
SUNDIALS, KETJU), and the defaults by which `make` and GIZMO then find it.

Checks (`test_stella_deps.py`):
1. clean HOME: the setup installs the release `tables-<ref>` of the pinned stella-tables commit into
   `~/.stella-deps/tables-<ref>`, hash-checked, and SUNDIALS (`sundials-<version>`, pinned in `src/Makefile`); the stamp
   holds no machine-local path; a rerun downloads and builds nothing (< 30 s);
2. `make ketju-path` reports the KETJU install `make` will use: `$STELLA_DEPS_PREFIX/ketju-main-kokkos-<key>` (the latest commit of branch main, no pin),
   different for another Kokkos install; an explicit `KETJU_PATH` wins;
3. a KETJU build without the install stops in `make` with "run scripts/get_stella_deps.sh --ketju", an SGChem build
   without SUNDIALS with "run scripts/get_stella_deps.sh";
4. (slow) `new_machine.sh`, the procedure of the docs end to end with an empty HOME: fresh clone, setup, build and a few
   steps of the cloud smoke run (`scripts/stella`, IC from the stella-tables `ics` release), the same with KETJU (make
   first stops naming the setup; the setup then picks KETJU up from `Config.sh`), rerun of the setup = check only, a run
   without the cache stops at start-up naming the setup.

The start-up defaults for `StellarTablesFile` are also checked in `test/stellar_evolution_tables` (unset, `SET_ME`, no cache).

Tests 1 and 4 need the network and are skipped without it. Test 4 runs in the caller's build environment (modules,
`LD_LIBRARY_PATH`, `SYSTYPE` and its Kokkos); `NP` sets ranks and make jobs (default 4).

    pytest -m 'not slow' test/stella_deps     # 1-3, about a minute
    pytest test/stella_deps                   # with 4: builds and runs twice, about half an hour
