# Build system PRs (issues #95, #96)

Series: build system, five PRs in two chains: bs/01 -> bs/02, and bs/01 -> bs/03 -> bs/04 -> bs/05. Independent of the Stella series.
Covers: unknown-flag check in `make`, Template_Config consistency warning, `prepare_config` in Python, a CMake build next to the Makefile, the flag check in CMake configure.
Base: <BASE>. Tip: <TIP>. Results below were measured on gpu_kokkos_unified 070c5b78f, Kokkos OpenMP, `tools/objcmp.py`, 140 objects each unless stated, except where marked as an older base.
Full suite: <SUITE RESULT>.

Series-level results (070c5b78f):
- Tests of the series at the bs/05 tip: `test/config_flag_check` + `test/cmake_build` 746 passed (no log kept; to be re-measured on <TIP>) (bs/01 3/3, bs/02 5/5, bs/03 95 with the flag tests). The source-list guard failed first on the base's spaced `NUCLEAR_NETWORK_SOLVER 1` condition, then passed after the reader accepted options with a space.
- Flag check on `Template_Config.sh`: passes. On all 92 `test/**/Config*.sh`: one catch, `test/sbcluster/Config.sh` `PM_HIRES_REGION_CLIPPING`.
- Template warning (bs/02): 4 template options read nowhere, 65 names tested in source and missing from the template (no log kept; to be re-measured on <TIP>).

---

## PR 1: bs/01-config-flag-check (#95)
**What and why.** Plain `make` stops when an active Config.sh option appears in no source file or Makefile, and names it. `make build` builds without the check and the message says so. A misspelled or removed option no longer gives a silently different build.

**Evidence before.** `OUTPUT_IN_DOUBLEPRECISSION` in a Config: `make` exits 0 (issue #95). Test: `test/config_flag_check` (2 of 3 fail before the change: no `check` target). Narrow.

**Change.** `src/buildsystem/check_config_flags.py` (76 lines, arguments `SRC_DIR CONFIG`); `src/Makefile`: `all` runs `check` then `build` one after the other (also under `-j`), plus the targets `check` and `build`; one docs paragraph. An option is "known" if its name appears outside comments in any source, header or Makefile under the source dir. Makefile options such as CHIMES and OPENMP are tested through CONFIGVARS and are found there. `Template_Config.sh` and the generated files do not count.

**Cost.** One Python run (about 1 s) per plain `make`; skipped with a warning if python3 is missing.

**Evidence after.** A misspelled config fails and names the flag, a correct one passes, `make build` runs no check: 3/3. On all 92 `test/**/Config*.sh`: one catch, `PM_HIRES_REGION_CLIPPING` in `test/sbcluster/Config.sh`. On the 160 compile-suite rows (measured on an older base; to be re-measured on the audit branch): `PM_HIRES_REGION_CLIPPING` (row with PMGRID=32), `OUTPUT_RECOMPUTE_POTENTIAL` (only in a comment in `core/run.cc`, retired), `OUTPUT_UNSPAWNED_SINKMASS` (template only), `RT_ISRF_BACKGROUNDs` (typo for `RT_ISRF_BACKGROUND`). None is fixed here; they will stop `make` for `test/sbcluster` and for those compile-suite rows until fixed.

**Flag off.** No source file touched; build products unchanged (the Makefile `build` path is the old rule). Flag off: to be re-run on <TIP>.

**Limits.** The four catches above stop plain `make` for those configs until fixed; `make build` skips the check.

**Depends on / followed by.** None. Followed by PR 2 and PR 5.

---

## PR 2: bs/02-template-consistency-warning (#95, reverse direction)
**What and why.** Plain `make` prints one warning line when `Template_Config.sh` and the source disagree; `make check_template` lists them. It never stops the build.

**Evidence before.** On the base: 4 template options are read nowhere and 65 names are tested in source and missing from the template.

**Change.** `src/buildsystem/check_template.py` (75 lines), two Makefile targets, one docs line. Test: `test/config_flag_check/test_template_check.py` (fails before, 5/5 pass after). Narrow.

**Cost.** One Python run per plain `make`.

**Evidence after.** `test/config_flag_check/test_template_check.py` 5/5 (narrow).

**Flag off.** No source file touched. Flag off: to be re-run on <TIP>.

**Limits.** Warns only; the 4 + 65 mismatches are not fixed here.

**Depends on / followed by.** Needs PR 1.

---

## PR 3: bs/03-prepare-config-python
**What and why.** `prepare_config.py` writes `GIZMO_config.h` and `compile_time_info.cc` without Perl, for CMake. The Makefile is unchanged and still uses the Perl script.

**Evidence before.** No Python equivalent exists; CMake needs one.

**Change.** `src/buildsystem/prepare_config.py`; no Makefile change.

**Cost.** None.

**Evidence after.** `test/cmake_build/test_prepare_config.py`: output byte-identical to `prepare-config.perl` for all 92 test Configs (95 tests with the flag tests). Narrow.

**Flag off.** Makefile untouched. Flag off: to be re-run on <TIP>.

**Limits.** The Makefile keeps the Perl script; two generators exist until the Makefile switches.

**Depends on / followed by.** None. Followed by PR 4.

---

## PR 4: bs/04-cmake-build
**What and why.** `src/CMakeLists.txt` next to the Makefile. Config.sh stays the interface (`-DGIZMO_CONFIG=`). Sources come from the Makefile itself: `src/buildsystem/makefile_sources.py` reads the Makefile's OBJS lists and the `ifeq (X,$(findstring X,$(CONFIGVARS)))` blocks (an option with a space is accepted). An object assignment inside a condition it cannot evaluate is an error, so the two cannot drift silently.

**Evidence before.** No CMake build exists.

**Change.**
- Libraries: MPI, OpenMP, HDF5, zlib, Kokkos (its CMake package), FFTW (when PMGRID or TURB_DRIVING_SPECTRUMGRID), HYPRE, SUNDIALS, Grackle (GRACKLE_PATH or PKG_CONFIG_PATH, like the Makefile), SkyNet (SKYNET_PATH, SKYNET_EXTRA_INCL and SKYNET_EXTRA_LIBS as in the Makefile; HAS_SKYNET tests `NUCLEAR_NETWORK_SOLVER 1`; solver 2 is sources only, no library). GSL only on request; the Makefile does not link it. HDF5 link directory so `-lhdf5` from `grackle.pc` resolves.
- Presets in `src/CMakePresets.json`: generic, orion-kokkos, rusty-genoa (the RUSTY_GENOA systype: -O2 -funroll-loops -g -Wall, Kokkos OpenMP from $HOME/opt/kokkos-openmp).
- Docs: HYPRE, SUNDIALS, Grackle and SkyNet are separate installs found through HYPRE_ROOT, SUNDIALS_DIR, GRACKLE_PATH, SKYNET_PATH.
- Each Stella PR that adds a Makefile library block adds the matching CMake lines (the KETJU_PATH and SGCHEM blocks come with the Stella PRs); `test_library_options.py` fails otherwise.

**Size.** CMakeLists 176 lines, `makefile_sources.py` 77, presets 35: 288 lines (249 without comments and blank lines), over the 200-line guide, under 300. It is kept in one PR because the CMake file does nothing without the source-list reader.

**Cost.** None for the Makefile build; CMake is an additional way to build.

**Evidence after.** `test/cmake_build/test_source_lists.py`: the CMake source list equals the Makefile's compile list for all 646 configs on 070c5b78f (92 test Configs, 162 compile-suite rows, 392 single Template options; counted from the test's own lists). `test_library_options.py` sees spaced options too (solver 2 listed as sources-only). Narrow for the library test, broad for the object comparison below.

Objects, Makefile vs CMake, same flags, 070c5b78f:
- gmc_cooling (SINGLE_STAR_STARFORGE_DEFAULTS): 87 byte-identical, rest same instructions, 0 differ.
- fire (FIRE_PHYSICS_DEFAULTS=3, `test/fire`): 81 byte-identical, 0 differ.
- STARFORGE (`test/gmc_cooling_rt`, SINGLE_STAR_STARFORGE_DEFAULTS + RT): 85, 0 differ.
- forgedinfire: 80, 0 differ.
- SkyNet solver 1 (`nuclear_skynet.o` alone, SKYNET_PATH and SKYNET_EXTRA_INCL from the environment for both): same instructions.
- Grackle 3.4.1 (compile-suite rows 153 and 154; measured on an older base; to be re-measured on the audit branch): both build systems link and build 140 objects each; 93 byte-identical, rest same instructions, 0 differ.
- SkyNet (nuclear_detonation Config with `NUCLEAR_NETWORK_SOLVER=1`; measured on an older base; to be re-measured on the audit branch): 144 objects, 97 byte-identical, 0 differ.
- Short starts of the CMake binary (measured on an older base; to be re-measured on the audit branch): gmc_cooling to TimeMax 0.03, gas snapshot bit-identical to the Makefile binary (0 of 54 datasets differ). fire: both binaries run the 131 steps to the final time; 5 of 60 datasets differ in a few cells (Coordinates last bit, CoolingRate, MolecularMassFraction); two runs of the Makefile binary differ in the same 6 datasets, so this is run-to-run noise, not the build.
- Configure-only: HYPRE and SUNDIALS/CHIMES configure with install prefixes.

**Limits.** Not covered: HYPRE/SUNDIALS/CHIMES beyond configure; the RUSTY_GENOA preset (no access to that machine); the other Makefile systypes (Frontera, Vista, Lonestar6, Frontier, Expanse, CaltechHPC, BigRed200, RUSTY, MacBook, github-ubuntu) are not ported; GPU backends (nvcc_wrapper, hipcc) untested; Stella production, tallbox and cloud configs not run.

**Flag off / Makefile.** Makefile untouched in this PR. Flag off: to be re-run on <TIP>.

**Depends on / followed by.** Needs PR 3. Followed by PR 5.

---

## PR 5: bs/05-cmake-config-check
**What and why.** CMake configure runs `check_config_flags.py` before any library search, so a misspelled option also stops a CMake configure. `-DGIZMO_SKIP_CONFIG_CHECK=ON` skips it, and the message says so.

**Evidence before.** `test/cmake_build/test_cmake_config_check.py`: 1 of 2 fails before.

**Change.** `src/CMakeLists.txt` runs the script; the script gets an optional third argument for the skip hint.

**Cost.** One Python run (about 1 s) per configure.

**Evidence after.** `test/cmake_build/test_cmake_config_check.py` 2/2. Narrow.

**Flag off.** Makefile untouched. Flag off: to be re-run on <TIP>.

**Limits.** Same catches as PR 1 stop a CMake configure.

**Depends on / followed by.** Needs PR 1 and PR 4.
