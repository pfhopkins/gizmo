# PR texts: KETJU series (kj), draft, nothing posted

**Series.** `kj/01` to `kj/24`. KETJU chain regularisation (MSTAR) of close star encounters, plus sinks, black-hole mergers and post-Newtonian (PN) terms. Everything is under `KETJU_REGULARIZATION` and off by default.

**Order.** Overall PR order is slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. Inside kj: 01, 02, 03, 03b, 04, 05, 06, 07, 08, 09, 10, 11, 12, 14, 15, 16, 17, 18, 19, 19b, 19c, 20, 21, 22, 23, 24. There is no `kj/13` (LET chain spheres are part of `kj/14`; no test can show them working without it). Base placeholder: `<BASE>`. Tip placeholder: `<TIP>`. Full-suite result: `<SUITE RESULT>`.

**Numbers in this file.** Numbers below were measured on an older base (the series on `3a4ad08b7`) and are to be re-measured on `<TIP>`. Every flag-off line is to be re-run on `<TIP>`. Each test is marked narrow (one flag, one bug or one bookkeeping path) or broad (a physical problem with a reference or a conservation law).

# State of the series

**KETJU is a separate install.** MSTAR (Kokkos library) and the C headers are installed outside this repository and found through `KETJU_PATH` (machine block, environment, or make line), as #112 asks for external packages; nothing is fetched at compile time. The build block is in the integration line (commit `d0284b30f`); `kj/01` describes it.

**Pending-momentum fix.** The first version of the series carried its own fix for pending kicks counted twice in the tree build, and a refresh loop in `kj/16`. Both are replaced by the port of the pending-momentum fix of starforge_dev #103 (merged in the base). The three earlier fix branches are gone from this series: the pending-momentum one is superseded by that port, the `readjust_timebase` one is a general fix with its own PR, and the empty-rank fix is now in the base (B2).

**Upstream.** starforge_dev #100 (merged) removes sink super-timestepping and points to KETJU. `SuperTimestepFlag` still exists in `timestep.cc` here. The series has two `SINGLE_STAR_TIMESTEPPING > 0` guards in `ketju_coupling.cc`, both harmless; they go when #100 reaches this branch. It makes the sink path (`kj/20`) more important, so a second look at the one subtraction path matters more.

**Corrections found while building.**
- Moment refresh does not need the pending-momentum clear in plain N-body; only the tree build does. Under KETJU both do. This is now handled by the #103 port.
- The empty-rank abort in the LET was not a domain-roundoff problem. It is fixed in the base (B2).
- A clamp abort in `gpu_topology_build` was suspected; it did not occur in any run of this series. Not shown fixed.
- Type-5 sinks use the exact pair kernel like stars, not the softened subtraction of starforge_dev (1.3e-3 against 1.4e-12 on the test binary). Mike should review this. A comparison against his fewbody and plummer_binaries Type-5 cases was not run.
- The `KetjuFinalVelocity` snapshot block of the reference implementation (stella_dev) is dropped: `Velocities` holds the value at output.

**Test bars changed after seeing a result, all named.** `kj/08`: the `ketju_tide` case with MaxSizeTimestep 1e-8 has the original bar 1e-5. Caps arrive in `kj/19b`, so `kj/08` alone holds a temporary 1e-3 (measured 5.4e-4). From `kj/19b` on the bar is 1e-5 again (measured 1.79e-6 on `kj/19b` and on the tip; earlier port 1.9e-6). That 1e-5 applies to this case only, one added for the series. The earlier port's two `ketju_tide` cases keep their own bars: KetjuTidalStepRes 1e-4 (measured 1.64e-5) and KetjuSmoothTide 3e-4 (measured 1.01e-4). `kj/19`: "MSTAR never stepped" was wrong; the Kepler path books one step per region. `kj/22`: the first test mixed 1PN into the osculating orbit; now 2.5PN alone, same bar. `kj/20` and `kj/22` first read `Velocities`, a mixed state; now `HermiteSync*`.

**Every PR: before and after.** Each PR's test fails on the commit before the change and passes after it, except `kj/17`, which is test only (no source change is needed, so it passes before too). `kj/16` and `kj/19c`: part of their tests pass before, the new part fails. Per-PR results are in the PR texts below.

**Measured, K-stack** (older base; tip = the series sources built before the last Type-5 change, which does not touch this path; port = `stella_kokkos-ketju` 6a6c2f7aa; same batch node, 8 cores; to be re-measured on `<TIP>`). Energy error |dE/E|:

| case | FX0 tip | FX0 port | KOf tip | KOf port |
|---|---|---|---|---|
| k1_circ | pass 9.9e-13 | pass 9.9e-13 | pass 2.6e-12 | pass 2.6e-12 |
| k1_e90 | pass 1.4e-10 | pass 1.4e-10 | pass 2.3e-11 | pass 2.3e-11 |
| k1_e99 | FAIL 2.3e-07 | FAIL 2.3e-07 | FAIL 3.5e-08 | FAIL 3.5e-08 |
| k2a | pass 2.0e-02 | pass 4.1e-02 | pass 6.0e-02 | pass 5.2e-02 |
| k2b | pass 8.1e-03 | pass 7.2e-03 | pass 1.4e-03 | pass 1.3e-03 |
| k2c | pass 3.0e-03 | pass 1.6e-03 | pass 1.1e-03 | pass 8.9e-04 |
| k2d | pass 2.7e-03 | pass 1.9e-03 | pass 1.2e-03 | pass 1.0e-03 |
| k3 | pass 7.4e-09 | pass 7.4e-09 | pass 1.4e-09 | pass 1.4e-09 |
| k4 | pass 3.0e-08 | pass 3.0e-08 | pass 4.1e-08 | pass 4.1e-08 |
| k5 | FAIL 2.5e-01 | FAIL 3.3e-01 | FAIL 1.6e-04 | FAIL 6.2e-04 |
| k6_np1 | pass 1.3e-04 | pass 1.4e-04 | pass 1.1e-04 | pass 1.0e-04 |
| k6_np2 | pass 1.1e-04 | pass 1.2e-04 | pass 2.6e-04 | pass 2.6e-04 |
| k8 | pass 1.0e+00 | pass 1.0e+00 | pass 1.0e+00 | pass 1.0e+00 |
| k9i | pass 1.8e-03 | pass 1.0e-03 | pass 1.0e-03 | pass 9.6e-04 |
| k9u | pass 2.0e-13 | pass 2.0e-13 | pass 8.6e-13 | pass 8.6e-13 |

13/15 for tip and port in both variants, same two failures. `k1_e99`: |dE/E| 2.3e-7 (FX0), 3.5e-8 (KOf) against 1e-8. `k5`: fails on every path of the old matrix too; contraction 2.1 against the required 4. Energy in k2c and k2d is up to 2x the port's in FX0, the price of the cheaper rebuild rule; KOf about 1.2x. On the reference implementation (stella_dev) k2b had 2.4 in FX0. On this branch `k5` FX0 energy is high because the tidal step prefactor is missing (see point 5).

**Measured, fewbody** (older base; 48 problems, R = 0.005, FX0, max |dE/E| < 0.1): tip 48 finish, 43 pass, median 3.2e-4, worst 0.72. Port 41 finish, 37 pass, median 3.2e-4, worst 0.44. The 7 port runs that did not finish all finish on tip; 6 of them hit the empty-rank LET abort that is now fixed in the base. The 5 tip failures (p040 0.72, p045 0.44, p023 0.40, p032 0.34, p036 0.19) each contain one stellar collision; not traced further. p040 is among those that aborted on the port.

**All KETJU tests on the final tip** (older base; `kj/24`, all test directories of the series): 34 passed, 3 skipped (legacy C library googletests). After that run, `kj/19b` onward was restacked for the `ketju_tide` bar only (the test file is the only change at the tip) and `ketju_tide` rerun: 3 passed. An earlier full pass caught a regression, 767 collectives against 756 in the quiet test, from the snapshot swap and the rebuild rule treating "no chain yet" at time 0 as live; fixed in `kj/09`, `kj/14`, `kj/18`. Final: 756 = 756. These late fixes touch only `ketju_coupling.cc`. The K-stack and fewbody binaries predate them; they change no integration on a run with regions, but K-stack and fewbody were not rerun. The full result on `<TIP>` is `<SUITE RESULT>`.

**Flag off** (older base; gmc_cooling and FIRE3 configs, objects against the previous branch of the series): all K branches: no object differs in instructions; 134 to 143 of 145 byte-identical, the rest line numbers and strings only. Flag off: to be re-run on `<TIP>`.

**Not measured.** The full old matrix (clusters, plummer, plummer_binaries, cloud matrix with RUMI paths); fewbody R = 0.001 and other variants; a Stella production-model cloud with chemistry and feedback (the cloud runs are the `ketju_cloud_restart` setup, gas with cooling, 21 steps); anything on a GPU; the stale-import force tail on the series; tree bounds with a very massive BH; a BH merger in a run; the Type-5 comparison with Mike's cases.

**Sizes**, source lines added/removed per PR against the previous branch (tests, docs, Template_Config not counted): 01: about 35 (install-path form, to be re-counted after the re-cut), 02: 197/-11, 03: 255, 03b: 149/-1, 04: 275, 05: 249/-3, 06: 324, 07: 180, 08: 328/-2, 09: 242/-1, 10: 112/-3, 11: 65, 12: 127, 14: 96, 15: 3, 16: 335, 17: 0, 18: 229, 19: 150, 19b: 191/-1, 19c: 58, 20: 122/-2, 21: 244, 22: 168/-9, 23: 39/-1, 24: 17. These are on the older base; the re-cut changes them.

Over the ~200 target: 03, 04, 05, 09, 18, 21, and those below. Over the ~300 limit, plain counts with comments: `kj/06` (324), `kj/08` (330 with removals), `kj/16` (335). Proposed splits:
- `kj/06`: 06a integrate and scatter; 06b velocity trick with motion bounds and node momentum.
- `kj/08`: 08a shared timebin and limiter; 08b chain-external field.
- `kj/16`: 16a contact detection; 16b merger products from the tables.

The texts of these three are written for the unsplit PR; the "Change" section lists which part goes where.

# Points raised on #69

From #69 (2026-10-29, 2026-11-01). Each point: the answer, the evidence, what is still open. Numbers are measured on Orion (Kokkos OpenMP, no GPU) on an older base unless marked; to be re-measured on `<TIP>`.

## 1. The velocity swap changes velocities outside the kick; the tree's motion bounds must follow

Both KETJU writes call `gizmo_motion_bound_raise`: the chord velocity in `scatter_results` before the step's flush, the end-of-step velocity in `ketju_set_final_velocities` before anything drifts on it. Both also book the change in `P.dp`, so the nodes' bulk velocity follows (the earlier port's first version did not; node error 24).

The audit ships with the series: `TEST_KETJU_TREE_MOTION_AUDIT` and `test/ketju_tree_motion` (256 hard binaries, `kj/12`). The 1 and 2 rank rows are the test's own asserted lines; the 4 rank row is one run of the same problem.

| ranks | chain members above an ancestor's vmax | on any rank's top-level copy | outside a widened box | node bulk velocity | imported leaf offset |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 1.4e-15 (mergers 1.6e-13) | none imported |
| 2 | 0 | 0 | 0 | 5.5e-2 in a few nodes, open | mean 1.7e-6, max 5.8e-5 kpc |
| 4 | 0 | 0 | 0 | not exercised: a build every step in this problem, so no reused tree to check | at their stars |

Open, named here on purpose:
- Node velocity residual on 2 ranks: traced to one multi-particle leaf; the error appears once, stays constant, and is about one star's momentum. It does not grow, and plain N-body on 2 and 4 ranks is clean (9e-16, 7e-16), so it comes from KETJU with several ranks. Cause not found.
- Stale-import force tail: two stars in a close passage on different ranks, neither in a chain, on a reused tree. Measured on the earlier port before this series (force error 8.1 in `k2b`, 0.92 in a plain run with no regions). Not re-measured on the series, not fixed: it is a property of reused imports, not of KETJU. Question for the reviewer, with those runs.
- Kicked non-members exceed `vmax` until the next tree update (the audit's `vmax` column: 73 on 2 ranks, 270 on 4, all non-members; on the earlier port the same showed in a plain run with no regions). starforge_dev #103 also raises those bounds before drifting; whether that part is in the base is to be checked on `<TIP>`.

## 2. KETJU works with the existing tree structures

There is no parallel tree. Stars stay in the tree, and the chain-internal force is removed by negative half-kicks through `grav_force_pair`, the kernel the walk itself uses. What the tree gains: one float per particle read in the opening decision on device and host (`kj/11`), a chain test in the LET essential check and the chain velocity for an imported member (`kj/14`), one line in `needs_new_treeforce` (`kj/15`). All under `KETJU_REGULARIZATION`.

## 3. Performance rules, and the known costs

What holds:
- No candidate star anywhere: not one collective more than the same run without KETJU, also on the final tip, counted from outside by an MPI interposer (`test/ketju_quiet_steps`, `kj/09`: 756 collectives without KETJU, 756 with it, 17 steps, 2 ranks). The flag rides on the reduction `gravity_tree` already does.
- No all-particle scans on the default path: loops run over the active list.
- Communicators are cached by task list and not built per step. Region counts come from the gathered group. The ~9 collectives per step found in the reference implementation (stella_dev) are gone.

Known costs, not removed:
- Candidates but no region: two world collectives per step (the gather; 788 against 756 in the quiet-step run). The payload is active candidates, not all particles. A filter by distance to the domain edge would remove most of it; not done.
- Per-step collectives while chains are alive: one Allgather pair per tree build (`kj/14`), and point-to-point plus collectives on each region's own communicator (`kj/10`).
- Communicators: one per distinct task list, kept for the run. Their number is not bounded by anything but the number of distinct task lists seen; not measured on a production-size run.
- Motion caps (default on): one pass over local particles on a step with a region (see point 5).
- Rebuild rule (`kj/14`). The tree is rebuilt at a sync point while a chain spans ranks, or an active star of another rank is within ten chain sizes of a live chain. The first version rebuilt through a chain's whole step:

| problem | builds, old rule | builds, new rule | energy error old / new | survival old / new |
|---|---|---|---|---|
| k2b, 4 ranks, every star in a binary (worst case) | 718 | 146 | 2.7e-3 / 8.7e-3 | 0.945 / 0.941 |
| k2c, 4 ranks | 1206 | 1226 | 1.2e-3 / 1.3e-3 | 0.918 / 0.918 |
| k9i, 4 ranks | 1032 | 1032 | 1.6e-3 / 1.6e-3 | 0.930 / 0.930 |

Without any rule, the small gas cloud (`ketju_cloud_restart` setup, 2 ranks, 21 steps) gives star energy error 4.0e-2 against 1.8e-6 with it, and the same 20 builds. So the rule is needed, and costs nothing there; that cloud is small and short, a production-size cloud has not been measured. `KETJU_NEIGHBOUR_RANK_FACTOR` = 10 is still one value on one problem, not tuned. A KETJU-aware walk could replace the rule.

Per-step moment refresh of the single-star model (`stella_luminosity.cc`): on the earlier port only. The series branches do not have it: there `TreeMomentsStaleFlag` is raised at gas collection, not per step (checked by grep on the branches at the time). KETJU raises it only on a mass change. Whether the radiation series needs a per-step source update stays with that series.

## 4. Order independence, ranks and threads

Regions are found from a gather that is the same on every rank, and each region is integrated on its own, so the result does not depend on region order. Measured on `k6` (512 stars; older base):
- Same run twice, 1 rank: bit-identical.
- 1 vs 2 OpenMP threads on 2 ranks: same difference from the 1-rank run to all digits printed; no thread dependence seen.
- 1 vs 2 ranks: not identical (max |dx| 2.1e-5 kpc, median 4e-7). With regions off the same pair of runs differs by 2.0e-4, ten times more, so KETJU adds no rank dependence of its own. Binary survival agrees (0.996 on both).

## 5. What KETJU cannot meet yet, and why

- Two world collectives per step when candidates exist and no region does (see point 3).
- Motion caps, default on as in the reference implementation: one pass over local particles on a step with a region. With caps off, a tidally kicked pair drifts 2% in `ketju_tide` against 7e-4, so the default stays, and the scan is not yet removed.
- Fly-by step, opt-in: gathers all stars, O(n^2). Direct-sum energy diagnostic: opt-in, at output times.
- GPU: never run on one (no GPU on Orion). The coupling is host code by design, like cooling's host path; MSTAR runs through its own Kokkos build. Walk changes are in the device kernels but only exercised on Kokkos OpenMP. Host writes to `P` invalidate the arena once per region step; cost on a GPU not measured.
- `P.KetjuOpenRadius` < 0 marks a chain on one rank: without that a walk with the cheaper rule forced open foreign nodes never shipped, and the run stopped on 4 ranks ("imported tree did not carry the structure"). Fixed in `kj/11`; such a member can take a multipole from another rank's node, which holds none of its partners.
- A rank empty from the first build can still abort in the topology emit (rc 3, 2 stars on 4 ranks). The case of a rank that empties later is fixed in the base (B2).
- Merged stars keep the survivor's age; table radius and rejuvenation need table fields the reader does not have yet.
- `k5` energy error on this branch is 0.28, against 1.1e-3 in the reference implementation (stella_dev). Not KETJU: 5100 steps against 46500, because this branch has no tidal step prefactor. With MaxSizeTimestep/16 it is 1.7e-4.

# Intermediate-mass black holes

The reference implementation (stella_dev) has no IMBH model: the word does not occur in its source. What it has, and what the series carries:

| reference implementation | series | test |
|---|---|---|
| Stars and sinks in chains, BH centres above KetjuMinBHMass | `kj/20` | `ketju_sinks` |
| BH mergers with GR remnant mass, spin, recoil; geometric coalescence of sinks | `kj/22` `KETJU_MERGE_BH` | compile; PN inspiral tested, a BH merger itself is not |
| PN terms (KetjuPNTerms), `KETJU_PN_COMPACT_OBJECTS` | `kj/22` | `test_pn_binary` (Peters rate, 2.5PN) |
| Remnant type kept, PN on NS/BH that stay Type 4 (`KETJU_PN_REMNANT_TAG`) | `kj/23`, in the star record | `test_remnant_pn` (log only, with a test flag marking stars as remnants) |
| BH promotion, experimental (`GALSF_RESOLVEDISM_BH_PROMOTION`) | `kj/24` `GALSF_SINGLE_STARS_BH_PROMOTION` | `single_stars_bh_promotion` |

Growth to an IMBH can only come from these: sink accretion and repeated mergers. Not in the reference implementation either, so listed and not invented: a runaway-collision product model (mass loss in the collision, rejuvenation for very massive stars), a seed or formation channel for IMBHs as such, an IMBH-specific accretion or feedback model, massive-star table entries beyond the current mass range. Tree bounds with a massive BH: not audited (the audit runs on equal-mass binaries).

# PR texts

## kj/01-build-mstar: KETJU via an install path: build block, KETJU_PATH, flag

**What and why.** KETJU regularises close star encounters with MSTAR. MSTAR (a Kokkos library) and the ketju-integrator C headers are external packages. Following the rule on #112, they are a separate install that the build finds; nothing is fetched at compile time and nothing is a submodule. This PR adds the home `src/gravity/ketju/`, the Makefile block that finds the install through `KETJU_PATH`, the flag `KETJU_REGULARIZATION` (default off) and the option refusals.

**Evidence before.** The base has no KETJU: `KETJU_REGULARIZATION` occurs in no source file or Makefile, so a Config that sets it builds a binary without chains and says nothing (GIZMO does not reject unknown options). No test can fail before; the need is the build itself.

**Change.**
- `src/gravity/ketju/Makefile.ketju`, included from `src/Makefile` when `KETJU_REGULARIZATION` is set. `KETJU_PATH` is the install prefix, set in the machine block, the environment or on the make line. Make stops with a message naming the missing piece if `KETJU_PATH` is unset or if `include/mstar/mstar.hpp`, `include/ketju_integrator/ketju_integrator.h` or `lib/libmstar.a` (or `lib64/`) is missing under it. It adds `-I$(KETJU_PATH)/include` and links `-lmstar` with the Kokkos libraries GIZMO already uses; MSTAR must be built against that same Kokkos.
- `src/gravity/ketju/ketju_precompiler_logic.h`, included at the end of `precompiler_logic.h`: `#error` with `FORCE_EQUAL_TIMESTEPS`, with adaptive star softening (bit 16 of `ADAPTIVE_GRAVSOFT_FORALL`), and with the two options that merge stars by distance (`GALSF_MERGER_STARCLUSTER_PARTICLES`, `SINGLE_STAR_SINK_DYNAMICS`) under single stars.
- One `Template_Config.sh` line.
- About 35 source lines (Makefile block, logic header, two includes); to be re-counted after the re-cut.

**Cost.** None at run time; the flag alone adds no field and no work. Build: needs the install.

**Evidence after.** The integration line with this block (commit `d0284b30f`) builds and links the KETJU config against an install given in `KETJU_PATH` (MSTAR and C headers installed from ketju-integrator); the other eight Stella configs build too (older base; to be re-measured on `<TIP>`). `test/ketju_unit` (narrow) runs the library's own tests from source checkouts given in `KETJU_SOURCE_DIR` and `KETJU_C_SOURCE_DIR` and is skipped without them; run on `<TIP>`: to be measured.

**Flag off.** Only `Makefile` (one conditional include) and `precompiler_logic.h` (one include under the flag) change. Flag off: to be re-run on `<TIP>`.

**Limits.** No CMake lines here; they come with the CMake work (#96), checking the same three files. Tested only on Kokkos OpenMP; no GPU build of MSTAR tried.

**Depends on / followed by.** Depends on the base. Followed by `kj/02`.

## kj/02-parameters-member-state: KETJU: parameters and member state in the star record

**What and why.** Parameters, and the state of a chain member. Only three fields stay on every particle; everything a member needs for its step goes into the compact star record of the single-star series, which already follows particles through domain exchange and restart. The `KetjuFinalVelocity` snapshot block of the reference implementation (stella_dev) is dropped: at output `Velocities` already holds the MSTAR value.

**Evidence before.** Narrow test. `1 failed in 42.88s` (assert not <re.Match object; span=(6544, 6581), match="Tag 'KetjuRegionRadius' was specified">) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_params.cc`, `ketju_particle_fields.h`, `ketju_global_fields.h`, `ketju_member_state.h`; guards `SINGLE_STAR_RECORDS` (= IMF stars or KETJU) in domain, restart, merge_split, ledger; parameters KetjuRegionRadius, KetjuMinStarMass, KetjuIntegrationTolerance, KetjuUseStarStarSoftening, KetjuMaxStepCount. Restart layout is the v3 table with `RESTARTFILE_LAYOUT_VERSION` 4; the record bytes are to be re-counted on `<TIP>`.

**Cost.** 16 B per particle under the flag (short, 8-byte chain ID, float opening radius; reference implementation about 200 B), 4 B StarIdx, 160 B per star that is in a chain, only while it is in one. Byte counts predate the v3 layout table; to be re-checked on `<TIP>`.

**Evidence after.** `1 passed in 44.52s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 141 byte-identical; 4 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 141 byte-identical; 4 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Record bytes predate the v3 restart layout table; to be re-counted on `<TIP>`.

**Depends on / followed by.** After `kj/01`. Followed by `kj/03`.


## kj/03-chain-centres: KETJU: chain candidates gathered across ranks

**What and why.** Chain candidates: synchronised stars above KetjuMinStarMass, gathered from all ranks. Nothing is grouped or integrated yet.

**Evidence before.** Narrow test. `1 failed in 46.05s` (AssertionError: assert None) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_coupling.cc` (candidate tests, gather, `ketju_find_regions`), one hook in `run.cc` after the first half-kick.

**Cost.** One Allgather plus one Allgatherv per step. `kj/09` removes both when no candidate exists.

**Evidence after.** `1 passed in 45.10s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 140 byte-identical; 5 same instructions (line numbers or strings); differing: none; FIRE3: same counts (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Two collectives per step while any candidate exists (removed for steps without candidates in `kj/09`).

**Depends on / followed by.** After `kj/02`. Followed by `kj/03b`.


## kj/03b-regions: KETJU: regions from pair orbits

**What and why.** Groups candidates into regions by pair orbit: bound and tight, or a pericentre inside the region radius within a capture window, so a dense cluster does not merge into one region. The grouping is the same on every rank, so members and chain IDs need no further reduction.

**Evidence before.** Narrow test. `2 failed in 90.84s (0:01:30)` (AssertionError: assert None) (older base; to be re-measured on `<TIP>`)

**Change.** Orbit grouping in `ketju_coupling.cc`.

**Cost.** O(candidates) through a cell list, no collective.

**Evidence after.** `2 passed in 90.62s (0:01:30)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 140 byte-identical; 5 same instructions (line numbers or strings); differing: none; FIRE3: same counts (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** The capture window is one value, not tuned on production clouds.

**Depends on / followed by.** After `kj/03`. Followed by `kj/04`.


## kj/04-mstar-wrapper: KETJU: MSTAR calls and natural units

**What and why.** Wraps MSTAR behind three calls and converts to its natural units (pc, Msun, km/s), where its GBS convergence is good whatever the host units are.

**Evidence before.** Narrow test. `1 failed in 44.78s` (assert (None)) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_mstar_kokkos.cc/.h`, unit conversion, test flag `TEST_KETJU_MSTAR_PAIR`.

**Cost.** None on the run path.

**Evidence after.** `1 passed in 45.97s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 140 byte-identical; 5 same instructions (line numbers or strings); differing: none; FIRE3: same counts (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** None beyond the series list (section 5).

**Depends on / followed by.** After `kj/03b`. Followed by `kj/05`. 275 source lines.


## kj/05-region-setup: KETJU: region setup in its centre-of-mass frame

**What and why.** Sets up each region: members move to the centre-of-mass frame, in physical coordinates, and the integrator state is created. Task groups are here because every later step speaks through them; communicators are cached by task list and not built per step.

**Evidence before.** Narrow test. `1 failed in 37.51s` (assert (None)) (older base; to be re-measured on `<TIP>`)

**Change.** `setup_integrator`, `KetjuTaskGroup`, test flag `TEST_KETJU_REGION_LOG`.

**Cost.** One group lookup per region per step.

**Evidence after.** `1 passed in 36.86s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 140 byte-identical; 5 same instructions (line numbers or strings); differing: none; FIRE3: same counts (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Communicators are cached per task list; their number is not bounded (section 3).

**Depends on / followed by.** After `kj/04`. Followed by `kj/06`.


## kj/06-integrate-scatter: KETJU: integrate a region, velocity trick with motion bounds and node momentum

**What and why.** Integrates a region and hands the result back. The velocity trick: a member gets the chord velocity that lands it on MSTAR's end position in the host drift, and its real velocity after the drift. Both writes raise the motion bound (`gizmo_motion_bound_raise`) and book the momentum in `P.dp`, so nodes and bounds follow a velocity change made outside the kick. The pending-momentum clears this relies on come from the #103 port.

**Evidence before.** Narrow test. `1 failed in 40.63s` (assert (np.float64(10.879545633750784) < 0.001)) (older base; to be re-measured on `<TIP>`)

**Change.** `integrate_region`, `scatter_results`, `ketju_run_integration`, `ketju_set_final_velocities`, hooks in `run.cc` and `apply_long_range_kick`. 324 source lines, over the ~300 limit. Proposed split: 06a integrate and scatter; 06b velocity trick with motion bounds and node momentum.

**Cost.** None without regions.

**Evidence after.** `1 passed in 36.89s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 139 byte-identical; 6 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 140 byte-identical; 5 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Over ~300 lines; split proposed above.

**Depends on / followed by.** After `kj/05`. Followed by `kj/07`. Needs the #103 pending-momentum port in the base.


## kj/07-pair-subtraction: KETJU: the tree pair force taken out by negative half-kicks

**What and why.** The tree kick contains the softened pair force of chain partners; MSTAR replaces it with the exact one. Negative half-kicks remove what the tree added, through the tree's own pair kernel (`grav_force_pair`), not a copy.

**Evidence before.** Narrow test. `1 failed in 36.89s` (assert (np.float64(3.7167514614184087) < 1e-06)) (older base; to be re-measured on `<TIP>`)

**Change.** `do_negative_halfstep_kick`, `tree_pair_softening`, `ketju_tree.h`. The softening snapshot of the reference implementation (stella_dev) is gone: star softening is fixed under KETJU, so 32 B per particle and a loop per gravity call are saved.

**Cost.** O(n^2) in chain members only.

**Evidence after.** `1 passed in 38.64s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 139 byte-identical; 6 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 140 byte-identical; 5 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** None beyond the series list (section 5).

**Depends on / followed by.** After `kj/06`. Followed by `kj/08`.


## kj/08-shared-step: KETJU: shared timebin, limiter, chain-external field

**What and why.** A region's members start and end their step together. Members take a shared bin, set by their own steps and by stars inside the limiting radius; a member's own step follows the field external to its chain, not the orbit MSTAR integrates.

**Evidence before.** Narrow test. `1 failed in 59.95s` (AssertionError: assert 0 > 0) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_limit_timesteps`, `ketju_compute_com_quantities`, `ketju_finish_step`, hooks in the `find_timesteps` call site and `get_timestep`; parameter KetjuTimestepLimitingFactor. 328 source lines, over the ~300 limit. Proposed split: 08a shared timebin and limiter; 08b chain-external field. Test bar: `ketju_tide` with MaxSizeTimestep 1e-8 has the original bar 1e-5, from the earlier port's 1.9e-6 with the motion caps on. Caps arrive in `kj/19b`, so this PR alone holds a temporary 1e-3 (measured 5.4e-4); `kj/19b` restores 1e-5.

**Cost.** No scan of all particles: loops run over the active list.

**Evidence after.** `1 passed in 56.65s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 138 byte-identical; 7 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 139 byte-identical; 6 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Over ~300 lines; split proposed above. The `ketju_tide` bar is temporarily 1e-3 in this PR alone.

**Depends on / followed by.** After `kj/07`. Followed by `kj/09`.


## kj/09-task-groups-quiet-steps: KETJU: task scheduling, cached communicators, no collective on quiet steps

**What and why.** Cost-based task allocation for several regions, and the quiet-step rule. A step with no candidate anywhere adds no collective: the candidate flag rides on the reduction `gravity_tree` already does for `TreeMomentsStaleFlag`. Region counts come from the gathered group without further reductions, statistics reductions are deferred, and steps with no chain alive skip the end-of-step work.

**Evidence before.** Narrow test. `1 failed in 115.16s (0:01:55)` (assert 788 == 756) (older base; to be re-measured on `<TIP>`)

**Change.** Scheduler, `ketju_active_candidates`, one replaced `MPI_Allreduce` in `gravtree.cc` under the flag (two ints instead of one).

**Cost.** No new collective without candidates. With candidates but no region, two per step (the gather); payload in active candidates, not in all particles. Known cost, see point 3.

**Evidence after.** `1 passed in 97.85s (0:01:37)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 137 byte-identical; 8 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 138 byte-identical; 7 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** With candidates but no region, two world collectives per step remain (section 3).

**Depends on / followed by.** After `kj/08`. Followed by `kj/10`. 242 source lines.


## kj/10-regions-across-ranks: KETJU: regions with members on several ranks

**What and why.** Regions with members on several ranks: gather to the compute root, scatter back.

**Evidence before.** Narrow test. `1 failed in 154.58s (0:02:34)` (RuntimeError: GIZMO exited with status 1 for test 'ketju_tree_motion'.) (older base; to be re-measured on `<TIP>`)

**Change.** Gather and forward paths in `setup_integrator` and `scatter_results`.

**Cost.** Point-to-point and collectives on the region's own communicator only. Known cost, see point 3.

**Evidence after.** `1 passed in 40.36s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 137 byte-identical; 8 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 138 byte-identical; 7 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Collectives on the region's communicator on every step while a region spans ranks.

**Depends on / followed by.** After `kj/09`. Followed by `kj/11`.


## kj/11-walk-chain-opening: KETJU: the walk takes chain partners as leaves

**What and why.** A chain member must see its partners as leaves, or the exact subtraction removes something the tree never added. The walk opens nodes within a per-particle radius of a member, read on the device from `P.KetjuOpenRadius`. Same-chain pairs inside the region radius do not set the sink step.

**Evidence before.** Narrow test. `1 failed in 45.54s` (assert (None)) (older base; to be re-measured on `<TIP>`)

**Change.** `gpu_gravtree.cc`, `forcetree.cc` opening; `gravtree_force_kernel.h` proximity filter; all in existing tree structures, no second tree.

**Cost.** One float compare per accepted multipole for a member target; nothing for others. One float per particle under the flag.

**Evidence after.** `1 passed in 47.00s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 135 byte-identical; 10 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 136 byte-identical; 9 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Device path exercised on Kokkos OpenMP only, not on a GPU. A one-rank chain member can take a multipole from another rank's node (section 5).

**Depends on / followed by.** After `kj/10`. Followed by `kj/12`. Only exercised on Kokkos OpenMP, not on a GPU.


## kj/12-tree-audit: KETJU: tree audit, chains under the motion bounds

**What and why.** The audit asked for in review. Test-only flag: every chain member under every ancestor's `vmax` on every rank's top-level copy, inside the widened box, node bulk velocity equal to the members, imported leaves at their stars.

**Evidence before.** Narrow test. `1 failed in 45.58s` (AssertionError: no KETJU_TREE_AUDIT line) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_tree_audit.cc`, test flag `TEST_KETJU_TREE_MOTION_AUDIT`, `test/ketju_tree_motion`.

**Cost.** Test flag only.

**Evidence after.** `1 passed in 47.02s` (older base; to be re-measured on `<TIP>`). Table in point 1.

**Flag off.** gmc_cooling: 145 objects: 135 byte-identical; 10 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 136 byte-identical; 9 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** On 2 ranks a node bulk-velocity residual of about one star's momentum stays open (section 1).

**Depends on / followed by.** After `kj/11`. Followed by `kj/14`. 127 source lines.


## kj/14-let-imported-chains: KETJU: chains in the LET, imported members move with their chain

**What and why.** Chains across ranks. The LET sender treats nodes overlapping a receiver's chain as essential; an imported member moves with its chain's centre-of-mass velocity, not its orbital one; while a chain spans ranks, or has active stars of another rank near it, the tree is rebuilt each step. (Includes the LET chain spheres; there is no `kj/13`.)

**Evidence before.** Narrow test. `1 failed in 42.56s` (assert (408105.0 > 0 and 2.372e-05 < 3e-06)) (older base; to be re-measured on `<TIP>`)

**Change.** `let_pack.cc` (chain sphere test, imported velocity), `ketju_let_gather_spheres`, rebuild rule in `ketju_run_integration`.

**Cost.** One Allgather pair per tree build while chains are alive, none otherwise. The rebuild rule costs tree builds; see point 3 for the measured table. Known cost.

**Evidence after.** `1 passed in 42.08s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** The rebuild rule's factor 10 is one value on one problem; a KETJU-aware walk could replace it.

**Depends on / followed by.** After `kj/12`. Followed by `kj/15`.


## kj/15-adaptive-treeforce: KETJU: chain stars take a fresh tree force

**What and why.** `ADAPTIVE_TREEFORCE_UPDATE` must not reuse a stale tree force for chain stars: the subtraction is exact only against this step's force.

**Evidence before.** Narrow test. `1 failed in 37.19s` (assert (np.float64(5.927365797764465) < 1e-06)) (older base; to be re-measured on `<TIP>`)

**Change.** One line in `needs_new_treeforce`.

**Cost.** None.

**Evidence after.** `1 passed in 37.27s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** None beyond the series list (section 5).

**Depends on / followed by.** After `kj/14`. Followed by `kj/16`.


## kj/16-star-mergers: KETJU: stellar collisions

**What and why.** Stars merge on contact inside MSTAR. Mass, momentum and metals go to the survivor, drawn masses add, the survivor keeps its age. Table radius and rejuvenation wait for table fields the reader does not have yet; the radius is the main-sequence relation. The refresh loop of the first version is gone: pending momentum is cleared by the #103 port, and this PR carries no refresh of its own.

**Evidence before.** Narrow test. `2 failed, 2 passed in 160.46s (0:02:40)` (AssertionError: assert ('KETJU STAR COLLISION' in '# launch: mpirun -np 1 --use-hwthread-cpus --bind-to no) (older base; to be re-measured on `<TIP>`). The first version's refresh loop is dropped, so the count of source lines changes.

**Change.** `handle_mergers`, `ketju_star_mergers.cc`, flag `KETJU_MERGE_STARS`. 335 source lines on the older base, over the ~300 limit. Proposed split: 16a contact detection; 16b merger products from the tables.

**Cost.** Work only on a merger.

**Evidence after.** `4 passed in 163.15s (0:02:43)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Over ~300 lines; split proposed above. Merged stars keep the survivor's age; table radius and rejuvenation need table fields the reader lacks.

**Depends on / followed by.** After `kj/15`. Followed by `kj/17`. Needs the #103 pending-momentum port in the base.


## kj/17-restart-test: KETJU: restart inside a chain step

**What and why.** Test only: a run restarted inside a chain step ends with the straight run. No source change is needed: the member state sits in the star record, which restart files already carry.

**Evidence before.** Narrow test. `1 passed in 36.97s` (test only, so it passes before too; older base; to be re-measured on `<TIP>`)

**Change.** `test/ketju_cloud_restart`.

**Cost.** None.

**Evidence after.** `1 passed in 35.77s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** A restart differs from a straight run in close-pair problems (max |dv| 1.5e-4 km/s, cause not traced).

**Depends on / followed by.** After `kj/16`. Followed by `kj/18`. Restart layout is the v3 table, `RESTARTFILE_LAYOUT_VERSION` 4.


## kj/18-snapshots: KETJU: snapshots inside a chain step, HDF5 chain output

**What and why.** Snapshots inside a chain step: a member part-way through its step sits on the chord, so a copy of the region is integrated to the output time and the state swapped in for the write. HDF5 chain output.

**Evidence before.** Broad test. `2 failed in 83.91s (0:01:23)` (assert np.float64(1.4821680290678898) < 0.002) (older base; to be re-measured on `<TIP>`)

**Change.** `record_midstep_output_state`, `ketju_swap_output_velocities`, HDF5 writers, hooks around `savepositions`.

**Cost.** Only at output times.

**Evidence after.** `2 passed in 82.90s (0:01:22)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** None beyond the series list (section 5).

**Depends on / followed by.** After `kj/17`. Followed by `kj/19`. 229 source lines.


## kj/19-kepler-expansion: KETJU: exact Kepler pairs, tight binary expansion

**What and why.** Two-body regions advance by an exact Kepler drift; a tight outlier binary can be expanded instead of stalling the integrator.

**Evidence before.** Broad test. `1 failed in 38.66s` (AssertionError: assert '204' == '827') (older base; to be re-measured on `<TIP>`)

**Change.** `kepler_pair_step`, `expand_tight_binaries`, parameters KetjuKeplerPairs, KetjuExpandBinariesFactor.

**Cost.** None by default for the expansion.

**Evidence after.** `1 passed in 38.43s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** None beyond the series list (section 5).

**Depends on / followed by.** After `kj/18`. Followed by `kj/19b`.


## kj/19b-flyby-caps: KETJU: fly-by step, motion and tide caps

**What and why.** Fly-by step for stars outside regions, and the motion and tide caps of Mannerkoski+ 2023. The fly-by step is opt-in; the caps are on by default as in the reference implementation (stella_dev) (with them off a tidally kicked pair drifts 2% in `ketju_tide` against 7e-4), and they scan the local stars on steps with a region.

**Evidence before.** Broad test. `4 failed, 1 passed in 230.10s (0:03:50)` (assert (np.float64(0.003959994001422107) < 0.0003)) (older base; to be re-measured on `<TIP>`)

**Change.** `flyby_steps`, `region_step_caps`, parameters KetjuRegionMotionCapFactor (default 1), KetjuFlybyStepEta. Also adds `ketju_tide` to this PR's test: the MaxSizeTimestep 1e-8 case returns to its original bar 1e-5 (measured 1.79e-6), and the KetjuTidalStepRes case runs here at the earlier port's bar 1e-4 (measured 1.64e-5).

**Cost.** Caps: one pass over local particles on a step with a region (known cost, see point 3). Fly-by: all stars, O(n^2), only when switched on.

**Evidence after.** `5 passed in 217.56s (0:03:37)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** The caps scan local particles on each step with a region; the fly-by step is O(n^2) when on.

**Depends on / followed by.** After `kj/19`. Followed by `kj/19c`. 191 source lines.


## kj/19c-smooth-tide: KETJU: external tide inside MSTAR

**What and why.** External tide inside MSTAR instead of host kicks at orbital phases.

**Evidence before.** Broad test. `1 failed, 2 passed in 147.40s (0:02:27)` (assert np.float64(0.000736137574971929) < 0.0003) (older base; to be re-measured on `<TIP>`). The two passing tests pass before; the new one fails.

**Change.** `ketju_region_tide_kick`, parameter KetjuSmoothTide.

**Cost.** Only when switched on.

**Evidence after.** `3 passed in 168.16s (0:02:48)` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Only when switched on; not compared with host-kick tides on a production run.

**Depends on / followed by.** After `kj/19b`. Followed by `kj/20`.


## kj/20-sinks: KETJU: Type-5 sinks as chain members

**What and why.** Type-5 sinks as chain members. They go through the same exact subtraction as stars. This differs from starforge_dev, which subtracts with a fixed softened kernel: on this tree that path gave |da/a| 1.3e-3 for the test binary, the exact kernel 1.4e-12, so there is one path for all types. Mike should review this.

**Evidence before.** Broad test (`ketju_sinks`). `1 failed in 37.30s` (AssertionError: assert 'Found 1 region(s) from 2 center(s)' in '# launch: mpirun -np 1 --use-hwthread-cpus) (older base; to be re-measured on `<TIP>`)

**Change.** Type-5 centres, `HermiteHistoryStale`, spin and PN class fields, parameter KetjuMinBHMass.

**Cost.** 10 B per particle under the sink flags.

**Evidence after.** `1 passed in 36.11s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Differs from starforge_dev's softened subtraction for sinks; not compared with Mike's fewbody and plummer_binaries Type-5 cases.

**Depends on / followed by.** After `kj/19c`. Followed by `kj/21`. The two `SINGLE_STAR_TIMESTEPPING > 0` guards go when #100 reaches this branch.


## kj/21-energy-diagnostic: KETJU: direct-sum energy diagnostic and coupling ledger

**What and why.** Direct-sum energy of the stars and a ledger of the coupling terms, to tell integrator error from kick work.

**Evidence before.** Narrow test. `1 failed in 39.17s` (assert ([])) (older base; to be re-measured on `<TIP>`)

**Change.** `ketju_energy_diag.cc`, ledger lines. The ledger has no deferred-heat row (heat is always injected in the base).

**Cost.** Direct sum only at output times.

**Evidence after.** `1 passed in 37.83s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Diagnostic only; the direct sum is O(n^2) at output times.

**Depends on / followed by.** After `kj/20`. Followed by `kj/22`. 244 source lines.


## kj/22-bh-mergers-pn: KETJU: BH mergers and PN terms

**What and why.** BH mergers and post-Newtonian terms inside MSTAR, geometric coalescence of sinks. PN forces on sinks tagged as relics, or on all sinks with `KETJU_PN_COMPACT_OBJECTS`.

**Evidence before.** Broad test (PN inspiral; a BH merger itself is not tested). `1 failed in 37.82s` (assert np.float64(1.0000000000556581) < 0.2) (older base; to be re-measured on `<TIP>`)

**Change.** Flags `KETJU_MERGE_BH`, `KETJU_PN_COMPACT_OBJECTS`, parameters KetjuPNTerms, KetjuEnableBHMergerKicks.

**Cost.** None without the flags.

**Evidence after.** `1 passed in 36.04s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** A BH merger itself is not tested in a run; only the PN inspiral.

**Depends on / followed by.** After `kj/21`. Followed by `kj/23`. 168 source lines added, 9 removed.


## kj/23-remnant-pn-tag: KETJU: NS and BH remnants of Type-4 stars get PN terms

**What and why.** Neutron stars and black holes that stay Type-4 stars take PN terms. The remnant type is kept in the star record.

**Evidence before.** Narrow test. `1 failed in 38.11s` (AssertionError: assert None) (older base; to be re-measured on `<TIP>`)

**Change.** Flag `KETJU_PN_REMNANT_TAG`, one field in the record, one line in `single_star_feedback.cc`.

**Cost.** 1 B per star record.

**Evidence after.** `1 passed in 39.94s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** The test checks the log only, with a test flag marking stars as remnants.

**Depends on / followed by.** After `kj/22`. Followed by `kj/24`. Restart layout is the v3 table, `RESTARTFILE_LAYOUT_VERSION` 4; record bytes to be re-checked on `<TIP>`.


## kj/24-bh-promotion: Single stars: massive remnants become sinks (BH promotion)

**What and why.** Experimental, as in the reference implementation (stella_dev): remnants from pair-pulsation, failed supernovae and direct collapse become Type-5 sinks. This is what makes an intermediate-mass black hole possible through mergers; nothing beyond the reference implementation is added.

**Evidence before.** Narrow test (`single_stars_bh_promotion`). `1 failed in 45.60s` (assert (2 > 0 and np.uint32(0) == 2)) (older base; to be re-measured on `<TIP>`)

**Change.** Flag `GALSF_SINGLE_STARS_BH_PROMOTION`, needs SINK_PARTICLES.

**Cost.** None without the flag.

**Evidence after.** `1 passed in 43.81s` (older base; to be re-measured on `<TIP>`)

**Flag off.** gmc_cooling: 145 objects: 134 byte-identical; 11 same instructions (line numbers or strings); differing: none; FIRE3: 145 objects: 135 byte-identical; 10 same instructions; differing: none (older base; to be re-measured on `<TIP>`). Flag off: to be re-run on `<TIP>`.

**Limits.** Experimental, as in the reference implementation.

**Depends on / followed by.** After `kj/23`. Last in the series.
