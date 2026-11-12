# Small fixes to gpu_kokkos_unified: B4, B5, C1, C2, and the extra passive metal slots

Measured on gpu_kokkos_unified 070c5b78f. Tests are offered as temporary checks unless they validate more than one bug. Full suite: <SUITE RESULT>.

## B4: the HII-region flag does not link without an RT solver (`up/B4-hii-without-rt`, tip 8bdcd06fb)
**What and why.** `GALSF_FB_FIRE_RT_HIIHEATING` calls `particle_ionizing_luminosity_in_cgs`, which is defined only under `#if RADTRANSFER || RT_USE_GRAVTREE`. Scope: `FIRE_PHYSICS_DEFAULTS` without an RT solver defines HIIHEATING together with LONGRANGE, which pulls in `RT_USE_GRAVTREE`, so the FIRE defaults link. Only a hand-set HIIHEATING without LONGRANGE or an RT module is hit.
**Test (temporary check).** One compile-suite row, `GALSF COOLING METALS GALSF_FB_FIRE_RT_HIIHEATING`. On 070c5b78f it fails to link: `radfb_local.cc: undefined reference to particle_ionizing_luminosity_in_cgs(long)`; with the change the same Config builds and links.
**Change.** `src/radiation/rt_functions.h`, `src/radiation/rt_utilities.cc`: the luminosity function and its host wrapper move out of the RT guard into one that also admits the HII flag (moved block, +23 -19, no new logic).
**Flag off.** 137 of 140 objects byte-identical in both Configs; the other three have the same instructions.
**Limits.** None known.

## B5: step 0 stops when MaxSizeTimestep is more than 8 times the run length (`up/B5-timestep-clamp-before-cast`, tip 099ec9c86)
**What and why.** `get_timestep` tests `(integertime)(dt / Timebase_interval) <= 1` before the existing clamp of dt to half the timeline. With dt more than 8 (TimeMax - TimeBegin) the quotient exceeds 2^63, the conversion is undefined (LLONG_MIN on x86) and the test fires as if the step were below `MinSizeTimestep`.
**Test (temporary check).** `test_gmc_cooling_max_timestep_above_run_length`: `test/gmc_cooling/gmc_cooling_quick.params` (TimeMax 0.001, MaxSizeTimestep 0.0252, ratio 25) with the `test/gmc_cooling` Config, 2 ranks. On 070c5b78f: `Controlled stop (code=888): STOP_WHEN_BELOW_MINTIMESTEP ... (get_timestep()/core/timestep.cc/line 1319)` at Time=0. With the change the run reaches TimeMax.
**Change.** `src/core/timestep.cc`: the clamp moves above the conversion test (+1 -1).
**Flag off.** Only steps longer than half the timeline change. 136 of 140 objects byte-identical in both Configs; `core/timestep.o` differs (the change).
**Limits.** None known.

## C1: readjust_timebase does not halve dt_step (`up/C1-readjust-timebase-dt-step`, tip f858d1612)
**What and why.** A restart that raises TimeMax past the end of the timeline halves the integer timeline in `readjust_timebase`. It halves `Ti_begstep`, `Ti_current` and the timebin but not `P[i].dt_step`, which `integertime_step()` returns. A particle in the middle of a step gets its second half-kick with the old integer length on a timeline whose interval has doubled.
**Test (temporary check).** `test/timeline_restart`: 64 stars with close pairs, stopped in the middle of long steps with the test flag `TEST_RESTART_AT_STEP` (environment `GIZMO_TEST_RESTART_AT_STEP`; a 3-line hook in `src/core/run.cc`, in the test commit), restarted with TimeMax doubled. The reference is the same stop with a plain restart on the same timeline: a restart differs from a straight run by 1.5e-4 km/s in this problem (close pairs, forces rebuilt), with or without a timeline change, so the reference is restart against restart. Bar 1e-12 km/s. On 070c5b78f: max |dx| 8.06e-8, max |dv| 4.09e-4 km/s. With the change both are 0.
**Change.** `src/core/begrun.cc`: `P[i].dt_step /= 2` in the halving loop (+1).
**Flag off.** Only restarts that halve the timeline change. 135 of 140 objects byte-identical in both Configs; `core/begrun.o` differs (the change), `core/run.o` has the same instructions.
**Limits.** The 1.5e-4 km/s restart-versus-straight difference is not traced.

## C2: rt_get_sigma divides 0 by 0 when an ionising band gets no photons (`up/C2-rt-sigma-bands-without-photons`, tip 8319f16e9)
**What and why.** With a cool source the blackbody integral of the highest ionising band underflows to exactly zero (for 1000 K only the 70-500 eV band; the lower bands stay tiny but non-zero). `rt_get_sigma` then divides 0 by 0 and the photoionisation rates become NaN. The same divisions are in starforge_dev's `radiation/rt_chem.cc`.
**Test (temporary check).** `test/rt_photoion_cool_source`: M1, `RT_CHEM_PHOTOION=2`, one star with star_Teff = 1000 K, 2 ranks; asserts that every gas field is finite. On 070c5b78f the run stops at step 0 with `dt_desired=-nan`, `accel_hydro=-nan`, `dtInternalEnergy=-nan`. With the change all fields are finite.
**Change.** `src/radiation/rt_chem.cc`: each division is guarded; a band without photons keeps its edge energy and cross-section 0 (+8 -8).
**Flag off.** Results change only where a denominator was 0. 137 of 140 objects byte-identical in both Configs; the other three have the same instructions.
**Limits.** Only bands with exactly zero photons are touched; sources hot enough to put photons in every band are unchanged.

## Extra passive metal slots (feature, opt-in; `up/C4-metals-extra-passive-species`, tip 0aee497a9)
**What and why.** Not a fix. `METALS_EXTRA_PASSIVE_SPECIES=n` appends n zero slots to `Metallicity`, carried, diffused, merged and written with the metals but not used by cooling. It is for Stella: the 27-element single-star yields need 16 slots outside the FIRE layout (`GALSF_SINGLE_STARS_27_ELEMENT_YIELDS` in our integration branch sets `METALS_EXTRA_PASSIVE_SPECIES` to at least 16), and the scalars that starforge_dev carries through `NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION` (a placeholder 0 on this branch) need somewhere to live. Without Stella it changes nothing; it belongs with the Stella series, not with the fixes.
**Test.** `test_SN_metals_extra_passive_species`: SN_singlestar with the flag off against `METALS_EXTRA_PASSIVE_SPECIES=3`; the snapshots must be bitwise equal apart from three extra zero `Metallicity` columns. This one checks the whole single-star supernova output, so it is worth keeping. On 070c5b78f the flag is unknown and silently ignored: `assert (262144, 4) == (262144, 7)`. With the change it passes.
**Change.** `precompiler_logic.h` (default 0, `#error` without METALS, offset macro, `NUM_METAL_SPECIES_BASE` for the layout checks) and the places that checked `NUM_METAL_SPECIES` against the cooling layout (`cooling_functions.h`, `init.cc`, `cosmic_ray_functions.h`, `eos_functions.h`, `sfr_eff.cc`, `stellar_evolution.cc`), +24 -15; plus a `Template_Config.sh` line.
**Cost.** n x 8 bytes per gas cell and star with the flag; none by default.
**Flag off.** Default 0 leaves `NUM_METAL_SPECIES` unchanged. 136 of 140 objects byte-identical in both Configs.
**Limits.** The extra slots stay zero until a module fills them.
