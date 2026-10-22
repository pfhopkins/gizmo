# Stella × FIRE-3 × STARFORGE: design of the coupled mode

Branch `stella-fire-starforge-unification` (off `stella_dev` dab9f937). Source: `/orion/ptmp/uli/orion/stella_hybrid_audit/AUDIT.md` plus Uli's decisions of 2026-10-22. Code references were re-checked by grep in this worktree. Claims that are not yet checked are marked *(unverified)*. The compile probe (sbatch 222629) is still pending, so nothing here comes from a build yet.

## 1. Goal

Stella becomes a module that can run **alongside** FIRE-3 and STARFORGE in one Forged-in-FIRE-style run, instead of being a separate world. The work has four parts:
- one behaviour-named mode flag plus a defaults preset;
- every star carries a fixed owner;
- each model's physics acts only on what it owns;
- clean hand-off where gas or radiation crosses between the models.

With the mode off, FIRE-only, STARFORGE-only and Stella-only builds stay bit-identical.

## 2. Decisions

| # | Decision (Uli) | Reason |
|---|---|---|
| A | **Three star-formation tiers by cell mass**: m > `M_SSP` gives a FIRE-3 SSP; `M_SINK` < m ≤ `M_SSP` gives Stella IMF sampling; m ≤ `M_SINK` gives a STARFORGE sink (`M_SINK=0` by default, so no sinks) | Each model is used where it is valid. LYRA sampling is meant for ~0.1–10 Msun cells. Sinks are only valid where protostellar collapse is resolved. |
| B | **The coupled mode uses FIRE's cooling and heating**, not CHEMCOOL | One cooling function on both sides, as upstream's hybrid requires (`COOLING /* only physical if include cooling for both sides, using same cooling functions */`, precompiler_logic.h:342). This also avoids running the non-equilibrium network in 10³ Msun cells. |
| C | **The coupled mode carries only FIRE's metal fields** (Z, He, C, N, O, Ne, Mg, Si, S, Ca, Fe + FIRE tracers). Stella yields are collapsed at deposit. The full 27-element layout is an optional extra. | Memory: 28 slots per particle vs FIRE's 11, plus Stella's per-element export buffers. |
| D | **Source physics follows the star's owner.** Stella stars use the Stella PI/G0/radp routines, FIRE SSPs the FIRE path, STARFORGE sinks the STARFORGE path. Our feedback loops are aligned with FIRE's and STARFORGE's (same call points and field meanings) but stay in separate files for now. | Keeps each model's calibration intact and keeps the diff local. |
| E | Upstream files may be edited | Needed: the tag gates and the star-formation switch have to sit inside FIRE/STARFORGE code. |
| BH | **Option 1** (section 4.8): no promotion in the mode. Remnants stay Type 4 with the remnant tag (`STELLA_REMNANT`); `GALSF_RESOLVEDISM_BH_PROMOTION` is an `#error` in the mode. | No Type-5 ambiguity with STARFORGE; BH physics can follow later. |
| F | **Shielding:** accept that Stella G0 (TREE_RAD columns) and FIRE UV (local Sobolev) shield differently for the prototype; measure the difference. | Prototype first; decide on one scheme from data. |
| G | **Radiation pressure:** measure first (Stella direct + IR vs a long-range term) before adding a long-range term for Stella stars. | Avoid adding physics before it is shown to matter. |
| H | **HII order:** the FIRE HII loop runs first, Stella PI on top (skips cells FIRE already ionised). | FIRE's HII state is the one the cooling reads. |
| I | **SN event field:** Stella gets its own event field (`StellaFBEvent_ThisTimeStep`) instead of sharing `SNe_ThisTimeStep`. | One meaning per field; no reset clobbering. |
| J | **Dust:** FIRE dust model or none first. | Stella dust needs the full element layout. |

Still open: the refinement criterion tied to `M_SSP` (open question 6) and main-sequence tables for STARFORGE sinks (open question 8, later).

**Verdict (Uli, 2026-10-22): "good for a prototype".**

## 3. The mode and its flags

| Flag | Meaning |
|---|---|
| `GALSF_RESOLVEDISM_PER_PARTICLE_STAR_MODEL` | Stella physics acts only on Stella-owned stars and Stella-channel cells, and may be compiled together with FIRE-3/STARFORGE. |
| `GALSF_RESOLVEDISM_HYBRID_DEFAULTS=(M_SSP)` | Preset. Sets `SINGLE_STAR_AND_SSP_HYBRID_MODEL=(M_SSP)`, the mode flag, `FIRE_PHYSICS_DEFAULTS=3` (FIRE cooling, mechanical FB, LOCALRP, LONGRANGE, HIIHEATING; precompiler_logic.h:138-162), and Stella SAMPLE_IMF, FB, STELLAR_TABLES, PHOTOION, G0_VARIABLE (TREE_RAD), RADPRESSURE. No CHEMCOOL, no METALS_INDIVIDUAL. |
| `GALSF_RESOLVEDISM_HYBRID_SINKS_BELOW=(M_SINK)` | Optional STARFORGE tier below `M_SINK` |
| `GALSF_RESOLVEDISM_HYBRID_CARRY_ALL_ELEMENTS` | Optional (phase P7): gas carries the full Stella element set instead of the FIRE subset |

As implemented in P1: the preset does **not** yet set `SINGLE_STAR_AND_SSP_HYBRID_MODEL` (without sink dynamics it leaves `cell_can_be_singlestar` undeclared at sfr_eff.cc:344 and switches on the hybrid refinement in merge_split.cc); P3 wires it with the SF switch, so in P1 all star formation in the mode is Stella. `GALSF_SFR_CRITERION` in Config.sh is an `#error` in the mode (FIRE-3 would silently override it). Outside the mode, `GALSF_RESOLVEDISM_METALS_INDIVIDUAL` with `GALSF_FB_FIRE_STELLAREVOLUTION` or `GALSF_ISMDUSTCHEM_MODEL` is now an `#error` (slot clash).

**`#error` changes**

| Today | In the mode |
|---|---|
| precompiler_logic.h:1317-1319 Stella FB vs `GALSF_FB_MECHANICAL`/`THERMAL` | MECHANICAL allowed; THERMAL stays an error |
| precompiler_logic.h:1320-1322 Stella FB vs `GALSF_FB_FIRE_RT_LOCALRP/LONGRANGE/HIIHEATING/STELLAREVOLUTION` | all allowed: each acts on SSPs only (decision D). Unlike the audit, **HIIHEATING stays on**, because under B FIRE's HII state is the one the cooling reads. |
| 1325-1340 G0/PI/RADP vs `RADTRANSFER` | unchanged (M1 preset is a later phase) |
| **new** in the mode | `CHEMCOOL` (B); `GALSF_RESOLVEDISM_METALS_INDIVIDUAL` unless `..._CARRY_ALL_ELEMENTS`; `GALSF_RESOLVEDISM_DUST`, `GALSF_ISMDUSTCHEM_MODEL` (dust deferred); `GALSF_RESOLVEDISM_GHOSTD_G0_CR` (G0 comes from sources); `CHIMES` (its HII branch in `do_the_local_ionization` is not mapped); `FIRE_PHYSICS_DEFAULTS<3` |

## 4. Design by area

### 4.1 Star ownership tag

Stella stars and FIRE SSPs are both Type 4:
- Stella: e.g. resolvedism_fb.cc:105;
- FIRE: mechanical_fb.cc:71-77 takes Types 2–4 non-cosmologically, 4 cosmologically.

So the particle type cannot carry the owner. Add an immutable `signed char StarModel` to `P`. It is set at birth, written to snapshots, and inferred on read when missing (`MstarSampleIMF` present gives Stella, otherwise SSP).

| `StarModel` | Type | Evolved by |
|---|---|---|
| 0 UNKNOWN | any | not a star, or not yet inferred (absent snapshot datasets are zero-filled on read, so 0 must mean "unknown") |
| 1 FIRE_SSP | 4 | FIRE-3 stellar evolution + mechanical FB |
| 2 STELLA | 4 | Stella tables + resolvedism_fb_* |
| 3 STARFORGE_SINK | 5 | STARFORGE |
| 4 STELLA_REMNANT | 4 | Stella (Type Ia of WD remnants) + gravity (+KETJU) |

As implemented in P1: codes are `STAR_MODEL_*` (particle_data.h); snapshot dataset `StarModel` (int, Types 4 and 5). Inference in `init()` when the dataset is absent: Type 5 → SINK; Type 4 → STELLA if `IMFSampledFlag==1` or `MstarSampleIMF[0]>0` (REMNANT if `MstarSampleIMF[0]==0`), else FIRE_SSP; fresh ICs (RestartFlag 0) carry no Stella star state, so their Type 4 are FIRE_SSP. Raw restart files (RestartFlag 1) carry the field.

Gates are the macros `STELLA_OWNS_STAR(p)` (Type 4 with tag STELLA or STELLA_REMNANT; plain `Type==4` with the mode off) and `STELLA_FB_EVENT(p)` (`StellaFBEvent_ThisTimeStep` in the mode, `SNe_ThisTimeStep` otherwise), so mode-off builds compile to the old conditions. Fresh-IC stars (RestartFlag 0) get `sampled`/`MstarSampleIMF` zeroed in the mode. Every top-level selection gets a one-line gate:
- FIRE: `determine_where_SNe_occur`/mechanical loop (mechanical_fb.cc:71-77), HII loop (radfb_local.cc:233-235), LOCALRP loop (radfb_local.cc:31), LEBRON source dispatcher (`rt_get_source_luminosity`, rt_utilities.cc:56-80);
- Stella: `Type == 4` selections in resolvedism_*.cc, including the `SNe_ThisTimeStep` reset at resolvedism_fb.cc:104-106 (today it clears every active Type 4, right after FIRE set it at run.cc:657-661);
- KETJU: `KETJU_TYPE4_STARS` is forced for Stella (precompiler_logic.h:1660); only tag 1/3 become chain members.

### 4.2 Star formation: three-way switch

- Upstream's `is_particle_single_star_eligible` (stellar_evolution.cc:22-36) returns 0 unless `SINGLE_STAR_SINK_DYNAMICS`. Without sinks, the hybrid has no single-star side. Add `sf_channel_for_cell(i)` → {SSP, STELLA, SINK}, by cell mass vs `M_SSP` / `M_SINK`.
- Fork point: FIRE criteria run first. `if(cell_can_be_singlestar==0) {return rateOfSF;}` (sfr_eff.cc:344) becomes `if(channel==SSP) return`. The Stella criteria (today chosen at compile time by `#if defined(GALSF_RESOLVEDISM)`, sfr_eff.cc:301-327) and the sink criteria below that line become runtime branches by channel.
- Spawn: SSP gives Type 4 tag 1 (generic spawn); STELLA gives the Stella IMF path (tag 2); SINK gives Type 5.
- Crossing the threshold is clean: Stella sampling completes within one step, no reservoir stays on the gas, and a star keeps its tag for life. Sinks keep their own reservoir.
- Guards:
  - a cell with m > `M_SSP` is never a Stella accretion donor;
  - refinement must keep star-forming gas in the Stella band. Upstream leaves this criterion open; the audit cites merge_split.cc:158 *(unverified)*.

As implemented in P3:
- **Threshold.** The preset defines `GALSF_RESOLVEDISM_HYBRID_M_SSP` (= the preset value, Msun). It does **not** define `SINGLE_STAR_AND_SSP_HYBRID_MODEL`. Upstream's "single-star eligible" (`is_particle_single_star_eligible`, mass ≤ threshold, sinks only) is our SINK tier, so in P5 that macro belongs to `M_SINK`, not `M_SSP`. In the mode, `SINGLE_STAR_SINK_DYNAMICS` and `SINGLE_STAR_AND_SSP_HYBRID_MODEL` are an `#error` until P5. The mode without the preset is an `#error` (no `M_SSP`).
- **Channel.** `sf_channel_for_cell(i)` (sfr_eff.cc) returns SSP if `P.Mass` > `M_SSP`, otherwise STELLA (SINK below `M_SINK` exists only behind the P5 `#error`). It is evaluated per cell when SF criteria are checked and again at spawn, with the same mass.
- **Criteria per channel** (`get_starformation_rate`):
  - Common to both: density threshold (`CritPhysDensity`, one parameter for both channels), Stella `Ionized` block, `IMFSpawnBlock`.
  - SSP cells: FIRE-3's bits 0+1+2+64 in FIRE form: hard virial threshold, FIRE convergent flow, FIRE Jeans mass (`MJ_crit` ≥ 100 Msun).
  - STELLA cells: no virial criterion; bits from `GALSF_RESOLVEDISM_HYBRID_STELLA_SFR_CRITERION` (default `(64)` = Stella production `GALSF_SFR_CRITERION=(64)`; only bits 2 and 64 have Stella versions, anything else is an `#error`) in Stella form (`Particle_DivVel`, `FacSfThreshMJ`).
  - `GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF` acts on STELLA cells only.
  - Both channels return before the sink-only criteria.
- **Spawn.** STELLA: LYRA path (IMF draw, `P=(M_cell/M_*)·pfac`, accretion walk), tag 2. SSP: generic GIZMO spawn compiled alongside it (whole cell with `GALSF_GENERATIONS=1`, FIRE probability, no IMF draw, `assign_imf_properties_from_starforming_gas`), Type 4 tag 1. The SSP's `sampled`, `MstarSampleIMF` and `StellaFBEvent_ThisTimeStep` are zeroed.
- **Donor guard.** The Stella accretion walk skips donors with m > `M_SSP` (resolvedism_imf_sampling.cc, both counting and removal passes).
- **Refinement.** What upstream does under `SINGLE_STAR_AND_SSP_HYBRID_MODEL` (merge_split.cc): only inside `FIRE_SUPERLAGRANGIAN_JEANS_REFINEMENT`. There it sets a 10 pc instead of 700 pc de-refinement floor, Jeans refinement without the 1e4 K floor and at 0.01 M_J instead of 0.001 M_J, and sink-distance radii 0.1/1/10 kpc. `target_mass_renormalization_factor_for_mergesplit` sets `ref_factor = 1` with the comment "need to determine appropriate desired refinement criterion": upstream leaves it open (this is the line the audit cited). None of this is switched on by the mode. P3 adds one guard: a gas merge whose target is in the Stella band is refused if the merged mass would exceed `M_SSP`. Splits only lower mass; an SSP cell that splits into the Stella band forms Stella stars from then on.
- **OPEN (Uli):** a refinement criterion keyed to `M_SSP`, e.g. splitting dense SSP-band gas into the Stella band, or a Jeans/`M_SSP` target in `target_mass_renormalization_factor_for_mergesplit`. Not implemented. Side effect of the guard to decide on: a cell below `MinMassForParticleMerger` whose only eligible neighbours are Stella-band cells just below `M_SSP` stays unmerged.
- **Not separated:** `CritPhysDensity` and `SfEffPerFreeFall` are shared by both channels.

### 4.3 Stellar evolution

- One driver per tag (4.1). Stella keeps its per-star tables, which are looked up by total Z: `BirthMetallicity = Metallicity[0]` (resolvedism_imf_sampling.cc:893; particle_data.h:135). That works unchanged with the FIRE layout.
- The SSP-only functions (`evaluate_light_to_mass_ratio`, `particle_ionizing_luminosity_in_cgs`) are never called on tag 1.

### 4.4 Metals (decision C)

- **Default layout in the mode: FIRE's.** Slots 0=Z, 1=He, 2=C, 3=N, 4=O, 5=Ne, 6=Mg, 7=Si, 8=S, 9=Ca, 10=Fe (init.cc:200-210), then FIRE r-process/age/STARFORGE tracers. There is no H slot.
- **Stella deposit.** The tables always give per-element yields (`stellar_type_ia_yield(k)` etc., resolvedism_fb_thermal.cc:93-99). Today they are packed into `ElemYields[]` only `#ifdef METALS_INDIVIDUAL` (:41-43, :95). In the mode, pack an 11-slot `FireYields[]` instead:
  - Z = Σ metals (all 25), the same total as now;
  - He…Fe copied from their table elements;
  - untracked metals count only in Z.
- **Skip the H closure.** Stella's H closure and renormalisation (resolvedism_fb_thermal.cc:482-514) assume an H slot and are skipped in the FIRE layout. Total-Z deposit (:432-439) is layout-agnostic and stays.
- **Invariant.** Deposited Z mass = Stella metal yield exactly, and Σ_{2..10} ≤ Z per cell. Assert it; never put a violation down to roundoff.
- **Unchanged:**
  - FIRE SSPs write their own slots and tracers as now; Stella stars never write age/r-process tracers;
  - ICs from FIRE parents need no conversion;
  - `TURB_DIFF_METALS` diffuses all slots.
- **Full layout:** only with `..._CARRY_ALL_ELEMENTS` (P7). That is the audit's superset with named indices at ~80 FIRE literal-index sites. Not on the critical path.
- **Isotopes:** the 40-isotope mode is not implemented (`#error`, resolvedism_stellar_tables.h:11), so it is out of scope.

### 4.5 Cooling and heating (decision B)

FIRE-3 cooling (`DoCooling`) serves every cell. Stella's inputs must reach it through the fields FIRE already reads:

| FIRE cooling input | Who writes it in the mode |
|---|---|
| HII state `DelayTimeHII` + ionisation heating (radfb_local.cc:409-457 `do_the_local_ionization`; cooling.cc:401-437, 571-578, 2257-2262) | SSPs: FIRE HII loop (tag-gated). **Stella stars: Stella PI calls `do_the_local_ionization(j, dt, star)` for each cell it claims**, in place of setting `CellP.Ionized` (which only CHEMCOOL reads: chemcool.cc:163-246). |
| FUV flux `Rad_Flux_UV` (Habing; read by `get_FUV_G0` cooling.cc:2550-2575, photoelectric cooling.cc:1550, eos.cc:324, sfr_eff.cc:88) | SSPs/sinks: LEBRON gravtree (forcetree.cc:3342/3461) + local shielding (`selfshield_local_incident_uv_flux`, cooling.cc:2236-2265). **Stella stars: Stella TREE_RAD G0, added to `Rad_Flux_UV` after that shielding step.** |
| CR ionisation (`Get_CosmicRayIonizationRate_cgs`, cooling.cc:2383, 2581) | unchanged |
| H2 fraction | FIRE's own (`COOL_MOLECFRAC_NONEQM` if on) |

Consequences:
- **G0 conversion moves out of `chem_interface.cc`.** Today the TREE_RAD flux → G0 conversion lives inside the CHEMCOOL interface (chem_interface.cc:215-252, `#ifdef GALSF_RESOLVEDISM_G0_VARIABLE`). Extract it into a CHEMCOOL-independent function `resolvedism_G0_from_tree_flux(i)`, with bit-identical output for current builds.
- **No double counting, by construction.** TREE_RAD node luminosities read `UV/LW_luminosity` of Types 4/5 (forcetree.cc:580-582). SSPs leave those at 0, so TREE_RAD G0 holds only Stella stars. LEBRON's dispatcher must return 0 for tag 1, otherwise Stella stars count twice. Today it calls `rt_get_lum_band_resolvedism` for *every* star when `GALSF_RESOLVEDISM_STELLAR_TABLES && RT_USE_GRAVTREE` (rt_utilities.cc:75-76).
- **Columns.** `TREE_RAD_H2`/`TREE_RAD_CO` columns need CHEMCOOL abundances (forcetree.cc:571-576, precompiler_logic.h:1517-1520), so they are off in the mode. TREE_RAD total/dust column shielding stays for Stella G0.
- **CHEMCOOL-only fields.** Stella code reading `CellP.Temp` (resolvedism_fb.cc:1376 diagnostics, resolvedism_dust.cc:90, sfr_eff.cc:14-16 logging) must be guarded or use FIRE's temperature. The compile probe will list them.

### 4.6 Radiation sources and radiation pressure

| Source | UV/G0 | Ionising | Radiation pressure |
|---|---|---|---|
| FIRE SSP | LEBRON (`rt_get_lum_band_stellarpopulation`) | FIRE HII loop | FIRE LOCALRP + LONGRANGE |
| Stella star | Stella TREE_RAD → `Rad_Flux_UV` | Stella PI → `do_the_local_ionization` | Stella RADPRESSURE (direct + IR) |
| STARFORGE sink | LEBRON (`rt_get_lum_band_singlestar`) | `SINK_HII_HEATING` (radfb_local.cc:232-233) | STARFORGE |

A cell already marked HII this step by one model is skipped by the other, without spending photons. Stella PI already uses that rule between Stella stars (`Ionized==2`, resolvedism_photoion.cc:348); extend it to `DelayTimeHII>0`.

### 4.7 Feedback loops

- Separate files are kept: `mechanical_fb.cc` (FIRE + STARFORGE single-star branches) and `resolvedism_fb_thermal.cc`/`_momentum.cc`.
- Alignment means the following:
  - **Same call points:** flagging in run.cc:657-661, injection in `compute_stellar_feedback` after `density()`.
  - **One meaning per shared field.** `SNe_ThisTimeStep` is a FIRE SN count but a Stella event code. With tag gating each particle is written by one driver only. Document both meanings in particle_data.h:210, or give Stella its own field (open question 5).
  - **Same CR hook.** FIRE injects CRs per SN inside the mechanical loop (mechanical_fb.cc:411, 757); Stella injects for its own stars. No double count after gating.
- **Receivers.** Stella stars may deposit into coarse cells and SSPs into fine ones. Both loops normalise by kernel mass, so no change is needed.

### 4.8 Remnants → black holes (decided: option 1)

Today `GALSF_RESOLVEDISM_BH_PROMOTION` turns a remnant into Type 5 with `SinkSubType=1` (resolvedism_fb.cc:1131-1132). In a build with STARFORGE every Type 5 counts as a single star (stellar_evolution.cc:28), so protostellar evolution would run on it.

| Option | What | Cost / risk |
|---|---|---|
| **1 (chosen 2026-10-22)** | No promotion in the mode: remnants stay Type 4, tag 3, gravity (+KETJU) only. `#error` for BH_PROMOTION in the mode. | none; loses BH accretion/feedback |
| 2 | Promote to Type 5 with tag 3; every STARFORGE/FIRE Type 5 path skips tag 3 | moderate; touches every Type 5 selection |
| 3 | Hand them to upstream's BH module (`SINK_PARTICLES` AGN path) as seeds | most physics; seeding/accretion parameters and resolution questions |

## 5. Test plan

All tests use the production model (adiabatic Sedov is the only exception). FB tests are never coarser than 1 Msun.

| Test | What | Pass |
|---|---|---|
| T0 mode off | Stella default goldens (tree-RT reference dumps, chemistry goldens), `test/fire*`, `test/forgedinfire`, STARFORGE tests | bit-identical to `stella_dev` |
| T1 ownership | box with 1 SSP + 1 Stella star + gas | each driver flags/injects only its own; mass, energy and metal ledgers close |
| T2 metals | single Stella SN and Ia in the FIRE layout | Z deposit = table metal yield exactly; He…Fe = table values; SSP slots/tracers untouched |
| T3 HII | Stella star of known Q in uniform gas; twin SSP of equal Q | Strömgren radius from Stella PI; cells carry `DelayTimeHII` and FIRE HII temperature; T matches the SSP twin |
| T4 UV | Stella star only, then SSP only, then both | `Rad_Flux_UV` = Stella G0 (case 1), LEBRON (case 2), sum (case 3); Stella star adds 0 to LEBRON |
| T5 SF switch | two-resolution box straddling `M_SSP` | SSP only above, Stella only below; SFR continuous across the boundary within noise |
| T6 crossing | refine/derefine donor and enriched cells across `M_SSP` | no stranded state, ledgers close |
| T7 harness | `test/forgedinfire` with the mode on (Stella middle tier, `M_SINK=0`), then with sinks | runs, conserves, profiles comparable to upstream |

## 6. Phases

| Phase | Content | Effort | Exit |
|---|---|---|---|
| P1 Tag + mode | `StarModel` + IO/restart inference; mode flag + preset; `#error` rewrite; tag gates (4.1) | 5–7 d | T0, T1 |
| P2 Metals, FIRE layout | 11-slot Stella packing, H closure skipped, invariant assert | 2–3 d | T2 |
| P3 SF switch | `sf_channel_for_cell`, runtime criteria, spawn fork, donor guard, refinement keyed to `M_SSP` | 5–7 d | T5, T6 |
| P4 Radiation/thermo | G0 extraction from chem_interface; Stella G0 → `Rad_Flux_UV`; Stella PI → `do_the_local_ionization`; LEBRON tag fork; CHEMCOOL-only field guards | 4–6 d | T3, T4 |
| P5 STARFORGE tier + BH | `M_SINK` band; BH option chosen (1 = `#error`) | 5–10 d | T7 with sinks |
| P6 Validation | hybrid dwarf (Stella zoom in a FIRE-3 halo), boundary figures, then a cosmological zoom parent | 2–3 wk + compute | conservation checkers, twins vs Stella-only / FIRE-only |
| P7 optional | full element layout (`..._CARRY_ALL_ELEMENTS`, named indices); M1 preset (Stella stars as M1 sources via `rt_get_lum_band_resolvedism`) | 4–6 d each | T0 + T2 in that layout |

Order: P1 → P2 → P3/P4 (parallel) → P5 → P6. That is ~5–8 weeks of development before production, against 7–10 weeks in the audit. The saving comes from B (no CHEMCOOL-everywhere cost and mapping work) and C (no 80-site index port on the critical path).

## 7. What changed from the audit

- **Cooling:** CHEMCOOL everywhere → FIRE cooling. CHEMCOOL is an `#error` in the mode.
- **HII:** "disable FIRE HIIHEATING" → **keep it for SSPs**; Stella PI writes FIRE's HII state.
- **Metals:** superset layout with ~80-site port → **FIRE layout + 11-slot collapse at Stella deposit**. The superset becomes optional P7.
- **UV:** SSP light into Stella G0 → each owner's own scheme feeding the one field FIRE reads (`Rad_Flux_UV`); LEBRON skips Stella stars.
- **Phase order:** the tag/mode now comes first; metals got smaller.
- **Isotopes:** the 40-isotope mode is listed as not implemented (the audit called it optional).

## 8. Open questions

Status after Uli's decisions of 2026-10-22 (section 2):

1. **BH promotion:** decided, option 1 (decision BH).
2. **Shielding mismatch:** decided for the prototype, keep both and measure (decision F).
3. **Long-range radiation pressure from Stella stars:** decided, measure first (decision G).
4. **Order of the HII loops:** decided, FIRE first, Stella on top (decision H).
5. **`SNe_ThisTimeStep`:** decided, Stella gets its own event field (decision I).
6. **Refinement criterion keyed to `M_SSP`:** still open; not settled upstream either.
7. **Dust in the mode:** decided, FIRE dust model or none first (decision J).
8. **Later:** STARFORGE sinks with Stella main-sequence tables and SNe: still open, later phase.
9. **Compile probe 222629** (done 2026-10-22): P1 (Stella default + FIRE-3) compiles and links, so the metal aliasing and the `SNe_ThisTimeStep` clash are silent; FIRE-3 also silently overrides `GALSF_SFR_CRITERION`. P2 (Stella M1 + hybrid defaults) has one hard error, a stale 2-argument `rt_ir_lambdadust` call at chemcool.cc:470 (CHEMCOOL + RT_INFRARED latent bug, fix: pass `P, CellP`). No other compile-level conflicts. Details in stella_hybrid_audit/AUDIT.md section 9.
