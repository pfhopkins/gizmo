# P4: radiation sources and heating in the Stella × FIRE-3 coupled mode

This document is the design only: no code, no runs. It builds on DESIGN.md (decisions B, D, F, G, H) and on TreeCol step 2 (`/orion/ptmp/uli/orion/treecol_step2/DESIGN.md`), which moves Stella's tree FUV/LW/NUV/OPT light onto upstream RT bins and `Nodes[].stellar_lum`. The code was checked by grep in three trees:

| tag | tree |
|---|---|
| **[P3]** | `wt_hybrid_p3` @1821df1e (P1+P3 of the mode) |
| **[S2]** | `wt_treecol_step2` @58b50d15 (TreeCol step 2) |
| **[SD]** | `test/gizmo` stella_dev @d5bc67a9 |

Star tags are those of [P3] declarations/particle_data.h:520-525: 1 = FIRE_SSP, 2 = STELLA, 3 = SINK, 4 = STELLA_REMNANT. DESIGN.md §4.5 still says "tag 1" for Stella; that is the old numbering.

## 0. Summary

- **One tree, owner-disjoint bins.** In the mode, SSPs fill only FIRE's bins (`FIRE_UV/OPT/IR`). Stella stars fill only Stella's bins (`FUV_6_8EV`, `PHOTOELECTRIC`, `LYMAN_WERNER`, `NUV`, `OPTICAL_NIR`). Both ride the same `stellar_lum[]` node sums and the same walk. The bins are separated in the source dispatcher, so neither model ever sees the other's light. That gives no double count by construction.
- **One heating field.** Stella's attenuated G0, computed with its own per-pixel shielding, is added to FIRE's `Rad_Flux_UV` right after FIRE's own shielding step. FIRE cooling reads nothing else for FUV.
- **One HII state.** Stella PI marks cells through FIRE's `do_the_local_ionization` and `DelayTimeHII`. FIRE's loop runs first. Each model claims a cell at most once, and a cell claimed by the other model is free for both.
- **Masks are required.** If step 2 and FIRE-3 are simply combined, the result is silently wrong in 6 places (§1). P4 adds one mode macro that keeps Stella's bins out of every FIRE all-bins consumer.
- **Effort:** 7–8 dev days after step 2 lands in stella_dev and is merged into this branch (§7).

## 1. What breaks if step 2 + FIRE-3 are combined with no P4 work

FIRE-3's LONGRANGE turns on `RT_LEBRON`, which turns on `RT_USE_GRAVTREE` and `..._SAVE_RAD_ENERGY` ([P3] precompiler_logic.h:843-851), plus `SAVE_RAD_FLUX` (:205). Step 2's G0_VARIABLE preset adds the PE/6-8/LW/NUV/OPT bins ([S2] precompiler_logic.h:812-826). FIRE-3 alone has none of these bins.

| # | Silent effect | Where |
|---|---|---|
| X1 | The SSP dispatcher fills PE/LW/NUV/OPT_NIR as soon as the flags exist. SSP light then enters Stella's pixel flux (double G0) and appears twice in LONGRANGE RP (FIRE bins + band bins). | [S2] rt_utilities.cc:465, 473, 477-482, 486-491; pixel reads forcetree.cc:2842-2848 |
| X2 | LONGRANGE RP loops over **all** bins, so Stella light is pushed by FIRE RP as well as by Stella TREERAD. | [S2] forcetree.cc:2991 (Rad_Flux save), gravtree.cc:887-901 |
| X3 | The EUV fallback sums all bins when `FIRE_IR==0` (any Stella-only node). Stella light, in erg/s, then enters `Rad_Flux_EUV`. | [S2] forcetree.cc:2955-2956 |
| X4 | FIRE cooling switches to the unshielded band field. `get_FUV_G0` adds `Rad_E_gamma[PE]` on top of `Rad_Flux_UV`. The molecular-fraction G0 and eos G0 are *replaced* by `Rad_E_gamma[LW]`. NUV cooling redistribution switches on. | [S2] cooling.cc:2559-2560, 2301-2307, 1588; eos.cc:326 |
| X5 | Dust temperature sums every `Rad_E_gamma_Pred` bin. That adds unshielded Stella light, with LW counted twice (it lies inside PE). | [S2] cooling.cc:2516-2523, 2628 |
| X6 | Step 2 sets `RT_SOURCES (16+32)` before FIRE's all-types default, so non-cosmological FIRE IC disk/bulge stars (Types 2/3) stop emitting. | [S2] precompiler_logic.h:824-826 vs :932-933 |

Two more findings that matter only in the mode:
- Without CHEMCOOL, `resolvedism_cell_DGR_solar` returns 1 for every Z ([SD] declarations/resolvedism_dgr.h:29). The dust-weighted TreeCol column, and with it Stella G0 shielding, would be solar at any metallicity.
- `COSMIC_RAY_SUBGRID_LEBRON` treats every Type 4 as an SSP SN source, Stella stars included ([P3] cosmic_ray_utilities.cc:1723-1736).

Already fine:
- FIRE keeps its own Sobolev shielding. With GALSF_RESOLVEDISM on, the HEALPix TreeCol does not feed `SigmaEff` ([SD] treecol_config.h:49-54).
- The Stella band cache is already tag-gated ([P3] resolvedism_imf_sampling.cc:198-204).
- The FIRE HII and LOCALRP loops are already tag-gated ([P3] radfb_local.cc:242-244, 34-35).

## 2. Q1: Stella ionising photons → FIRE HII state

**Order (already correct, no change).** Within one step:
1. FIRE HII loop inside `compute_stellar_feedback` ([P3] accel.cc:191; run.cc:813);
2. Stella PI (run.cc:945);
3. cooling (run.cc:988).

So FIRE goes first and Stella sits on top (decision H).

**What FIRE cooling reads.**
- `DelayTimeHII > 0`: fully ionised (cooling.cc:2770, 2802), H2 = 0 (:2287), compact-HII flux added in shielding (:2257-2262).
- When the timer runs out it goes negative, and the next cooling call recombines the cell at fixed T (:405-407, :571-575).
- Heating: `do_the_local_ionization` (FIRE-3 branch, radfb_local.cc:433-452) raises u by about 6.4 eV per atom, capped at the Z-dependent HII equilibrium temperature. It sets `Ne`, adds the compact-HII `Rad_Flux_UV/EUV`, and sets `DelayTimeHII = min(dt, 10 Myr)` (:459). `source` is read only under CHIMES (an `#error` in the mode), so a Stella star can be passed directly.

**Design.**

| Step (in `resolvedism_photoionize`) | Rule |
|---|---|
| entry, before the global reset (photoion.cc:396-402) | Cell is **FIRE-held** ⇔ `DelayTimeHII>0 && Ionized==0`. Mark it with a transient `Ionized=3`. `Ionized==1` already means "Stella-held last step", so no new field is needed. |
| histogram sweep (:348) | FIRE-held cells cost nothing (same rule as `Ionized==2`). |
| marking sweep (:367-369) | FIRE-held cells are not marked. |
| 2→1 transition (:725-736) | **Newly** claimed cells (`DelayTimeHII<=0`) call `do_the_local_ionization(j, t_hold, star)`. **Re-claimed** cells only refresh the timer: `DelayTimeHII = max(DelayTimeHII, t_hold)`, with no second heating. |
| exit | `3 → 0` |

- `t_hold = 2 × cell timestep`, capped at 10 Myr. This keeps the timer positive through the next step's shielding call and FIRE loop. If Stella stops re-claiming the cell (source dead or moved), the timer runs out within ≤2 cell steps and FIRE's fixed-T recombination applies.
- FIRE never takes a Stella cell: it skips `DelayTimeHII>0` and `Ne>0.8` (radfb_local.cc:292-295).
- `CellP.Ionized` is kept as Stella's ownership and wake flag. Upstream has no owner bit for the HII state; this is the only Stella-own HII field, and it is named as such.

**Photon conservation across both models.**
- Each model spends only its own budget, and only on cells it claims:
  - FIRE: `mionizable ≤ Q·dt` (radfb_local.cc:264);
  - Stella: Σ recombination cost ≤ Q/NPIX per pixel.
- A cell held by one model is transparent and free for the other. This is the standard overlapping-HII assumption: the holding source pays that cell's recombinations.
- Ledger, printed per step and asserted:
  - Σ claimed by both = 0;
  - Stella billed ≤ Q_Stella;
  - FIRE: the existing `FdHIIHeating` line.
- Stella keeps billing hot, already-ionised gas. FIRE skips it for free. See open question 4.

**Readers of the Stella HII flag** (the SF block at sfr_eff.cc:222 and the donor block at imf_sampling.cc:555): in the mode, they also block on `DelayTimeHII>0` for the Stella channel. See open question 8.

## 3. Q2: Stella FUV/LW → FIRE heating

**Source dispatcher** (rt_get_source_luminosity; P1 gate at [P3] rt_utilities.cc:75-76; step 2 at [S2] :74-79). In the mode:

| Owner | Call | Bins written | Bins forced to 0 |
|---|---|---|---|
| STELLA / REMNANT | `rt_get_lum_band_resolvedism` (step-2 cache, [S2] :306-320) | 6-8, PE, LW, NUV, OPT_NIR | FIRE_UV/OPT/IR |
| FIRE_SSP (and Types 2/3 non-cosmo) | `rt_get_lum_band_stellarpopulation` | FIRE_UV/OPT/IR | 6-8, PE, LW, NUV, OPT_NIR (restores exactly what FIRE-3 alone emits) |

- `RT_SOURCES` in the mode is the upstream default (fixes X6).
- Bin units can differ between owners (Stella erg/s, SSP code units) because no bin mixes owners. Only owner-aware consumers read each set.

**Mode macro `GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY`** (internal, set by the mode). Stella's bins feed only the per-pixel flux:
- They are excluded from the Rad_E_gamma save (forcetree.cc:2935), the Rad_Flux save (:2991) and the EUV fallback sum (:2955). This fixes X2, X3 and X5.
- Step 2's consumer guard `RADTRANSFER || SAVE_RAD_ENERGY` becomes one predicate that is false in the mode, so FIRE cooling stays on its `Rad_Flux_UV` branch. This fixes X4.
- The mask is a per-bin skip in three loops plus one predicate in 7 guards. FIRE-only and Stella-only builds compile it away, so they stay bitwise.

**Stella G0 into `Rad_Flux_UV`.**
- *Extract.* Write `resolvedism_G0_attenuated(i)`, which does not depend on CHEMCOOL:
  - `G0 = All.G0 · (fac_flux2energy/u_Hab_FUV) · Σ_pix UV_flux[pix] · exp(-2.5·A_V,pix)`
  - `A_V,pix = AV_conv·DGR·col[pix] + AV_local (+AV_ext)`
  - This is the net-5 photoelectric field `G_dust/f_rsc` ([SD] chem_interface.cc:231-238; chem_calc_photo.cc:212-213, 296-332, 569).
  - `AV_conversion_factor` (5.348e-22) and `AV_ext` currently live in CHEMCOOL COMMON, so the mode needs its own copies.
  - On stella_dev, CHEMCOOL calls the same function. Regression: bitwise.
- *Hook.* Call it right after `selfshield_local_incident_uv_flux()` ([P3] accel.cc:118), on the same `ActiveParticleList`, and add the result to `Rad_Flux_UV`. That field is already in Habing G0 after shielding (cooling.cc:2244).
  - No pixel floor (chem_interface.cc:229-235) and no UVB floor (:243-249): FIRE adds the UVB itself in `get_FUV_G0` (cooling.cc:2556).
  - Pixel fluxes are not overwritten in place in the mode.
- *Consumers that then see Stella light automatically:* photoelectric heating (cooling.cc:1546-1550 → `get_FUV_G0` :2555), molecular fraction (:2299), eos dust G0 (eos.cc:324), SF bit 256 (sfr_eff.cc:88).
- *DGR without CHEMCOOL* (fixes the resolvedism_dgr.h:29 finding): use FIRE's `return_dust_to_metals_ratio_vs_solar(i,0,…)·Z/Z☉` (eos.cc:246), the field FIRE shielding already uses. Use it for the dust-weighted column (forcetree.cc:570) and for Stella radp, so both see one DGR.
- *LW:* FIRE has no separate LW field. The H2 model reads `Rad_Flux_UV`. Stella's `G0_LW` is not consumed in the mode (open question 6).
- *Normalisation:* FIRE's G0 unit is 7.84e-4 erg s⁻¹ cm⁻² (1/1276.19, its narrow FIRE_UV band). Stella's is 1.59e-3 (c·5.29e-14, 6-13.6 eV). Both are "G0" to the consumers, so Stella's G0 is added unscaled (open question 1).

**Why not a single shared FUV field.** One option is that both kinds of source fill one bin set and one shielding scheme serves all. That would change either FIRE's SSP calibration or Stella's shielding, which goes against decisions D and F. The shared part is everything upstream already has: the walk, `stellar_lum[]` node sums, the RT bin indices, and the single heating field `Rad_Flux_UV`. The only Stella-own radiation field left is the 12-pixel directional flux (step 2 §3.1). To measure F, there is an optional twin, `GALSF_RESOLVEDISM_HYBRID_STELLA_LIGHT_THROUGH_LEBRON` (§5).

## 4. Q3 and Q4: SSP → Stella, cosmic rays, radiation pressure

**SSP light → Stella fields: none.** With owner-disjoint bins, SSPs contribute 0 to:
- Stella's pixel flux and hence Stella's G0 term;
- TREERAD radp;
- the TREERAD photon cap (`L_emit` sums Stella stars only, [P3] resolvedism_fb_momentum.cc:858-868).

The models are coupled only through the shared outputs: `Rad_Flux_UV`, `DelayTimeHII` and the gas itself. TreeCol columns count all gas, whatever its owner, which is correct.

**Cosmic rays.**

| CR treatment | Mode behaviour |
|---|---|
| FIRE-3 default (no CR flag) | constant 1 eV cm⁻³, ζ = 1.6e-17 (+Z terms); density ramp in cosmological runs ([SD] cosmic_ray_utilities.cc:1775-1795, 1798-1815). Unchanged. |
| `COSMIC_RAY_FLUID` | each owner injects its own SNe: FIRE [P3] mechanical_fb.cc:414/760, Stella resolvedism_fb_thermal.cc:394-407. ζ comes from the fluid. No double count once tags gate the loops. |
| `COSMIC_RAY_SUBGRID_LEBRON` | add a tag gate: Stella stars → 0 in `cr_get_source_injection_rate` ([P3] :1723). Open question 5. |
| Stella `GHOSTD_G0_CR`, `CR_SCALE_WITH_G0` | CHEMCOOL inputs only (chem_interface.cc:217-229, 273), so silently inert in the mode → `#error`. |

**Radiation pressure.**

| Owner | Term | Sees |
|---|---|---|
| FIRE SSP | LOCALRP (radfb_local.cc:31-35, gated) + LONGRANGE (Rad_Flux of FIRE bins only after the mask) | SSP light only |
| Stella | star-side IR boost + TREERAD per-cell direct term (auto with G0_VARIABLE, [P3] precompiler_logic.h:1482), photon-capped | Stella light only |

- Stella's TREERAD is already Stella's long-range analogue of LONGRANGE. So decision G ("measure before adding a long-range term") needs no new term. The measurement is the T4b ledger plus the optional LEBRON twin.
- Known in-Stella double count, kept as it is for now: TREERAD sums UV(6-13.6) + LW + NUV, which counts LW and 6-8 eV twice (step 2 §6).

## 5. Q5: Stella-only vs shared; flags; #error

| Item | Status in the mode |
|---|---|
| Gravity-tree walk, `stellar_lum[]` node sums, RT bin indices | **shared** (upstream) |
| `Rad_Flux_UV` heating field, FIRE cooling, `get_FUV_G0` | **shared**; Stella adds its G0 term |
| `DelayTimeHII` + `do_the_local_ionization` | **shared** HII state and heating |
| CR rate (`Get_CosmicRayIonizationRate_cgs`), CR fluid | **shared** |
| Dust-to-gas (`return_dust_to_metals_ratio_vs_solar`) | **shared** (Stella columns and radp switch to it in the mode) |
| HEALPix TreeCol columns | upstream machinery, Stella consumer |
| 12-pixel band flux, luminosity centroid, per-star band cache | **Stella-only** (step 2 §3: upstream has no such field) |
| Stella PI front solver, `Ionized` owner/wake flag | **Stella-only** |
| `resolvedism_G0_attenuated` (extracted from chem_interface + calc_photo) | **Stella-only** function, CHEMCOOL-independent |
| TREERAD radp + IR boost | **Stella-only** |
| LEBRON FIRE bins + Sobolev shielding, FIRE HII loop, LOCALRP, LONGRANGE | **FIRE-only**, tag-gated |

**Flags (named by behaviour).**
- `GALSF_RESOLVEDISM_TREE_BANDS_PIXEL_FLUX_ONLY`: internal, set by the mode (§3).
- `GALSF_RESOLVEDISM_HYBRID_STELLA_LIGHT_THROUGH_LEBRON`: optional, measurement only. Stella stars emit into FIRE_UV/OPT/IR instead, and Stella G0 and TREERAD are off for them. Needs code-unit Stella luminosities and a band mapping (FIRE_UV = NUV 3.4-8 + PE 8-13.6, FIRE_OPT = OPT, FIRE_IR = rest of L_bol). Phase P4b.

**New `#error`s in the mode.**

| Flag | Reason |
|---|---|
| `TREE_RAY`, `TREE_RAY_PI` | attenuated-in-walk fluxes and CHEMCOOL rt_phot rates are not mapped |
| `GALSF_RESOLVEDISM_GHOSTD_G0_CR`, `CR_SCALE_WITH_G0` | silently inert without CHEMCOOL. DESIGN §3 lists GHOSTD, but P1 does not have it yet (renamed on stella_dev after the branch point). |
| `GALSF_RESOLVEDISM_PHOTOION` without `GALSF_FB_FIRE_RT_HIIHEATING`; `G0_VARIABLE` without `GALSF_FB_FIRE_RT_LONGRANGE` | Stella writes FIRE's HII state and `Rad_Flux_UV` |
| `RT_SEPARATELY_TRACK_LUMPOS` | its all-bin `l_tot` centroid ([S2] forcetree.cc:784-788) would mix owners |
| `CellP.Temp` readers | listed in DESIGN §4.5; guard or use FIRE's T |

## 6. Q6: tests (production model, coupled preset, gas ≤ 1 Msun, 0.1 Msun twin for T3a)

| Test | Setup | Pass |
|---|---|---|
| T0 | mode off: stella_dev+step 2 goldens; FIRE-3 / forgedinfire dumps | bitwise |
| T0b | stella_dev CHEMCOOL build: `resolvedism_G0_attenuated` vs `G_dust/f_rsc` | bitwise per cell |
| T3a Strömgren | Stella star, pinned Q (TestFixedQion), uniform n = 100 cm⁻³; SSP twin of equal Q (age/mass chosen, or `HIIRegion_fLum_Coupled`) | R_S within the pisub ladder tolerances; cells carry `DelayTimeHII>0`, `Ne` ionised; T equals the SSP twin's within 10% |
| T3b overlap | Stella star + SSP at one position | 0 cells claimed twice; billed ≤ own Q for each model; ionised mass ≈ M_S(Q1+Q2) within 10% |
| T3c switch-off | Q → 0 mid-run | cells recombine via FIRE fixed-T within ≤2 cell steps; no stale `Ionized` |
| T4 UV | Stella only / SSP only / both | `Rad_Flux_UV` = Stella G0 term / FIRE-3 reference (bitwise) / exact sum; Stella bins in `Rad_E_gamma`, `Rad_Flux` = 0; `Rad_Flux_EUV` unchanged by the Stella star |
| T4b radp | same three cases | LONGRANGE acceleration has no Stella part; TREERAD has no SSP part; momentum ledger per owner |
| T4c CR | SUBGRID_LEBRON on, Stella star only | SubGrid CR energy density = 0 (or the chosen rate) |
| T4d Z | T4 at 0.1 Z☉ | Stella G0 shielding uses the Z-scaled DGR (≠ solar) |

## 7. Effort

| Work | Days |
|---|---|
| Dispatcher owner fork, bin zeroing, `RT_SOURCES`, mask macro + 7 guards | 1.5 |
| G0 extraction (bitwise on stella_dev), hook after FIRE shielding, DGR branch | 1.5 |
| PI → `DelayTimeHII` (ownership states, hold timer, ledger), SF/donor block | 1.5 |
| CR gate, `#error`s, `CellP.Temp` guards | 0.5–1 |
| T0/T0b/T3/T4 + write-up | 2 + compute |
| **Total** | **7–8 d** (DESIGN.md said 4–6 d; the masks, DGR and PI ownership states were not in that estimate) |

**Prerequisite:** step 2 must be merged to stella_dev and then into this branch. The branch is off dab9f937, so it also lacks the d5bc67a9 TreeCol rename and the GHOSTD rename. The hybrid preset then sets `RT_USE_TREECOL_FOR_NH` (HEALPix) itself.

## 8. Open questions (Uli)

1. **G0 normalisation:** add Stella G0 (6-13.6 eV Habing, 1.59e-3 erg s⁻¹ cm⁻²) to FIRE's `Rad_Flux_UV` (unit 7.84e-4, narrow band) unscaled, or rescale by 2.03? Recommendation: unscaled; T4 reports both.
2. **Stella ionising escape:** Stella has no EUV leakage into `Rad_Flux_EUV` (FIRE SSPs leak via LEBRON). Accept 0 for the prototype?
3. **HII hold timer:** 2 cell steps, so recombination comes up to 2 cell steps after source death (standalone Stella: immediate unflag, then chemistry). OK?
4. **Hot gas:** keep Stella billing already-ionised hot gas (production algorithm), or skip `Ne>0.8` cells as FIRE does?
5. **CR under SUBGRID_LEBRON:** Stella stars give 0, or a Stella time-averaged rate?
6. **LW:** FIRE has no LW field, so Stella's `G0_LW` is unused in the mode. OK?
7. **Stella bin units:** keep erg/s (step 2 open question; consistent here because bins are owner-disjoint). Code units are needed only for the LEBRON twin.
8. **SF/donor block:** extend to FIRE-held HII (`DelayTimeHII>0`) for the Stella channel?
9. **DGR source:** use FIRE's `return_dust_to_metals_ratio_vs_solar·Z/Z☉` for Stella columns and radp in the mode (reuse upstream), rather than Stella's Z/Z☉?
