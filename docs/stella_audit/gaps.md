# Small gaps to the reference implementation: PR drafts (series gap)

Covers: three small differences between the port and the reference implementation (stella_dev) in single-star radiation: the mass cut for radiating stars, the minimum mass for radiation-pressure sources, and 16-bit per-pixel storage in TreeCol.

Order and dependencies: PR order of the whole work is slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. This series stacks on `tc/06c-snapshot-radiation-fields` (gap/01 -> gap/02 -> gap/03).

Base: `<BASE>`. Tip: `<TIP>`. Full-suite result: `<SUITE RESULT>`.

Numbers below were measured on an older base and are to be re-measured on `<TIP>`. Source lines are recomputed against the previous PR (tests, docs, Template_Config excluded).

| branch | src +/- | test | scope |
|---|---|---|---|
| gap/01-imf-sample-mass-cut | 13/1 | `test_radiation_mass_cut` | narrow |
| gap/02-radpressure-min-mass | 4/1 | `test_sources` | narrow |
| gap/03-treecol-16bit-storage-default | 44/10 | `test_pixels_stored_as_16bit_log` | broad (accuracy against float storage and the reference implementation) |

All three are under 300 source lines. Tests: `pytest test/treecol_pixels test/single_stars_radiation_pressure` (needs `SUNDIALS_DIR`; stellar tables via `GIZMO_STELLA_TABLES_DIR`).

Four gaps were listed. Gaps 1 and 3 have a PR each (gap/01, gap/02), gap 4 is gap/03 (16-bit storage, now the default); gap 2 (NUV/optical light of stars without FUV) does not exist, see the section after gap/02. A shielding bug found on the way (SGChem `iphoto` switch) is part of tc/05.

---
## gap/01 Single stars: no radiation at or below IMFSampleStellarMassCut

**What and why.** the reference production (its cloud and tall-box parameter files: `IMFSampleStellarMassCut 0.5`) gives stars at or below 0.5 Msun drawn mass zero luminosity in every band (`galaxy_sf/resolvedism_imf_sampling.cc`: `finalize_sampled_star` and the per-step `recompute_resolvedism_fuv_luminosities`: `if(M_drawn <= All.IMFSampleStellarMassCut) continue;`). The port lets them shine, mostly in the optical. The cut also skips their table lookups: with the full IMF (down to 0.08 Msun) about three quarters of the stars by number are below 0.5 Msun.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test_radiation_mass_cut` on tc/06: the parameter is ignored ("Tag 'IMFSampleStellarMassCut' was specified, but it is being ignored") and a 0.2 Msun main-sequence star shines (NUV 1.28e28 erg/s, optical 1.7e31 erg/s). Against the reference (83b7ed68, dump build, same test cloud gas, eight main-sequence stars 0.1-2 Msun, first gravity call): port tc/06 vs the reference production cut 0.5, optical energy density per cell median 3.0e-3, p99 9.3e-2, max 0.73 (NUV 3.444-6 eV p99 5.4e-4).

**Change.** Parameter `IMFSampleStellarMassCut` [Msun] (the reference's name, so its parameter files work), read with `GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES` or `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS`, default 0.5 when absent (printed). `single_star_band_luminosities_cgs` (`galaxy_sf/single_star_luminosity.cc`, the one band provider of the tree sources and of radiation pressure) returns dark for drawn mass <= cut. Files: `global_data_all_struct.h`, `begrun.cc`, `single_star_luminosity.cc`. No new flag.

**Cost.** 8 B in All. No per-particle bytes, no collectives; fewer table lookups (stars below the cut return before them).

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_radiation_mass_cut` (stars 0.1, 0.2, 0.3, 0.45, 0.5, 0.6, 1, 8 Msun, cut 0.5): stars <= 0.5 Msun dark in all five bands, the others equal the table interpolation (1e-10), root node = sum of the shining stars (1e-12). PASS; `test_star_band_luminosities` (1, 2 ranks) PASS. Against the reference (same setup as before): port cut 0.5 vs the reference cut 0.5, per cell UV 6-13.6 / LW / NUV / optical / 6-8 eV median 3.4e-7 / 2.8e-6 / 2.0e-7 / 4.0e-7 / 8.8e-8, max <= 4.1e-6.

**Flag off.** See "Flag off" at the end.

**Limits.** The default cut of 0.5 Msun applies when the parameter is absent (printed at start-up), so a run without it changes against tc/06c.

**Depends on / followed by.** tc/06c; gap/02. Source lines: 13 added, 1 removed.

---
## gap/02 Single-star radiation pressure: sources from 2 Msun drawn mass

**What and why.** In the reference only sampled stars of at least 2 Msun are radiation-pressure sources (`galaxy_sf/resolvedism_fb_momentum.cc`, the active test of loop pair (-2, 1): `if(P[i].sampled) Mstar = P[i].MstarSampleIMF[0]; ... if(Mstar >= 2.0) return 1;`). The port lets every main-sequence star push. Stars without a single-star record were already consistent: the reference needs `sampled` (also in `get_star_info`), the port's band provider returns 0 without a record, so neither code makes them a source. The new test keeps that fixed.

**Evidence before** (older base; to be re-measured on `<TIP>`). `test_sources` on gap/01 (40, 3 and 1.5 Msun stars with records and a 40 Msun star particle without a record, n_H = 100 lattice, 2 ranks): 3 sources fire every step (the 1.5 Msun star included), the record-less particle does not. The reference 83b7ed68 built with its default production config, same four stars, same lattice: "RESOLVEDISM RADPRESSURE: 2 active stars" at every full step.

**Change.** `single_star_radpressure_momentum_cgs` (`galaxy_sf/single_star_luminosity.cc`) returns 0 below `SINGLE_STAR_RADPRESSURE_MIN_MASS` = 2.0 Msun drawn mass (constant, as the reference) and without a record, before any table lookup. No new flag or parameter.

**Cost.** None; stars below 2 Msun skip the five table interpolations of the estimator each step.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_sources`: two sources every step, offered momentum = sum of the 40 and 3 Msun estimators (2 %). PASS; `test_momentum_budget` and `test_count_6_8ev_twice` unchanged, PASS. Source set equal to the reference production.

**Flag off.** See "Flag off" at the end.

**Limits.** The 2 Msun threshold is a constant, as in the reference. Not ported: the channel-3 ledger of the reference adds the star-side momentum of every owned star, including those below 2 Msun that never inject (`resolvedism_fb.cc`, loop after `resolvedism_fb_apply_recoil`). That is a bookkeeping inconsistency in the reference; the port books only what it offers.

**Depends on / followed by.** gap/01; gap/03. Source lines: 4 added, 1 removed.

---
## Not a gap: NUV and optical light of stars without FUV

The reference deposits a source's light only if its 6-13.6 eV or LW luminosity is non-zero (`gravity/forcetree.cc`: `if((uv_lum > 0 || lw_lum > 0 ...) && g0_flux_r2 > 0)`); the port admits NUV/optical-only sources. On the reference this gate never removes a living star: its band luminosities are doubles in erg/s, and a star whose 6-13.6 eV table corners are all dead (-99) gets 10^-99 erg/s, which opens the gate. In the tables such stars exist only below 0.25 Msun (0.1 and 0.2 Msun on the main sequence; scan of all masses 0.1-28 Msun, 400 ages each), i.e. below the production cut of gap/01, where both codes are dark anyway. Run (test cloud gas, only a 0.1 and a 0.2 Msun main-sequence star, cut 0.08 on the reference, none on the port; older base, to be re-measured on `<TIP>`): the reference NUV and optical non-zero in all 5296 cells and equal to the port (median 2.5e-8 / 4.8e-8, max 7e-7). No PR.

Remaining node-level difference (known, small): the light centroid of a node is weighted by UV + LW on the reference and by all bands in the port (`rt_s`). With the production cut it does not show at float precision in the run above (all bands <= 4e-6); without the cut (FUV-dead 0.1/0.2 Msun stars present) optical p99 1.2e-4.

---
## gap/03 TreeCol pixels: per-pixel cell values as 16-bit log10 by default

**What and why.** The 48 per-pixel cell values (dust, H2, CO columns and 6-13.6 eV energy densities, 12 pixels) are stored as 16-bit fixed-point log10 instead of float (layout B of the TreeCol extension design note). Saves 96 B per gas cell (208 -> 112 B with 12 pixels and the band fluxes). Default whenever `GALSF_RESOLVEDISM_TREECOL_PIXELS` is on, after the accuracy check below; float stays available for parity builds.

**Evidence before.** `test_pixels_stored_as_16bit_log` on gap/02: the default build stores floats (offset from the 16-bit grid up to 0.5 step). FAIL.

**Change.** Opt-in flag `GALSF_RESOLVEDISM_TREECOL_PIXELS_STORED_AS_FLOAT` (keep float per-pixel values; `#error` without `GALSF_RESOLVEDISM_TREECOL_PIXELS`). `precompiler_logic.h`: storage type `TREECOL_PIXEL_STORAGE_T` (uint16_t by default, float under the flag) and internal `TREECOL_PIXELS_LOG16`. `cell_data.h`: the two per-pixel fields take the type. `gravtree_treecol_pixels.h`: `treecol_pixel_store` / `treecol_pixel_load` (`KOKKOS_INLINE_FUNCTION`; 20 dex over 65535 steps, 3.05e-4 dex, rounded to nearest, 0 = zero, saturating at the top; columns from 1e-16 g/cm^2 at the code normalisation, energy densities from 1e-24 erg/cm^3) at the walk-end cell write (host and device); readers through `TREECOL_PIXELS_COLUMN` / `TREECOL_PIXELS_EGY_UV` (`sgchem_cell.cc`, test dump). Float build: identity casts. `test_host_equals_device` runs on the float build (it compares the two walks at float precision).

**Cost.** -96 B per gas cell. Per active cell per walk 48 log10; per chemistry call 48 pow. No collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_pixels_stored_as_16bit_log`: all 57144 non-zero column and 4941 energy-density values on the grid (offset <= 7e-6 step), max deviation from the float build 3.51e-4 (half a step); zeros stay zero; species masses and band totals identical; G0/G0_LW/G0_dust within 3.5e-4. PASS. At the tip: `test/treecol_pixels` + `test/single_stars_radiation_pressure` + `test/sgchem_box` 26 passed; `test/sgchem_onezone` 9 passed.
- Float build unchanged: the test-cloud cell dump is byte-identical to the build before this PR.
- Test-cloud snapshot vs the reference (704b4545; 5296 cells, 5 stars, first gravity call, background G0 1.7): default (16-bit) G0 median 1.1e-4 / p99 3.2e-4 / max 3.5e-4, G0_dust 9.4e-5 / 2.8e-4, fac_uv 1.1e-4 / 4.2e-4, UV per pixel 1.8e-4 / 3.5e-4, dust columns per cell median 3.6e-4 / p99 1.7e-2 (float build: G0 1.4e-8 / 4.6e-8, dust columns 3.5e-4 / 1.7e-2). Design bar median <= 1e-3: met.
- Chemistry downstream (accuracy check, with shielding on), PDR slab (n_H ~ 1e3, N_H ~ 1e22, 40 Msun star, 0.5 Myr), max over 7 depth bins of |16-bit - float| / float vs |float 2 ranks - float 1 rank| / float: 0.1 Myr x_H2 3.8e-4 vs 1.3e-2, x_CO 6.0e-3 vs 4.2e-2, T 1.1e-3 vs 4.8e-3; 0.3 Myr 1.9e-4 vs 4.2e-3, 2.5e-2 vs 6.2e-2, 3.5e-4 vs 3.7e-3; 0.5 Myr 4.0e-3 vs 8.5e-3, 2.3e-3 vs 8.3e-3, 6.1e-4 vs 6.9e-3. Inside the run-to-run scatter throughout.
- Cloud (12377 cells of 1 Msun, 5 Myr, star formation, single-star feedback, radiation pressure, shielding on): 4 float runs (1-4 ranks), 3 16-bit runs (1-3 ranks), 1 float run with another IMF realisation; f_H2, x_CO, T median/p10/p90, cold mass fraction, stars and stellar mass at 1-5 Myr. 16-bit inside the float range in 82/120 (same-distribution expectation 3/5 = 72/120). After 2 Myr feedback makes all runs diverge (float at 5 Myr: T median 212-9061 K). Largest early 16-bit deviation (1 rank, 2 Myr): T median 20.9 vs float 18.6-19.0 K, the other IMF realisation 19.2 K.

**Flag off.** See "Flag off" at the end.

**Limits.** Values below 1e-16 g/cm^2 (columns) or 1e-24 erg/cm^3 (energy densities) store as zero; the top saturates. Each chemistry call pays 48 pow. `test_host_equals_device` runs on the float build only.

**Depends on / followed by.** gap/02 / -. Source lines: 44 added, 10 removed.

---
## Production-model PDR slab: port vs the reference (reference for the tc/05 shielding switch)

Setup (older base; to be re-measured on `<TIP>`): the slab above, open box, star formation off (`CritPhysDensity 1e30`, `FacSfThreshMJ 1e-20`), 0.5 Myr. Port: series tip, the cloud production config (SGChem net 5, TreeCol pixels + band fluxes, 16-bit default, single-star feedback, radiation pressure; no photoionisation in this series yet). The reference 83b7ed68 with its default production config (clean build), its production units (kpc, 1e10 Msun), `RestartFlag 2`.

Mass-weighted, 7 bins of 0.5 pc into the slab; port / the reference in the interior (bins 4-7, > 1.5 pc deep):
- 0.1 Myr: x_H2 1.05-1.09, x_CO 1.08-1.35, T 0.93-1.08 (port x_H2 0.038-0.040, T 32-34 K).
- 0.3 Myr: x_H2 1.02-1.35, x_CO 1.08-1.32, T 0.99-2.75.
- 0.5 Myr: x_H2 0.98-1.27, x_CO 0.91-1.19, T 0.82-1.29 except the back edge (port 70 K, the reference 26 K).

The first 1-1.5 pc differ by construction: The reference photoionises the face (T ~ 1e4 K, H2 and CO gone); the port has no photoionisation in this series, so its face is a warm PDR (T 150-330 K).

Notes on a valid reference run (behaviour of the reference, not changed):
- `RestartFlag 0` re-initialises IC stars (`InitStellarAge`, sampled flag and IMF mass cleared): the star is dark.
- `RestartFlag 2` reads metals from `ElementAbundance` (27 elements); without that block the gas ends with Z = 0 or garbage, even with a full 28-slot `Metallicity` block and `Flag_Metals 28`. The reference IC therefore carries `ElementAbundance` and `Metallicity` copied from the reference solar initialisation.
- A `RestartFlag 2` start of a uniform-Z slab (pc units, no `ElementAbundance`) lost its metals within 0.1 Myr through the metal-exchange update (`hydro_toplevel.cc`); consistent with the missing-`ElementAbundance` init, not investigated further.

---
## Series-wide results

**Flag off (all PRs).** objcmp, each branch against its parent (tc/04 -> tc/05 -> tc/06 -> gap/01 -> gap/02 -> gap/03), configs `test/gmc_cooling` and `test/fire`: 147/147 objects, 144 byte-identical in every gap row, the rest "same instructions" (`cooling.o`, `main.o`: data/strings only; `begrun.o` in gap/01: line-number immediates). No object with different instructions. Older base. Flag off: to be re-run on `<TIP>`.

**Tests.** On the old tc/06: `test/treecol_pixels`, `single_stars_radiation_pressure`, `sgchem_box` 23 passed, `sgchem_onezone` 9 passed. At the old series tip (gap/03): 26 passed, `sgchem_onezone` 9 passed (older base; to be re-measured on `<TIP>`). Full suite: `<SUITE RESULT>`.

**Sizes (source lines added/removed, tests and Template_Config excluded):** gap/01 13/1, gap/02 4/1, gap/03 44/10.

**Open.** Photoionisation in the port (the slab face against the reference); the reference ledger quirk for radiation pressure below 2 Msun (not ported); the reference `RestartFlag 2` metal initialisation without `ElementAbundance` (above).
