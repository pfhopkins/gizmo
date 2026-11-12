# TreeCol extension: PR drafts (series tc)

**Series:** tc/01 ... tc/06c. Extended TreeCol in the gravity tree: single-star band luminosities as sources, dust/H2/CO masses in tree nodes, 12-pixel columns and band fluxes in the host and device gravity walks, chemistry hookup, band-resolved radiation pressure, and radiation fields in snapshots. The reference implementation (stella_dev) is used for comparison numbers.
**Base:** `<BASE>`. **Tip:** `<TIP>`.
**Order:** slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj.
**Inside the series:** tc/01 -> 01b -> 02 -> 03 -> 04 -> 05 -> 06 -> 06b -> 06c -> 06d. The series base is a merge of slice2/15 (step limit) and chem/12.
**Build and test needs:** `pytest test/treecol_pixels`; needs `SUNDIALS_DIR` and the stella tables (`GIZMO_STELLA_TABLES_DIR` or `scripts/get_stella_tables.sh`).
**Gravity kernel cost:** extended TreeCol in the gravity kernel needs a register budget of 96 doubles per target, behind a compile-time gate. A design note comes first.
**Numbers:** numbers below were measured on an older base (43025c83 and earlier) and are to be re-measured on `<TIP>`, except where a line says "earlier run on the audit line". Source lines are recomputed against the previous PR (tests, docs, scripts, params, Template_Config excluded). Full suite: `<SUITE RESULT>`.

| branch | commits | source lines (+/-) | test | breadth |
|---|---|---|---|---|
| tc/01-star-band-luminosities | 3 | 165/12 | `test_star_band_luminosities` | narrow |
| tc/01b-merge-split-chemistry-test | 1 | 0/0 | `single_stars_merge_split` | narrow |
| tc/02-node-species-masses | 3 | 202/5 | `test_node_species_masses` | narrow |
| tc/03-host-walk-deposit | 6 | 192/3 | `test_pixel_columns_and_fluxes` | broad |
| tc/04-device-walk | 5 | 67/5 | `test_host_equals_device` | broad |
| tc/05-chemistry-hookup | 6 | 68/7 | `test_chemistry_inputs`, `test_far_field_g0`, `test_slab_shielding` | broad |
| tc/06-radiation-pressure | 3 | 126/27 | `test_momentum_budget`, `test_count_6_8ev_twice` | narrow |
| tc/06b-radpressure-direct-per-cell | 2 | 124/2 | `single_stars_radpressure_cells` | broad (momentum against direct sums and the photon cap) |
| tc/06c-snapshot-radiation-fields | 2 | 171/0 | `single_stars_radpressure_cells::test_snapshot_fields` | narrow |
| tc/06d star-side radiation pressure direction (no branch yet) | 2 | 3/3 | `single_stars_radiation_pressure::test_outward` | narrow |

Notes on the table:
- tc/03: columns are unsoftened within the shielding length, as in the reference implementation (`test_columns_unsoftened` fails before); dust, H2 and CO columns change. `grav_target_valid_gas_for_rt` is defined for the pixels without band fluxes too (the tall box config did not build).
- tc/04 keeps only the tidal criterion for gravity in the gas timestep when the only tree radiation is the band fluxes.
- tc/01b is kept apart to stay near 200 source lines. The merge test needs SGChem and the TreeCol IC writer, so it cannot run in the slice or chem series. It covers the split threshold of slice2/05 and the mass-weighted chemistry state of chem/05; it fails on the base before the fix.
- tc/06b is direct radiation pressure per gas cell. tc/06c is radiation fields in snapshots, after 06b because its test lives there. The CR_IonizationRate line of 06c is `sgchem_cr_ion_rate` from cr/01 on. tc/06d sets the direction of the star-side kick for the base's new sign convention (see tc/06d).
- tc/05 contains the shielding switch (3 commits: test, change, README); tc/06 is stacked on it.

---
## tc/01 Single stars: band luminosities as gravity-tree radiation sources

**What and why.** IMF-sampled single stars become sources of the gravity tree's radiation arrays (`stellar_lum`, light centroid), with band luminosities from the stella-tables at their current age, in disjoint bands: PE 8-13.6, LW 11.2-13.6 (inside PE), 6-8 eV (the generic band), NUV 3.444-6, OPT 0.4133-3.444 eV. This is the source model for the per-direction fluxes of the following PRs.

**Evidence before.** On the base, `GALSF_SINGLE_STARS_FROM_IMF` with any gravity-tree radiation stops at compile time (`galaxy_sf/single_star_record.cc`: "no single-star branch in the stellar radiation sources yet; sampled stars would get stellar-population luminosities"); `test_star_band_luminosities` cannot be built.

**Change.** New flag `GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES` (default off; needs SGCHEM and the single-star tables; `#error` with RADTRANSFER/LEBRON/FIRE RT). It sets `RT_USE_GRAVTREE`, `RT_SEPARATELY_TRACK_LUMPOS`, the five band flags and `RT_NUV_SPLIT_AT_6EV_WITH_6_8EV_IN_GENERIC_USER_BAND` (`rt_utilities.cc`: NUV 3.444-6, generic 6-8 eV). `rt_get_source_luminosity`: type-4 stars return `single_star_rt_band_luminosities` (new `galaxy_sf/single_star_luminosity.cc`, host; PMS and dead stars dark). Lazy device source evaluation is excluded under the flag (host-only tables; the eager host prepass is used). Two readers in `cooling.cc` that the band flags activate (`Lambda_RadiativeCooling_toRHDBins`, `Rad_E_gamma` in `get_FUV_G0`) are guarded off under the flag: no always-zero RT cell field is declared. Test-only `TEST_DUMP_TREECOL_PIXELS_AND_EXIT` (`gravity/treecol_pixels_test_dump.cc`, one call in `accel.cc`).

**Cost.** Node: the RT band arrays of `RT_USE_GRAVTREE` (+64 B/node vs the default build, only under the flag). Per star: 0 B (luminosities computed at each tree build/refresh: 5 3D + 2 2D table interpolations per star). No collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_star_band_luminosities` (1, 2 ranks; narrow): 11 stars 1-100 Msun before/on/after the main sequence and dead; each band = Python reimplementation of the table interpolation to 1e-10; OPT+NUV+6-8+PE = table OPT+NUV(3.444-8)+PE to 1e-12; LW <= PE; root node = sum of the stars to 1e-12. PASS.

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Limits.** Differences to the reference implementation (stella_dev): luminosities always at the current age (the reference refreshes only active stars); no IMFSampleStellarMassCut (the slice has no such parameter); NUV/OPT of stars without FUV still shine (the reference skips the deposit when UV and LW are both zero).

**Depends on / followed by.** base / tc/02.

---
## tc/02 TreeCol pixels: dust-weighted gas, H2 and CO masses in tree nodes

**What and why.** The per-pixel columns need species masses in every node. Three masses ride every gas-mass node path: host top-level update and DomainMoment exchange, the shared moment kernel, the device moment refresh, pseudo/import update, tree SoA mirror, LET wire.

**Evidence before.** No node carries dust/H2/CO mass; the walk can only build the total gas column (the 6 bins of the existing TreeCol).

**Change.** New flag `GALSF_RESOLVEDISM_TREECOL_PIXELS=N` (12/48/192, `#error` otherwise; needs SGCHEM). Node field `treecol_species_mass[3]` next to `gasmass` at all its sites (`allvars.h`, `forcetree.cc`, `gravtree_moment_kernel.h`, `gpu_moment_refresh.cc`, `gpu_pseudo_update.cc`, `gpu_gravity_tree.{h,cc}`, `let_pack.cc`). Particle values from one `KOKKOS_INLINE_FUNCTION` (`gravity/gravtree_treecol_pixels.h`): dust = DGR x m (the chemistry's DGR, factored into `cooling/sgchem/sgchem_dust_to_gas.h` and used by the chemistry input too, no change in value; 0 above 1e5 K), H2 = 2 x_H2 m/(1+4 ABHE), CO = 28 x_CO m/(1+4 ABHE). Test hook: optional `TEST_DUMP_TREECOL_PIXELS_AT_STEP=N`.

**Cost.** +24 B per node (and on the LET wire), only under the flag. No collectives; work only in existing moment passes.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_node_species_masses` (narrow): cells as the chemistry inputs (1e-12); root node = sum of cells to 1e-12 at step 0 (1, 2 ranks) and at step 3 after chemistry and drift (2 ranks; CO changed by up to 84%, so the refresh paths are exercised). PASS.

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Limits / open question.** Deviation from the reference implementation: H nuclei per mass is 1/(1+4 ABHE) = 0.714, the chemistry's own n_H, where the reference uses HYDROGEN_MASSFRAC = 0.7065 for the H2/CO node masses. H2 and CO columns here are about 1% higher than the reference's. Is the chemistry's own n_H the right choice for the node masses?

**Depends on / followed by.** tc/01 / tc/03.

---
## tc/03 TreeCol pixels: columns and band fluxes in the host gravity walk

**What and why.** Per gas target, the host walk accumulates 12-pixel columns (dust-weighted, H2, CO) and stellar band fluxes (6-13.6 eV and 11.2-13.6 eV per pixel; 3.444-6, 0.4133-3.444, 6-8 eV totals) at the existing TreeCol deposit site, with the softened pair weight. Walk end writes the cell (float layout).

**Evidence before.** The reference implementation's production runs use these inputs; here the chemistry is optically thin under a uniform field.

**Change.** `gravity/gravtree_treecol_pixels.h`: HEALPix RING lookup (polynomial atan with exact fallback near pixel edges, no atan2 in the common case), per-target sums, `treecol_pixels_accumulate`, `treecol_pixels_write_cell`. `forcetree.cc`: member sums, species per accepted source, one call next to `grav_treecol_accumulate`, write at walk end. Cell fields `TreeColColumn[3][N]` (code surface density), `TreeColEgyUV[N]`, `TreeColEgyLW/NUV/OPT/68` (erg/cm^3: fluxes in erg/s/pc^2 overflow float). LW floored per pixel at walk end (0.1 x 0.324e-2 SGChemBackgroundG0/N Habing, as the reference), then summed. Parameter `TreeColShieldingLength` (physical code length, 0 = no limit; `ShieldingLength` in the reference). Until tc/04 the flag routes the gravity walk to the host.

Accuracy check of the one-bin rule (pixel of the mass centre for the flux too): FAILED on the reference snapshot (48% of cells had part of their 6-13.6 eV flux in another pixel than the reference, p99 68% of a cell's flux). Fluxes therefore go to the pixel of the light centroid; the second lookup is taken only when the light centroid differs from the mass centre (luminous nodes); leaves and dark nodes do one lookup.

**Cost.** Per gas cell: 36 + 12 + 4 floats = 208 B (under the flag). Per target during the walk: 63 doubles. No collectives; no work for non-gas targets.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_pixel_columns_and_fluxes` (1, 2 ranks; broad), vs direct sums: columns median 6.5e-3 / p99 3.3e-2 (tree error); 6-13.6 eV total 3e-8 / 2.6e-5, per pixel 3.5e-8 / 8.9e-3; band totals 4e-8 / at most 1.9e-4. PASS.
- Parity vs the reference implementation (stella_dev, 704b4545; snapshot with 5296 cells and 5 stars of 8-60 Msun, first gravity call, 1 and 2 ranks): 6-13.6 eV per pixel median 2.1e-8 / p99 5.5e-8 (float); LW, NUV 3.444-6, OPT, 6-8 eV totals median 2.1e-8 / p99 5.7e-8; dust columns per cell median 3.5e-4 / p99 1.7e-2 (baseline of this port 5e-4 / 1.9e-2).

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Limits.** Columns carry the tree error of the walk (p99 3.3e-2 against direct sums). Luminous nodes cost a second pixel lookup. Fluxes are stored as float energy densities; 16-bit storage follows in gap/03.

**Depends on / followed by.** tc/02 / tc/04.

---
## tc/04 TreeCol pixels: columns and band fluxes in the device gravity walk

**What and why.** The same deposit in the device walk (`gpu_gravtree.cc`), through the packet engine: per-member sums in `gpu_grav_member_optional_sums_t` with the sum combine rule, lane partials folded in lane order. No atomics.

**Evidence before.** After tc/03 the flag routes the gravity walk to the host.

**Change.** `gpu_gravtree.cc`: sums field + combine + zero, species in the leaf/node payload, accumulate next to `grav_treecol_accumulate`, cell write in `gpu_grav_member_finish`, copy-back to the host cell; `sums_in_scratch` is also true under the flag (`grav_walk_carries_treecol_pixels`): the ~63-double sums live in team scratch. Host routing of tc/03 removed.

**Cost.** Team scratch per lane +504 B under the flag. No collectives.

**Evidence after** (older base; to be re-measured on `<TIP>`). `test_host_equals_device` (broad): device = host cell outputs (identical floats) and device repeat bitwise, 2 ranks x 2 threads and 1 rank x 4 threads (all targets on the device walk). The parity numbers of tc/03 are unchanged on the device walk. 
**Limits.** Kokkos OpenMP backend only (no GPU available for this work): register/spill report (`-Xptxas -v`) and occupancy not measured.

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Depends on / followed by.** tc/03 / tc/05.

---
## tc/05 SGChem: pixel columns and the stars' field from the gravity walk

**What and why.** The chemistry takes its shielding columns and (with the band fluxes) its field from the walk instead of the uniform parameter field: per-pixel dust (divided by the local DGR, as the reference), H2 and CO columns; G0, G0_LW and band-resolved G0_dust through the shared `chem_fuv_field_from_band_energy`, 6-13.6 eV pixels floored at 0.324e-2 SGChemBackgroundG0/12 Habing (as the reference), fac_uv per pixel.

**Evidence before.** `sgchem_config.h` on the base: "without them the chemistry is optically thin (no dust, H2 or CO shielding) under a uniform field".

**Change.** `sgchem.cc`: `iphoto = 6` (TreeCol mode, as the reference) under `GALSF_RESOLVEDISM_TREECOL_PIXELS`, else 0 as before; without it the chemistry ignored the columns (no dust, H2 or CO shielding). `sgchem_config.h`: under `GALSF_RESOLVEDISM_TREECOL_PIXELS` NPIX and the chemistry's pixel switches (`RT_TREECOL_DUST_WEIGHTED`, `RT_TREECOL_H2_CO`; `GALSF_RESOLVEDISM_G0_VARIABLE` with the band fluxes). `sgchem_cell.cc`: `sgchem_treecol_columns`, `sgchem_treecol_field` (replaces `sgchem_uniform_field` under the band flag). The band flag now requires `GALSF_RESOLVEDISM_TREECOL_PIXELS` (`#error`). No new parameters. The parameter field SGChemG0/SGChemG0_LW is unused under the band flag (as the reference).

**Cost.** None beyond tc/03 (reads the cell fields in the host cooling loop).

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_chemistry_inputs` (narrow): G0, G0_LW, G0_dust from the cell energy densities as the reference formula (1e-6). PASS.
- `test_far_field_g0` (broad): after 5 steps with chemistry, one 60 Msun star dying at ~150 yr inside the run, the tree 6-13.6 eV field of every cell = direct sum over the current star luminosities: median 3.4e-8, p99 2.7e-5; the dead star is dark (node luminosities refreshed at every moment path). PASS (2 ranks).
- Reference snapshot (G0 background 1.7, the reference's `chem_interface.cc` formulas applied to its gravity dump): G0 median 1.4e-8 / p99 4.6e-8, G0_LW 2.1e-8 / 5.4e-8, G0_dust 1.2e-8 / 4.1e-8, fac_uv 1.4e-8 / 5.5e-8 (float storage). Median G0 133 Habing.
- `test_slab_shielding` (narrow): n_H ~ 1e3 slab (3.5 pc, N_H ~ 1e22) lit by a 40 Msun star 3 pc in front, 20 steps (0.15 Myr). Before `iphoto = 6`: x_H2 6.1e-7 at the face, 1.6e-6 deep (unshielded everywhere, although the walk's columns were right). After: on the star's axis 1.4e-4 at the face, 2.6e-2 more than 2 pc deep (bar: deep > 1e-2 and at least 30x face). PASS.

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Limits.** No chemistry cloud run against the reference implementation yet (T, H2, CO distributions): the reference run with these parameters writes a MinSizeTimestep warning every step and needs a proper paired setup. CR attenuation by the gas column (the reference's `CR_BACKGROUND_IONRATE_ATTENUATED`) is in cr/01, not here.

**Depends on / followed by.** tc/04 / tc/06.

---
## tc/06 Single stars: band-resolved radiation pressure in the local loop

**What and why.** Radiation pressure of single stars in the local radiation-pressure loop (`radfb_rp`), with the reference implementation's validated estimator at the star: L_abs = f_UV (L_PE + L_6-8) + f_NUV L_NUV(3.444-6) + f_OPT L_OPT, f_b = 1 - exp(-sigma_b DGR N_H), sigma = 2.0/1.3/0.5 x 1e-21 cm^2, N_H = (rho/m_p) h at the star, capped at L_bol; IR boost tau_IR = 5 DGR Sigma on the absorbed light; budget L_abs (1 + tau_IR) dt / c. Default disjoint bands; opt-in `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE` = the reference's production setting (6-8 eV also in the NUV term).

**Evidence before.** The base refuses `GALSF_FB_FIRE_RT_LOCALRP` with single stars (`single_star_record.cc` `#error`); FIRE's loop would give them stellar-population luminosities. With the FIRE-2 gates the variant first fired in 1 of 4 steps and coupled to 0 cells, so the coupling is deterministic for single stars.

**Change.** Flags `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS` (sets `GALSF_FB_FIRE_RT_LOCALRP`; needs the single-star flags) and `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE`. `single_star_luminosity.cc`: shared band provider `single_star_band_luminosities_cgs` (also used by tc/01's source branch), `single_star_radpressure_momentum_cgs`. `radfb_rp_loop.cc`: the budget replaces the population luminosity, no source gate, no age cut. `radfb_rp_loop.h`: every neighbour gets wk x budget radially; no per-cell absorption and no separate IR kick (both are in the budget). Non-sampled type-4 particles get no radiation pressure.

**Cost.** 0 B persistent; work per active star as in the existing loop; no tree payload, no RT band flags needed.

**Evidence after** (older base; to be re-measured on `<TIP>`).
- `test_momentum_budget` (40 Msun, n_H 100 cm^-3, 2 ranks; narrow): offered momentum per step = table estimator within 2% for all steps; coupled = offered (log). PASS.
- `test_count_6_8ev_twice` (narrow): disjoint / counted twice = 0.8682 (tables 0.8681). PASS.
- Compile-suite row without SGCHEM (DGR from Z / solar).

**Flag off.** See the series-wide flag-off result at the end. Flag off: to be re-run on `<TIP>`.

**Limits.** Not compared against reference-implementation builds at the star (the per-cell form is, in tc/06b); recoil not booked to a ledger. The star kernel holds max(DesNumNgb, 64) neighbours under GALSF (`density_loop.h`), as in the reference.

**Depends on / followed by.** tc/05 (shares the band provider of tc/01) / tc/06b. TREE_RAY shells not started here.

---
## tc/01b Single stars: chemistry state and split threshold across merges (test only)

**What and why.** A test for two small changes that sit in earlier PRs: under single stars a cell splits above 2.01 times the largest IC cell (slice2/05, as the reference implementation's `init.cc`; other builds keep 3.01), and a cell merge mass-weights the SGChem abundances, dust temperature and temperature (chem/05; before, the survivor kept its own abundances and the H2, H+ and CO of the swallowed cell were lost). The test needs SGChem and the TreeCol IC writer, so it cannot run in the slice or chemistry series; it is its own commit here.

**Evidence before** (older base). Without the two changes: `test_split_threshold` reads 3.01 in the snapshot header and fails. In a run with 47 merges of 0.3 Msun H2-rich cells the total m x(H2) fell from 5.64 to 8.8e-6, the rest being network formation.

**Change.** `test/single_stars_merge_split` (Config, parameters, test, README). No source lines.

**Cost.** None (test only).

**Evidence after** (older base; the same values in an earlier run on the audit line). `single_stars_merge_split` (narrow, 2 tests): light cells merge with mass conserved to 1e-10; total m x: H2 5.64 -> 5.6400001, H+ 0.4501 -> 0.45009983, CO 1.41e-4 unchanged (bar 1e-6); split threshold 2.01. PASS.

**Flag off.** No source change.

**Limits.** Dust temperature and temperature are mixed by the same formula, but this test cannot measure them: the IC cannot carry the dust temperature and the solve overwrites both on active steps. The reference implementation weights species by X_H; here by mass at the one H fraction of the chemistry. The two differ only with variable X_H, which is off in production.

**Depends on / followed by.** tc/01, slice2/05, chem/05 / tc/02.

---
## tc/06b Single stars: direct radiation pressure per gas cell from the pixel fluxes

**What and why.** The reference implementation's production runs deposit the direct radiation-pressure term in the cell that absorbs the light, from the unattenuated pixel fluxes, capped at the emitted luminosity; only the infrared boost stays at the star. tc/06 put all of it at the star, from the kernel column there. This PR moves the direct term to the absorbing cells when the band fluxes of the walk exist.

**Evidence before** (older base). One 40 Msun star in a uniform box, chemistry and gravity on, radiation pressure the only stellar channel; radial momentum about the star at 20 kyr, run minus a twin without radiation pressure, in Msun km/s, same ICs and script for both codes:

| n_H, bands | reference | before | after | after / reference |
|---|---|---|---|---|
| 1, production | 34.7 | -2.48 | 34.5 | 0.995 |
| 1, disjoint | 24.4 | -2.19 | 24.3 | 0.995 |
| 100, production | 110.5 | -42.3 | 110.8 | 1.003 |
| 100, disjoint | 76.6 | -37.1 | 76.8 | 1.002 |
| 1e4, production | 114.8 | -108.7 | 120.4 | 1.049 |
| 1e4, disjoint | 79.3 | -92.1 | 84.1 | 1.061 |

"Before" is negative because on that base the star-side kick followed a separation vector that pointed at the star; this PR also flipped that sign. The base has since changed the vector to gas minus star (`eb4a2bd57`), so on a re-cut this sign flip is dropped (tc/06d). These matrix values were not kept in a run log and are to be re-measured on `<TIP>`.

**Change.** New flag `GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL`, set automatically with `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS` and the band fluxes; opt-out `GALSF_FB_RADPRESSURE_DIRECT_TERM_AT_STAR` keeps tc/06's form. Without band fluxes the term stays at the star (as the reference implementation's own branch, used by its tall box). `gravtree_treecol_pixels.h`: four more sums in the walk (cross-section-weighted flux along the pixel centres, and its scalar), same pixel as the flux. `single_star_luminosity.cc`: `single_star_radpressure_per_cell` gives each active cell L_abs = sum_b sigma_b F_b DGR N_H(cell) / 4 pi, pushed away from the pixel centres, all deposits scaled by L_emit / sum L_abs when the sum exceeds L_emit (the photon cap), L_emit from the root node; the star keeps only the infrared term. `radfb_rp_loop.{cc,h}`: the call and the infrared-only budget. With `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE` the LW band is also counted twice in deposit and cap, as the reference implementation's production does. Cosmological a^-2 on the flux (the reference has none; identical at a = 1).

**Cost.** 20 B per gas cell (four flux floats and the cell's absorbed luminosity), 32 B per target during the walk. Work on active cells only; a full recount over local cells on steps where all bins are active. One `MPI_Allreduce` of one double per call of the radiation-pressure loop, with the flag on.

**Evidence after** (older base; the same values in an earlier run on the audit line). `single_stars_radpressure_cells` (broad): thin box (n_H = 1) momentum per step 591.6 against the direct sum over cells 591.5 (before 204.4); thick box (n_H = 1e4, cap active) 3903 and 5538 against L_emit/c plus the infrared term 3903 and 5538, disjoint and counted twice; the gas gains outward radial momentum, 0.94 of the logged 1.18 (before -0.51). PASS. Matrix above: 5 of 6 cases within 5 % of the reference; n_H = 1e4 disjoint at 6.1 %. There the summed logged momentum agrees (95.0 against 96.4, production 136.1 against 137.8); the radial projection in the collapsing dense box is 5-6 % higher, cause not measured.

**Flag off.** Older base: objcmp of the whole parity stack (which contained this PR) against its base, gmc_cooling and FIRE-3 configs: 147/147 objects with identical instructions, 140 byte-identical, the rest line-number immediates and strings. Flag off: to be re-run on `<TIP>`.

**Limits.** The collective runs on quiet steps too (flag on only, one double); this conflicts with the no-collective-on-quiet-steps rule and needs a gate on "any active gas cell with flux". Inactive cells enter the cap with the share of their last update (the reference implementation uses their current mass and Z). The 1e4 disjoint case misses the 5 % bar.

**Depends on / followed by.** tc/06 / tc/06c.

---
## tc/06c SGChem: radiation fields in snapshots

**What and why.** Snapshots carry no radiation field, so cloud analysis cannot see what the chemistry saw. With `OUTPUT_SGCHEM_RADIATION_FIELDS` gas cells get G0_FUV, G0_LW, G0_FUV_atten, ShieldFac_Dust, G0_NUV, G0_OPT and CR_IonizationRate, and stars log10 UV_Luminosity and LW_Luminosity [erg/s], under the reference implementation's names and units.

**Evidence before.** None of these blocks exists; `test_snapshot_fields` fails on the missing fields.

**Change.** Flag `OUTPUT_SGCHEM_RADIATION_FIELDS` (default off; `#error` without the band fluxes; sets `OUTPUT_SHIELD_FAC`). `sgchem_cell.cc`: `sgchem_radiation_field_for_snapshot` computes each value at output from the pixel fields of the last walk; only the dust shielding factor of the last solve is stored. `io.cc`: nine blocks. `allvars.h`, `cell_data.h`, `proto.h`, `precompiler_logic.h`. CR_IonizationRate is `sgchem_cr_ion_rate` from cr/01 on (the parameter rate before).

**Cost.** 4 B per gas cell under the flag. Work only at output; none on other steps.

**Evidence after.** `single_stars_radpressure_cells::test_snapshot_fields` (narrow): fields present; G0_FUV against the analytic field of the star beyond three cell sizes, median 1.0000 (bar 3 %); star luminosities equal the tables (1e-3 dex). PASS (earlier run on the audit line). Against the reference implementation, cell by cell, n_H = 100 at 5 kyr (older base, values not kept in a run log, to be re-measured on `<TIP>`): G0_FUV 1.0002 (p5-p95 0.9998-1.0005), G0_LW 1.0001, G0_FUV_atten and ShieldFac_Dust 1.023 (0.99-1.33), luminosities 1.6e-4 dex.

**Flag off.** As tc/06b (same stack). Flag off: to be re-run on `<TIP>`.

**Limits.** G0_NUV and G0_OPT are written but never set in the reference implementation; here they hold the real band field. CR_IonizationRate on the older base was the parameter rate, 2.06 times the reference's attenuated rate; from cr/01 on it is the rate the chemistry uses. The ionising photon rate of stars belongs to the photoionisation series.

**Depends on / followed by.** tc/06b / tc/06d.

---
## tc/06d Single stars: star-side radiation pressure points away from the star

**What and why.** The base changed the separation vector of the local radiation-pressure loop to gas minus star (`eb4a2bd57`, after #105). tc/06b had flipped the sign of the single-star kick for the old vector (star minus gas). On the new base that flip sends the star-side kick toward the star again. This PR takes the vector as it is now: the kick points away from the star. The same change applies to the direction sum of the mass x kernel weights (hp/02), which also carried the old sign.

**Evidence before.** `single_stars_radiation_pressure::test_outward` (new, narrow): one star, the star-side kick the only source of radial momentum; gas radial momentum about the star at the end against the momentum coupled: -4.82 against 4.97 (code units), i.e. inward. Bar: outward and above half the coupled momentum.

**Change.** `radfb_rp_loop.h`: single-star kick scale `+dv_ss` along gas minus star (was `-dv_ss`); under `GALSF_FB_MECHANICAL_MASS_KERNEL_WEIGHTS` the direction sum `wt_dir += w dp / r` (was `-=`). 3 lines changed. No flags, no parameters. On a re-cut onto the current base this is not a separate PR: tc/06b drops its sign flip and hp/02 takes the `+=`; kept as its own commit pair here so the history can be read.

**Cost.** None.

**Evidence after.** `test_outward`: +4.82 against 4.97 coupled. PASS. (Run on the audit line before and after the change; to be re-run on `<TIP>`.)

**Flag off.** Single-star branch only; flag-off objects unchanged. To be re-run on `<TIP>`.

**Limits.** The star-side path is used only without band fluxes (tc/06b) or with the opt-out `GALSF_FB_RADPRESSURE_DIRECT_TERM_AT_STAR`; the default per-cell direct term (tc/06b) has its own direction (pixel centres) and was not affected. The test is a single-bug check and is offered as a temporary gate.

**Depends on / followed by.** tc/06c (and hp/02 for the mass x kernel part) / gap/01.

---
## Series-wide results

**Flag off (all PRs).** (Older base.) objcmp, series base vs each tip tc/01..tc/06, same-length dirs, configs `gmc_cooling` and `fire`: 145/145 objects, 136-143 byte-identical (143 for tc/01, 138 for tc/02, 136 from tc/03 on), the rest "same instructions", the only differences being immediates (`__LINE__`) and strings; no object with different instructions. Flag off: to be re-run on `<TIP>`.

Files touched in the shared tree: `forcetree.cc`, `gpu_gravtree.cc`, `gpu_moment_refresh.cc`, `gpu_pseudo_update.cc`, `gpu_gravity_tree.{h,cc}`, `gravtree_moment_kernel.h`, `let_pack.cc`, `gravtree.cc` (none left after tc/04), `allvars.h`, `cell_data.h`, `global_data_all_struct.h`, `precompiler_logic.h`, `begrun.cc`, `accel.cc`, `rt_functions.h`, `rt_utilities.cc`, `cooling.cc` (2 guards), `radfb_rp_loop.{h,cc}`, `Makefile`, `Template_Config.sh`.

**Tests on the tip** (clean harness builds; older base; to be re-run on `<TIP>`). 11 passed. 2 failed: `test_star_band_luminosities` used a band-flag build without the pixel flag, which tc/05 had broken. Fixed in tc/05 (the band flag now requires `GALSF_RESOLVEDISM_TREECOL_PIXELS`, `#error`) and re-run: PASS on 1 and 2 ranks. Full suite: `<SUITE RESULT>`.

**Sizes (source lines added/removed, tests excluded):** tc/01 165/12, tc/02 202/5, tc/03 192/3, tc/04 67/5, tc/05 68/7 (incl. the 6-line shielding switch), tc/06 126/27, tc/06b 124/2, tc/06c 171/0, tc/06d 3/3. All are below the ~300 line limit.

**Not done / open:**
- TREE_RAY shells and sweep (opt-in);
- chemistry cloud run vs the reference implementation (T, H2, CO distributions);
- radiation pressure of tc/06 (at the star) vs reference-implementation builds; the per-cell form is compared in tc/06b;
- device register/spill report and occupancy (no GPU available; Kokkos OpenMP only);
- recoil ledger.
