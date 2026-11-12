# Extra capabilities series (ext/01 to ext/05)

Series name: ext. Base: <BASE>. Tip: <TIP>. Branch names are series names.
Covers: single-star event logs (ext/01), stochastic IMF (ext/02), and the Hu et al. ISM dust model in three steps (ext/03 to ext/05).
Order: slice2 -> chem (chem/12 before tc) -> tc -> gap -> cr (cr/04 before pi/06) -> met -> pi -> ext -> tr -> kj. ext/01 and ext/02 sit on slice2/15 and are independent of the rest. ext/03 to ext/05 need met/05 and are serial. The ext/03 branch is stacked on met/06 (the exact-metal-diffusion defaults, Stella switch pending); its own content does not use met/06, so it can be re-cut onto met/05.

**ext/02 (stochastic IMF) is not rebased onto the current base and needs a re-cut.** The branch and numbers below are from the older base.

| PR | branch | base | src lines +/- | test before -> after | flag off (gmc, FIRE3) |
|---|---|---|---|---|---|
| 01 | ext/01-single-star-event-logs | slice2/15-step-limit | 140/1 | no logs -> 3 passed | 145 objects, instruction-identical |
| 02 | ext/02-single-star-stochastic-imf | slice2/15-step-limit | 72/3 | no draws -> 3 passed | 145 objects, instruction-identical |
| 03 | ext/03-ism-dust-hu-kernels | met/05 | 194/0 | no header -> 2 passed | new header, included by no object |
| 04 | ext/04-ism-dust-hu-processes | ext/03 | 140/0 | base rates -> 3 passed | 147 objects, instruction-identical |
| 05 | ext/05-ism-dust-hu-seed-and-stars | ext/04 | 37/2 | 2 failed -> 5 passed | 147 objects, instruction-identical |

All PRs are under 300 source lines. "Instruction-identical": objcmp against the base built in the same path; objects differ at most in immediates and strings (build date, struct offsets). All flag-off lines and numbers below were measured on older bases and are to be re-measured on <TIP>. Flag off: to be re-run on <TIP>. Full suite: <SUITE RESULT>.
Test breadth: narrow: single_stars_event_logs, single_stars_stochastic_imf, ism_dust_hu_kernels. Broad: ism_dust_hu_model (rates against analytic and reference runs).

---
## ext/01: Single stars, SNinfo, AGBinfo and WINDinfo logs written at output

**What and why.** One log row per supernova, AGB death and wind delivery of a single star, in the reference implementation's (stella_dev) columns, so analysis scripts carry over. A rank keeps its rows until the next snapshot or restart file; rank 0 then gathers and writes them sorted by time and ID. Between outputs there is no collective.

**Evidence before.** `test/single_stars_event_logs`: the files do not exist.

**Change.** New `galaxy_sf/single_star_event_logs.cc` (built only with the flag); hooks in `single_star_feedback_finalize`, `savepositions`, `restart`; the density Spec gains one accumulator (mass-weighted gas energy around the star, read into the star's record). New flag `GALSF_SINGLE_STARS_EVENT_LOGS`, default off. No parameters.

**Cost.** 8 B per star record, under the flag; 8 B in the density accumulator; 152 B per row until the next output. One gather at each snapshot and restart file; on quiet steps no work and no collective. WINDinfo has one row per massive star per delivery, so its rows dominate between outputs.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 3 passed: rows equal the source model's own record (ID, class, drawn mass, ejecta, metals, energy); failed SN status 1, no energy; ambient density and energy those of the lattice; the 64 wind rows of the 35 Msun star add up to its mass loss to 1e-6, at the table speed (3265.6-3267.4 km/s, table 3265.7); same rows on 1, 2, 4 ranks.

Comparison with the reference (same initial conditions, its SNinfo.txt): classes and lifetimes equal; ages within 2 yr; ejecta + delivered wind 18.45843 / 112.1023 / 24.56595 Msun against 18.4580 / 112.102 / 24.566; metals 3.51197 / 8.81212 / 0.33849 against 3.512 / 8.812 / 0.338 (to be re-measured on <TIP>) (the reference's rows there have winds off, so its death row holds the wind mass; here it is in WINDinfo). Column 8 of a Type Ia row is the drawn mass here; the reference writes the slot it has already zeroed.

**Flag off.** gmc and FIRE3: 145 objects, 140 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** The existing `single_star_events.txt` still has one integer reduction on steps with active feedback sources. It could use the same output path; left alone as it is the series' own log. Row-level status 2 (no gas reached) is written but no test star reaches that case.

**Depends on / followed by.** slice2/15. Independent of the rest.

---
## ext/02: Single stars, stochastic IMF

**Limits.** Not rebased onto the current base; needs a re-cut before review. Text and numbers below are from the older base.

**What and why.** An alternative to LYRA-style sampling: a star particle of the normal star formation keeps its mass and carries, with probability M/`MassPerStarIMF`, one massive star from a Salpeter slope on [`MinMassIMF`, `MaxMassIMF`]. Draws use the keyed generator (`gpu_rng.h`), so they do not depend on rank count or order. The star takes a compact record and drives the table feedback; a particle without a star is no source.

**Evidence before.** `test/single_stars_stochastic_imf`: no drawn masses in the snapshots.

**Change.** `single_star_imf.h` (the draw), `sfr_eff.cc` (the record at birth), `single_star_feedback.cc` (two guards, flag only), parameters in `begrun.cc`. New flag `GALSF_SINGLE_STARS_STOCHASTIC_IMF` (implies `GALSF_SINGLE_STARS_FROM_IMF`, excludes `GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS`). New parameters `MinMassIMF`, `MaxMassIMF`, `MassPerStarIMF`; `SingleStarsRNGRealisation` as for LYRA.

**Cost.** No per-particle memory; one existing record per star that holds a massive star. Two random numbers at a birth. No collective.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 3 passed: 3745 stars, 741 with a massive star (expected 749 +- 24); KS against Salpeter D = 0.0323, p = 0.413; identical stars on 1, 2, 4 ranks, another realisation differs; total mass conserved to 1e-12 at death of a particle heavier and of one lighter than its star.

**Comparison with the reference: not possible by run.** `GALSF_RESOLVEDISM_STOCHASTIC_IMF` does not compile at the reference HEAD: `resolvedism_fb.cc` and `resolvedism_fb_momentum.cc` read `MstarSampleIMF` outside `GALSF_RESOLVEDISM_SAMPLE_IMF`. Checked against the source instead: same probability, same inverse CDF (sfr_eff.cc:936-945); the key differs (the base generator).

**Open question for the reviewer: death of a particle heavier than its star.** As the reference's `resolvedism_fb.cc` reads (code that has not run), a death returns the whole particle but the table remnant, with the yields scaled to that mass: a 100 Msun particle with a 20 Msun star returned 98.46 Msun, about 5.3x the table's ejecta and metals. Two guards are new here, because the reference has none: a wind stops at the table remnant, and a particle lighter than its remnant keeps its mass. Such a particle's supernova has no ejecta and so couples no energy. The rule is physically doubtful for particles much heavier than the star. Returning only the table ejecta would be a small change in `single_star_event_of`. Which rule should it be?

**Flag off.** gmc and FIRE3: 145 objects, 141 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** slice2/15. Independent of the rest.

---
## ext/03: ISM dust, kernels of the Hu et al. model

**What and why.** The microphysics of Hu et al. 2019, 2023, 2026 on plain doubles, host and device, ported unchanged from the reference's `resolvedism_dust_kernels.h`: Nozawa+06 sputtering fits, drag and grain charge, the dust-gas drift, logistic growth.

**Evidence before.** `test/ism_dust_hu_kernels` does not compile: no header.

**Change.** New `solids/ism_dust_hu_kernels.h`. Nothing includes it yet.

**Cost.** None until ext/04.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 2 passed (plain -O2 and production flags), all checks against analytic solutions or fine-step references, as the reference's unit test.

**Flag off.** No object changes: no source includes the header. Flag off: to be re-run on <TIP>.

**Limits.** Kernels only; no process uses them until ext/04.

**Depends on / followed by.** met/05; ext/04.

---
## ext/04: ISM dust, processes of Hu et al. on the fields of the dust model

**What and why.** `GALSF_ISMDUSTCHEM_HU_MODEL` replaces the accretion and sputtering of the live dust model with Hu's: thermal and nonthermal sputtering per MRN size bin with the cell's dust-gas drift, growth below 300 K, using SGChem's temperature, ionisation and CO. It keeps the model's fields, so transport, snapshots and restart are the base's; only the rates change. The default stays dust-to-gas from Z. Silicate is FeMgSiO4 as a new composition bit (16). Metals never change, only their dust share.

**Evidence before.** `test/ism_dust_hu_model` with the model's own rates: silicate dust gone in 20 kyr of 5e6 K gas, where Hu's rate leaves a quarter per 10 kyr; no carbon growth in cold gas.

**Change.** New `solids/ism_dust_hu.h` (one inline function, host and device), an early call in `update_dust_processes`, drift field in `cell_data.h`, dust-mass-weighted drift in the cell merge, composition bit in `ism_dust_chemistry.cc`, parameters `DustGrowthTimescaleGyr` (1.5), `DustGrowthCoulombFactor` (1). Flags `GALSF_ISMDUSTCHEM_HU_MODEL` (needs `SGCHEM` and `GALSF_ISMDUSTCHEM_MODEL=(1)` or `(1+2)`), `GALSF_ISMDUSTCHEM_HU_SIZE_BINS=8` (default 1), `GALSF_ISMDUSTCHEM_HU_NO_BETATRON`.

**Cost.** 12 B per cell and size bin, on top of the dust model (with `(1+2)` measured 576 -> 704 B per particle in met/04, plus its cell fields). Per active cell one sub-stepped drift integration (up to 1e5 sub-steps in a shock kick); no collective. The reference's `dust_processes.txt`, reduced every 16 steps, is not ported.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 3 passed: thermal sputtering against the analytic rate, ratio 1.0000 for C and silicate (bar 1 %); growth C 0.013278 against 0.013147, silicate 0.011688 against 0.011598 (bar 2 %); metals unchanged slot by slot to 1e-12 with diffusion off; dust inside its elements; 2 and 4 ranks agree to 1e-8. Comparison with the reference in ext/05.

**Found while testing.** SGChem builds without the tables keep GIZMO's solar metallicity 0.02, which made growth 0.7 of the rate. The code uses `SGCHEM_SOLAR_Z` (0.014), as the reference.

**Flag off.** gmc and FIRE3: 147 objects, 143 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Limits.** Sub-stepped drift up to 1e5 sub-steps per kick in shocks; GPU not run. The reference's `dust_processes.txt` is not ported.

**Depends on / followed by.** ext/03; ext/05.

---
## ext/05: ISM dust, seed, Milky Way anchor, supernova dust as options

**What and why.** Completes the model as the reference runs it. A fresh start seeds half of each cell's carbon and 82 % of its silicon in silicate. The dust-to-gas ratio takes the seed as its Milky Way value (4.74e-3, not 0.01), so a seeded solar cell has ratio 1. Supernovae make no dust and destroy none by swept mass (Hu+23: the resolved blast sputters it); AGB deaths and slow winds still condense.

**Evidence before.** `test_seed_on_a_fresh_start`, `test_supernova_dust_is_opt_in` fail: no dust at the start; the supernova condenses dust whatever the flag.

**Change.** `ism_dust_hu_seed_cell` in `ism_dust_hu.h`, call in `Initialize_ISMDustChem_Particle_Variables`; anchor in `sgchem_dust_to_gas.h`; gates in `single_star_dust_yields` and `ISMDustChem_Return_Mass_Where_Dust_Shocked`. Flags `GALSF_ISMDUSTCHEM_HU_SN_EJECTA_CONDENSATION`, `GALSF_ISMDUSTCHEM_HU_SN_SWEPT_MASS_DESTRUCTION`, both off. With the second, the shared loop's own McKee rule runs (efficiency 0.4, as the reference).

**Cost.** None beyond ext/04.

**Evidence after** (measured on an older base; to be re-measured on the audit branch). 5 passed (seed exact to 1e-6). Default: 0.0111 Msun of dust, all AGB carbon. With condensation 0.0545 Msun. Swept-mass destruction: 0.0394 Msun lost against 0.0073 Msun by sputtering alone, 50 yr after the event (ratio 5.4). The first bar here was 10x; it was set on a wrong picture and revised to 2x after seeing the result: the destruction saturates in the kernel of 1 Msun cells, and sputtering acts within decades. The reviewer should weigh that.

Comparison with the reference (same lattices, production cooling flags; bars set before running: hot and cold 2 %, shock 5 %). Dust total, C, Si are port / reference ratios:

| box | time | dust total | C | Si | T [K] (port, reference) |
|---|---|---|---|---|---|
| hot, n_H = 1 | 20 kyr | 0.9994 | 1.0000 | 1.0000 | 5.209e6, 5.209e6 |
| cold, n_H = 1e3 | 150 kyr | 0.9989 | 1.0000 | 1.0004 | 42.37, 42.39 |
| shock, half 1e7 K | 50 kyr | 0.9951 | 0.9976 | 0.9959 | 1.586e6, 1.588e6 |

The 0.1 % in the total of the first two boxes is the silicate mass: here the model's atomic masses, in the reference 16/24/28/56. The first comparison failed the shock bar (+11 %) because the reference build lacked `WSS_CIE_COOL`; with the production flag both codes agree.

**Limits (differences to the reference).** Swept-mass destruction is the shared loop's, applied per pair; `DGR_around` (radiation pressure with live dust) is still not ported (met/05); `dust_processes.txt` is not ported; no resolved supernova remnant has been run with this model.

**Flag off.** gmc and FIRE3: 147 objects, 143 byte-identical, the rest same instructions (measured on an older base; to be re-measured on the audit branch). Flag off: to be re-run on <TIP>.

**Depends on / followed by.** ext/04.
