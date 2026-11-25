# Stella: new options (2026-11)

All options on this page are **in development (developer: Ulrich Steinwandel, GitHub Ulli1991); please contact the
developer before using them.** They follow GIZMO's convention for development modules
(scripts/gizmo_documentation.md, "Important Notes on Stable vs. Development Branches"). Each switch is listed in
Template_Config.sh and Template_Config_stella.sh with the same note.

Defaults below are the values the code uses when a parameter is not set in the parameter file (core/begrun.cc).

## GALSF_RESOLVEDISM_POPIII_STARS: Pop III star formation

Gas below a critical metallicity forms Pop III stars through its own criterion and IMF. A cell forms a star when all of these hold:
n_H above a threshold, a converging flow (div v < 0), an H2 fraction above a floor and a Jeans mass below a reservoir mass.
The mass comes from the Pop III IMF. A star is made with its full drawn mass or not at all. These cells are never formed
by the forced high-density path. Stellar tables use their Z = 1e-4 row until a Z = 0 row exists.
Needs GALSF_RESOLVEDISM_SAMPLE_IMF, SGCHEM and METALS.

| parameter | meaning | unit | default |
|---|---|---|---|
| PopIII_Zcrit_solar | metallicity below which the gas is Pop III | Zsun | 1e-4 |
| PopIII_nH_thresh | density threshold | cm^-3 | 1e6 |
| PopIII_xH2_min | minimum H2 fraction n(H2)/n_H | - | 1e-3 |
| PopIII_Mreservoir | the Jeans mass must be below this | Msun | 1000 |
| PopIII_IMF_Type | 0: dN/dlnM = M^-1.3 exp[-(Mchar/M)^1.6] (Wise+12a form); 1: flat in ln M | - | 0 |
| PopIII_IMF_Mchar | characteristic mass of type 0 | Msun | 20 |
| PopIII_IMF_Mmin, PopIII_IMF_Mmax | IMF mass range | Msun | 1, 300 |

On in: Config_stella_zoom.sh. Tests: test/popIII_onezone (pristine gas forms Pop III stars at n_H ~ 1e6, one-zone;
criterion; IMF), test/unit/test_popIII_sf.cc.

## SGCHEM_DGR_REMY_RUYER14: dust-to-gas ratio at low metallicity

Without a live dust model, the dust-to-gas ratio follows Remy-Ruyer et al. (2014; Table 1, X_CO,Z broken power law) below
0.26 Zsun and stays linear in Z above that (the factor is exactly 1 there, so solar-metallicity runs are unchanged).
It applies everywhere the code uses this ratio: dust cooling, H2 formation on grains, photoelectric heating, A_V, the
TreeCol dust column and radiation pressure. Normalised by DGRnormalized, as before.
Cannot be combined with GALSF_RESOLVEDISM_DUST or SGCHEM_DGR_INDEPENDENT_OF_METALLICITY. Needs SGCHEM and METALS.

On in: Config_stella_default.sh, Config_stella_cloud.sh, Config_stella_tallbox.sh, Config_stella_zoom.sh.
Tests: test/unit/test_dgr_rr14.cc; test/popIII_onezone (a 1e-3 Zsun one-zone cloud fragments at the CMB floor).

## SGCHEM_PRIMORDIAL_HELIUM: primordial helium

He/H = 0.079 by number (Y = 0.24) in the chemistry and hydrogen mass fraction X_H = 0.76 for the gas, for any network
(net 1 always uses these). Without it nets 5 and 17 use He/H = 0.1 and X_H = 0.7065. For cosmological zooms.
Do not also set ABHE in Config.sh (compile error).

On in: Config_stella_zoom.sh. Test: test/unit/test_primordial_helium.cc.

## SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS: H- photodetachment by starlight

The H- photodetachment rate comes from the stellar radiation in the tree bands instead of the ISRF calibration
4.2e-7 G0. Bands: optical/near-IR 0.4133-3.444 eV, NUV 3.444-6 eV, FUV 6-13.6 eV, on the upstream bin edges.
It uses the Shapiro & Kang (1987) cross-section with a 3e4 K shape inside each band. The ISRF floor and the UV
background part keep 4.2e-7 G0. The CMB terms are unchanged.
Needs the C++ backend and GALSF_RESOLVEDISM_G0_VARIABLE with either RT_NUV + RT_OPTICAL_NIR or
GALSF_RESOLVEDISM_RADPRESSURE. RT_NUV and RT_OPTICAL_NIR now switch on the tree band fields without radiation pressure.
Not with TREE_RAY or RADTRANSFER. No parameters.

On in: Config_stella_zoom.sh (with RT_NUV and RT_OPTICAL_NIR). Tests: test/unit/test_hminus_bands.cc (blackbody field
against the spectral integral, upstream bin edges), test/unit/test_band_flags_without_radpressure.cc.

## SGCHEM_COSMIC_RAYS_ZERO_BEFORE_FIRST_STAR: no cosmic rays before the first star

The SGChem cosmic-ray ionisation rate is exactly 0 until star formation has set in: no cosmic-ray ionisation of H, D, He and H2 and
no cosmic-ray heating, also below the 1e-21 s^-1 limit of `CR_SCALE_WITH_G0` and without the radioactive terms of
`CR_BACKGROUND_IONRATE_ATTENUATED`. Electrons then come only from the ionisation in the initial conditions, collisions and
radiation, and H2 forms through H- from those (primordial gas). From then on the rate is that of the build, from the parameter
`CosmicRayIonRate` (3e-19 in the zoom), as without the flag. Star formation has set in when a star or sink (type 4 or 5) is in
the initial conditions or in the snapshot the run starts from, or when the first one has formed in the run; this holds for the
whole box at once. A Stella star counts once it has survived its IMF draw: a spawned star whose draw fails is removed again and
does not switch the rate on. The state is one integer, set from totals the star-formation and IMF routines reduce over tasks
anyway, kept in the restart files and written to the HDF5 snapshot header (`StarFormationHasSetIn`). A snapshot start reads it,
so the rate stays on when every star has gone again (a fully disrupted first star); a snapshot without the entry (older builds,
binary formats) falls back to the scan for stars and sinks. Restart files of a build without the flag do not load. No effect with `COSMIC_RAY_FLUID`. On in
`Config_stella_zoom.sh`. Test: test/sgchem_cr_zero_before_first_star (no star, star in the IC, first star formed in the run, failed first
draws, restart from restart files and from snapshots with and without stars and header entry, 2 ranks).

## RESOLVED_ISM_VARIABLE_SN_ENERGY: explosion energies by type (off by default)

By default every exploding death (CCSN, ECSN, PPISN, PISN) injects 1e51 erg (Uli 2026-11-19); failed SNe and direct
collapse inject nothing. With this switch (Uli's choice 2026-11-19) a PISN injects 1e52 erg and an ECSN 5e50 erg; CCSNe and
PPISNe stay at 1e51. For reference: the He-core fit of Heger & Woosley 2002 (via Wise+12a) gives 4.87e52 erg for a 200 Msun
PISN, and Kitaura, Janka & Hillebrandt 2006 ~1e50 erg for an ECSN; the fit is no longer in the code. Deposit, FeedbackBudget, the energy trace, SNinfo and the ejecta-cell payload use the same energy. It
replaces the former GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE.

Tests: test/unit/test_pisn_energy.cc and test_sn_energy_variable.cc; test/stella_fb_conservation (a PISN, an ECSN and a
CCSN with and without the switch, booked = deposited).

## SGCHEM_UVB_TREECOL_SHIELDING: self-shielding of the UV background

For cosmological runs. The TREECOOL UV background (FG20) photoionisation and heating rates of HI, HeI and HeII are
attenuated per HEALPix pixel. The attenuation uses 10 FG20 bands and Verner+96 cross-sections. Each pixel's column is the
TreeCol gas column times the cell's own neutral H, He0 and He+ fractions, and the result is the mean over pixels.
With zero column the gas is exactly optically thin. Heating hardens behind the column. No Rahmati fit is used.
Needs RT_USE_TREECOL_FOR_NH (HEALPix). No parameters. Design and numbers: reports/uvb_selfshielding.md.

On in: no shipped config yet (zoom test runs in progress). Test: test/sgchem_uvb_shielding. Zero columns are bitwise
equal to the unshielded rates. The band factors agree with the FG20 spectrum to within 10% where they exceed 0.01.
A dense sphere ends up neutral and cold inside.

## GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF: forced star formation above a density

A cell above nHcutoffSF forms its star in that step, whatever the Jeans test says. New: the forced path keeps the same
vetoes as the normal SF rate. Zero-mass cells and ionised (HII-region) cells never form stars this way, and neither does a
spawned wind cell (SINK_WIND_SPAWN). Pop III cells only form through their own criterion.

| parameter | meaning | unit | default |
|---|---|---|---|
| nHcutoffSF | hydrogen number density above which SF is forced | cm^-3 (physical) | none (must be set) |

On in: no shipped config (the default is the pure Jeans criterion). Test: test/sf_cutoff_pi_guard.

## SN ejecta cell first (GALSF_RESOLVEDISM_SN_SPAWN=f)

A supernova whose kernel holds less gas than f x its ejecta mass becomes a new gas cell. The cell holds the ejecta mass,
its metals and the SN energy. This decision is now taken on the first weighting pass. The kernel is widened (x1.4, up to
three times) only for events that do not make a cell: winds, radiation, failed SNe, or builds without SN_SPAWN.
Before, an SN in an empty bubble first grew its kernel and could couple to distant gas.
SN_SPAWN itself is a prototype and is not in the shipped configs.

Test: test/stella_sn_ejecta_cell_first. An SN that finds no gas makes one cell holding its ejecta, with no kernel growth.
Without SN_SPAWN, the kernel-growth fallback still delivers the SN once gas is in reach.

## Comoving runs: star density refresh (no switch)

In comoving runs the star's neighbour density (DensityAroundParticle, kernel radius, NumNgb) was never updated after
birth. Now it is refreshed for every Stella event star and for stars younger than 35 Myr, as isolated runs already
did (hydro/density.cc). Without this, supernovae, photoionisation and winds used the birth density. Isolated runs
(a = 1) are bitwise unchanged.

## Death-log columns (no switch)

SNinfo.txt gains columns 21-26 and AGBinfo.txt gains columns 19-24. They hold the gas around the star at the event
(n_H, T, kernel radius, NumNgb), the birth density, and the birth density at the current scale factor.
Details: docs/stella_death_log.md.

## STARS_APPROACH_TIMESTEP: approach-time step for stars

Type-4 stars also take STARFORGE's two-body step: dt <= sqrt(2 eta) 0.3 / (1/t_approach + 1/t_freefall), with
t_approach = r/|dv| and t_freefall = sqrt(r^3 / G (m1 + m2)) to the nearest star (softened r). The gravity walk fills both
times (Min_Sink_Approach_Time, Min_Sink_Freefall_time; stars are the "special" type of the tree's sink moments), so there
is no extra walk. Comoving runs convert them to physical time. With KETJU, a chain member ignores its own chain partners
within the region radius (MSTAR integrates them); it is still limited by approaching stars outside the chain.
Without it, star encounters are resolved only by MaxStellarTimestep. Not combinable with sinks, STARS_HERMITE_DIRECT_GRAVITY
or STARS_INTEGRATED_BY_RUMI. On by default in every Stella build with single stars (GALSF_RESOLVEDISM_SAMPLE_IMF), set in
declarations/precompiler_logic.h; skipped where one of those is set; opt out with STARS_APPROACH_TIMESTEP_OFF.

Checks (reports/sn_boundary_and_approach_dt.md): 1 + 10 Msun flyby, eps = b = 0.033 pc, v = 3 km/s, cloud build. With a
50 kyr step cap the deflection is 0.31 deg off and dE/E = -4e-3; with no cap 63 deg off. With the switch, 0.002 deg and
|dE/E| <= 5e-5 at any cap. Cost in the 1e4 Msun cloud: 9% more steps, 19% more time per step.

## RESOLVED_ISM_FB_AFTER_CLOSING_KICK: feedback at the step boundary

Feedback is deposited after the closing kick, and the hydro pass is redone on feedback steps (gizmo2017 order). The redone
pass no longer repeats the turbulent metal exchange, which is applied in the pass itself rather than by the kicks. Not
combinable with MFV. On in Config_stella_zoom.sh. Adiabatic SN at 1 Msun: energy excess +0.96% (max) with mid-step
deposit, +0.40% at the boundary.

## RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB: hydro recompute after every boundary deposit

With RESOLVED_ISM_FB_AFTER_CLOSING_KICK alone, the hydro pass is redone after SN and Ia deposits only. With this switch, wind, AGB and
radiation-pressure deposits (star-side walk and the per-cell tree term) also flag their receivers, so the next opening kick
uses forces from the post-deposit state there too. Requires RESOLVED_ISM_FB_AFTER_CLOSING_KICK. Off by default and set in no
production config, the zoom included (Uli 2026-11-19); with continuous winds
the recompute then runs on almost every step. Cost on the zoom smoke test (L10 zoom, 17.7M particles, restart from snapshot 17, 8 ranks, 23 steps): 245 s without, 282 s with the flag (+15 %, a recompute on 23 of 23 steps). Adiabatic 60 Msun wind + AGB box (test/stella_fb_boundary_wind): energy excess +0.0017/+0.0020 % with, +0.0028/+0.0031 % without (100 K / 1e4 K), so no measurable energy gain there.
Test: test/stella_fb_boundary_wind (flag on and off).

## RESOLVED_ISM_FB_RAISES_RECEIVER_SIGNAL_SPEED: feedback receivers on their own timestep (off by default)

A feedback deposit (wind, AGB, SN, Ia ejecta) raises the receiving cell's signal speed to 2 c_s of the post-injection u plus the
velocity change of the deposit. Together with WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP, which it requires, a woken receiver then starts
on the timestep of its heated state instead of the highest active bin. Without it, receivers that were inactive at the deposit apply
the forces of the heated state over a step that is too long at their next closing kick; the u limiter catches them and creates
energy of about the wind energy at the first wind (STL-183). r26 cloud without MHD, first wind (job 242310): energy added by the
limiter 1.05 x the booked wind energy without, 2.0e-3 with the flags (exact kick), 3.1e-3 with the pair-flux exchange.
For configs that deposit feedback before the hydro forces (cloud, default, galaxy, tallbox). Set in no config yet.
Test: test/stella_fb_receiver_wake. TEST_KICK_FLOOR_DUMP=<energy> (test only) prints one line per kick whose u limiter adds more
than that energy and one line per feedback receiver.

## MERGE_INACTIVE_PARENT_COMPLETES_STEP_WITH_OWN_RATES: merges into inactive cells (off by default)

When an active cell merges into an inactive one, the inactive parent has been kicked to the midpoint of its step. With this switch
the merge first kicks that parent from the midpoint to the current time with its own rates and mass (in either half of its step);
the wake that follows skips its own completion kick (marker MergeOldStepCompleted on the particle; the step bounds are untouched).
Without it the wake does that kick with the merged rates, so a hot cell merged into a long-step cell drives u to the floor, which
creates energy: centre-merge Sedov case, |dE|/E 3.5 -> 5.9e-3 (pair-flux exchange off) and 3.9 -> 8.1e-5 (on), job 242294, both
with WAKEUP_MERGE_SPLIT_FROM_OWN_TIMESTEP; alone it does not help (63, 89). MFM and SPH only (an #error with MFV); needs WAKEUP.
The kick's energy change is booked in the merge stage of the energy step trace (merge_parent_kick) and handed back to the
integration residual (merge_parent_kick_to_int). Set in no config yet. Test: test/pairflux_exact_exchange (test_merge_centre).

## MERGE_SPLIT_NEIGHBOR_SEARCH_ACROSS_RANKS: merge and split independent of the rank count (off by default)

Without the switch the search for a merge partner, and for the nearest neighbour that sets how far apart a split's daughters are
placed, sees the cells of the calling rank only, and the merge choice is made in local index order; the result of a merge/split
pass then depends on the number of ranks (STL-324: 77 / 75 / 71 merges on 1 / 2 / 4 ranks in a wind box, energy different by
1e-2 three steps later). With the switch every active cell that needs a merge or a split is a query answered by all ranks. The
partner is the candidate of lowest mass, then lowest IDs (position as the last tie-break). A proposal i -> j is accepted if i's key
(mass, IDs, position) is the smallest
among all proposals that touch i or j, so a cell takes part in at most one merge per pass and nothing depends on an order; the
others wait for the next pass. Landing points of the accepted merges are compared over all ranks. A merge with a partner on
another rank is carried out on the partner's rank by the unchanged merge routine, from a byte copy of the absorbed cell; its
owner removes it on the partner's acknowledgement. Merges of a pass are done before its splits. The choice differs from the
one without the switch, also on a single rank. Not available with the spawned-wind, star-cluster and RIAF merge clauses or
CHIMES. A merge across ranks waits for the next pass (reported in the log) if pair-flux records are pending or the partner's
rank has no free slot for the copy; a split's neighbour distance is taken inside the kernel radius only. Cost per pass: one gather
of the queries, one neighbour search per query and rank, small message rounds for answers, proposals and verdicts, and one message
with the full particle and cell record per merge across ranks.
Code: mesh/merge_split_across_ranks.h. Test: test/merge_split_rank_independence.

## TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS (test only)

Each step, recomputes the KETJU limiting-particle steps by a full particle scan and aborts if the cell grid gives a
different answer. Slow; for checking the grid (which now also works in periodic boxes). KetjuRegionRadius is a physical
length in every use.

## Energy-conserving merges (no switch)

Every Stella build now switches on MERGE_SPLIT_CONSERVE_ENERGY, MERGE_SPLIT_LIMIT_KINETIC_DISSIPATION and
WAKEUP_TRUNCATE_STEP_ON_DEMOTION (declarations/precompiler_logic.h), as spawning builds already did. Before, a gas merge
kept momentum but dropped the pair's centre-of-mass kinetic energy (1/2) mu |dv|^2. Now that energy goes into the
merged cell's thermal energy. Merges that would turn most of the energy involved into heat are refused. The wake-up
switch has no effect with MFM, which already ends the step at the system time. MERGE_SPLIT_DISCARD_ENERGY,
MERGE_SPLIT_ALLOW_KINETIC_DISSIPATION and WAKEUP_REVERSE_KICK_ON_DEMOTION switch the three off again.

Tests: test/merge_split_conservation_stella_default (with the Stella default config, a light cell merged into a heavier one at
8 km/s relative speed: 6.1% of the pair's energy lost before, conserved to 1e-16 after); test/stella_sn_merge_energy
(adiabatic SN at 1 Msun with merge/split on: gas energy matches the coupled energy to 0.3%).

## Names on gku_stella_dev

| stella_dev | gku_stella_dev |
|---|---|
| GALSF_RESOLVEDISM_POPIII_STARS | GALSF_SINGLE_STARS_POPIII (same PopIII_* parameters) |
| SGCHEM_DGR_REMY_RUYER14 | SGCHEM_DGR_REMY_RUYER14 |
| SGCHEM_PRIMORDIAL_HELIUM | SGCHEM_PRIMORDIAL_HELIUM |
| SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS | SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS |
| SGCHEM_UVB_TREECOL_SHIELDING | SGCHEM_UVB_TREECOL_SHIELDING |
| GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF (nHcutoffSF) | GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY (nHcutoffSF) |
| GALSF_RESOLVEDISM_SN_SPAWN | GALSF_SINGLE_STARS_SN_EJECTA_CELL |
| TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS | TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS |
| star density refresh, death-log columns | no switch on either branch |

### Star softening rule (in development)
Stars use a fixed gravitational softening (adaptive softening for gas only), set per run from the gas mass resolution:
eps_star = 1 pc x (m_gas/Msun)^(1/3) x (n_SF / 1e3 cm^-3)^(-1/3), with n_SF = the hard star-formation threshold (1e6 cm^-3), i.e. eps_star = 0.1 pc x (m_gas/Msun)^(1/3) (0.1 pc at 1 Msun, 0.047 pc at 0.1 Msun). This is 3x the Hopkins et al. (2018) / RIGEL scaling: it keeps stars clearly above the smallest gas cells, so birth clusters do not shrink below the softening and set the global timestep. Comoving runs set it as a physical cap (SofteningStarsMaxPhys) with a comoving value large enough that the cap binds before the first stars form. The gas smoothing length is not capped (tiny gas softening floor).

## Feedback audit 2026-11-17: changes that affect runs

- **Feedback tests.** test/stella_fb_conservation, stella_fb_wakeup and stella_fb_boundary_wind build with
  HYDRO_EXACT_PAIR_FLUX_EXCHANGE and HYDRO_TIMESTEP_FROM_PAIR_FORCE (Uli 2026-11-21), so their energy traces test the feedback,
  not the cross-timebin integration error; the default build's 1-3 % residual after hot deposits is a recorded strict expected
  failure (test_default_build_integration_residual; STL-55, STL-127).

- **Restart files.** The restart layout changed: the element ledger (CumulElemToGas, CumulElemFromStars) is written after
  the metal ledger, the metal-ledger accumulators are long double, and every particle carries FB_DepositedP and
  FB_DepositedZ. Restart files
  written before this change cannot be read by this code and vice versa (no size check guards it). Resume running
  productions from a snapshot (RestartFlag 2). Reading ElementAbundance now sets H = 1 - Z - Y, so single-precision
  snapshots and ICs start with a closed composition (an IC with an all-zero ElementAbundance becomes pure H; fresh starts
  still take the composition from InitMetallicity).
- **Explosion energies.** Default: every SN 1e51 erg, as before the audit. With RESOLVED_ISM_VARIABLE_SN_ENERGY: PISN 1e52 erg,
  ECSN 5e50 erg.
- **Death ejecta.** Remnant and ejecta are fractions of the star's mass at death. Failed SNe and direct collapse eject
  nothing (all table ejecta are 0). Remnants between grid points differ from the earlier absolute table values.
- **Type Ia.** Each Ia delivers the table's 1.378 Msun of Ia matter (0.744 Msun Fe): the white dwarf particle plus
  1.378 - M_WD of the receivers' gas turned into Ia composition in place, shared with the ejecta's kernel weight (or the
  HEALPix weight), so it vanishes at the kernel edge (cell mass, momentum and specific internal energy unchanged; its dust
  removed). A cell whose share exceeds its mass is converted whole; the rest is not converted and is reported per step
  (RESOLVEDISM TYPE_IA CLAMP, with running totals). The mean molecular weight, and so T at fixed u,
  changes in the converted part. At 0.1 Msun cells the kernel holds a few Msun, so up to ~25 % of the kernel gas becomes
  Ia matter. The rate normalisation is unchanged, so Ia iron per formed mass is rate x table yield. In the Stella x FIRE-3
  coupled mode the conversion acts on every FIRE slot, tracer slots included: the converted gas keeps no tracer.
- **Metal diffusion.** Stella builds book every pair's turbulent metal exchange to both cells (METALS_EXACT_PAIRWISE_EXCHANGE,
  replacing the forced TURB_DIFF_METALS_EXACT_CONSERVATION, which stays opt-in): with an inactive neighbour the active cell owns the
  pair, with both active the shorter step owns it (equal steps half each, or the single evaluation of HYDRO_PAIR_ONCE), and the owner
  computes it over its own step with the existing flux and limiters. At the end of every hydro pass every touched cell, active or not,
  applies its exchange and clears it, so nothing stays pending. Each pair's 27 element-slot amounts are closed (residual shared in
  proportion to |amount|, slot 0 = sum of metals) and then scaled by one factor so that no cell gives more than its holding / N per
  evaluation, N counted in the gradient pass: a slot can never go below zero. Metal and element masses are exact with individual
  timesteps. The 1 % per-pair cap is not applied on this path (the other paths keep it). The apply floors only a debit below zero
  and each task reports it if it ever happens; a slot another module left below zero takes its credit unfloored. MFM only: Stella
  builds with MFV, SPH, GALSF_ISMDUSTCHEM_MODEL, TRANSPORT_SUBCYCLE or SGCHEM without the element slots keep the pending path, as does
  METALS_EXACT_PAIRWISE_EXCHANGE_OFF (#error only when the flag is set by hand there); the SGChem tracer slots are not diffused (as
  before: they were never applied). Restart layout changes (two ints per cell).
- **Mass trace.** mass_balance_steps.txt counts mass leaving through an outflow boundary in M_bary (also in non-Stella
  builds), and mass_balance_elements.txt closes every element; MbaryOutflow is not restarted (the trace re-baselines after a
  restart).
