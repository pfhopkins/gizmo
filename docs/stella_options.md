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

## GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE: PISN energy (opt-in)

By default every exploding death (CCSN, ECSN, PPISN, PISN) injects a flat 1e51 erg, by design. Failed SNe and direct
black holes inject nothing. With this switch a PISN instead injects [5.0 + 1.304 (M_He - 64)] 1e51 erg. Here
M_He = (13/24)(M - 20), held to 64-133 Msun (Heger & Woosley 2002, as fitted by Wise+12a). The same energy goes to an
ejecta cell (see the ejecta-cell rule below). No parameters.

On in: no shipped config. Tests: test/unit/test_pisn_energy.cc (default flat 1e51), test/unit/test_pisn_energy_helium_core.cc.

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

## TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS (test only)

Each step, recomputes the KETJU limiting-particle steps by a full particle scan and aborts if the cell grid gives a
different answer. Slow; for checking the grid (which now also works in periodic boxes). KetjuRegionRadius is a physical
length in every use.

## Names on gku_stella_dev

| stella_dev | gku_stella_dev |
|---|---|
| GALSF_RESOLVEDISM_POPIII_STARS | GALSF_SINGLE_STARS_POPIII (same PopIII_* parameters) |
| SGCHEM_DGR_REMY_RUYER14 | SGCHEM_DGR_REMY_RUYER14 |
| SGCHEM_PRIMORDIAL_HELIUM | SGCHEM_PRIMORDIAL_HELIUM |
| SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS | SGCHEM_HMINUS_PHOTODETACHMENT_FROM_STELLAR_BANDS |
| GALSF_RESOLVEDISM_PISN_ENERGY_FROM_HELIUM_CORE | GALSF_SINGLE_STARS_PISN_ENERGY_FROM_HELIUM_CORE |
| SGCHEM_UVB_TREECOL_SHIELDING | SGCHEM_UVB_TREECOL_SHIELDING |
| GALSF_RESOLVEDISM_SF_INSTANT_CUTOFF (nHcutoffSF) | GALSF_SINGLE_STARS_SF_INSTANT_ABOVE_DENSITY (nHcutoffSF) |
| GALSF_RESOLVEDISM_SN_SPAWN | GALSF_SINGLE_STARS_SN_EJECTA_CELL |
| TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS | TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS |
| star density refresh, death-log columns | no switch on either branch |
