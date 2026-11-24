# Stella production configurations

`Config_stella_{default,cloud,galaxy,zoom,tallbox}.sh`, `stella_{cloud,galaxy,zoom,tallbox}.params`: the production
set in this tree's flag and parameter names; `Template_Config_stella.sh` lists the options of the paper runs
(CR fluid, KETJU, nets 15 and 17, moment transport). `STELLA_DEFAULTS` is the default set; the configs add the
opt-ins for the production runs: `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`, and radiation pressure
(`GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS` with `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE`) in cloud, default, galaxy and tall box; the zoom has no radiation pressure (zoom reference, Uli 2026-11-18).
Set `InitCondFile` (and `BoxSize` for the zoom) in the params. `StellarTablesFile` and `SGChemTablesPath` stay `SET_ME`:
GIZMO then reads the tables from the cache of `scripts/get_stella_deps.sh`.

New machine (in development): clone, then once `scripts/get_stella_deps.sh` (tables and SUNDIALS; with `--ketju`, or a `Config.sh` with
`KETJU_REGULARIZATION`, also MSTAR against the Kokkos of your `SYSTYPE`), then `make` and run. Run the setup in the build
environment; a rerun only checks. Cache: `~/.stella-deps`, or `$STELLA_DEPS_PREFIX` (export it for `make` and runs too).
Details: Stella section of `docs/gizmo_documentation.md`; test: `test/stella_deps`.

- cloud: periodic box with isolated gravity, `SGCHEM_KEEP_IC_ABUNDANCES`, attenuated CR rate (MakeCloud ICs).
- galaxy: open boundaries; dwarf-galaxy parameters (0.1 solar, 400 kpc box, 1 Gyr).
- zoom: comoving, periodic with a PM grid and nested PM grid; needs SGChem's UV background support. Softenings, `InitGasTemp`,
  the PM grid and the split and merge masses are placeholders to be set from the IC. `RESOLVED_ISM_FB_AFTER_CLOSING_KICK`: all
  feedback after the closing kick, and the hydro pass redone after SN/Ia deposits (gizmo2017 order; the repeat applies no second
  metal exchange). `RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB` (off; decision pending its cost) adds wind, AGB and radiation-pressure
  deposits. The old names `FB_G17_BOUNDARY_INJECTION` and `GALSF_FB_HYDRO_RECOMPUTE_AFTER_FEEDBACK` stop the build.
- tall box: `GRAVITY_PERIODIC_XY` (2D Ewald) plus the Hu+21 potential; `STELLA_DEFAULTS` without band fluxes: a uniform FUV field and CR rate
  scaled with the recent SFR (`ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR`), radiation pressure at the star.

`check_config_reads.py SRC_DIR FILES...` lists every flag and parameter the code does not read (test/stella_configs).
`MULTIPLEDOMAINS`, `NO_ISEND_IRECV_IN_DOMAIN`, `MaxMemSize` and `BufferSize` are not read in this tree and are not set. `TimeBetStatistics` is the cadence of ledger.txt rows (read with
`OUTPUT_MASS_METAL_ENERGY_LEDGER`; unset, one row at the start).
`TimeInstantSN` is read only with `GALSF_SINGLE_STARS_INSTANT_SN_BEFORE_TIME`, which no config sets. Absent from the params: `MaxMassForParticleSplit` (derived from the IC), `CritOverDensity` (fixed 1000).
`PhotoionMaxTimestep` is not set: the default of 10 kyr applies, so a source need not solve every step.
`MaxStellarTimestep` 1e4 yr (code default 50 kyr) is the whole single-star step rule: a star dies at its first step past its lifetime, at most one step late.
