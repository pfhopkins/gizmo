# Stella production configurations

`Config_stella_{default,cloud,galaxy,zoom,tallbox}.sh`, `stella_{cloud,galaxy,zoom,tallbox}.params`: the production
set in this tree's flag and parameter names; `Template_Config_stella.sh` lists the options of the paper runs
(CR fluid, KETJU, nets 15 and 17, moment transport). `STELLA_DEFAULTS` is the default set; the configs add the
opt-ins for the production runs: `GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`, `GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE`.
Set `InitCondFile`, `StellarTablesFile` and `SGChemTablesPath` (and `BoxSize` for the zoom) in the params.

- cloud: periodic box with isolated gravity, `SGCHEM_KEEP_IC_ABUNDANCES`, attenuated CR rate (MakeCloud ICs).
- galaxy: open boundaries; dwarf-galaxy parameters (0.1 solar, 400 kpc box, 1 Gyr).
- zoom: comoving, periodic with a PM grid and nested PM grid; needs SGChem's UV background support. Softenings, `InitGasTemp`,
  the PM grid and the split and merge masses are placeholders to be set from the IC.
- tall box: `GRAVITY_PERIODIC_XY_IMAGES`, no band fluxes, scaled with the recent SFR.

`check_config_reads.py SRC_DIR FILES...` lists every flag and parameter the code does not read (test/stella_configs).
`MULTIPLEDOMAINS`, `NO_ISEND_IRECV_IN_DOMAIN`, `MaxMemSize` and `BufferSize` are not read in this tree and are not set; `TimeBetStatistics` is read only with `DEVELOPER_MODE` (ledger.txt replaces the energy statistics).
Not ported, so absent from the params: `TimeInstantSN`, `MaxMassForParticleSplit` (derived from the IC), `CritOverDensity` (fixed 1000).
`PhotoionMaxTimestep` is not set: the default of 10 kyr applies, so a source need not solve every step.
`MaxStellarTimestep` 1e4 yr (code default 50 kyr) is the whole single-star step rule: a star dies at its first step past its lifetime, at most one step late.
