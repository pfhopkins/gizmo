# stella_configs

Static test of the production set in `scripts/stella`: `Config_stella_{default,cloud,galaxy,zoom,tallbox}.sh`,
`Template_Config_stella.sh` and `stella_{cloud,galaxy,zoom,tallbox}.params`. No build, no run.

- `test_every_setting_is_read`: every flag appears in `src/` outside `Template_Config.sh`, every parameter is a tag in
  `begrun.cc`. GIZMO only warns about an unknown parameter and ignores an unknown flag, so nothing may be set that the code
  does not read. `MULTIPLEDOMAINS`, `NO_ISEND_IRECV_IN_DOMAIN`, `MaxMemSize`, `BufferSize` and `TimeBetStatistics` (read only with `DEVELOPER_MODE`) were taken out for this reason.
- `test_checker_reports_unread`: the checker names an unknown flag and an unknown parameter and exits 1.
- `test_setups`: the four setups start from `STELLA_DEFAULTS`; only the zoom is comoving, with `TimeBetSnapshot` > 1.

Limit: a flag read somewhere in `src/` can still be inert in this combination; the check is by name.
The zoom needs the SGChem UV background (comoving floor): SGChem stops on `ComovingIntegrationOn` without it.
- `test_gas_kernel_radius_uncapped`: every tracked params file of a Stella build (`STELLA_DEFAULTS` or `GALSF_SINGLE_STARS_FROM_IMF`)
  with adaptive gas softening sets `MinGasKernelRadiusFractional`: unset, it defaults to 1 there (`begrun.cc`) and floors every gas
  kernel radius at 2.8 x `SofteningGas` (`gravtree.cc`). The floor it gives (`SofteningGasMaxPhys` / h for comoving runs) is below 1e-3 pc.
- `test_radiation_pressure_per_config`: each production Config through the preprocessor with `precompiler_logic.h`: cloud,
  default, galaxy and tall box set `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS` themselves and push the gas; the zoom build defines
  no radiation-pressure macro (zoom reference, Uli 2026-11-18). `test_stella_defaults_without_radiation_pressure`:
  `STELLA_DEFAULTS` alone sets none.
