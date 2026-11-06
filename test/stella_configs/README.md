# stella_configs

Static test of the production set in `scripts/stella`: `Config_stella_{default,cloud,galaxy,zoom,tallbox}.sh`,
`Template_Config_stella.sh` and `stella_{cloud,galaxy,zoom,tallbox}.params`. No build, no run.

- `test_every_setting_is_read`: every flag appears in `src/` outside `Template_Config.sh`, every parameter is a tag in
  `begrun.cc`. GIZMO only warns about an unknown parameter and ignores an unknown flag, so nothing may be set that the code
  does not read. `MULTIPLEDOMAINS`, `NO_ISEND_IRECV_IN_DOMAIN`, `MaxMemSize` and `BufferSize` were taken out for this reason.
- `test_checker_reports_unread`: the checker names an unknown flag and an unknown parameter and exits 1.
- `test_setups`: the four setups start from `STELLA_DEFAULTS`; only the zoom is comoving, with `TimeBetSnapshot` > 1.

Limit: a flag read somewhere in `src/` can still be inert in this combination; the check is by name.
The zoom needs `final/01-sgchem-uv-background` to run (SGChem stops on `ComovingIntegrationOn` without it).
