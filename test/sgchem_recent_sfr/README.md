# sgchem_recent_sfr

SGChem background field and cosmic-ray rate scaled with the recent star formation rate
(`ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR` with `CR_SCALE_WITH_G0`, Hu+21):
F = [M(stars younger than 30 Myr) / 30 Myr] / [M_gas,0 / 4 Gyr]; G0 = G0_LW = G0_dust = max(F SGChemG0, 0.324e-2);
zeta = F SGChemCosmicRayIonRate. The young-star mass is a running sum kept per rank by birth time (births, wind and
death mass changes of young stars, removal at 30 Myr), reduced over ranks only on calls where it changed.

An 8^3 box of 1 Msun cells, SGChem, single stars from the tables, two births at the first step, 2 ranks.
`TEST_RECENT_SFR_SCALE_PRINT_FULL_SCAN_EVERY_CALL`
(test only) prints, at every star-formation call, F, F from a full scan over all particles, whether the sum was reduced,
and the G0 and zeta the chemistry gets.

| test | checks |
|---|---|
| `test_recent_sfr_scale` | three 1 Msun stars pass 30 Myr, a young 8 Msun star loses wind mass (0.59 Myr): F = full scan at every call (1e-12); G0 and zeta follow F; F drops as each old star passes 30 Myr and with the wind mass loss |
| `test_quiet_calls_reduce_nothing` | only an 8 Msun star of 31.5 Myr loses mass (0.29 Myr): the young-star mass is reduced only at the births and when a star passes 30 Myr (2 reductions); F = full scan at every call |

Inputs: `SUNDIALS_DIR`; stellar and chemistry tables from `GIZMO_STELLA_TABLES_DIR` (else installed by
`scripts/get_stella_tables.sh`); `SGCHEM_TABLES_DIR` overrides the chemistry-table directory.
