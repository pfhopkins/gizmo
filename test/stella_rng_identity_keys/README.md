# stella_rng_identity_keys

With `GALSF_SINGLE_STARS_FROM_IMF` (on in every Stella build), the per-particle random draws of the single-star model are
keyed on the particle's identity (`gizmo_rng_particle_identity_key`: ID, child number, generation, type) and the integer
time, through `single_star_draw_rng_key` / `single_star_random_number` (`system/system.cc`) and `rp_draw_rng_key`
(`galaxy_sf/radfb_rp_loop.h`): IMF mass draws and the star-formation acceptance (`sfr_eff.cc`), Type Ia
(`single_star_feedback.cc`), HII sources (`radfb_local.cc`), radiation-pressure source and pair gates (`radfb_rp_loop.*`),
ejecta-cell and split directions and the split axis sign (`merge_split.cc`). Other models keep the event key (identity plus
single-precision position). The position differs by round-off between runs on different rank counts, so with the event key
those runs part at the first such draw (2026-11-15 study of 113 cloud runs: 0 shared stars); with the identity key they share
their draws until the dynamics part. stella_dev keys these draws on ID, child, generation and time (`resolvedism_rng.h`).

Problem: the Stella cloud production flags and parameters (`scripts/stella/Config_stella_cloud.sh`, `stella_cloud.params`)
on the cloud IC named by `STELLA_CLOUD_IC` (a MakeCloud cloud; the study used gmc_Z1_m1.0, 12377 cells), 2.7 Myr
(TimeMax 0.0028), 1, 2 and 4 ranks one after the other. The test skips without `STELLA_CLOUD_IC` or `SUNDIALS_DIR`.

Check (bar written before the run): the births logs (`single_star_births.txt`: time, ID, child, generation, drawn mass) of
the three runs agree row by row for at least the first 10 stars. (An identity-keyed test build of the same cloud shared
22-51 stars at 1-8 ranks.)
