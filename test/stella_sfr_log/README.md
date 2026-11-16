# stella_sfr_log

`sfr.txt` in Stella builds (`GALSF_SINGLE_STARS_FROM_IMF`): rows are written on full-sync steps only, so each row must book the
stars formed on every step since the previous row. Columns 2, 4 and 5 cover that interval; column 4 is column 5 over the
interval, in Msun/yr; column 3 is the SFR at the row.

Problem: the Stella cloud production flags and parameters (`scripts/stella/Config_stella_cloud.sh`, `stella_cloud.params`)
on the cloud IC named by `STELLA_CLOUD_IC` (a MakeCloud cloud, e.g. gmc_Z1_m1.0, 12377 cells), 2.7 Myr (TimeMax 0.0028), 2 ranks.
The test skips without `STELLA_CLOUD_IC` or `SUNDIALS_DIR`.

Check (bar written before the run): the last row is at `TimeMax`; summed over the rows, column 5 equals the collected masses
of the stars of `single_star_births.txt` (rows with ok = 1) to 1e-3, and column 4 times the interval since the previous row
gives the same mass to 1e-3.
