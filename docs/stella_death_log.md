# Stella death-event logs (`output/SNinfo.txt`, `output/AGBinfo.txt`)

One row per SN, Ia, PISN, direct-collapse or AGB death, written when the event is booked
(`fb_death_row_push`, galaxy_sf/resolvedism_fb.cc). Columns 1-20 (SNinfo) / 1-18 (AGBinfo)
are listed in the file header (core/begrun.cc). The last six columns give the gas around the
star at the step of the event:

| SNinfo | AGBinfo | quantity |
|---|---|---|
| 21 | 19 | n_H around the star at this step [cm^-3, physical], from DensityAroundParticle |
| 22 | 20 | T of the neighbours [K], from their mass-weighted u with mu = 1.22 |
| 23 | 21 | star kernel radius h [pc, physical] |
| 24 | 22 | NumNgb |
| 25 | 23 | n_H at the star's birth [cm^-3], from FormationDensity |
| 26 | 24 | birth comoving density at the current a [cm^-3]: the value a comoving run used before the star density refresh |

In isolated runs (a = 1) column 26 equals column 25. In a comoving run, columns 21 and 26
agree only if the star's density was not refreshed after birth.
