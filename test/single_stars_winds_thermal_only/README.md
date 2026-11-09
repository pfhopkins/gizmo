# single_stars_winds_thermal_only

`GALSF_SINGLE_STARS_WINDS_THERMAL_ONLY`: winds and AGB ejecta give the gas no kick; their whole 1/2 M v^2 lands as heat, with the
mass and metals as usual. It brackets the production wind from the side opposite to a momentum-only wind (Lahen+23 fig. 3).

Problem: 12^3 cells of 1 Msun (n_H about 30 cm^-3), SGChem net 5, table stars, 2 ranks, 100 yr; a 35 Msun star halfway through
its life, then a 3 Msun star at its AGB death.

Checks: radial gas momentum below 1/4 of the booked momentum; ledger heat row = ejecta energy row (1 %); kinetic row below 1e-3 of
it; mass delivered to the gas = mass taken from the star (1e-12). Before the change the radial momentum is about the booked one.

The flag has no radiation-pressure kick either, as stella_dev: GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS is left out by STELLA_DEFAULTS and refused if set.
Before the change: radial momentum 1.10 (wind) and 0.997 (AGB) of the booked one. After: 0.169 and 0.0017, heat row 1.00000 of the ejecta row.
