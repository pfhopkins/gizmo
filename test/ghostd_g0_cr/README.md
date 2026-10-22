# ghostd_g0_cr

`GHOSTD_G0_CR` scales the `RT_ISRF_BACKGROUND` field and the background cosmic-ray rate by
`FactorG0 = [M(type-4 stars younger than 30 Myr) / 30 Myr] / [Mgas0 / 4 Gyr]`, recomputed at every star-formation call.

A 50 pc box of cold gas at n_H ~ 100 cm^-3 forms stars (no feedback, so star masses stay fixed) while 20 pre-existing
stars (ages 0-40 Myr from `InitStellarAge`) age through the 30 Myr window. The test parses the `GHOSTD:` line printed at
every call and requires, to 1e-12, that `Mgas0` is the initial gas mass and that `M_young` and `FactorG0` match the
stars in the final snapshot younger than 30 Myr at that time. `FactorG0` must rise with star formation and fall when
an old star leaves the window. Runs in seconds on 2 ranks.
