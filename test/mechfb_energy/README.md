# mechfb_energy

The energy a mechanical feedback event gives the gas must not depend on the star's mass or the cell's.

One star in a 20^3 periodic lattice, gas at rest, adiabatic, no gravity, n_H = 0.1 cm^-3. Total gas energy (thermal + kinetic) in the star's frame, from the snapshots.

- Supernovae, example model (`GALSF_FB_MECHANICAL` alone): 14.8 Msun at 1e51 erg each, star of 3e4 Msun, star:cell mass 1:1, 100:1, 1000:1. Bar: 3 % of 1e51 erg per event.
- Moving star: the same supernovae at 1:1 with the star at 300 and 1000 km/s, and the same winds at 300 km/s. In its own frame the star delivers the same energy; same bars.
- Stellar mass return, FIRE-3 (`GALSF_FB_FIRE_STELLAREVOLUTION=3`), star:cell 100:1, young star, `GasReturnFraction` raised so that several 1 % packets fire, `GasReturnEnergy` lowered to keep the steps long. Bar: gas energy gained = sum of mass returned x v_wind^2 / 2 to 2 %.

The density is low on purpose. Above it the apply step takes the deferred branch for heating weaker than an assumed cooling rate (`mechanical_fb.cc`), and in an adiabatic run part of that heat is never applied: 0.73 x 1e51 erg per SN at 1:1, n_H = 10, same code. That is a separate effect, not tested here.
