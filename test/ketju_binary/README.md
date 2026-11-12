# ketju_binary

An isolated binary (1 + 1 Msun, a = 1000 AU, e = 0.9, ten orbits) and a far star, built up with the coupling: parameters read, regions found (1 and 2 ranks), the MSTAR wrapper against the Kepler orbit (`TEST_KETJU_MSTAR_PAIR`), region frame (`TEST_KETJU_REGION_LOG`), orbit held with a huge softening (no subtraction needed), then with a small one (exact subtraction), then with the two stars on different ranks. The hard binary again with HubbleParam 0.7 and the length and mass units times 0.7 (the same physical binary): the conversion to the integrator's pc, Msun, km/s units carries 1/h.

Bars: wrapper |da/a|, |de| below 1e-9; region frame below 1e-12; soft binary 1e-3; hard binary 1e-6, also with h = 0.7.

Periodic and comoving (KETJU radii are physical lengths): the binary with the far star and a 0.3 Msun field star 50 region radii away (a limiting star, no chain centre), once centred in the open box and once on the edge x = 0 of a periodic box with periodic gravity; the edge orbit matches the open one (a, e within 1e-3; before the fix a = 11.9 x 1000 AU) and, under `TEST_KETJU_LIMITER_SCAN_CROSSCHECK_ABORTS`, the wrapped cell grid of the limiter finds the same limiting star as a full scan at every step. At a = 0.01 two stars at rest 20 region radii apart in comoving units (0.2 physical) form one region (none before the fix).
