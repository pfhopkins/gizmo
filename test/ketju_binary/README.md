# ketju_binary

An isolated binary (1 + 1 Msun, a = 1000 AU, e = 0.9, ten orbits) and a far star, built up with the coupling: parameters read, regions found (1 and 2 ranks), the MSTAR wrapper against the Kepler orbit (`TEST_KETJU_MSTAR_PAIR`), region frame (`TEST_KETJU_REGION_LOG`), orbit held with a huge softening (no subtraction needed), then with a small one (exact subtraction), then with the two stars on different ranks.

Bars: wrapper |da/a|, |de| below 1e-9; region frame below 1e-12; soft binary 1e-3; hard binary 1e-6.
