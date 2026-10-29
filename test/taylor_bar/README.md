# taylor_bar

The Taylor anvil test as the Abaqus "impact of a copper rod" benchmark sets it up: a copper rod (32.4 mm by 3.2 mm
radius in the benchmark; 40 layers, 32.0 mm, on this lattice) hits a rigid frictionless wall at 227 m/s, flows plastically (perfectly plastic, yield strength 3.14e9) and
stops. The wall is a symmetry plane between two mirror-image rods. R/dp = 4 (2080 cells per rod), lattice jittered by
2%. The test first calibrates the masses so every cell starts at the reference density (`make_ics.py --calibrate`
after runs to TimeMax ~0), then runs to 80 us after contact.

Reference (Abaqus, converged): shortening 13.10-13.18 mm, widening 5.52-6.06 mm. R/dp = 4 is far from converged
(shortening 9.7 mm, widening 5.1 mm, energy of both rods +4.5% of their initial kinetic energy; 11.0 and 11.6 mm at R/dp = 6 and 8), and the widening rises past the benchmark range with
resolution (5.1, 5.9, 7.1 mm), an open discrepancy. The bands are those of this resolution and are meant to catch
failures of the method, above all the tensile runaway that destroys the rod when the tensile-instability correction
acts in plastically flowing material.

Checks: energy of both rods conserved to 10% of their initial kinetic energy; the rod stops and stays one body without crossing the
wall; shortening 8-13.5 mm, widening 4-7 mm.

    pytest test/taylor_bar                    # ~3 minutes
