# mechfb_mass_kernel_weights

`GALSF_FB_MECHANICAL_MASS_KERNEL_WEIGHTS` in the mechanical feedback loop: each cell inside the source's own kernel gets the share
mass x kernel. Winds and AGB deaths go along q = w (r - mean of w r), r the unit vector to the cell, normalised by sum |q|, so net momentum
is zero and the booked momentum is coupled. The cell closes the energy of the deposit, heat lands at once, and sources overlapping in a
cell add their momenta in full. All gas along one ray: radial, the star recoils. The star-side radiation pressure kick uses the same weight
and direction, over the star's kernel.

Problem: the gradient cells of mechfb_healpix_weights (1 Msun, n_H 1 to 1000 cm^-3), hydro frozen (`FREEZE_HYDRO`), no gravity, no cooling,
1 yr steps, on 1 and 2 ranks. A second build adds `GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS` for the radiation test. Each check reads the
snapshot pair around one deposit.

Checks (closure limit 1e-12, relative; a cell's share 1e-9, absolute, since a share comes from differences of masses near 1):
1. 20 Msun supernova: shares mass x kernel, energy with the same shares; energy, mass and metals against the event log, no net momentum;
2. 35 Msun wind and 3 Msun AGB death: shares |q| / sum |q|, kicks along q, momentum with the mass;
3. closure: AGB against the log and 30 km/s; wind: kinetic + heat = 1/2 M v^2 with v = sum |p| / M, net momentum, one composition;
4. gas and star moving: lab-frame closure, the wind speed that of the run at rest;
5. all cells along one ray: radial kicks, the star takes the opposite momentum;
6. two winds 0.5 pc apart in the same step: energy and momentum close;
7. 7 Msun star, radiation only: kick shares |q| / sum |q|, no net momentum.

The star's kernel radius is not written; the test takes the radius that fits the shares.
Reference: stella_dev resolvedism_fb_thermal.cc, resolvedism_fb_momentum.cc (passes -1, -4, 0 and -2, 1).
