# single_star_imf

Unit test of the IMF draw for single stars (`src/galaxy_sf/single_star_imf.h`); no GIZMO build or run.

One initial mass is drawn from the mass-weighted Kroupa IMF (M dN/dM; slopes -0.3 below 0.5 Msun, -1.3 above; upper end
350 Msun, lower end `IMFSampleMinMass`, at least 0.08) by the exact inverse of its cumulative distribution, with one
uniform from the keyed generator of `gpu_rng.h` (key = the particle's event key, counter = salt, integer time, draw
index). With the star-formation acceptance probability M_cell/M_star the formed stars then follow the number-weighted
Kroupa IMF (LYRA, Gutcke et al. 2021).

Checks (10^6 draws each, compiled with the host compiler):
1. Kolmogorov-Smirnov test against the analytic cumulative distribution, for m_min = 0.08 and 4 Msun;
2. weighting each draw by 1/M gives the number-weighted Kroupa mean mass to 1%;
3. the same (key, event) gives bitwise the same masses, another event other masses;
4. two-sample Kolmogorov-Smirnov test against the reference implementation's rejection sampler (stella_dev
   `resolvedism_imf_sampling.cc`, same IMF and limits), for m_min = 0.08 and 4 Msun.

Reference: the analytic IMF (checks 1-2) and the reference implementation's sampler, reimplemented in `imf_draws.cc`
(check 4).
