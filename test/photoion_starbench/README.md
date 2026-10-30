# photoion_starbench

StarBench D-type expansion of an HII region (Bisbas et al. 2015) with the per-direction photoionisation
(`GALSF_FB_PHOTOIONIZATION_PER_DIRECTION`). This is the benchmark ladder that closed the reference implementation
(stella_dev, galaxy_sf/resolvedism_photoion.cc). It is graded under the benchmark's own thermal conditions, as Hu et al.
(2017, App. A) and Smith et al. (2021) ran it: `TEST_PHOTOIONIZATION_FIXED_RATE_TWO_TEMPERATURE_GAS` gives the star
Q = 1e49 /s, holds ionised gas at 1e4 K and neutral gas at 1000 K, and runs no SGChem network.

Problem: a 57 pc box of uniform gas, n_H ~ 100 cm^-3, no gravity, a lattice of N^3 cells (N = 40, 86, 185: 9.9, 1.0 and
0.1 Msun cells, rungs m10, m1, m0.1), one star at the centre; 4.5 Myr.

Grading (`starbench.py`, as stella_dev's pi_set grader): the ionisation-front radius from the volume of ionised gas (H+
above the far-field level), against the StarBench combined solution R_II + (1 - 0.733 e^{-t/Myr}) (R_I - R_II) on
references measured in the run (ambient n_H, T_II, c_i, c_o). The figure of merit is the median of R_IF / R_StarBench - 1
over the D-type phase (t > 0.2 Myr).

Pass: within 2 points of stella_dev's -28.8 / -16.9 / -6.7 % (m10 / m1 / m0.1). m10 runs in the test (a few minutes on 4
cores). m1 and m0.1 run as batch jobs: `run_ladder.sh DIR`, then `GIZMO_STARBENCH_DIR=DIR pytest test/photoion_starbench`.

Measured (Orion, Kokkos OpenMP, 1 rank): m10 -28.3 % (3 min, 16 threads), m1 -16.2 % (26 min, 32 threads); stella_dev
-28.8 / -16.9 %. A 4x finer photoionisation cadence moves R_IF(t) at m10 by at most 0.29 %. The default HII-region loop
on the same ICs: -72 / -78 / -81 %.
