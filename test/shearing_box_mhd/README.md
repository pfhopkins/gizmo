# shearing_box_mhd

Ideal MHD in the 3D shear-periodic box (`BOX_SHEARING=3`, q = 3/2, Omega = L = c_s = rho = 1, v_A = B / sqrt(rho)).
B is shear-periodic, so the boundary remaps positions and v_phi but leaves B unchanged.

| case | setup | checks |
|---|---|---|
| windup | uniform B = (B0, 0, B0) in the equilibrium shear flow, 16^3 lattice, 2 orbits | B uniform and flow at equilibrium to 1e-6 (measured ~1e-12); B_phi / B_r = -q Omega t to 1e-6 (measured 1e-13) |
| mri | fastest-growing axisymmetric MRI eigenmode of a vertical field (k = 2 pi / L, k v_A = sqrt(15/16) Omega, s = 0.75 Omega), beta ~ 84, 16^3 and 32^3 | growth rate fitted to rms B_r within 2% of theory at 32^3 (measured -0.5%; -3.0% at 16^3) |

|B| oscillates by ~1e-3 in the wind-up with the kernel volume estimate of the sliding lattice (the density does the same
in the hydro-only `shearing_box` equilibrium), so the wind-up is checked through B_phi / B_r.

Runtime about 6 minutes on 4 ranks.
