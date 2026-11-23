#!/usr/bin/env python3
"""16^3 jittered lattice, rho = 1, warm background u = 1e-2, E = 1 in the centre cell, bulk velocity (0.3, 0.1, 0) plus random
velocities (rms 0.05), periodic unit box: a deep timestep hierarchy with wake-ups. A second file drives the merge arm: a shell of
cells far from the blast (0.30 < r < 0.36) at 0.45 x the cell mass (u raised to keep the pressure) and the merge threshold
0.49 x cell mass in the header, so only the shell merges."""
import numpy as np, h5py
n = 16; L = 1.0; dx = L / n; rng = np.random.default_rng(7)
x = (np.indices((n, n, n)).reshape(3, -1).T + 0.5 + 0.1 * rng.uniform(-1, 1, (n**3, 3))) * dx
m = np.full(n**3, L**3 / n**3); u = np.full(n**3, 1e-2)
v = np.array([0.3, 0.1, 0.0]) + 0.05 * rng.standard_normal((n**3, 3))
c = np.argmin(((x - 0.5 * L)**2).sum(1)); u[c] = 1.0 / m[c]
r = np.sqrt((((x - 0.5 * L) + 0.5 * L) % L - 0.5 * L)**2 @ np.ones(3)); far = np.logical_and(r > 0.30 * L, r < 0.36 * L)  # a shell well outside the blast
m_merge = m.copy(); u_merge = u.copy(); m_merge[far] *= 0.45; u_merge[far] /= 0.45  # lighter at equal pressure: below the merge threshold
for name, mm, uu, merge_min in (("pairflux_exact_exchange_ics", m, u, 0.), ("pairflux_exact_exchange_merge_ics", m_merge, u_merge, 0.49 * L**3 / n**3)):
  m, u = mm, uu
  with h5py.File(name + ".hdf5", "w") as f:  # merge ICs: only the shell cells are below the merge threshold
    h = f.create_group("Header"); N = n**3
    if merge_min > 0: h.attrs["Minimum_Mass_For_Cell_Merge"] = merge_min
    for k, val in dict(NumPart_ThisFile=np.array([N,0,0,0,0,0], np.int32), NumPart_Total=np.array([N,0,0,0,0,0], np.uint32), NumPart_Total_HighWord=np.zeros(6, np.uint32),
                       MassTable=np.zeros(6), Time=0.0, Redshift=0.0, BoxSize=L, NumFilesPerSnapshot=1, Omega0=0., OmegaLambda=0., HubbleParam=1., Flag_Sfr=0, Flag_Cooling=0,
                       Flag_StellarAge=0, Flag_Metals=0, Flag_Feedback=0, Flag_DoublePrecision=0).items(): h.attrs[k] = val
    g = f.create_group("PartType0")
    for k, a in dict(Coordinates=x, Velocities=v, Masses=m, InternalEnergy=u).items(): g[k] = a.astype(np.float32)
    g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
