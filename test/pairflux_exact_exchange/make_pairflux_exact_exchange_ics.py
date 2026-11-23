#!/usr/bin/env python3
"""16^3 jittered lattice, rho = 1, warm background u = 1e-2, E = 1 in the centre cell, bulk velocity (0.3, 0.1, 0) plus random
velocities (rms 0.05), periodic unit box: a deep timestep hierarchy with wake-ups."""
import numpy as np, h5py
n = 16; L = 1.0; dx = L / n; rng = np.random.default_rng(7)
x = (np.indices((n, n, n)).reshape(3, -1).T + 0.5 + 0.1 * rng.uniform(-1, 1, (n**3, 3))) * dx
m = np.full(n**3, L**3 / n**3); u = np.full(n**3, 1e-2)
v = np.array([0.3, 0.1, 0.0]) + 0.05 * rng.standard_normal((n**3, 3))
c = np.argmin(((x - 0.5 * L)**2).sum(1)); u[c] = 1.0 / m[c]
with h5py.File("pairflux_exact_exchange_ics.hdf5", "w") as f:
    h = f.create_group("Header"); N = n**3
    for k, val in dict(NumPart_ThisFile=np.array([N,0,0,0,0,0], np.int32), NumPart_Total=np.array([N,0,0,0,0,0], np.uint32), NumPart_Total_HighWord=np.zeros(6, np.uint32),
                       MassTable=np.zeros(6), Time=0.0, Redshift=0.0, BoxSize=L, NumFilesPerSnapshot=1, Omega0=0., OmegaLambda=0., HubbleParam=1., Flag_Sfr=0, Flag_Cooling=0,
                       Flag_StellarAge=0, Flag_Metals=0, Flag_Feedback=0, Flag_DoublePrecision=0).items(): h.attrs[k] = val
    g = f.create_group("PartType0")
    for k, a in dict(Coordinates=x, Velocities=v, Masses=m, InternalEnergy=u).items(): g[k] = a.astype(np.float32)
    g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
