#!/usr/bin/env python
"""Small DM-only Hernquist halo IC for the hernquist_sidm activating test
used to validate the B2 AGSForce GPU port.

Adapted from test/hernquist/hernquist.py by stripping the interactive
matplotlib call, reducing the default resolution, and writing to a
fixed output filename inside test/hernquist_sidm/.

Physics: pure Type=1 DM, Hernquist profile (rho ~ 1/(r(1+r)^3)),
isotropic velocity distribution drawn via Von Neumann sampling against
the DF f(E) at each radius. Units are whatever the calling params file
declares — scale radius a, halo mass m, and G=1 here are all free.

Usage: python make_ic.py [N_PER_SIDE [MASS [SCALE_RADIUS]]]
  defaults: N_PER_SIDE=10 (so N=1000 DM particles), mass=1, a=1.
"""
import sys
import numpy as np
import h5py
from numba import jit

G = 1.0

# args: (0) N_per_side (default 10 → 10^3 = 1000 particles)
#       (1) total halo mass (default 1.0)
#       (2) scale radius (default 1.0)
N_side = int(sys.argv[1]) if len(sys.argv) > 1 else 10
M_tot = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
a = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

N = N_side ** 3
print(f"Building Hernquist halo: N={N} particles, M={M_tot}, a={a}")

rng = np.random.default_rng(42)  # fixed seed so IC is reproducible
x_u = np.arange(0, 1, 1.0/N) + rng.random(N)/N
# Hernquist inverse-CDF for radius from mass enclosed x_u = r^2/(1+r)^2
r = (np.sqrt(x_u) + x_u) / (1 - x_u)

# Isotropic angles
phi = rng.random(N) * 2 * np.pi
theta = np.arccos(2.0 * rng.random(N) - 1.0)
pos = np.c_[r*np.cos(phi)*np.sin(theta),
            r*np.sin(phi)*np.sin(theta),
            r*np.cos(theta)]

# Hernquist potential (G=M=a=1 normalization):  Phi = -1/(1+r)
pot = -1.0 / (1.0 + r)
v_esc = np.sqrt(-2.0 * pot)

# Hernquist DF — Von Neumann acceptance-rejection for v/v_esc at each radius
# f(E) distribution (integrated over direction) following Binney & Tremaine
def Fq(rv, rr):
    return (2*rr*(1 + rr)**4 * rv**2 *
            ((np.sqrt(-((-1 + rv**2)*(rr + rv**2))) *
              (-1 + rr + 2*rv**2) *
              (-3 - 14*rr - 3*rr**2 + 8*(-1 + rr)*rv**2 + 8*rv**4)) / (1 + rr)**4
             + 3*np.arcsin(np.sqrt((1 - rv**2)/(1 + rr))))
            ) / (np.pi * (rr + rv**2)**2.5)


@jit(nopython=False, forceobj=True)
def von_neumann_sampling(r_arr, Fq_fn):
    out = np.empty_like(r_arr)
    for i in range(len(r_arr)):
        radius = r_arr[i]
        upper = 200.0 if radius < 0.1 else 10.0
        while True:
            Y = upper * np.random.rand()
            X = np.random.rand()
            if Y < Fq_fn(X, radius):
                out[i] = X
                break
    return out


print("Sampling velocities via Von Neumann rejection...")
# Seed numpy so sampling is reproducible too
np.random.seed(42)
q = von_neumann_sampling(r, Fq)
v_mag = v_esc * q
phi_v = rng.random(N) * 2 * np.pi
theta_v = np.arccos(2.0 * rng.random(N) - 1.0)
vel = np.c_[v_mag*np.cos(phi_v)*np.sin(theta_v),
            v_mag*np.sin(phi_v)*np.sin(theta_v),
            v_mag*np.cos(theta_v)]

# Rescale to requested (M, a). Positions scale as a, velocities as sqrt(G*M/a).
boxsize = 10.0 * a  # halo fits comfortably inside
pos = pos * a + boxsize/2
vel = vel * np.sqrt(G * M_tot / a)
pot = pot * M_tot / a

# Pure DM halo — no gas particles. The code must handle TotN_gas==0 cleanly
# so this IC doubles as a no-gas regression case.

# Output
fname = "hernquist_sidm_ics.hdf5"
with h5py.File(fname, "w") as F:
    F.create_group("Header")
    F["Header"].attrs["NumPart_ThisFile"] = [0, N, 0, 0, 0, 0]
    F["Header"].attrs["NumPart_Total"]    = [0, N, 0, 0, 0, 0]
    F["Header"].attrs["MassTable"]        = [0, M_tot/N, 0, 0, 0, 0]
    F["Header"].attrs["BoxSize"]          = boxsize
    F["Header"].attrs["Time"]             = 0.0
    F["Header"].attrs["Redshift"]         = 0.0
    F["Header"].attrs["NumFilesPerSnapshot"] = 1
    F["Header"].attrs["Omega0"]           = 0.0
    F["Header"].attrs["OmegaLambda"]      = 0.0
    F["Header"].attrs["HubbleParam"]      = 1.0
    F["Header"].attrs["Flag_Sfr"]         = 0
    F["Header"].attrs["Flag_Cooling"]     = 0
    F["Header"].attrs["Flag_StellarAge"]  = 0
    F["Header"].attrs["Flag_Metals"]      = 0
    F["Header"].attrs["Flag_Feedback"]    = 0
    F["Header"].attrs["Flag_DoublePrecision"] = 0

    g = F.create_group("PartType1")
    g.create_dataset("Coordinates",  data=pos.astype(np.float32))
    g.create_dataset("Velocities",   data=vel.astype(np.float32))
    g.create_dataset("Masses",       data=np.full(N, M_tot/N, dtype=np.float32))
    g.create_dataset("ParticleIDs",  data=np.arange(1, N+1, dtype=np.uint32))
    g.create_dataset("Potential",    data=pot.astype(np.float32))

print(f"Wrote {fname}: {N} Type=1 DM particles, M={M_tot}, a={a}, box={boxsize}")
