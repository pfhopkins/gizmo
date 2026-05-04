"""IC generator for the Jutzi 2008 P-alpha cratering smoke test.

Creates a small 2D (z=0) basalt-on-basalt impact at 3 km/s with ~250 target
particles + ~25 impactor particles.  CGS throughout.

CompositionType = 2  (MATERIAL_TILLOTSON_BASALT in composition_registry.h).
Distention / Damage / ActiveCracks are NOT written; init.cc seeds them from
the material table (Distention = alpha_0 = 1.275 for basalt, Damage = 0,
ActiveCracks = 0).

Units: cm, g, s throughout.
"""

import numpy as np
import h5py
import os

RHO_BASALT     = 2.7          # g/cm^3
V_IMPACT       = 3.0e5        # cm/s (3 km/s downward)
R_TARGET       = 5.0          # cm  target radius
R_IMPACTOR     = 1.2          # cm  impactor radius
COMPOSITION    = 2            # basalt
# Cold basalt: Tillotson at rho=rho0, u≈0 gives P≈0, c_s = sqrt(A/rho0)
# A = 2.67e11 dyne/cm², rho0 = 2.7 g/cm³  → c_s ≈ 3.14e5 cm/s
CS_BASALT      = 3.14e5       # cm/s (approximate cold bulk sound speed)
U_COLD         = 0.0          # internal energy (cold state)


def hex_disk(R, dx):
    """2D hexagonally-packed disk of radius R with particle spacing dx."""
    dy = dx * np.sqrt(3.0) / 2.0
    pts = []
    ny = int(R / dy) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        offset = (j % 2) * 0.5 * dx
        nx = int(np.sqrt(max(R * R - y * y, 0.0)) / dx) + 2
        for i in range(-nx, nx + 1):
            x = i * dx + offset
            if x * x + y * y <= R * R:
                pts.append([x, y, 0.0])
    return np.array(pts, dtype=np.float64)


def make_ics(outfile="jutzi_crater_ics.hdf5"):
    # Target: disk at origin, at rest
    dx = 0.55          # particle spacing (cm) — gives ~270 target particles
    pos_t = hex_disk(R_TARGET, dx)
    vel_t = np.zeros_like(pos_t)

    # Impactor: disk centred just above target, moving downward
    pos_i = hex_disk(R_IMPACTOR, dx)
    y_offset = R_TARGET + R_IMPACTOR + 1.2 * dx   # start just above
    pos_i[:, 1] += y_offset
    vel_i = np.zeros_like(pos_i)
    vel_i[:, 1] = -V_IMPACT

    pos = np.vstack([pos_t, pos_i])
    vel = np.vstack([vel_t, vel_i])
    N   = len(pos)

    # Particle mass from volume element
    A_part   = dx ** 2                 # 2D "area" per particle (approximate)
    mass_one = RHO_BASALT * A_part     # g
    masses   = np.full(N, mass_one, dtype=np.float32)

    rho = np.full(N, RHO_BASALT, dtype=np.float32)
    u   = np.full(N, U_COLD, dtype=np.float32)
    P   = np.zeros(N, dtype=np.float32)
    cs  = np.full(N, CS_BASALT, dtype=np.float32)
    comp = np.full(N, COMPOSITION, dtype=np.uint32)

    # SmoothingLength: from 2D neighbour sphere: pi*h^2 * rho = Nngb * m
    Nngb = 32
    h = np.sqrt(Nngb * mass_one / (np.pi * RHO_BASALT)) * np.ones(N, dtype=np.float32)

    stress = np.zeros((N, 9), dtype=np.float32)   # zero initial stress

    ids = np.arange(1, N + 1, dtype=np.uint32)
    child_ids    = np.zeros(N, dtype=np.uint32)
    gen_ids      = np.zeros(N, dtype=np.uint32)

    with h5py.File(outfile, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["NumPart_ThisFile"]      = np.array([N,0,0,0,0,0], dtype=np.int32)
        hdr.attrs["NumPart_Total"]         = np.array([N,0,0,0,0,0], dtype=np.uint32)
        hdr.attrs["NumPart_Total_HighWord"]= np.array([0,0,0,0,0,0], dtype=np.uint32)
        hdr.attrs["MassTable"]             = np.zeros(6, dtype=np.float64)
        hdr.attrs["Time"]                  = np.float64(0.0)
        hdr.attrs["Redshift"]              = np.float64(0.0)
        hdr.attrs["BoxSize"]               = np.float64(0.0)
        hdr.attrs["NumFilesPerSnapshot"]   = np.int32(1)
        hdr.attrs["Omega0"]                = np.float64(1.0)
        hdr.attrs["OmegaLambda"]           = np.float64(0.0)
        hdr.attrs["HubbleParam"]           = np.float64(1.0)
        hdr.attrs["Flag_Sfr"]              = np.int32(0)
        hdr.attrs["Flag_Feedback"]         = np.int32(0)
        hdr.attrs["Flag_Cooling"]          = np.int32(0)
        hdr.attrs["Flag_StellarAge"]       = np.int32(0)
        hdr.attrs["Flag_Metals"]           = np.int32(0)
        hdr.attrs["Flag_DoublePrecision"]  = np.int32(0)
        hdr.attrs["Flag_IC_Info"]          = np.int32(0)
        hdr.attrs["Flag_AgeTracers"]       = np.int32(0)

        g = f.create_group("PartType0")
        g.create_dataset("Coordinates",              data=pos.astype(np.float64))
        g.create_dataset("Velocities",               data=vel.astype(np.float64))
        g.create_dataset("Masses",                   data=masses)
        g.create_dataset("Density",                  data=rho)
        g.create_dataset("InternalEnergy",           data=u)
        g.create_dataset("Pressure",                 data=P)
        g.create_dataset("SoundSpeed",               data=cs)
        g.create_dataset("SmoothingLength",          data=h)
        g.create_dataset("StressTensor",             data=stress)
        g.create_dataset("CompositionType",          data=comp)
        g.create_dataset("ParticleIDs",              data=ids)
        g.create_dataset("ParticleChildIDsNumber",   data=child_ids)
        g.create_dataset("ParticleIDGenerationNumber", data=gen_ids)

    print(f"Written {N} particles ({len(pos_t)} target + {len(pos_i)} impactor) to {outfile}")
    return N


if __name__ == "__main__":
    make_ics()
