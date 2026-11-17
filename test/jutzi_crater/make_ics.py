"""IC generator for the basalt cratering test.

Creates a small 2D (z=0) impact at 3 km/s of a basalt projectile (~20
particles) into a basalt target (~300 particles), both the compact basalt
preset (CompositionType 2).  CGS throughout.  The target starts at rest at its
reference density (the masses are set with the lattice's own kernel density)
and near-zero pressure.  Damage, ActiveCracks and the deviatoric stress start
at zero.

The target is compact: a porous target with basalt's crush pressure (P_e = 1e8
dyn/cm^2) crushes at its own free surface before contact, from the remaining
free-surface density error of the kernel estimate; the P-alpha model is
validated separately (test/jutzi_crush_unit for the crush curve).

Both bodies are hexagonal lattices; the projectile starts one kernel radius
above the target, so the two bodies do not interact before contact.

Units: cm, g, s throughout.
"""

import numpy as np
import h5py
import os

RHO_BASALT     = 2.7          # g/cm^3 (reference density of the preset)
V_IMPACT       = 3.0e5        # cm/s (3 km/s downward)
R_TARGET       = 5.0          # cm  target radius
R_IMPACTOR     = 1.2          # cm  impactor radius
COMPOSITION    = 2            # the basalt preset
# Cold basalt: Tillotson at rho=rho0, u≈0 gives P≈0, c_s = sqrt(A/rho0)
# A = 2.67e11 dyne/cm², rho0 = 2.7 g/cm³  → c_s ≈ 3.14e5 cm/s
CS_BASALT      = 3.14e5       # cm/s (approximate cold bulk sound speed)
# A small positive specific energy: its thermal pressure, ~(a+b) rho0 u = 5e6 dyn/cm^2,
# is negligible, and the energy does not start at its floor.
U_COLD         = 1.0e6        # erg/g
DES_NUM_NGB    = 32


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


def lattice_kernel_density_and_radius(dx, des_num_ngb):
    """GIZMO's kernel density on an infinite hex lattice of spacing dx, relative to (mass per lattice area), and the
    kernel radius h: 2D Wendland C2 (KERNEL_FUNCTION=6 in this test's Config), h solving pi h^2 sum_j W_ij = DesNumNgb.
    The lattice kernel density is ~0.4% above m/A (~1 GPa for basalt), so the masses are set with this ratio."""
    dy = dx * np.sqrt(3.0) / 2.0
    pts = np.array([[i * dx + (j % 2) * 0.5 * dx, j * dy] for i in range(-20, 21) for j in range(-20, 21)])
    r = np.linalg.norm(pts - pts[np.argmin(np.linalg.norm(pts, axis=1))], axis=1)
    w = lambda q: np.where(q < 1, (7.0 / np.pi) * (1 - q) ** 4 * (1 + 4 * q), 0.0)
    lo, hi = 0.5 * dx, 20.0 * dx
    for _ in range(100):
        h = 0.5 * (lo + hi)
        if np.pi * np.sum(w(r / h)) < des_num_ngb: lo = h
        else: hi = h
    return dx * dy * np.sum(w(r / h)) / h ** 2, h


def make_ics(outfile="jutzi_crater_ics.hdf5"):
    # Target: disk at origin, at rest
    dx = 0.55          # particle spacing (cm) — gives ~270 target particles
    pos_t = hex_disk(R_TARGET, dx)
    vel_t = np.zeros_like(pos_t)

    # 2D area per particle of a hexagonal lattice; the lattice's kernel density (relative to m / A_part) and radius
    A_part = dx ** 2 * np.sqrt(3.0) / 2.0
    kernel_overdensity, h_kernel = lattice_kernel_density_and_radius(dx, DES_NUM_NGB)

    # Impactor: disk above the target, moving downward; the gap between the two
    # lattices' nearest particles is one kernel radius
    pos_i = hex_disk(R_IMPACTOR, dx)
    pos_i[:, 1] += pos_t[:, 1].max() - pos_i[:, 1].min() + h_kernel
    vel_i = np.zeros_like(pos_i)
    vel_i[:, 1] = -V_IMPACT

    pos = np.vstack([pos_t, pos_i])
    vel = np.vstack([vel_t, vel_i])
    N   = len(pos)

    masses = np.full(N, RHO_BASALT * A_part / kernel_overdensity, dtype=np.float32)   # interior kernel density = RHO_BASALT

    rho = np.full(N, RHO_BASALT, dtype=np.float32)
    u   = np.full(N, U_COLD, dtype=np.float32)
    P   = np.zeros(N, dtype=np.float32)
    cs  = np.full(N, CS_BASALT, dtype=np.float32)
    comp = np.full(N, COMPOSITION, dtype=np.uint32)

    h = np.full(N, h_kernel, dtype=np.float32)

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
