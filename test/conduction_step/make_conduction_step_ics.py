"""Initial conditions for the conduction_step test.

2D periodic box of side BOXSIZE, uniform density RHO0 on a square N_SIDE x N_SIDE lattice (z = 0), at rest,
with a single sinusoidal internal-energy mode along x:

    u(x) = U0 * (1 + AMP * sin(2 pi x / BOXSIZE))

With theta_deg given, a uniform in-plane magnetic field of magnitude B0 at angle theta_deg to the x axis
(the direction of grad u) is added; otherwise no MagneticField dataset is written.
"""
import argparse

import h5py
import numpy as np

N_SIDE = 64
BOXSIZE = 1.0
RHO0 = 1.0
U0 = 1.0
AMP = 0.2
B0 = 0.1  # only the direction matters: FREEZE_HYDRO keeps B fixed and conduction uses only bhat


def lattice_positions(n_side=N_SIDE, boxsize=BOXSIZE):
    """Cell-centred square lattice, so sum_i sin(k x_i) = 0 and sum_i sin^2(k x_i) = N/2 exactly."""
    g = (np.arange(n_side) + 0.5) * (boxsize / n_side)
    x, y = np.meshgrid(g, g, indexing="ij")
    return np.c_[x.ravel(), y.ravel(), np.zeros(n_side * n_side)]


def make_conduction_step_ics(outfile, theta_deg=None, n_side=N_SIDE, boxsize=BOXSIZE, rho0=RHO0, u0=U0, amp=AMP, b0=B0):
    pos = lattice_positions(n_side, boxsize)
    n = len(pos)
    k = 2 * np.pi / boxsize
    u = u0 * (1 + amp * np.sin(k * pos[:, 0]))
    m = rho0 * boxsize**2 / n
    with h5py.File(outfile, "w") as f:
        h = f.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total"] = [n, 0, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"] = [0.0, 0, 0, 0, 0, 0]
        h.attrs["BoxSize"] = boxsize
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Flag_DoublePrecision"] = 1
        g = f.create_group("PartType0")
        g.create_dataset("Coordinates", data=pos)
        g.create_dataset("Velocities", data=np.zeros((n, 3)))
        g.create_dataset("Masses", data=np.full(n, m))
        g.create_dataset("ParticleIDs", data=np.arange(1, n + 1, dtype=np.uint32))
        g.create_dataset("InternalEnergy", data=u)
        if theta_deg is not None:
            th = np.radians(theta_deg)
            # exact zero for the 90 degree case: a cos(pi/2) ~ 6e-17 x-component would leak conduction along x
            bx = 0.0 if theta_deg == 90 else b0 * np.cos(th)
            bvec = np.array([bx, b0 * np.sin(th), 0.0])
            g.create_dataset("MagneticField", data=np.tile(bvec, (n, 1)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("outfile")
    p.add_argument("--theta", type=float, default=None, help="B angle to x in degrees; omit for no B field")
    p.add_argument("--n_side", type=int, default=N_SIDE)
    a = p.parse_args()
    make_conduction_step_ics(a.outfile, a.theta, a.n_side)
