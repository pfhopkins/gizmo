"""ICs for the MHD double rarefaction: a 2D periodic strip [0, 4] x [0, 0.25] on a square lattice,
gas receding from x = 2 at -/+v0 with a uniform normal field Bx (and the inverted, colliding image of
the jump at x = 0 = 4). Particle mass is proportional to density; all fields are in code units with
magnetic pressure B^2/2 (MU_0 = 1)."""

import h5py
import numpy as np

# rho, vx, vy, vz, By, Bz, P on the left; the right state has vx -> -vx
CASES = {
    "near_vacuum": dict(v0=2.0, t_end=0.3),  # density at the centre drops to ~1% of the initial
    "vacuum": dict(v0=4.0, t_end=0.2),       # density at the centre drops to ~3e-4
}
RHO, P, BX, BY, GAMMA = 1.0, 0.45, 0.75, 0.5, 5.0 / 3.0


def left_right(case):
    v0 = CASES[case]["v0"]
    left = (RHO, -v0, 0.0, 0.0, BY, 0.0, P)
    return left, (RHO, v0, 0.0, 0.0, BY, 0.0, P)


def make_mhd_double_rarefaction_ics(path, case, nx=896):
    ny = nx // 16
    dx = 4.0 / nx
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    left, right = left_right(case)
    W = np.where((X < 2.0)[:, None], np.array(left)[None, :], np.array(right)[None, :])
    rho, vx, vy, vz, By, Bz, P_ = W.T
    N = len(X)
    with h5py.File(path, "w") as F:
        h = F.create_group("Header")
        h.attrs.update(dict(NumPart_ThisFile=np.array([N, 0, 0, 0, 0, 0], dtype=np.int32),
                            NumPart_Total=np.array([N, 0, 0, 0, 0, 0], dtype=np.uint32),
                            NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32), MassTable=np.zeros(6), Time=0.0,
                            Redshift=0.0, BoxSize=0.25, NumFilesPerSnapshot=1, Omega0=0.0, OmegaLambda=0.0,
                            HubbleParam=1.0, Flag_Sfr=0, Flag_Cooling=0, Flag_StellarAge=0, Flag_Metals=0,
                            Flag_Feedback=0, Flag_DoublePrecision=1))
        g = F.create_group("PartType0")
        g["Coordinates"] = np.c_[X, Y, np.zeros(N)]
        g["Velocities"] = np.c_[vx, vy, vz]
        g["MagneticField"] = np.c_[BX * np.ones(N), By, Bz]
        g["Masses"] = rho * dx * dx
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        g["InternalEnergy"] = P_ / ((GAMMA - 1) * rho)
