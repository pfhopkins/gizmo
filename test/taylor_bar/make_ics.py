"""IC for test/taylor_bar: the Taylor anvil test as the Abaqus "impact of a copper rod" benchmark sets it up (rod length
32.4 mm, radius 3.2 mm, 227 m/s onto a rigid frictionless wall; on this lattice the rod is 40 layers, 32.0 mm, long; copper, density 8.97 g/cc, bulk modulus 9.1667e11 and
shear modulus 4.2308e11 dyn/cm^2, yield strength 3.14e9 dyn/cm^2, perfectly plastic). The wall is modelled by symmetry:
two identical rods, mirror images in the plane x = 10, approach each other at +-227 m/s, starting a gap of 3 lattice
spacings apart. Each rod is a cubic lattice with R/dp = 4 (2080 cells per rod) whose positions are jittered by up to 2%
of the spacing (fixed seed), so the run does not rely on a perfectly aligned lattice.
(R/dp = 4 gives 2080 cells per rod: 52 per cross-section by 40 layers.)

  python3 make_ics.py                 writes taylor_bar_ics.hdf5 with masses rho_0 dp^3
  python3 make_ics.py --calibrate     rescales the masses of taylor_bar_ics.hdf5 by rho_0 / rho, with rho each cell's
                                      kernel density in output/snapshot_000.hdf5 (a run to TimeMax ~0), and prints the
                                      largest relative density error before the rescale

The calibration matters: with uniform masses the kernel density at the free surface falls up to ~20% below rho_0 (at
the edges and corners), so the rod starts with spurious surface tension and stress. A few passes bring every cell to
rho_0 within 0.3%."""
import sys
import numpy as np
import h5py

L_ROD, R_ROD, V_IMPACT, RHO_0 = 3.24, 0.32, 2.27e4, 8.97    # cm, cm, cm/s, g/cc
R_OVER_DP, JITTER, SEED, BOX = 4, 0.02, 12345, 20.0
U_INIT = 1.16e9                                           # erg/g: copper at room temperature (c_v T, 0.385 J/g/K x 300 K)
DP = R_ROD / R_OVER_DP
GAP = 3 * DP
CENTER = 0.5 * BOX


def lattice():
    nx = int(round(L_ROD / DP))
    g1 = (np.arange(-R_OVER_DP - 1, R_OVER_DP + 1) + 0.5) * DP
    Y, Z = np.meshgrid(g1, g1, indexing="ij")
    keep = Y**2 + Z**2 < R_ROD**2
    yz = np.c_[Y[keep], Z[keep]]
    xs = GAP / 2 + (np.arange(nx) + 0.5) * DP
    rod_b = np.array([[x, y, z] for x in xs for (y, z) in yz])     # the rod at x > CENTER, moving in -x
    rod_a = rod_b * np.array([-1, 1, 1])
    pos = np.vstack([rod_a, rod_b])
    pos += JITTER * DP * np.random.default_rng(SEED).uniform(-1, 1, pos.shape)
    vel = np.zeros_like(pos)
    vel[:len(rod_a), 0] = +V_IMPACT
    vel[len(rod_a):, 0] = -V_IMPACT
    return pos + CENTER, vel


def write(path="taylor_bar_ics.hdf5"):
    pos, vel = lattice()
    N = len(pos)
    with h5py.File(path, "w") as f:
        h = f.create_group("Header")
        for key, val in [("NumPart_ThisFile", np.array([N, 0, 0, 0, 0, 0], np.int32)), ("NumPart_Total", np.array([N, 0, 0, 0, 0, 0], np.uint32)),
                         ("NumPart_Total_HighWord", np.zeros(6, np.uint32)), ("MassTable", np.zeros(6)), ("Time", 0.0), ("Redshift", 0.0),
                         ("BoxSize", BOX), ("NumFilesPerSnapshot", 1), ("Omega0", 0.0), ("OmegaLambda", 0.0), ("HubbleParam", 1.0),
                         ("Flag_DoublePrecision", 1)]:
            h.attrs[key] = val
        for key in ["Flag_Sfr", "Flag_Feedback", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals"]:
            h.attrs[key] = 0
        g = f.create_group("PartType0")
        g["Coordinates"] = pos
        g["Velocities"] = vel
        g["Masses"] = np.full(N, RHO_0 * DP**3)
        g["InternalEnergy"] = np.full(N, U_INIT)
        g["ParticleIDs"] = np.arange(1, N + 1, dtype=np.uint32)
        g["CompositionType"] = np.zeros(N, np.int32)


def calibrate(ic_path="taylor_bar_ics.hdf5", snapshot="output/snapshot_000.hdf5"):
    """One calibration pass; returns the largest |rho/rho_0 - 1| before the rescale."""
    with h5py.File(snapshot, "r") as s:
        rho = dict(zip(s["PartType0/ParticleIDs"][:].tolist(), s["PartType0/Density"][:].tolist()))
    with h5py.File(ic_path, "r+") as f:
        g = f["PartType0"]
        r = np.array([rho[i] for i in g["ParticleIDs"][:].tolist()])
        g["Masses"][...] = g["Masses"][:] * RHO_0 / r
    return np.abs(r / RHO_0 - 1).max()


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        print(f"{calibrate():.3e}")
    else:
        write()
