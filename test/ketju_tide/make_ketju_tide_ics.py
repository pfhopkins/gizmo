#!/usr/bin/env python3
"""ICs for test/ketju_tide (kpc, km/s, 1e10 Msun): a 1 + 1 Msun pair (a = 200 AU, e = 0.5, P ~ 2000 yr) in a KETJU region
and a 20 Msun star on a circular orbit 5000 AU from it (P_out ~ 0.07 Myr), whose tide reaches the pair. Centred on the
CoM at the origin."""
import h5py
import numpy as np

AU = 1.0 / 206264.806 / 1e3  # kpc
G = 4.300710573e-6  # kpc (km/s)^2 / Msun


def write(name, m, x, v):
    with h5py.File(name, "w") as f:
        h = f.create_group("Header"); n = len(m)
        h.attrs["NumPart_ThisFile"] = [0, 0, 0, 0, n, 0]; h.attrs["NumPart_Total"] = [0, 0, 0, 0, n, 0]
        h.attrs["NumPart_Total_HighWord"] = [0] * 6; h.attrs["MassTable"] = [0.0] * 6
        h.attrs["Time"] = 0.0; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = 1.0; h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0; h.attrs["OmegaLambda"] = 0.0; h.attrs["HubbleParam"] = 1.0
        for k in ("Flag_Sfr", "Flag_Cooling", "Flag_StellarAge", "Flag_Metals", "Flag_Feedback"): h.attrs[k] = 0
        g = f.create_group("PartType4")
        g["Coordinates"] = np.array(x) * AU; g["Velocities"] = np.array(v); g["Masses"] = np.array(m) / 1e10
        g["ParticleIDs"] = np.arange(1, n + 1, dtype=np.uint64)


def kepler_state(m1, m2, a, e, f):
    mu = G * (m1 + m2); p = a * (1 - e * e); r = p / (1 + e * np.cos(f))
    x = r * np.array([np.cos(f), np.sin(f), 0]); v = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0])
    M = m1 + m2; return [-m2 / M * x, m1 / M * x], [-m2 / M * v, m1 / M * v]

def state():
    m = np.array([1.0, 1.0, 20.0]); xs, vs = kepler_state(1, 1, 200 * AU, 0.5, 0.7)
    # tilt the inner orbit out of the outer plane so the tide changes e and a
    c, s = np.cos(0.6), np.sin(0.6); Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    xs = [Rx @ x for x in xs]; vs = [Rx @ v for v in vs]
    r3 = 5000 * AU; v3 = np.sqrt(G * m.sum() / r3)
    xo, vo = np.array([r3, 0, 0]), np.array([0, v3, 0]); M = m.sum()
    x = np.array([xx - m[2] / M * xo for xx in xs] + [2.0 / M * xo]); v = np.array([vv - m[2] / M * vo for vv in vs] + [2.0 / M * vo])
    return m, x, v

def main():
    m, x, v = state(); write("ketju_tide_ics.hdf5", m, x / AU, v)

if __name__ == "__main__":
    main()
