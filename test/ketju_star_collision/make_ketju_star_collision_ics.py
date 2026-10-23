#!/usr/bin/env python3
"""ICs for test/ketju_star_collision (kpc, km/s, 1e10 Msun). A 1 + 0.5 Msun pair 30 AU apart approaching on a hyperbolic
orbit (v_inf = 5 km/s) with pericentre 0 (head-on), 0.8 (R1 + R2) (graze, with a 0.7 Msun third star on a 600 AU orbit
around the pair, so MSTAR goes on with two bodies after the merger) or 1.25 (R1 + R2) (miss). A 0.3 Msun bystander sits
at 0.97 pc. Radii: R/Rsun = M^0.8 below 1 Msun, M^0.57 above (the code's fallback without Stella tables). Centred on
the origin: GIZMO reads positions in single precision."""
import numpy as np, h5py

AU = 1.0 / 206264.806 / 1e3          # kpc
G = 4.300710573e-6                   # kpc (km/s)^2 / Msun, as GIZMO (GRAVITY_G_CGS = 6.672e-8)
RSUN_AU = 6.957e10 / 1.495978707e13
M1, M2, M3 = 1.0, 0.5, 0.7


def radius_au(m):
    return RSUN_AU * (m ** 0.8 if m < 1 else m ** 0.57)


R_CONTACT = radius_au(M1) + radius_au(M2)
CASES = {"headon": 0.0, "graze": 0.8, "miss": 1.25}   # pericentre / (R1 + R2)


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


def pair(rp_au, r0_au=30.0, vinf=5.0):
    mu = G * (M1 + M2); a = -mu / vinf ** 2; e = 1 - rp_au * AU / a
    h = np.sqrt(mu * a * (1 - e * e)); v0 = np.sqrt(vinf ** 2 + 2 * mu / (r0_au * AU))
    vt = h / (r0_au * AU); vr = -np.sqrt(max(v0 ** 2 - vt ** 2, 0.0))
    x = np.array([r0_au, 0, 0]); v = np.array([vr, vt, 0]); M = M1 + M2
    return [-M2 / M * x, M1 / M * x], [-M2 / M * v, M1 / M * v]


def main():
    for case, f in CASES.items():
        x, v = pair(f * R_CONTACT); m = [M1, M2]
        if case != "headon":
            r3 = 600.0; v3 = np.sqrt(G * (M1 + M2 + M3) / (r3 * AU)); M = M1 + M2 + M3; m.append(M3)
            x = [xx - M3 / M * np.array([0, 0, r3]) for xx in x] + [(M1 + M2) / M * np.array([0, 0, r3])]
            v = [vv - M3 / M * np.array([v3, 0, 0]) for vv in v] + [(M1 + M2) / M * np.array([v3, 0, 0])]
        m.append(0.3); x.append(np.array([2e5, 0, 0])); v.append(np.zeros(3))
        write("ketju_star_collision_ics.hdf5" if case == "headon" else f"ketju_star_collision_{case}_ics.hdf5", m, x, v)


if __name__ == "__main__":
    main()
