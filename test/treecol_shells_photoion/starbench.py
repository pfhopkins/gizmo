"""Grading of a StarBench D-type run (as test/photoion_starbench) against the combined StarBench solution (Bisbas et al. 2015, eqs 8, 11, 28-29),
on references measured in the run: ambient n_H, ionised-gas temperature and the ionised and neutral sound speeds (the
gas holds helium, so the pure-hydrogen c_i = 12.85 km/s does not apply). As stella_dev's pi_set grader."""
import glob
import os
import numpy as np
import h5py

UL, UM, XH, MP, PC, MYR, Q = 3.08568e21, 1.989e43, 0.7065, 1.6726e-24, 3.086e18, 3.15576e13, 1e49
UT = UL / 1e5


def load(snap):
    with h5py.File(snap, "r") as f:
        g = f["PartType0"]
        d = dict(x=g["SGChemTracAbund"][()][:, 1], pos=g["Coordinates"][()], m=g["Masses"][()], rho=g["Density"][()].astype(np.float64),
                 u=g["InternalEnergy"][()], T=g["Temperature"][()] if "Temperature" in g else None)
        box, t = float(f["Header"].attrs["BoxSize"]), float(f["Header"].attrs["Time"])
    d["rn"] = np.linalg.norm(d["pos"] - box / 2, axis=1) * UL; d["t"] = t * UT; d["V"] = d["m"] / d["rho"]
    return d


def raga(t, ref):
    """Raga-I (with ambient back-pressure) and Raga-II (shell inertia and pressure) radii [cm] at times t [s]"""
    rs = (3 * Q / (4 * np.pi * 2.6e-13 * (ref["T_ii"] / 1e4) ** -0.7 * ref["n_amb"] ** 2)) ** (1 / 3)
    ci, co = ref["c_i_kms"] * 1e5, ref["c_o_kms"] * 1e5
    beta, n = (co / ci) ** 2, 4000
    tg = np.linspace(0, t.max(), n + 1); dt = tg[1]
    R1 = np.empty(n + 1); R1[0] = rs
    R2 = np.empty(n + 1); V2 = np.empty(n + 1); R2[0] = rs; V2[0] = np.sqrt(4 / 3) * ci
    for i in range(n):
        x = R1[i] / rs
        R1[i + 1] = R1[i] + dt * ci * max(x ** -0.75 - beta * x ** 0.75, 0.0)
        V2[i + 1] = V2[i] + dt * (3 * rs ** 1.5 * ci ** 2 / R2[i] ** 2.5 - 3 * co ** 2 / R2[i] - 3 / R2[i] * V2[i] ** 2)
        R2[i + 1] = max(R2[i] + dt * V2[i + 1], 0.5 * rs)
    return np.interp(t, tg, R1), np.interp(t, tg, R2)


def grade(outdir, t_max_myr=99.0):
    """times [s], ionisation-front radii [cm], StarBench radii [cm], references; and the D-type median error (t > 0.2 Myr)"""
    snaps = [s for s in sorted(glob.glob(os.path.join(outdir, "snapshot_*.hdf5"))) if os.path.getsize(s) > 10000]
    snaps = [s for s in snaps if "SGChemTracAbund" in h5py.File(s, "r")["PartType0"]]
    ts, rs, d0 = [], [], None
    for s in snaps:
        d = load(s); d0 = d0 or d
        rmax = 0.75 * d["rn"].max(); far = d["rn"] > 0.75 * rmax
        x0 = np.median(d["x"][far]) if far.sum() > 10 else 0.0
        xe = np.clip(d["x"] - x0, 0, 1); ins = d["rn"] < rmax
        ts.append(d["t"]); rs.append((3 * np.sum(xe[ins] * d["V"][ins]) * UL ** 3 / (4 * np.pi)) ** (1 / 3))
    ts, rs = np.array(ts), np.array(rs)
    nH = d0["rho"] * UM / UL ** 3 * XH / MP; amb = d0["rn"] / PC > 10
    ref = {"n_amb": float(np.sum(nH[amb] * d0["V"][amb]) / np.sum(d0["V"][amb]))}
    dm = load(snaps[len(snaps) // 2]); ion = dm["x"] > 0.9; V = dm["V"][ion]
    ref["T_ii"] = float(np.median(dm["T"][ion])) if dm["T"] is not None else 1e4
    ref["c_i_kms"] = float(np.sqrt(2 / 3 * np.sum(dm["u"][ion] * V) / np.sum(V) * 1e10) / 1e5)
    neu = (dm["x"] < 0.05) & (dm["rn"] > 0.55 * dm["rn"].max()); Vn = dm["V"][neu]
    ref["c_o_kms"] = float(np.sqrt(2 / 3 * np.sum(dm["u"][neu] * Vn) / np.sum(Vn) * 1e10) / 1e5)
    R1, R2 = raga(ts, ref)
    sb = R2 + (1 - 0.733 * np.exp(-ts / MYR)) * (R1 - R2)
    w = (ts > 0.2 * MYR) & (ts < t_max_myr * MYR)
    return ts, rs, sb, ref, float(np.median(rs[w] / sb[w] - 1))


def photon_ledger(snap):
    """recombinations per ionising photon: sum alpha_B x^2 n_H^2 V / Q"""
    d = load(snap); n = d["rho"] * UM / UL ** 3 * XH / MP
    return d["t"], float(np.sum(2.6e-13 * d["x"] ** 2 * n ** 2 * d["V"] * UL ** 3) / Q)


ABHE = 0.1
ALPHA_HE = 2.7e-13


def load_helium(snap):
    """net 17: H+, He+, He++ per H nucleus and the cell's n_H, volume"""
    d = load(snap)
    with h5py.File(snap, "r") as f:
        a = f["PartType0"]["SGChemTracAbund"][()]
    d["xhe"], d["xhepp"] = a[:, 3], a[:, 4]
    d["n"] = d["rho"] * UM / UL ** 3 * XH / MP
    return d


def band_ledger(snap, q1_over_q):
    """recombinations per photon at Q = 1e49: all gas / Q, helium / Q1; alpha at 1e4 K, case B"""
    d = load_helium(snap); ne = (d["x"] + d["xhe"] + 2 * d["xhepp"]) * d["n"]; V = d["V"] * UL ** 3
    r_h, r_he = np.sum(2.6e-13 * d["x"] * d["n"] * ne * V), np.sum(ALPHA_HE * d["xhe"] * d["n"] * ne * V)
    return dict(t_myr=d["t"] / MYR, total=float((r_h + r_he) / Q), helium=float(r_he / (q1_over_q * Q)))


def helium_front(snap):
    """R_He / R_IF by the volume integrals of x(He+) / ABHE and x(H+); mean x(He+) / ABHE where x(H+) > 0.9; mean x(He++) / ABHE"""
    d = load_helium(snap); ion = d["x"] > 0.9; V = d["V"]
    far = d["rn"] > 0.75 * 0.75 * d["rn"].max(); ins = d["rn"] < 0.75 * d["rn"].max()
    he = np.clip(d["xhe"] / ABHE - np.median(d["xhe"][far]) / ABHE, 0, 1); h = np.clip(d["x"] - np.median(d["x"][far]), 0, 1)
    r = (np.sum((he * V)[ins]) / np.sum((h * V)[ins])) ** (1 / 3) if np.sum((h * V)[ins]) > 0 else 0.0
    return dict(R_he_over_R_if=float(r), x_he_in_hii=float(np.sum((d["xhe"] * V)[ion]) / max(np.sum(V[ion]), 1e-99) / ABHE) if ion.any() else 0.0,
                x_hepp=float(np.sum((d["xhepp"] * V)[ion]) / max(np.sum(V[ion]), 1e-99) / ABHE) if ion.any() else 0.0)

