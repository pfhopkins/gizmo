#!/usr/bin/env python3
"""Time series of stratified shearing-box runs: gas/stellar surface densities, SFR surface density, gas scale height,
mid-plane pressures vs the weight of the gas, velocity dispersion, mid-plane phase volume fractions, and the residual of
v_y from the background shear. Prints a table per run and plots all runs together.

    python tallbox_diagnostics.py <run>/output [<run2>/output ...] [--labels a,b] [-o diagnostics.png]
"""

import argparse
import glob
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MYR = 0.9778          # code time unit in Myr (pc / (km/s))
G = 4.30092e-3        # pc (km/s)^2 / Msun
NH = 1.989e33 / (3.0857e18) ** 3 * 0.70 / 1.6726e-24  # code density -> n_H
KB = 1.380649e-16
PK = 1.989e33 / (3.0857e18) ** 3 * 1e10 / KB  # code pressure (rho v^2) -> P/k [K cm^-3]
PHASES = (("cold", 0, 5e3), ("warm", 5e3, 2e4), ("ionized", 2e4, 5e5), ("hot", 5e5, np.inf))
COLS = ["t", "Sgas", "Sstar", "SFR10", "H", "n_mid", "Pth", "Pturb", "W", "sig1d", "vy_res"] + [p[0] for p in PHASES] + ["N", "h_mid"]


def weight(z, m, L, a):
    """W = int rho |g_z| dz over one half of the disk (|z| < 1 kpc): external field plus the gas sheet interior to |z|"""
    gext = 2 * np.pi * G * a.sigma_star * np.abs(z) / np.sqrt(z**2 + a.z_star**2) + 4 * np.pi * G * a.rho_dm * np.abs(z) / (1 + z**2 / a.R0**2)
    o = np.argsort(np.abs(z))
    gself = 2 * np.pi * G * np.cumsum(m[o]) / L**2
    sel = np.abs(z[o]) < 1000
    return (m[o] * (gext[o] + gself))[sel].sum() / (2 * L * L) * PK


def analyze(outdir, a):
    rows = []
    for fn in sorted(glob.glob(f"{outdir}/snapshot_*.hdf5")):
        with h5py.File(fn, "r") as f:
            t, L = f["Header"].attrs["Time"], f["Header"].attrs["BoxSize"]
            g = f["PartType0"]
            x, v, m = g["Coordinates"][:], g["Velocities"][:], g["Masses"][:]
            rho, T, P = g["Density"][:], g["Temperature"][:], g["Pressure"][:]
            hsml = g["KernelMaxRadius"][:]
            mst = f["PartType4/Masses"][:] if "PartType4" in f else np.array([])
            age = (t - f["PartType4/StellarFormationTime"][:]) if "PartType4" in f else np.array([])
        z = x[:, 2] - 0.5 * a.long_z * L
        sfr10 = mst[age < 10 / MYR].sum() / 10e6 / (L / 1e3) ** 2  # Msun/yr/kpc^2 over the last 10 Myr
        H = np.sqrt(np.average(z**2, weights=m))
        mid = np.abs(z) < 0.05 * L
        vol = m / rho
        dv = v.copy(); dv[:, 1] += a.q * a.omega * (x[:, 0] - 0.5 * L)
        dv -= np.average(dv, axis=0, weights=m)
        # mid-plane pressures: volume-weighted means over |z| < 0.05 L (thermal is dominated by hot gas when present)
        Pth = np.average(P[mid], weights=vol[mid]) * PK
        Pturb = np.average((rho * dv[:, 2] ** 2)[mid], weights=vol[mid]) * PK
        disk = np.abs(z) < 3 * H
        sig1d = np.sqrt(np.average((dv[disk] ** 2).sum(axis=1), weights=m[disk]) / 3)
        xb = np.minimum((x[:, 0] / L * 8).astype(int), 7)  # mean v_y residual in 8 radial bins: shear-periodic boundaries
        vy_res = np.abs([np.average(dv[xb == i, 1], weights=m[xb == i]) for i in range(8)]).max()
        fv = [vol[mid & (T >= lo) & (T < hi)].sum() / vol[mid].sum() for _, lo, hi in PHASES]
        rows.append([t * MYR, m.sum() / L**2, mst.sum() / L**2, sfr10, H, (m[mid].sum() / vol[mid].sum()) * NH,
                     Pth, Pturb, weight(z, m, L, a), sig1d, vy_res, *fv, len(m), np.median(hsml[mid])])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--labels", default=None, help="comma-separated, one per run")
    ap.add_argument("-o", "--output", default=None, help="plot path [default: <first outdir>/diagnostics.png]")
    ap.add_argument("--omega", type=float, default=0.028, help="km/s/pc")
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--long_z", type=float, default=8.0)
    ap.add_argument("--sigma_star", type=float, default=42.0)
    ap.add_argument("--z_star", type=float, default=245.0)
    ap.add_argument("--rho_dm", type=float, default=0.0064)
    ap.add_argument("--R0", type=float, default=8000.0)
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else [os.path.basename(os.path.dirname(os.path.abspath(d))) for d in a.outdirs]
    runs = []
    for d, lab in zip(a.outdirs, labels):
        r = analyze(d, a)
        runs.append((lab, r))
        print(f"# {lab}\n" + " ".join(COLS))
        for row in r:
            print(" ".join(f"{v:.4g}" for v in row))
    c = {k: i for i, k in enumerate(COLS)}
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    ax = ax.ravel()
    multi = len(runs) > 1
    for j, (lab, r) in enumerate(runs):
        col = f"C{j}"
        t = r[:, c["t"]]
        ax[0].plot(t, r[:, c["Sgas"]], color=col, label=lab if multi else "gas")
        ax[0].plot(t, r[:, c["Sstar"]], color=col, ls="--", label=None if multi else "new stars")
        ax[1].semilogy(t, np.maximum(r[:, c["SFR10"]], 1e-5), color=col)
        ax[2].plot(t, r[:, c["H"]], color=col)
        ax[3].semilogy(t, r[:, c["Pth"]] + r[:, c["Pturb"]], color=col, label=None if multi else r"$P_{\rm th}+\rho v_z^2$")
        ax[3].semilogy(t, r[:, c["W"]], color=col, ls=":", label=None if multi else r"weight $\mathcal{W}$")
        ax[4].plot(t, r[:, c["sig1d"]], color=col)
        ax[5].semilogy(t, np.maximum(r[:, c["hot"]], 1e-4), color=col, label=None if multi else r"hot ($>5\times10^5$ K)")
        ax[5].semilogy(t, np.maximum(r[:, c["ionized"]], 1e-4), color=col, ls="--", label=None if multi else r"$2\times10^4$-$5\times10^5$ K")
    ax[0].set_ylabel(r"$\Sigma$ [M$_\odot$ pc$^{-2}$]  (dashed: new stars)" if multi else r"$\Sigma$ [M$_\odot$ pc$^{-2}$]")
    ax[1].axhspan(2e-3, 5e-3, color="0.9"); ax[1].set_ylabel(r"$\Sigma_{\rm SFR,10\,Myr}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]")
    ax[2].set_ylabel("gas rms height [pc]")
    ax[3].set_ylabel(r"mid-plane $P/k$ [K cm$^{-3}$]" + ("  (dotted: weight)" if multi else ""))
    ax[4].set_ylabel(r"$\sigma_{\rm 1D}$, $|z|<3H$ [km/s]")
    ax[5].set_ylabel("mid-plane volume fraction" + ("  (dashed: ionized)" if multi else ""))
    ax[5].set_ylim(1e-4, 1)
    for i in (0, 3, 5):
        if ax[i].get_legend_handles_labels()[0]:
            ax[i].legend(frameon=False, fontsize=8)
    for x in ax[3:]:
        x.set_xlabel("t [Myr]")
    fig.tight_layout()
    out = a.output or f"{a.outdirs[0]}/diagnostics.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
