"""Diagnostic analysis + figures for the magnetized rotating core-collapse test.

This module turns a finished `core` run (a directory of snapshot_*.hdf5) into a
set of publication-style diagnostic figures and a printed summary table that
together demonstrate whether the collapse is behaving correctly:

  * central-density runaway and first-core bounce vs the free-fall time
  * exact adherence to the forced barotropic EOS  u(rho)=P/(rho(gamma-1))
  * virial ratio 2K/|W| evolution
  * radial profiles (density, infall velocity, rotation, |B|, timestep)
  * thermodynamic / flux-freezing / resolution phase diagrams
  * conservation of momentum, angular momentum, and core position
  * per-particle timestep correctness against the Courant / accel / self-gravity criteria
  * div.B cleaning error (here with the default Dedner cleaner; CG / MG variants
    should do strictly better)
  * face-on / edge-on density slices with the B field overlaid (meshoid)

It is imported by test_core.py and run automatically at the end of the test, so
any future user of this problem gets the figures for free.  It can also be run
standalone:   python3 core_analysis.py [output_dir] [plot_dir]

Figures follow the Hopkins plotting conventions: single-panel transparent vector
PDFs, white background (black for the 2D images), descriptive axis labels with
units, explicit limits, restored log minor ticks.
"""

import os
import glob
import numpy as np
import h5py

# ----------------------------------------------------------------------------
# physical constants for this test, taken directly from the source + run header
# ----------------------------------------------------------------------------
G_CODE   = 0.00430093        # All.G for (pc, Msun, km/s) units (run stdout)
RHO_C    = 1.47705e8         # barotropic transition density  (eos_functions.h)
GAMMA    = 5.0 / 3.0         # EOS_GAMMA
COURANT  = 0.2               # CourantFac        (core.params)
ETOL     = 0.01              # ErrTolIntAccuracy (core.params)
KCORE    = 0.5               # KERNEL_CORE_SIZE  (cubic spline)
SOFT     = 5e-6              # SofteningGas      (core.params)
B0       = 6.1019e-5         # initial uniform |B| (BiniZ)
CENTER0  = np.array([0.075, 0.075, 0.075])   # geometric box center

# barotropic relations the gas is *forced* onto every step
def u_eos(rho):     return (0.04 / (GAMMA - 1.0)) * np.sqrt(1.0 + (rho / RHO_C) ** (4.0 / 3.0))
def press_eos(rho): return 0.04 * rho * np.sqrt(1.0 + (rho / RHO_C) ** (4.0 / 3.0))
def cs_eos(rho):    return np.sqrt(GAMMA * press_eos(rho) / rho)


# ----------------------------------------------------------------------------
# matplotlib style (Hopkins conventions)
# ----------------------------------------------------------------------------
_USETEX = None   # cached: probe the TeX install exactly once per process


def _setup_mpl():
    global _USETEX
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pylab as pylab
    import matplotlib.ticker  # noqa: F401
    if _USETEX is None:
        _USETEX = True
        try:
            fig = pylab.figure(); fig.text(0.5, 0.5, r"$x$"); fig.savefig(os.devnull); pylab.close(fig)
        except Exception:
            _USETEX = False
    matplotlib.rc("text", usetex=_USETEX)
    if _USETEX:
        matplotlib.rc("text.latex", preamble=r"\usepackage{amsmath}")
    matplotlib.rcParams.update({"font.size": 14})
    pylab.close("all")
    return pylab


def _minor_ticks_x(pylab, n=9):
    import matplotlib.ticker as mt
    ax = pylab.gca()
    ax.xaxis.set_major_locator(mt.LogLocator(base=10, numticks=1000))
    ax.xaxis.set_minor_locator(mt.LogLocator(base=10.0, subs=np.linspace(0, 1, n + 2)[1:-1], numticks=1000))
    ax.xaxis.set_minor_formatter(mt.NullFormatter())


def _minor_ticks_y(pylab, n=9):
    import matplotlib.ticker as mt
    ax = pylab.gca()
    ax.yaxis.set_major_locator(mt.LogLocator(base=10, numticks=1000))
    ax.yaxis.set_minor_locator(mt.LogLocator(base=10.0, subs=np.linspace(0, 1, n + 2)[1:-1], numticks=1000))
    ax.yaxis.set_minor_formatter(mt.NullFormatter())


def _save(pylab, path):
    pylab.subplots_adjust(left=0.16, bottom=0.14, right=0.99, top=0.99)
    pylab.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    pylab.close("all")


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
_FIELDS = dict(pos="Coordinates", vel="Velocities", rho="Density", m="Masses",
               u="InternalEnergy", pot="Potential", h="KernelMaxRadius",
               dt="TimeStep", acc="Acceleration", divB="DivergenceOfMagneticField",
               soft="Softening_KernelRadius")


def load_snapshot(path):
    """Load one snapshot into a dict of arrays (+ derived |B|).  Diagnostic
    fields (Potential, TimeStep, ...) are optional so the loader still works on
    a minimal run; absent fields come back as None."""
    d = {}
    with h5py.File(path, "r") as F:
        d["t"] = float(F["Header"].attrs["Time"])
        g = F["PartType0"]
        for k, name in _FIELDS.items():
            d[k] = g[name][:] if name in g else None
        d["B"] = g["MagneticField"][:] if "MagneticField" in g else None
    d["Bmag"] = np.linalg.norm(d["B"], axis=1) if d["B"] is not None else None
    return d


def free_fall_time(rho0):
    return np.sqrt(3 * np.pi / (32 * G_CODE * rho0))


# ----------------------------------------------------------------------------
# global time-series diagnostics + table
# ----------------------------------------------------------------------------
def compute_global_diagnostics(snaps, rho0=None):
    """Return a dict of time-series arrays over the list of snapshot files."""
    if rho0 is None:
        d0 = load_snapshot(snaps[0]); rho0 = np.median(d0["rho"][d0["rho"] > 1e4])
    tff = free_fall_time(rho0)
    out = {k: [] for k in
           ("t", "rho_max", "eos_max", "vir", "KE", "U", "W", "Emag",
            "dt_min", "dt_dense", "h_dense_min", "Bc99", "divBerr_med", "cen_drift",
            "Pmag", "Lz", "n_above_crit")}
    for s in snaps:
        d = load_snapshot(s)
        rho, m, vel, pos = d["rho"], d["m"], d["vel"], d["pos"]
        cloud = rho > 1e4
        ic = np.argmin(d["pot"]) if d["pot"] is not None else np.argmax(rho)
        vcom = (m[:, None] * vel).sum(0) / m.sum(); vrel = vel - vcom
        KE = 0.5 * np.sum(m * np.sum(vrel ** 2, 1)); U = np.sum(m * d["u"])
        W = 0.5 * np.sum(m * d["pot"]) if d["pot"] is not None else np.nan
        Emag = np.sum(0.5 * d["Bmag"] ** 2 * (m / rho)) if d["Bmag"] is not None else np.nan
        out["t"].append(d["t"])
        out["rho_max"].append(rho.max())
        out["eos_max"].append((np.abs(d["u"] - u_eos(rho)) / u_eos(rho)).max())
        out["vir"].append(2 * KE / abs(W) if np.isfinite(W) else np.nan)
        out["KE"].append(KE); out["U"].append(U); out["W"].append(W); out["Emag"].append(Emag)
        out["dt_min"].append(d["dt"].min() if d["dt"] is not None else np.nan)
        out["dt_dense"].append(d["dt"][np.argmax(rho)] if d["dt"] is not None else np.nan)
        dense01 = rho > np.percentile(rho, 99.9)
        out["h_dense_min"].append(d["h"][dense01].min() if d["h"] is not None else np.nan)
        if d["Bmag"] is not None:
            out["Bc99"].append(np.percentile(d["Bmag"][cloud], 99))
            err = (d["h"] * np.abs(d["divB"]) / (d["Bmag"] + 1e-30))[cloud] if d["divB"] is not None else np.array([np.nan])
            out["divBerr_med"].append(np.median(err))
        else:
            out["Bc99"].append(np.nan); out["divBerr_med"].append(np.nan)
        r0 = pos - CENTER0
        out["cen_drift"].append(np.linalg.norm(pos[ic] - CENTER0))
        out["Pmag"].append(np.linalg.norm((m[:, None] * vel).sum(0)))
        out["Lz"].append(np.sum(m * (r0[:, 0] * vel[:, 1] - r0[:, 1] * vel[:, 0])))
        out["n_above_crit"].append(int((rho > RHO_C).sum()))
    for k in out:
        out[k] = np.array(out[k])
    out["tff"] = tff; out["rho0"] = rho0
    return out


def print_diagnostics_table(diag):
    tff = diag["tff"]; Lz0 = diag["Lz"][0]; rho0 = diag["rho0"]
    print(f"\n  magnetized core-collapse diagnostics   t_ff={tff:.5f}   "
          f"rho0={rho0:.3e}   rho_crit/rho0={RHO_C/rho0:.0f}")
    hdr = ("  t/tff   rho_max/rho0  EOSdev    2K/|W|   dt_min    Bcl99/B0  "
           "divBerr  cen_drift  dLz/Lz0   N>rhoc")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i in range(len(diag["t"])):
        print(f"  {diag['t'][i]/tff:5.2f}  {diag['rho_max'][i]/rho0:10.2e}  "
              f"{diag['eos_max'][i]:7.1e}  {diag['vir'][i]:6.3f}  "
              f"{diag['dt_min'][i]:8.2e}  {diag['Bc99'][i]/B0:7.2f}  "
              f"{diag['divBerr_med'][i]:7.1e}  {diag['cen_drift'][i]:.2e}  "
              f"{(diag['Lz'][i]-Lz0)/Lz0:+.2e}  {diag['n_above_crit'][i]:5d}")


# ----------------------------------------------------------------------------
# individual Hopkins-style figures (one PDF each)
# ----------------------------------------------------------------------------
def fig_central_density(diag, path):
    pylab = _setup_mpl(); tff = diag["tff"]; rho0 = diag["rho0"]
    pylab.figure(1, figsize=(6., 4.5)); pylab.yscale("log")
    x = diag["t"] / tff; y = diag["rho_max"] / rho0
    pylab.plot(x, y, "-", color="black", linewidth=2.5, label=r"$\rho_{\rm max}/\rho_{0}$")
    pylab.axhline(RHO_C / rho0, linestyle="--", color="red", linewidth=2.,
                  label=r"$\rho_{\rm crit}/\rho_{0}$ (first core)")
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(0.7, max(5 * RHO_C / rho0, y.max() * 1.5))
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Central Density $\rho_{\rm max}/\rho_{0}$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_y(pylab); _save(pylab, path)


def fig_timestep_floor(diag, path):
    """min timestep and min core kernel size vs time: shows the runaway dt
    descent bottoming out at the resolution-limited hydrostatic first core."""
    pylab = _setup_mpl(); tff = diag["tff"]
    pylab.figure(1, figsize=(6., 4.5)); pylab.yscale("log")
    x = diag["t"] / tff
    pylab.plot(x, diag["dt_min"], "-", color="black", linewidth=2.5, label=r"$\Delta t_{\rm min}$")
    pylab.plot(x, diag["h_dense_min"], "-", color="dodgerblue", linewidth=2.,
               label=r"core kernel $h_{\rm min}$")
    pylab.axhline(SOFT, linestyle="--", color="red", linewidth=2., label=r"softening floor $\epsilon$")
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(1e-7, 1e-3)
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Timestep / Resolution  $[{\rm code},\,{\rm pc}]$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_y(pylab); _save(pylab, path)


def fig_virial(diag, path):
    pylab = _setup_mpl(); tff = diag["tff"]
    pylab.figure(1, figsize=(6., 4.5))
    x = diag["t"] / tff
    pylab.plot(x, diag["vir"], "-", color="black", linewidth=2.5, label=r"$2K/|W|$")
    pylab.axhline(1.0, linestyle=":", color="grey", linewidth=1.5)
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(0, max(1.1, np.nanmax(diag["vir"]) * 1.1))
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Virial Ratio $2K/|W|$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _save(pylab, path)


def fig_energies(diag, path):
    pylab = _setup_mpl(); tff = diag["tff"]
    pylab.figure(1, figsize=(6., 4.5))
    x = diag["t"] / tff
    norm = abs(diag["W"][0])
    for key, color, lw, lab in [("KE", "orange", 2., r"Kinetic $K$"),
                                ("U", "lime", 2., r"Thermal $U$"),
                                ("Emag", "magenta", 2., r"Magnetic $E_{B}$"),
                                ("W", "black", 2.5, r"$|W|$ (grav.)")]:
        y = np.abs(diag[key]) / norm
        pylab.plot(x, y, "-", color=color, linewidth=lw, label=lab)
    pylab.yscale("log")
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(1e-3, 5.)
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Energy $/\,|W_{0}|$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_y(pylab); _save(pylab, path)


def fig_divb(diag, path):
    pylab = _setup_mpl(); tff = diag["tff"]
    pylab.figure(1, figsize=(6., 4.5)); pylab.yscale("log")
    x = diag["t"] / tff
    pylab.plot(x, diag["divBerr_med"], "-", color="blue", linewidth=2.5,
               label=r"median $h\,|\nabla\!\cdot\!B|/|B|$ (cloud)")
    pylab.axhline(0.1, linestyle="--", color="red", linewidth=2., label=r"$10\%$ tolerance")
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(1e-5, 1.)
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Cleaning Error $h\,|\nabla\!\cdot\!B|/|B|$")
    pylab.legend(loc="best", fontsize=12., frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_y(pylab); _save(pylab, path)


def fig_conservation(diag, path):
    pylab = _setup_mpl(); tff = diag["tff"]; Lz0 = diag["Lz"][0]
    pylab.figure(1, figsize=(6., 4.5)); pylab.yscale("log")
    x = diag["t"] / tff
    pylab.plot(x, np.abs((diag["Lz"] - Lz0) / Lz0) + 1e-16, "-", color="blue", linewidth=2.5,
               label=r"$|\Delta L_{z}/L_{z}|$")
    pylab.plot(x, diag["cen_drift"], "-", color="black", linewidth=2.,
               label=r"core offset from center")
    pylab.axhline(2.6e-4, linestyle=":", color="grey", linewidth=1.5)
    pylab.text(0.02 * x.max(), 2.9e-4, r"particle spacing", fontsize=10, color="grey", va="bottom")
    pylab.xlim(0, max(1.05, x.max())); pylab.ylim(1e-5, 1e-2)
    pylab.xlabel(r"Time $t / t_{\rm ff}$"); pylab.ylabel(r"Conservation Diagnostics")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_y(pylab); _save(pylab, path)


def fig_eos_phase(snap, path, tff=None):
    pylab = _setup_mpl()
    d = load_snapshot(snap); rho = d["rho"]
    if tff is None: tff = free_fall_time(np.median(rho[rho > 1e4]))
    pylab.figure(1, figsize=(6., 4.5)); pylab.xscale("log"); pylab.yscale("log")
    pylab.plot(rho, d["u"], ".", color="dodgerblue", markersize=1.5, alpha=0.25, rasterized=True)
    rr = np.logspace(np.log10(rho.min()), np.log10(rho.max()), 300)
    pylab.plot(rr, u_eos(rr), "-", color="black", linewidth=2.,
               label=r"$u=P/[\rho(\gamma\!-\!1)]$ (forced)")
    pylab.xlim(rho.min() * 0.8, rho.max() * 1.3); pylab.ylim(u_eos(rho.min()) * 0.9, u_eos(rho.max()) * 1.2)
    pylab.xlabel(r"Density $\rho$  $[{\rm M_\odot\,pc^{-3}}]$"); pylab.ylabel(r"Internal Energy $u$  $[{\rm km^2\,s^{-2}}]$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    pylab.text(rho.min() * 1.5, u_eos(rho.max()) * 0.9,
               r"$t/t_{\rm ff}=%.2f$" % (d["t"] / tff),
               fontsize=11, ha="left", va="top")
    _minor_ticks_x(pylab); _minor_ticks_y(pylab); _save(pylab, path)


def fig_bfield_phase(snap, path, rho0):
    pylab = _setup_mpl()
    d = load_snapshot(snap); rho = d["rho"]; cl = rho > 1e4
    pylab.figure(1, figsize=(6., 4.5)); pylab.xscale("log"); pylab.yscale("log")
    pylab.plot(rho[cl], d["Bmag"][cl], ".", color="magenta", markersize=1.5, alpha=0.2, rasterized=True)
    rr = np.logspace(np.log10(rho[cl].min()), np.log10(rho[cl].max()), 200)
    pylab.plot(rr, B0 * (rr / rho0) ** (2. / 3.), "-", color="black", linewidth=2., label=r"$|B|\propto\rho^{2/3}$")
    pylab.plot(rr, B0 * (rr / rho0) ** (0.5), "--", color="green", linewidth=2., label=r"$|B|\propto\rho^{1/2}$")
    pylab.xlim(rho[cl].min() * 0.8, rho[cl].max() * 1.3)
    pylab.xlabel(r"Density $\rho$  $[{\rm M_\odot\,pc^{-3}}]$"); pylab.ylabel(r"Magnetic Field $|B|$  $[{\rm G}]$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_x(pylab); _minor_ticks_y(pylab); _save(pylab, path)


def fig_timestep_check(snap, path):
    pylab = _setup_mpl()
    d = load_snapshot(snap)
    if d["dt"] is None or d["soft"] is None:
        return
    rho, m = d["rho"], d["m"]
    L = (m / rho) ** (1. / 3.)
    amag = np.linalg.norm(d["acc"], axis=1)
    ic = np.argmin(d["pot"]); rel = d["pos"] - d["pos"][ic]; rr = np.linalg.norm(rel, axis=1) + 1e-30
    vcom = (m[:, None] * d["vel"]).sum(0) / m.sum()
    vrad = np.abs(np.sum((d["vel"] - vcom) * rel / rr[:, None], axis=1))
    dt_grav = np.sqrt(ETOL / (G_CODE * rho))
    dt_acc = np.sqrt(2 * ETOL * KCORE * d["soft"] / (amag + 1e-30))
    dt_cour = COURANT * L / (cs_eos(rho) + 0.5 * vrad)
    dt_pred = np.minimum(np.minimum(dt_grav, dt_acc), dt_cour)
    pylab.figure(1, figsize=(6., 4.5)); pylab.xscale("log"); pylab.yscale("log")
    pylab.plot(rho, dt_pred, ".", color="dodgerblue", markersize=1.5, alpha=0.25, rasterized=True,
               label=r"$\min(\,$grav, accel, Courant$)$")
    pylab.plot(rho, d["dt"], ".", color="black", markersize=1.5, alpha=0.3, rasterized=True, label=r"actual $\Delta t$")
    pylab.xlim(rho.min() * 0.8, rho.max() * 1.3)
    pylab.xlabel(r"Density $\rho$  $[{\rm M_\odot\,pc^{-3}}]$"); pylab.ylabel(r"Timestep $\Delta t$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_x(pylab); _minor_ticks_y(pylab); _save(pylab, path)


def fig_resolution(snap, path):
    pylab = _setup_mpl()
    d = load_snapshot(snap); rho = d["rho"]
    pylab.figure(1, figsize=(6., 4.5)); pylab.xscale("log"); pylab.yscale("log")
    pylab.plot(rho, d["h"], ".", color="dodgerblue", markersize=1.5, alpha=0.25, rasterized=True, label=r"kernel $h$")
    pylab.axhline(SOFT, linestyle="--", color="red", linewidth=2., label=r"softening floor $\epsilon$")
    pylab.xlim(rho.min() * 0.8, rho.max() * 1.3)
    pylab.xlabel(r"Density $\rho$  $[{\rm M_\odot\,pc^{-3}}]$"); pylab.ylabel(r"Resolution Length $h$  $[{\rm pc}]$")
    pylab.legend(loc="best", fontsize=12.6, frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_x(pylab); _minor_ticks_y(pylab); _save(pylab, path)


# ----------------------------------------------------------------------------
# radial profiles (centered on deepest-potential particle), one quantity / PDF
# ----------------------------------------------------------------------------
def _profile_curves(snaps, indices, quantity, tff):
    """yield (label, r_centers, median_q) for selected snapshots."""
    from scipy.stats import binned_statistic
    for k in indices:
        d = load_snapshot(snaps[k])
        ic = np.argmin(d["pot"]) if d["pot"] is not None else np.argmax(d["rho"])
        cen = d["pos"][ic]; rel = d["pos"] - cen; r = np.linalg.norm(rel, axis=1)
        if quantity == "rho":   q = d["rho"]
        elif quantity == "vr":
            vcom = (d["m"][:, None] * d["vel"]).sum(0) / d["m"].sum()
            q = np.sum((d["vel"] - vcom) * rel / (r[:, None] + 1e-30), axis=1)
        elif quantity == "vphi":
            Rc = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
            q = (-rel[:, 1] * d["vel"][:, 0] + rel[:, 0] * d["vel"][:, 1]) / (Rc + 1e-30)
        elif quantity == "B":   q = d["Bmag"]
        elif quantity == "dt":  q = d["dt"]
        rb = np.logspace(np.log10(max(r[r > 0].min(), 1e-5)), np.log10(r.max()), 45)
        med = binned_statistic(r, q, "median", rb)[0]
        rc = np.sqrt(rb[1:] * rb[:-1])
        yield (r"$t/t_{\rm ff}=%.2f$" % (d["t"] / tff), rc, med)


def fig_profile(snaps, path, quantity, tff):
    pylab = _setup_mpl()
    n = len(snaps); indices = sorted(set([n // 4, n // 2, 3 * n // 4, n - 1]))
    colors = ["black", "green", "dodgerblue", "red"][-len(indices):]
    lws = [1.5, 2., 2.5, 3.][-len(indices):]
    logy = quantity in ("rho", "B", "dt")
    pylab.figure(1, figsize=(6., 4.5)); pylab.xscale("log")
    if logy: pylab.yscale("log")
    rmin, rmax = 1e9, 0
    for (lab, rc, med), color, lw in zip(_profile_curves(snaps, indices, quantity, tff), colors, lws):
        pylab.plot(rc, med, "-", color=color, linewidth=lw, label=lab)
        good = np.isfinite(med); rmin = min(rmin, rc[good].min()); rmax = max(rmax, rc[good].max())
    if quantity == "rho":
        pylab.axhline(RHO_C, linestyle="--", color="grey", linewidth=1.5)
        pylab.text(rmin * 1.3, RHO_C * 1.2, r"$\rho_{\rm crit}$", fontsize=10, color="grey", va="bottom")
    pylab.axvline(SOFT, linestyle=":", color="grey", linewidth=1.2)
    if quantity in ("vr", "vphi"): pylab.axhline(0, color="black", linewidth=0.6)
    labels = dict(rho=(r"Radius $r$  $[{\rm pc}]$", r"Density $\rho$  $[{\rm M_\odot\,pc^{-3}}]$"),
                  vr=(r"Radius $r$  $[{\rm pc}]$", r"Radial Velocity $v_{r}$  $[{\rm km\,s^{-1}}]$"),
                  vphi=(r"Radius $r$  $[{\rm pc}]$", r"Rotation $v_{\phi}$  $[{\rm km\,s^{-1}}]$"),
                  B=(r"Radius $r$  $[{\rm pc}]$", r"Magnetic Field $|B|$  $[{\rm G}]$"),
                  dt=(r"Radius $r$  $[{\rm pc}]$", r"Timestep $\Delta t$"))
    pylab.xlabel(labels[quantity][0]); pylab.ylabel(labels[quantity][1])
    pylab.xlim(max(SOFT * 0.5, rmin * 0.7), rmax * 1.3)
    pylab.legend(loc="best", fontsize=12., frameon=False, labelcolor="linecolor", handlelength=3.)
    _minor_ticks_x(pylab)
    if logy: _minor_ticks_y(pylab)
    _save(pylab, path)


# ----------------------------------------------------------------------------
# 2D density slices with B-field streamlines (meshoid), face-on + edge-on
# ----------------------------------------------------------------------------
def fig_density_map(snap, path, view="face", zoom=None, tff=None):
    """view='face' -> x-y (perp to rotation/B axis z);  view='edge' -> x-z."""
    pylab = _setup_mpl()
    try:
        from meshoid import Meshoid
    except Exception:
        return
    d = load_snapshot(snap)
    if tff is None: tff = free_fall_time(np.median(d["rho"][d["rho"] > 1e4]))
    ic = np.argmin(d["pot"]) if d["pot"] is not None else np.argmax(d["rho"])
    cen = d["pos"][ic]; rel = d["pos"] - cen
    rabs = np.linalg.norm(rel, axis=1)
    if zoom is None:
        # zoom on the collapsed core / centrifugal disk: enclose the densest 1%
        dense = d["rho"] > np.percentile(d["rho"], 99)
        zoom = max(2.5 * np.percentile(rabs[dense], 90), 8 * SOFT)
    res = 400
    # meshoid SurfaceDensity always projects along z; reorder coords so the
    # desired perpendicular axis maps onto z (face-on: perp=z; edge-on: perp=y)
    if view == "face":   i, j, kperp, ttl, order = 0, 1, 2, r"face-on  $(x,y)$", [0, 1, 2]
    else:                i, j, kperp, ttl, order = 0, 2, 1, r"edge-on  $(x,z)$", [0, 2, 1]
    M = Meshoid(d["pos"][:, order], d["m"], d["h"])
    sigma = M.SurfaceDensity(d["m"], center=cen[order], size=2 * zoom, res=res)
    sigma = np.maximum(sigma, np.percentile(sigma[sigma > 0], 1))
    import matplotlib.colors as mcolors
    fig = pylab.figure(1, figsize=(5.4, 5.0)); ax = fig.add_subplot(111)
    ext = [-zoom, zoom, -zoom, zoom]
    im = ax.imshow(np.log10(sigma).T, origin="lower", extent=ext, cmap="inferno", aspect="equal")
    # B streamlines from the raw particles in a thin slab
    slab = np.abs(rel[:, kperp]) < zoom / 4
    if d["B"] is not None and slab.sum() > 50:
        g = 26
        edges = np.linspace(-zoom, zoom, g + 1); ctr = 0.5 * (edges[1:] + edges[:-1])
        Bi, _, _ = np.histogram2d(rel[slab, i], rel[slab, j], bins=edges, weights=(d["m"] * d["B"][:, i])[slab])
        Bj, _, _ = np.histogram2d(rel[slab, i], rel[slab, j], bins=edges, weights=(d["m"] * d["B"][:, j])[slab])
        W, _, _ = np.histogram2d(rel[slab, i], rel[slab, j], bins=edges, weights=d["m"][slab])
        with np.errstate(invalid="ignore", divide="ignore"):
            U = (Bi / W).T; V = (Bj / W).T
        try:
            ax.streamplot(ctr, ctr, U, V, color="cyan", density=0.8, linewidth=0.6, arrowsize=0.6)
        except Exception:
            pass
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_xlabel(r"$%s$  $[{\rm pc}]$" % "xyz"[i], color="black")
    ax.set_ylabel(r"$%s$  $[{\rm pc}]$" % "xyz"[j], color="black")
    ax.tick_params(color="white", labelcolor="black")
    for sp in ax.spines.values(): sp.set_color("white")
    ax.text(0.04, 0.95, ttl + r",  $t/t_{\rm ff}=%.2f$" % (d["t"] / tff),
            color="white", transform=ax.transAxes, ha="left", va="top", fontsize=11)
    pylab.subplots_adjust(left=0.16, bottom=0.13, right=0.99, top=0.99)
    pylab.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02); pylab.close("all")


# ----------------------------------------------------------------------------
# top-level driver
# ----------------------------------------------------------------------------
def make_all_core_diagnostics(output_dir, plot_dir=None, verbose=True):
    """Compute the table and write every diagnostic PDF for a finished run."""
    snaps = sorted(glob.glob(os.path.join(output_dir, "snapshot_*.hdf5")))
    if len(snaps) < 2:
        if verbose: print(f"[core_analysis] <2 snapshots in {output_dir}; skipping diagnostics")
        return
    if plot_dir is None:
        plot_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")), "diag_plots")
    os.makedirs(plot_dir, exist_ok=True)
    diag = compute_global_diagnostics(snaps)
    if verbose: print_diagnostics_table(diag)
    p = lambda name: os.path.join(plot_dir, name)
    # time series (one quantity per single-panel PDF)
    fig_central_density(diag, p("core_central_density.pdf"))
    fig_timestep_floor(diag, p("core_timestep_floor.pdf"))
    fig_virial(diag, p("core_virial.pdf"))
    fig_energies(diag, p("core_energies.pdf"))
    fig_divb(diag, p("core_divB.pdf"))
    fig_conservation(diag, p("core_conservation.pdf"))
    # phase diagrams + timestep + resolution (latest snapshot)
    fig_eos_phase(snaps[-1], p("core_phase_eos.pdf"), tff=diag["tff"])
    fig_bfield_phase(snaps[-1], p("core_phase_bfield.pdf"), diag["rho0"])
    fig_timestep_check(snaps[-1], p("core_timestep_check.pdf"))
    fig_resolution(snaps[-1], p("core_resolution.pdf"))
    # radial profiles
    for q in ("rho", "vr", "vphi", "B", "dt"):
        fig_profile(snaps, p(f"core_profile_{q}.pdf"), q, diag["tff"])
    # 2D maps with B field
    fig_density_map(snaps[-1], p("core_map_faceon.pdf"), view="face", tff=diag["tff"])
    fig_density_map(snaps[-1], p("core_map_edgeon.pdf"), view="edge", tff=diag["tff"])
    if verbose: print(f"[core_analysis] wrote diagnostic PDFs to {plot_dir}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "output"
    plots = sys.argv[2] if len(sys.argv) > 2 else None
    make_all_core_diagnostics(out, plots)
