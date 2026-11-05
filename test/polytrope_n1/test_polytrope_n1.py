"""polytrope_n1: hydrostatic equilibrium of a self-gravitating n=1 (gamma=2) MFM polytrope.

The analytic Lane-Emden n=1 sphere (G = M = R = 1, see make_polytrope_n1_ics.py for the
formulas) is set up at rest and must stay put: tree gravity has to balance the MFM pressure
gradient. Checks, each against the analytic solution:

  (a) t=0 forces, binned in enclosed mass fraction 5-90%: gravity ("Acceleration", gravity-only
      because OUTPUT_HYDROACCELERATION is also set) vs -G M(r)/r^2; hydro ("HydroAcceleration")
      vs -grad(P)/rho = -2 K drho/dr; and their sum vs zero. Also the t=0 shell density.
  (b) persistence over one pulsation period (TimeMax = 4): 10/50/90% Lagrange radii vs t=0,
      interior mass-weighted rms speed vs the central sound speed, and the final shell-density
      profile. The horizon is short on purpose: the MFM free surface launches a rarefaction that
      reaches the interior at t ~ 4.5 and disassembles the star afterwards (see README).
  (b') the snapshot kernel density vs analytic over 5-90% of the mass: catches a density estimator
      that is wrong on a stratified body (HYDRO_KERNEL_SURFACE_VOLCORR is, and is xfailed).
  (c) energy: total (thermal + kinetic + 0.5 sum m phi) conserved, and U(0) = 1/4, W(0) = -3/4.
  (d) spurious centre-of-mass drift (momentum_drift_common).

Variants: baseline (fixed SofteningGas), ags (ADAPTIVE_GRAVSOFT_FORGAS: gravity softened with the
hydro kernel), tidal_adaptive (TIDAL_TIMESTEP_CRITERION + ADAPTIVE_TREEFORCE_UPDATE=0.06: gravity
re-walked only every fraction of the tidal time).

Tolerances were calibrated on the 2026-11-05 first run (4 ranks x 2 threads); the
measured values are quoted next to each. All checks are evaluated before any assertion, so one
run reports every metric.
"""

from os import path
import glob
import importlib.util
import os
import sys

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from momentum_drift_common import (  # noqa: E402
    DRIFT_SANITY_CEILING, assert_randomized_drift, measure_and_record, plot_momentum_drift,
    report_momentum_drift,
)

from gizmo.test import (  # noqa: E402
    assert_energy_conserved,
    assert_final_time,
    build_and_run_test,
    default_mpi_ranks,
    default_omp_threads,
    gas_energy,
    variant_output_dir,
)

TEST_NAME = "polytrope_n1"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_polytrope_n1_ics",
        path.join(path.dirname(path.abspath(__file__)), "make_polytrope_n1_ics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


poly = _load_generator()
N_PARTICLES = poly.N_DEFAULT

# enclosed-mass-fraction window for the profile/force checks: inside 5% the bins hold few
# particles; outside 90% the free surface (rho -> 0, one-sided kernels) dominates
CHECK_MASS_FRACTIONS = (0.05, 0.90)
N_CHECK_BINS = 12
# The final density profile is compared over a narrower window: the free-surface rarefaction
# reaches the outer third of the mass before one pulsation period (at t=4: +0.1 dex at 60-75%,
# -0.2 dex at 80-90%), while 5-60% stays within 0.01 dex.
FINAL_RHO_MASS_FRACTIONS = (0.05, 0.60)
LAGRANGE_FRACTIONS = (0.1, 0.5, 0.9)
# global force scale for the per-cell residual: the peak of GM(r)/r^2 (1.37 at r = 0.66)
G_PEAK = float(np.max(np.abs(poly.g_analytic(np.linspace(1e-3, poly.RADIUS, 100001)))))

# --- tolerances (measured 2026-11-05 in brackets; "~" = not yet measured at TimeMax = 4) -------
GRAV_L1_TOL = 0.01          # binned |<a_grav,r> - g| / |g|            [7.6e-4 baseline, 1.1e-3 ags]
HYDRO_L1_TOL = 0.05         # binned |<a_hydro,r> + g| / |g|           [1.5e-2; 1.1e-2 crk_faces]
NET_L1_TOL = 0.05           # binned |<a_grav,r + a_hydro,r>| / |g|    [1.4e-2]
RHO_LOG_L1_TOL_T0 = 0.01    # dex, shell density at t=0                 [1.1e-3]
RHO_LOG_L1_TOL_FINAL = 0.05  # dex, shell density at TimeMax over 5-60% mass  [0.013 at t=4]
KERNEL_RHO_RMS_TOL = 0.02   # rms(rho_kernel/rho_analytic - 1), 5-90% of the mass  [4e-3; 0.1 volcorr]
LAGRANGE_TOL = (0.05, 0.05, 0.05)  # |dr/r| at 10/50/90%  [~; raw radii at t=4: +1.7%, -3.3%, -0.6%]
VRMS_TOL = 0.05             # interior v_rms / c_s,central              [~; 0.022 at t=4]
U0_TOL = 0.03               # |U(0)/U_exact - 1|                        [<1e-5]
W0_TOL = 0.03               # |W(0)/W_exact - 1|                        [9e-4]
ENERGY_TOL = 0.01           # |E_f/E_0 - 1|, E_0 ~ -1/2                 [1.1e-3 at t=10]

VARIANTS = [
    pytest.param((), id="baseline"),
    pytest.param(("ADAPTIVE_GRAVSOFT_FORGAS",), id="ags"),
    pytest.param(("TIDAL_TIMESTEP_CRITERION", "ADAPTIVE_TREEFORCE_UPDATE=0.06"), id="tidal_adaptive"),
    # The free surface is where MFM faces fail to close (one-sided neighbours); these two options exist
    # for exactly that and are exercised by no other test. Measured 2026-11-05 (see README):
    # HYDRO_KERNEL_SURFACE_VOLCORR keys on neighbour asymmetry, which a density gradient also
    # produces, so on this stratified star it inflates the kernel density by +4% at 60-80% of the
    # mass, +11% at 80-90% and +75% at the surface (true surface bias: +4%). xfail(strict) on the
    # kernel-density check so a fix shows up as an unexpected pass. KERNEL_CRK_FACES halves the
    # t=0 pressure-gradient noise but makes the surface imbalance worse (+33% outward in the outer 2%).
    pytest.param(("HYDRO_KERNEL_SURFACE_VOLCORR",), id="surface_volcorr",
                 marks=pytest.mark.xfail(strict=True, reason="HYDRO_KERNEL_SURFACE_VOLCORR misreads "
                                         "a density gradient as a free surface and inflates the "
                                         "kernel density of the outer 40% of the mass")),
    pytest.param(("KERNEL_CRK_FACES",), id="crk_faces"),
    # Density SPH on the same IC and analytic solution (HYDRO_SPH takes precedence over the MFM flag in
    # Config.sh, hydro/hydro_evaluate.h). The SPH viscosity/conductivity parameters are in the params file.
    pytest.param(("HYDRO_DENSITY_SPH",), id="sph"),
]


def _ensure_ic():
    if path.isfile(IC_FILE):
        with h5py.File(IC_FILE, "r") as F:
            if (int(F["Header"].attrs["NumPart_Total"][0]) == N_PARTICLES
                    and int(F.attrs.get("make_polytrope_n1_ics_version", -1)) == poly.IC_VERSION):
                return
    poly.make_polytrope_n1_ics(IC_FILE, n_part=N_PARTICLES)


def _load_snapshot(snap):
    out = {}
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        out["time"] = float(F["Header"].attrs["Time"])
        for key, name in (("pos", "Coordinates"), ("vel", "Velocities"), ("mass", "Masses"),
                          ("u", "InternalEnergy"), ("pot", "Potential"), ("rho", "Density"),
                          ("a_grav", "Acceleration"), ("a_hydro", "HydroAcceleration")):
            out[key] = g[name][:].astype(float) if name in g else None
    com = np.average(out["pos"], axis=0, weights=out["mass"])
    out["dx"] = out["pos"] - com
    out["r"] = np.sqrt(np.sum(out["dx"]**2, axis=1))
    # radial smoothing width for the mass profile: one local spacing (the IC is layered in shells)
    out["width"] = (out["mass"] / out["rho"]) ** (1.0 / 3.0)
    return out


def _radial_tangential(dx, r, vec):
    rhat = dx / np.maximum(r, 1e-30)[:, None]
    a_r = np.sum(vec * rhat, axis=1)
    a_t = np.sqrt(np.maximum(np.sum(vec**2, axis=1) - a_r**2, 0.0))
    return a_r, a_t


def _check_edges():
    return poly.radius_of_mass_fraction(np.linspace(*CHECK_MASS_FRACTIONS, N_CHECK_BINS + 1))


def _binned_mean(r, values, edges):
    idx = np.digitize(r, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = values[sel].mean()
    return out


def _binned_relative_error(r, sim, ref, edges):
    """Per-bin |<sim> - <ref>| / <|ref|> over the particles of each radial bin."""
    return np.abs(_binned_mean(r, sim, edges) - _binned_mean(r, ref, edges)) / _binned_mean(r, np.abs(ref), edges)


def _kernel_density_rms(s, q_window=(0.05, 0.90)):
    """rms of (snapshot Density / rho_analytic(r) - 1) over cells in the enclosed-mass window."""
    r = np.sqrt(np.sum(s["dx"] ** 2, axis=1))
    order = np.argsort(r)
    q = np.empty_like(r)
    q[order] = (np.arange(len(r)) + 0.5) / len(r)
    sel = (q >= q_window[0]) & (q < q_window[1])
    return float(np.sqrt(np.mean((s["rho"][sel] / poly.rho_analytic(r[sel]) - 1.0) ** 2)))


def _shell_density(s, edges):
    """Shell density from the enclosed-mass profile with each cell's mass spread over one local
    spacing (poly.smoothed_enclosed_mass): a raw histogram of the shell IC measures only which
    bin each particle shell falls in."""
    msum = np.diff(poly.smoothed_enclosed_mass(edges, s["r"], s["mass"], s["width"]))
    return msum / ((4.0 / 3.0) * np.pi * np.diff(edges**3))


def _log_l1(rho, rho_ref):
    good = (rho > 0) & (rho_ref > 0)
    return float(np.mean(np.abs(np.log10(rho[good] / rho_ref[good])))) if good.any() else np.inf


def _lagrange_radii(r, mass, fractions=LAGRANGE_FRACTIONS):
    """Unsmoothed: steps by a whole particle shell on the shell IC; used only as a selection cut."""
    order = np.argsort(r)
    cum = np.cumsum(mass[order]) / mass.sum()
    return np.array([r[order][np.searchsorted(cum, f)] for f in fractions])


def _energy_components(s):
    u = float(np.sum(s["mass"] * s["u"]))
    k = float(0.5 * np.sum(s["mass"] * np.sum(s["vel"]**2, axis=1)))
    w = float(0.5 * np.sum(s["mass"] * s["pot"])) if s["pot"] is not None else np.nan
    return u, k, w


def _vrms_interior(s, max_fraction=CHECK_MASS_FRACTIONS[1]):
    """Mass-weighted rms speed of the cells inside the max_fraction Lagrange radius."""
    r_cut = _lagrange_radii(s["r"], s["mass"], (max_fraction,))[0]
    sel = s["r"] <= r_cut
    v2 = np.sum(s["vel"][sel]**2, axis=1)
    return float(np.sqrt(np.average(v2, weights=s["mass"][sel])))


def _t0_force_metrics(s0, edges):
    for key, flag in (("a_grav", "OUTPUT_ACCELERATION"), ("a_hydro", "OUTPUT_HYDROACCELERATION")):
        if s0[key] is None:
            raise KeyError(f"snapshot_000 lacks the {key} field: is {flag} in Config.sh?")
    r, dx = s0["r"], s0["dx"]
    g_an = poly.g_analytic(r)
    h_an = poly.hydro_accel_analytic(r)
    ag_r, ag_t = _radial_tangential(dx, r, s0["a_grav"])
    ah_r, ah_t = _radial_tangential(dx, r, s0["a_hydro"])
    an_r, _ = _radial_tangential(dx, r, s0["a_grav"] + s0["a_hydro"])
    in_win = (r >= edges[0]) & (r < edges[-1])
    grav_bins = _binned_relative_error(r, ag_r, g_an, edges)
    hydro_bins = _binned_relative_error(r, ah_r, h_an, edges)
    net_bins =np.abs(_binned_mean(r, an_r, edges)) / _binned_mean(r, np.abs(g_an), edges)
    net_vec = np.sqrt(np.sum((s0["a_grav"] + s0["a_hydro"])**2, axis=1))
    return {
        "grav_bins": grav_bins, "hydro_bins": hydro_bins, "net_bins": net_bins,
        "grav_l1": float(np.nanmean(grav_bins)), "hydro_l1": float(np.nanmean(hydro_bins)),
        "net_l1": float(np.nanmean(net_bins)),
        # tangential/radial rms: the radial-only analytic force has no tangential part, so this is
        # particle-scale noise (tree and lattice for gravity; gradient estimate for hydro)
        "grav_tan_over_rad": float(np.sqrt(np.mean(ag_t[in_win]**2) / np.mean(ag_r[in_win]**2))),
        "hydro_tan_over_rad": float(np.sqrt(np.mean(ah_t[in_win]**2) / np.mean(ah_r[in_win]**2))),
        # per-cell residual net force (what the cells actually feel, noise included), over a global
        # scale: per-cell |g| -> 0 at the centre
        "net_rms_per_cell": float(np.sqrt(np.mean(net_vec[in_win]**2)) / G_PEAK),
        "rc": np.sqrt(edges[:-1] * edges[1:]),
        "g_bin": _binned_mean(r, g_an, edges),
        "ag_bin": _binned_mean(r, ag_r, edges), "ah_bin": _binned_mean(r, ah_r, edges),
        "anet_bin": _binned_mean(r, an_r, edges),
    }


def _summary_npz_path(variant_id):
    return f"{TEST_DIR}/summary_{variant_id}.npz"


def _plot_variant(variant_id, fm, times, energy):
    fig, (ax_a, ax_e) = plt.subplots(1, 2, figsize=(11, 4.5))
    rr = np.linspace(0.02, 0.98, 200)
    ax_a.plot(rr, np.abs(poly.g_analytic(rr)), "k-", lw=1.5, label=r"$GM(r)/r^2$ (analytic)")
    ax_a.plot(fm["rc"], -fm["ag_bin"], "o", mfc="none", label=r"$-a_{\rm grav,r}$")
    ax_a.plot(fm["rc"], fm["ah_bin"], "s", mfc="none", label=r"$a_{\rm hydro,r}$")
    ax_a.plot(fm["rc"], np.abs(fm["anet_bin"]), "^", label=r"$|a_{\rm grav,r} + a_{\rm hydro,r}|$")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("r")
    ax_a.set_ylabel("radial acceleration at t = 0 (binned)")
    ax_a.legend(fontsize=8)

    u, k, w = energy[:, 0], energy[:, 1], energy[:, 2]
    e = u + k + w
    norm = abs(e[0])
    ax_e.plot(times, (u - u[0]) / norm, label=r"$\Delta U$")
    ax_e.plot(times, (w - w[0]) / norm, label=r"$\Delta W$")
    ax_e.plot(times, k / norm, label=r"$K$")
    ax_e.plot(times, (e - e[0]) / norm, "k-", lw=2, label=r"$\Delta E$")
    ax_e.axhline(0, color="0.6", lw=0.5)
    ax_e.set_xlabel("t")
    ax_e.set_ylabel(r"energy change / $|E_0|$")
    ax_e.legend(fontsize=8)
    fig.suptitle(f"{TEST_NAME} [{variant_id}]")
    fig.tight_layout()
    out = f"{TEST_DIR}/{TEST_NAME}_{variant_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _plot_summary():
    files = sorted(glob.glob(f"{TEST_DIR}/summary_*.npz"))
    required = {"variant_id", "rc", "rho_exact", "rho_initial", "rho_final", "times", "lagrange"}
    data = []
    for f in files:
        d = np.load(f)
        if required.issubset(d.files):
            data.append({k: d[k] for k in d.files})
    if not data:
        return None
    fig, (ax_rho, ax_lag) = plt.subplots(2, 1, figsize=(7, 9))
    ax_rho.semilogy(data[0]["rc"], data[0]["rho_exact"], "k-", lw=1.5, label="analytic")
    ax_rho.semilogy(data[0]["rc"], data[0]["rho_initial"], "k.", label="t = 0")
    styles = ("--", "-", ":")
    for d in data:
        vid = str(d["variant_id"])
        rc_f = d["rc_final"] if "rc_final" in d else d["rc"]
        line, = ax_rho.semilogy(rc_f, d["rho_final"], "o", mfc="none", ms=4, label=f"{vid}, final")
        lag = d["lagrange"]
        for j, f in enumerate(LAGRANGE_FRACTIONS):
            ax_lag.plot(d["times"], lag[:, j] / lag[0, j] - 1.0, styles[j], color=line.get_color(),
                        label=f"{vid} {int(100 * f)}%")
    ax_rho.set_xlabel("r")
    ax_rho.set_ylabel(r"$\rho$ (shell average)")
    ax_rho.legend(fontsize=8)
    ax_lag.axhline(0, color="0.6", lw=0.5)
    ax_lag.set_xlabel("t")
    ax_lag.set_ylabel(r"$r_{\rm Lagrange}(t) / r_{\rm Lagrange}(0) - 1$")
    ax_lag.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    out = f"{TEST_DIR}/{TEST_NAME}_summary.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("extra_config_flags", VARIANTS)
def test_polytrope_n1(num_mpi_ranks, num_omp_threads, extra_config_flags, request):
    poly.check_analytic_identities()
    _ensure_ic()
    build_and_run_test(TEST_NAME, num_mpi_ranks, num_omp_threads, extra_config_flags)

    variant_id = request.node.callspec.id.split("-")[0]
    outputdir = variant_output_dir(TEST_NAME, extra_config_flags)
    snaps = sorted(glob.glob(outputdir + "/snapshot_*.hdf5"))
    if len(snaps) < 2:
        raise RuntimeError(f"GIZMO did not produce enough snapshots in {outputdir}")
    assert_final_time(snaps[-1], TEST_NAME)

    s0 = _load_snapshot(snaps[0])
    sf = _load_snapshot(snaps[-1])
    assert s0["time"] == 0.0, f"first snapshot is at t={s0['time']}, expected the t=0 output"
    edges = _check_edges()
    rho_exact = poly.shell_mean_density(edges)

    # (a) t=0 force balance and density
    fm = _t0_force_metrics(s0, edges)
    rho0 = _shell_density(s0, edges)
    rho_l1_0 = _log_l1(rho0, rho_exact)
    kernel_rho_rms = _kernel_density_rms(s0)

    # (b) persistence
    edges_f = poly.radius_of_mass_fraction(np.linspace(*FINAL_RHO_MASS_FRACTIONS, N_CHECK_BINS + 1))
    rhof = _shell_density(sf, edges_f)
    rho_l1_f = _log_l1(rhof, poly.shell_mean_density(edges_f))
    vrms_f = _vrms_interior(sf) / poly.CS_CENTRAL

    # trajectories (Lagrange radii, energy components) over every snapshot
    times, lagrange, energy = [], [], []
    for s in snaps:
        d = s0 if s == snaps[0] else (sf if s == snaps[-1] else _load_snapshot(s))
        times.append(d["time"])
        lagrange.append(poly.smoothed_lagrange_radii(d["r"], d["mass"], d["width"], LAGRANGE_FRACTIONS))
        energy.append(_energy_components(d))
    times, lagrange, energy = np.array(times), np.array(lagrange), np.array(energy)
    lag_rel = np.abs(lagrange[-1] / lagrange[0] - 1.0)

    # (c) energy at t=0; total-energy conservation is asserted below via the harness
    u0, _, w0 = energy[0]
    e_ratio = gas_energy(snaps[-1], include_potential=True) / gas_energy(snaps[0], include_potential=True)

    np.savez(_summary_npz_path(variant_id), variant_id=variant_id, rc=np.sqrt(edges[:-1] * edges[1:]),
             rho_exact=rho_exact, rho_initial=rho0, rho_final=rhof,
             rc_final=np.sqrt(edges_f[:-1] * edges_f[1:]), times=times, lagrange=lagrange,
             energy=energy, grav_bins=fm["grav_bins"], hydro_bins=fm["hydro_bins"], net_bins=fm["net_bins"])
    plots = [_plot_variant(variant_id, fm, times, energy), _plot_summary()]

    # (d) spurious COM drift from correlated tree-force errors
    traj = measure_and_record(TEST_DIR, variant_id, outputdir, parttype="PartType0")
    final_drift = np.nan
    if traj is not None:
        final_drift = report_momentum_drift(TEST_DIR, TEST_NAME, variant_id, traj)
        plots.append(plot_momentum_drift(TEST_DIR, TEST_NAME))

    print(f"\n[{TEST_NAME}/{variant_id}] metrics (window: enclosed mass {CHECK_MASS_FRACTIONS}):")
    print(f"  t=0 gravity   binned L1 {fm['grav_l1']:.3e} (max bin {np.nanmax(fm['grav_bins']):.3e}), "
          f"rms tangential/radial {fm['grav_tan_over_rad']:.3e}")
    print(f"  t=0 hydro     binned L1 {fm['hydro_l1']:.3e} (max bin {np.nanmax(fm['hydro_bins']):.3e}), "
          f"rms tangential/radial {fm['hydro_tan_over_rad']:.3e}")
    print(f"  t=0 net       binned L1 {fm['net_l1']:.3e} (max bin {np.nanmax(fm['net_bins']):.3e}), "
          f"per-cell rms |a_net| / max|g| {fm['net_rms_per_cell']:.3e}")
    print(f"  shell density log L1: t=0 {rho_l1_0:.3e} dex, final {rho_l1_f:.3e} dex; "
          f"t=0 kernel density rms(rho/rho_analytic - 1) over 5-90% mass {kernel_rho_rms:.3e}")
    print(f"  Lagrange |dr/r| at {LAGRANGE_FRACTIONS}: {np.array2string(lag_rel, precision=4)}")
    print(f"  final interior v_rms / c_s,c = {vrms_f:.3e}")
    print(f"  U(0) = {u0:.5f} (exact {poly.U_EXACT}), W(0) = {w0:.5f} (exact {poly.W_EXACT}), "
          f"E_f/E_0 - 1 = {e_ratio - 1:.3e}, COM drift {final_drift:.3e}")
    print("  plots: " + ", ".join(p for p in plots if p))

    failures = []

    def check(ok, msg):
        if not ok:
            failures.append(f"[{variant_id}] {msg}")

    check(fm["grav_l1"] < GRAV_L1_TOL,
          f"t=0 gravity vs -GM(r)/r^2: binned L1 {fm['grav_l1']:.3e} >= {GRAV_L1_TOL} "
          f"(tree force or softening error)")
    check(fm["hydro_l1"] < HYDRO_L1_TOL,
          f"t=0 hydro acceleration vs -grad(P)/rho: binned L1 {fm['hydro_l1']:.3e} >= {HYDRO_L1_TOL} "
          f"(MFM pressure-gradient error)")
    check(fm["net_l1"] < NET_L1_TOL,
          f"t=0 net radial acceleration / |g|: binned L1 {fm['net_l1']:.3e} >= {NET_L1_TOL} "
          f"(gravity and pressure do not balance)")
    check(rho_l1_0 < RHO_LOG_L1_TOL_T0,
          f"t=0 shell density vs analytic: log L1 {rho_l1_0:.3e} dex >= {RHO_LOG_L1_TOL_T0} (IC or read-in)")
    check(kernel_rho_rms < KERNEL_RHO_RMS_TOL,
          f"t=0 kernel density vs analytic: rms(rho/rho_an - 1) {kernel_rho_rms:.3e} >= {KERNEL_RHO_RMS_TOL} "
          "(density estimator wrong on a stratified body)")
    check(rho_l1_f < RHO_LOG_L1_TOL_FINAL,
          f"final shell density vs analytic: log L1 {rho_l1_f:.3e} dex >= {RHO_LOG_L1_TOL_FINAL}")
    for f, rel, tol in zip(LAGRANGE_FRACTIONS, lag_rel, LAGRANGE_TOL):
        check(rel < tol, f"{int(100 * f)}% Lagrange radius moved |dr/r| = {rel:.4f} >= {tol} "
                         f"(r0={lagrange[0][LAGRANGE_FRACTIONS.index(f)]:.4f})")
    check(vrms_f < VRMS_TOL, f"final interior v_rms = {vrms_f:.3e} c_s,c >= {VRMS_TOL} (spurious motions)")
    check(abs(u0 / poly.U_EXACT - 1) < U0_TOL, f"U(0) = {u0:.5f}, exact {poly.U_EXACT} (tol {U0_TOL:.0%})")
    check(np.isfinite(w0) and abs(w0 / poly.W_EXACT - 1) < W0_TOL,
          f"W(0) = 0.5 sum m phi = {w0:.5f}, exact {poly.W_EXACT} (tol {W0_TOL:.0%})")
    check(np.isfinite(final_drift) and final_drift < DRIFT_SANITY_CEILING,
          f"spurious COM velocity {final_drift:.3e} of v_grav (sanity ceiling {DRIFT_SANITY_CEILING})")
    if traj is not None:
        try:
            assert_randomized_drift(TEST_NAME, variant_id, final_drift)   # no-op without RANDOMIZE_GRAVTREE
        except AssertionError as err:
            failures.append(str(err))
    try:
        assert_energy_conserved(TEST_NAME, extra_config_flags, tol=ENERGY_TOL, include_potential=True)
    except AssertionError as err:
        failures.append(f"[{variant_id}] {err}")

    assert not failures, "\n".join(failures)
