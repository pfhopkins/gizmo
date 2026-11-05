"""conduction_step: isotropic and field-aligned thermal conduction of a sinusoidal internal-energy mode
in a frozen 2D periodic box, against the analytic diffusion solution.

    rho du/dt = div(kappa_eff grad u),  kappa_eff = kappa (isotropic) or kappa cos^2(theta) (B at theta to grad u)
    u(x, t)   = u0 + u0 A sin(k x) exp(-D_eff k^2 t),  D_eff = kappa_eff / rho

Cases: "iso" (Config.sh, no MAGNETIC) and "theta00"/"theta45"/"theta90" (MAGNETIC, uniform in-plane B).
See README.md for the derivation, the parameter choices and what each assertion is meant to catch.
"""
import os
import sys
from glob import glob
from pathlib import Path
from shutil import copyfile, rmtree

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

from gizmo.test import (
    TRUNCATED_MARKER,
    assert_final_time,
    build_gizmo_for_test,
    default_mpi_ranks,
    default_omp_threads,
    parse_params,
    run_test,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_conduction_step_ics as ics  # noqa: E402

TEST_NAME = "conduction_step"
TEST_DIR = f"test/{TEST_NAME}"  # relative to the repo root, where pytest runs
PARAMS = parse_params(Path(__file__).resolve().parent / f"{TEST_NAME}.params")
KAPPA = float(PARAMS["ConductionCoeff"])
K_MODE = 2 * np.pi / ics.BOXSIZE

# case -> (B angle to grad u in degrees, or None for no field; extra Config flags)
CASES = {
    "iso": (None, ()),
    "theta00": (0.0, ("MAGNETIC",)),
    "theta45": (45.0, ("MAGNETIC",)),
    "theta90": (90.0, ("MAGNETIC",)),
}

# All tolerances PROVISIONAL: untuned, not yet run. A numpy re-implementation of conduction.h on this
# lattice predicts fitted/analytic rates of 0.998 (theta00) and 1.0007 (theta45), and exactly zero
# decay at theta90. For iso it predicts 0.79: the per-component MINMOD(q_grad, q_direct) in the
# isotropic branch of hydro/conduction.h scales the flux through a face at angle phi to grad u by
# cos^2(phi). See README.
RATE_RTOL = 0.02  # PROVISIONAL: untuned, not yet run
LEAK_FRAC = 1e-3  # PROVISIONAL: untuned, not yet run -- theta90 decay rate / (kappa k^2 / rho)
PROFILE_L1_TOL = 5e-3  # PROVISIONAL: untuned, not yet run -- mean |u - u_exact| / (u0 A) at TimeMax
ENERGY_RTOL = 1e-10  # PROVISIONAL: untuned, not yet run -- conduction is a pairwise exchange
OVERSHOOT_TOL = 1e-3  # PROVISIONAL: untuned, not yet run -- excursion outside the IC range, in units of u0 A

# One binary per Config flag set, several runs per binary. The three field angles differ only in the
# IC, so they share one MAGNETIC build and are kept apart by overriding InitCondFile and OutputDir
# per run (run_test writes the override file; mark_run_truncated honours the overridden OutputDir).
# The harness's variant directories (output_<flags>) key on Config flags alone and cannot separate
# them, so this module manages output_<case>/ itself and does not use get_final_snapshot.
_BUILT_FLAGS = [None]


def _ic_stem(case):
    return f"{TEST_NAME}_{case}_ics"


def _outdir(case):
    return f"{TEST_DIR}/output_{case}"


def _ensure_ic(case):
    """(Re)write this case's IC. Always regenerated: it is cheap and can then never be stale."""
    theta, _ = CASES[case]
    ics.make_conduction_step_ics(f"{TEST_DIR}/{_ic_stem(case)}.hdf5", theta_deg=theta)


def _build_and_run(case, num_mpi_ranks, num_omp_threads):
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    _, flags = CASES[case]
    if _BUILT_FLAGS[0] != flags:
        _BUILT_FLAGS[0] = None  # a failed build leaves no binary
        build_gizmo_for_test(TEST_NAME, num_omp_threads, flags)
        _BUILT_FLAGS[0] = flags
    _ensure_ic(case)
    rmtree(_outdir(case), ignore_errors=True)
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        run_test(TEST_NAME, num_mpi_ranks, num_omp_threads,
                 param_overrides={"InitCondFile": _ic_stem(case), "OutputDir": f"output_{case}"})
    finally:
        os.chdir(cwd)
        log = f"{TEST_DIR}/test_{TEST_NAME}.out"
        if os.path.isfile(log) and os.path.isdir(_outdir(case)):
            copyfile(log, f"{_outdir(case)}/test_{TEST_NAME}.out")


def _snapshots(case):
    d = _outdir(case)
    marker = f"{d}/{TRUNCATED_MARKER}"
    if os.path.isfile(marker):
        with open(marker) as f:
            raise RuntimeError(f"{d} is from a run that did not finish: {f.read().strip()}")
    snaps = sorted(glob(f"{d}/snapshot_*.hdf5"))
    if len(snaps) < 3:
        raise RuntimeError(f"need >= 3 snapshots in {d} to fit a decay rate, found {len(snaps)}")
    return snaps


def _load(snap):
    with h5py.File(snap, "r") as f:
        g = f["PartType0"]
        o = np.argsort(g["ParticleIDs"][:])
        out = {k: g[k][:][o] for k in ("Coordinates", "Velocities", "Masses", "InternalEnergy", "Density")}
        out["MagneticField"] = g["MagneticField"][:][o] if "MagneticField" in g else None
        out["Time"] = float(f["Header"].attrs["Time"])
    return out


def _mode_amplitude(x, u):
    """Projection of u onto sin(kx); exact on the cell-centred lattice (sum sin^2 = N/2)."""
    return 2.0 / len(u) * np.sum((u - u.mean()) * np.sin(K_MODE * x))


def _plot(case, theta, x, u_final, u_exact, t, amp, lam_exact, lam_fit):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    order = np.argsort(x)
    ax0.plot(x[order], u_final[order], ".", ms=2, color="tab:blue", label="GIZMO")
    ax0.plot(x[order], u_exact[order], "-", lw=1, color="k", label="analytic")
    ax0.set_xlabel("x")
    ax0.set_ylabel(f"u at t = {t[-1]:g}")
    ax0.legend(frameon=False)
    a0 = ics.U0 * ics.AMP
    tt = np.linspace(0, t[-1], 100)
    ax1.semilogy(t, amp / a0, "o", ms=5, color="tab:blue", label=f"GIZMO (rate {lam_fit:.4f})")
    ax1.semilogy(tt, np.exp(-lam_exact * tt), "-", lw=1, color="k", label=f"analytic (rate {lam_exact:.4f})")
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"mode amplitude / $u_0 A$")
    ax1.legend(frameon=False)
    label = "isotropic" if theta is None else rf"$\theta_B = {theta:g}^\circ$"
    fig.suptitle(f"{TEST_NAME}: {label}")
    fig.tight_layout()
    fig.savefig(f"{TEST_DIR}/{TEST_NAME}_{case}.png", dpi=120)
    plt.close(fig)


# iso measured 0.793 of the analytic rate on 2026-11-05 before hydro/conduction.h stopped limiting the face
# gradient per component against the pair-direction gradient (which scaled oblique-face fluxes by cos^2 phi).
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(4),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
@pytest.mark.parametrize("case", list(CASES))
def test_conduction_step(case, num_mpi_ranks, num_omp_threads):
    theta, _ = CASES[case]
    _build_and_run(case, num_mpi_ranks, num_omp_threads)
    snaps = _snapshots(case)
    assert_final_time(snaps[-1], TEST_NAME)

    data = [_load(s) for s in snaps]
    s0, sf = data[0], data[-1]
    x = s0["Coordinates"][:, 0]
    t = np.array([d["Time"] for d in data])
    amp = np.array([_mode_amplitude(x, d["InternalEnergy"]) for d in data])

    # D = kappa/rho with the code's own (kernel) density, the rho in its discrete operator
    rho = float(np.mean(s0["Density"]))
    cos2 = 1.0 if theta is None else np.cos(np.radians(theta)) ** 2
    lam_iso = KAPPA / rho * K_MODE**2
    lam_exact = lam_iso * cos2
    lam_fit = -np.polyfit(t, np.log(np.abs(amp)), 1)[0]
    u_exact = ics.U0 * (1 + ics.AMP * np.sin(K_MODE * x) * np.exp(-lam_exact * t[-1]))
    _plot(case, theta, x, sf["InternalEnergy"], u_exact, t, amp, lam_exact, lam_fit)

    # premise: FREEZE_HYDRO holds the cells and (under MAGNETIC) the field fixed, so only conduction acts
    assert abs(rho / ics.RHO0 - 1) < 0.01, f"[{case}] mean kernel density {rho:.6f} vs IC rho0={ics.RHO0}"
    assert abs(amp[0] / (ics.U0 * ics.AMP) - 1) < 1e-10, (
        f"[{case}] t=0 mode amplitude {amp[0]:.12g} != u0 A = {ics.U0 * ics.AMP:g}: IC not read as written")
    for d in data[1:]:
        dx = np.max(np.abs(d["Coordinates"] - s0["Coordinates"]))
        vmax = np.max(np.abs(d["Velocities"]))
        assert dx < 1e-10 and vmax < 1e-10, (
            f"[{case}] t={d['Time']:g}: cells moved (max |dx|={dx:.3g}, max |v|={vmax:.3g}); FREEZE_HYDRO not in effect")
        if theta is not None:
            b0 = s0["MagneticField"] / np.linalg.norm(s0["MagneticField"], axis=1)[:, None]
            b = d["MagneticField"] / np.linalg.norm(d["MagneticField"], axis=1)[:, None]
            db = np.max(np.abs(b - b0))
            assert db < 1e-8, f"[{case}] t={d['Time']:g}: field direction changed by up to {db:.3g}; B is not frozen"

    # (c) conservation: conduction only exchanges energy between pairs, and nothing else acts
    e = np.array([np.sum(d["Masses"] * d["InternalEnergy"]) for d in data])
    de = np.max(np.abs(e / e[0] - 1))
    assert de < ENERGY_RTOL, (
        f"[{case}] total internal energy not conserved: max |E(t)/E(0) - 1| = {de:.3g} (tol {ENERGY_RTOL:g}); "
        "the pairwise conduction exchange is not antisymmetric, or cells are on different timesteps")

    # (d) no new extrema: a diffusion operator must not overshoot the initial range
    u_ic = s0["InternalEnergy"]
    tol = OVERSHOOT_TOL * ics.U0 * ics.AMP
    umax = max(d["InternalEnergy"].max() for d in data)
    umin = min(d["InternalEnergy"].min() for d in data)
    assert umax <= u_ic.max() + tol and umin >= u_ic.min() - tol, (
        f"[{case}] u left the initial range [{u_ic.min():.6g}, {u_ic.max():.6g}]: min {umin:.6g}, max {umax:.6g} "
        f"(tol {tol:.2g}); overshoot or instability in the conduction step")

    # (a) decay rate of the mode
    if theta is not None and cos2 < 1e-12:
        assert abs(lam_fit) < LEAK_FRAC * lam_iso, (
            f"[{case}] B perpendicular to grad u, yet the mode decays at {lam_fit:.4g} = "
            f"{lam_fit / lam_iso:.3g} of the field-aligned rate (bound {LEAK_FRAC:g}): cross-field leakage")
    else:
        rel = lam_fit / lam_exact - 1
        assert abs(rel) < RATE_RTOL, (
            f"[{case}] decay rate {lam_fit:.5g} vs analytic D_eff k^2 = {lam_exact:.5g} "
            f"(kappa={KAPPA:.6g}, rho={rho:.6g}, cos^2 theta={cos2:.4g}): relative error {rel:+.4f}, tol {RATE_RTOL}")

    # (b) final profile against the analytic solution
    l1 = np.mean(np.abs(sf["InternalEnergy"] - u_exact)) / (ics.U0 * ics.AMP)
    assert l1 < PROFILE_L1_TOL, (
        f"[{case}] final profile L1 = {l1:.4g} of u0 A (tol {PROFILE_L1_TOL:g}) at t={t[-1]:g}")
