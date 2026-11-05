"""sn_cluster_adiabatic: energy budget of FIRE-3 supernova feedback + MFM hydro in an adiabatic box.

A 1e3 Msun FIRE-3 stellar population (12 O stars) sits in a static, uniform, periodic box of warm
gas, with no cooling, gravity, radiation or stellar mass loss. The only gas energy source is the
supernovae, so the gas thermal + kinetic energy must equal its initial value plus the energy the
supernovae inject. Each case is checked in two independent halves, from GIZMO's in-code tallies
(ENERGY_BUDGET_DIAGNOSTIC):

  test_hydro_conserves_energy:  E(t) - E(0) == E_fb(t), the energy feedback actually wrote into
      the gas. This is the conservation law of the hydro + time integration alone.
  test_feedback_delivers_sn_energy:  E_fb(t) == N_SN(t) * 1e51 erg, with N_SN from
      MechFeedbackEvents.txt. This is the coupling algorithm alone.

Their product is the full budget (E - E0 - N_SN*1e51)/(N_SN*1e51), which is printed for each case.

Cases differ in gas cell mass and in TimeMax at fixed physical duration (TimeMax sets the integer
timestep ladder; the long-TimeMax runs stop at T_END on a 'stop' file). The 'resolved' case caps the
timestep at ~5 kyr, so cells hit by a supernova are already on steps short compared with their
post-explosion Courant time: it is the control showing that the instrument closes when the time
integration is resolved.
"""

import re
import subprocess
import time
from os import environ, makedirs
from pathlib import Path

import numpy as np
import h5py
import pytest

from gizmo.test import (
    build_gizmo_for_test,
    mpi_launch_command,
    default_mpi_ranks,
    default_omp_threads,
    _resolve_test_timeout,
    _robust_replace_dir,
)

TEST_NAME = "sn_cluster_adiabatic"
TEST_DIR = Path(__file__).parent
E51_CODE = 1.0e51 / 1.989e43  # 1e51 erg in Msun (km/s)^2
T_END = 10.24  # code units (~10 Myr): the end time of every case
NUM_RANKS = default_mpi_ranks(max_ranks=2)
NUM_THREADS = default_omp_threads()

# Tolerances.
# Hydro: MFM with individual timesteps is not exactly conservative, since a pair's fluxes are applied on
# each member's own step. With the time integration resolved (the 'resolved' control) the budget
# closes to ~1e-4 at 100 Msun and ~3e-3 at 10 Msun (cap 5 kyr, measured), so 5% is >10x that floor.
TOL_HYDRO = 0.05
# Coupling: the FIRE scheme is built to couple exactly E_SN per event in the star's frame, except that
# ejecta mass joins a cell at the cell's specific internal energy, adding m_ej*u_cell. Inside a hot
# superbubble that is percent-level (0.4% at 100 Msun, 1.9% at 10 Msun, measured with the weighting
# defect below removed), so 5% bounds it with margin.
TOL_FEEDBACK = 0.05

# name: (gas cell mass [Msun], TimeMax, MaxSizeTimestep)
CASES = {
    "m100_resolved": (100.0, T_END, 0.005),
    "m100_TimeMax1x": (100.0, T_END, T_END),
    "m100_TimeMax100x": (100.0, 100 * T_END, T_END),
    "m10_TimeMax1x": (10.0, T_END, T_END),
    "m10_TimeMax100x": (10.0, 100 * T_END, T_END),
}

_HYDRO_XFAIL = (
    "Supernova energy coupled into cells on long timesteps is not conserved: measured (dE - E_fb)/E_fb = "
    "+0.45 (m100, TimeMax 1x; -0.48 on 1 rank), +2.78 (m100, 100x), +2.49 (m10, 1x), +1.63 (m10, 100x), "
    "nearly all of it at the first supernova into quiescent gas. Feedback is deposited after the density "
    "pass and before the hydro force pass, so the closing half-kick of a receiving cell's old (long) step "
    "applies the post-explosion pressure forces; capping the step at 5 kyr closes the budget to <0.3%."
)
_FEEDBACK_XFAIL = (
    "Mechanical feedback does not couple E_SN when the ejecta mass differs from the star particle mass: "
    "measured E_fb/(N_SN*1e51 erg) = 1.056-1.082 at 100 Msun (0.87 at 1 Msun). The weighting pass (loop -1 "
    "in mechanical_fb.cc) sets the ejecta mass to the star particle mass, so the mass-loading factor in the "
    "kinetic-energy coefficients is wrong and the momentum solution mis-sizes the kinetic energy; using the "
    "true ejecta mass there gives 1.004."
)

HYDRO_XFAIL_CASES = {"m100_TimeMax1x", "m100_TimeMax100x", "m10_TimeMax1x", "m10_TimeMax100x"}
FEEDBACK_XFAIL_CASES = {"m100_resolved", "m100_TimeMax1x", "m100_TimeMax100x"}


def _param(case, xfail_cases, reason):
    marks = [pytest.mark.xfail(strict=True, reason=reason)] if case in xfail_cases else []
    return pytest.param(case, marks=marks, id=case)


_BUILT = False
_RESULTS = {}


def _build_once():
    global _BUILT
    if not _BUILT:
        build_gizmo_for_test(TEST_NAME, NUM_THREADS)
        _BUILT = True


def _ic_path(m_gas):
    return f"ics/ics_m{m_gas:g}_000"


def _write_params(case, m_gas, time_max, max_dt):
    """Per-case parameterfile (relative to the test directory)."""
    text = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    subs = {"InitCondFile": _ic_path(m_gas), "OutputDir": f"output_{case}",
            "TimeMax": f"{time_max:.10g}", "MaxSizeTimestep": f"{max_dt:.10g}"}
    for key, val in subs.items():
        text, n = re.subn(rf"^{key}\s+\S+", f"{key}    {val}", text, flags=re.M)
        assert n == 1, key
    makedirs(TEST_DIR / "params", exist_ok=True)
    name = f"params/{case}.params"
    (TEST_DIR / name).write_text(text)
    return name


def _synced_times(logfile):
    with open(logfile) as f:
        return [float(m.group(1)) for m in re.finditer(r"^Energy \(synced\) t=(\S+)", f.read(), re.M)]


def _run_case(case):
    """Run GIZMO for one case from the test directory, stopping it once a full synchronization at or
    past T_END has been reported."""
    m_gas, time_max, max_dt = CASES[case]
    from make_sn_cluster_adiabatic_ics import make_ics
    makedirs(TEST_DIR / "ics", exist_ok=True)
    ic = TEST_DIR / f"{_ic_path(m_gas)}.hdf5"
    if not ic.exists():
        make_ics(str(ic), m_gas=m_gas)
    params = _write_params(case, m_gas, time_max, max_dt)
    _robust_replace_dir(str(TEST_DIR / f"output_{case}"))
    makedirs(TEST_DIR / f"output_{case}")
    env = dict(environ, OMP_NUM_THREADS=str(NUM_THREADS), GIZMO_STOPFILE_INTERVAL_SEC="0")
    cmd = mpi_launch_command(NUM_RANKS, NUM_THREADS) + ["./GIZMO", params, "2"]
    out, err = TEST_DIR / f"test_{TEST_NAME}_{case}.out", TEST_DIR / f"test_{TEST_NAME}_{case}.err"
    timeout = _resolve_test_timeout(None)
    t_start = time.time()
    with open(out, "w") as fo, open(err, "w") as fe:
        proc = subprocess.Popen(cmd, cwd=TEST_DIR, stdout=fo, stderr=fe, env=env)
        while proc.poll() is None:
            time.sleep(2)
            times = _synced_times(out)
            if times and times[-1] >= T_END * (1 - 1e-9):
                (TEST_DIR / f"output_{case}" / "stop").touch()
            if timeout is not None and time.time() - t_start > timeout:
                proc.kill()
                pytest.skip(f"{case} exceeded {timeout}s timeout; GIZMO subprocess killed")
    if proc.returncode != 0:
        raise RuntimeError(f"GIZMO exited with status {proc.returncode} for {case}; see {out}, {err}")


def energy_budget(case):
    """(E - E0, E_fb, N_SN) at the last full synchronization at or before T_END."""
    if case in _RESULTS:
        return _RESULTS[case]
    m_gas = CASES[case][0]
    if not environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        _build_once()
        _run_case(case)
    rows = []
    with open(TEST_DIR / f"test_{TEST_NAME}_{case}.out") as f:
        for line in f:
            if line.startswith("Energy (synced)"):
                kv = dict(x.split("=") for x in line.split()[2:])
                rows.append([float(kv[k]) for k in ("t", "E_tot", "E_fb")])
    rows = np.array(rows)
    reached = rows[rows[:, 0] <= T_END * (1 + 1e-9)]
    assert len(reached) and reached[-1, 0] > 0.95 * T_END, f"{case}: run never reached t={T_END}"
    t, E, E_fb = reached[-1]
    with h5py.File(TEST_DIR / f"{_ic_path(m_gas)}.hdf5", "r") as F:  # the gas starts static
        E0 = float(np.sum(F["PartType0/Masses"][:] * F["PartType0/InternalEnergy"][:]))
    events = np.atleast_2d(np.loadtxt(TEST_DIR / f"output_{case}" / "MechFeedbackEvents.txt"))
    n_sn = float(events[events[:, 0] <= t, 3].sum())
    assert n_sn >= 2, f"{case}: only {n_sn:g} supernovae by t={t:g}; the test needs overlapping events"
    total = (E - E0 - n_sn * E51_CODE) / (n_sn * E51_CODE)
    print(f"{case}: t={t:.4g} N_SN={n_sn:g} (dE-E_fb)/E_fb={(E - E0 - E_fb) / E_fb:+.4f} "
          f"E_fb/(N_SN E51)={E_fb / (n_sn * E51_CODE):.4f} total (dE-N_SN E51)/(N_SN E51)={total:+.4f}")
    _RESULTS[case] = (E - E0, E_fb, n_sn)
    return _RESULTS[case]


@pytest.mark.parametrize("case", [_param(c, HYDRO_XFAIL_CASES, _HYDRO_XFAIL) for c in CASES])
def test_hydro_conserves_energy(case):
    dE, E_fb, n_sn = energy_budget(case)
    err = (dE - E_fb) / E_fb
    assert abs(err) < TOL_HYDRO, (
        f"{case}: gas energy change differs from the energy feedback deposited: (dE - E_fb)/E_fb = {err:+.4f} "
        f"(tolerance {TOL_HYDRO:.0%}); dE = {dE / E51_CODE:.3f} E51, E_fb = {E_fb / E51_CODE:.3f} E51, N_SN = {n_sn:g}"
    )


@pytest.mark.parametrize("case", [_param(c, FEEDBACK_XFAIL_CASES, _FEEDBACK_XFAIL) for c in CASES])
def test_feedback_delivers_sn_energy(case):
    dE, E_fb, n_sn = energy_budget(case)
    err = E_fb / (n_sn * E51_CODE) - 1
    assert abs(err) < TOL_FEEDBACK, (
        f"{case}: mechanical feedback deposited {E_fb / E51_CODE:.3f} E51 for {n_sn:g} supernovae: "
        f"E_fb/(N_SN*1e51 erg) - 1 = {err:+.4f} (tolerance {TOL_FEEDBACK:.0%})"
    )
