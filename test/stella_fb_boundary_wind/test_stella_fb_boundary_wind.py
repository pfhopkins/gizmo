"""Boundary feedback (FB_G17_BOUNDARY_INJECTION, the zoom production build): with RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB, wind, AGB
and radiation-pressure deposits made after the closing kick redo the hydro pass as SN deposits do; without it only SN/Ia deposits
do (the control arm).

Config_stella_zoom.sh plus the isolated-feedback test hooks, adiabatic (RESOLVEDISM_ADIABATIC_NOCOOL); non-comoving. Uniform n_H = 10,
1 Msun cells, 16 pc periodic box, at 100 K and at 1e4 K. A 60 Msun star blows its wind from birth and a 3 Msun star dies as an AGB
star at 2 kyr (forced); no SN occurs. Stellar radiation off.

Metric and bars, fixed before the runs (2026-11-18):
  excess(t) = [E_th + E_kin](t) - [E_th + E_kin](0) + E_sf_removed(t) - E_fb_booked(t), divided by E_fb_booked(t_end),
  at every ENERGYinfo.txt output (columns 5, 6, 14, 10). E_sf_removed is the booked gas energy taken by star formation.
  - with the flag: |excess| <= 0.40 % in both boxes (the SN bar of this build family, reports/sn_boundary_and_approach_dt.md),
    the hydro pass is redone (G17RECOMPUTE), and the mass, element and energy step traces close;
  - without the flag: no hydro recompute (no SN in the run), i.e. the SN-only behaviour;
  - the flag must matter: max|excess| without it minus max|excess| with it >= 0.05 percentage points in at least one box.
"""

import os
import shutil
import subprocess
from os.path import abspath, dirname, join

import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "stella_fb_boundary_wind"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
EXTRA_FLAGS = ("GALSF_RESOLVEDISM_ISOLATED_FB_TEST", "RESOLVEDISM_ADIABATIC_NOCOOL", "OUTPUT_ENERGY_STEP_TRACE")
FLAG = "RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB"
BAR_PERCENT = 0.40
DIFF_PERCENT = 0.05
RUN_TIMEOUT = 5400  # s


def _excess(out):
    a = np.loadtxt(join(out, "ENERGYinfo.txt"), comments="#", ndmin=2)
    E, Efb, Esf = a[:, 4] + a[:, 5], a[:, 9], a[:, 13]
    return (E - E[0] + Esf - Efb) / Efb[-1] * 100


def _run(binary, fbc, name, T):
    d = join(TEST_DIR, "run_" + name); out = join(d, "output")
    shutil.rmtree(d, ignore_errors=True); os.makedirs(out)
    os.symlink(REPO, join(d, "gizmo"))
    fbc.make_box(join(d, "box_ics.hdf5"), fbc.ZSUN, T=T)
    base = open(join(REPO, "test", "stella_fb_conservation", "stella_fb_conservation.params")).read().splitlines()
    over = {"TestNumStars": 2, "TestStarMass": 60.0, "TestStarMass2": 3.0, "TestSNTime0": 1.0e-3, "TestSNSpacing": -0.998e-3,
            "TimeMax": 2.0e-5, "TimeBetStatistics": 2.0e-7, "TestFUVZero": 1}
    lines = [l for l in base if not (l.split() and l.split()[0] in over)] + [f"{k} {v}" for k, v in over.items()]
    open(join(d, "run.params"), "w").write("\n".join(lines) + "\n")
    with open(join(d, "run.log"), "w") as log:
        r = subprocess.run(mpi_launch_command(2) + [binary, "run.params"], stdout=log, stderr=subprocess.STDOUT, cwd=d, timeout=RUN_TIMEOUT)
    text = open(join(d, "run.log")).read()
    assert r.returncode == 0 and "Simulation ends." in text, f"{name}: " + text[-1500:]
    assert "RESOLVEDISM AGB:" in text and "RESOLVEDISM WIND:" in text and "RESOLVEDISM SN:" not in text
    return text, out


def test_boundary_wind_energy():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fbc", join(REPO, "test", "stella_fb_conservation", "test_stella_fb_conservation.py"))
    fbc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fbc)
    cfg = open(join(REPO, "Config_stella_zoom.sh")).read()
    open(join(TEST_DIR, "Config.sh"), "w").write(cfg + "\n# test hooks\n" + "\n".join(EXTRA_FLAGS) + "\n")
    binaries = {}
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        for arm, extra in (("on", (FLAG,)), ("off", ())):
            build_gizmo_for_test(TEST_NAME, extra_config_flags=extra)
            binaries[arm] = join(TEST_DIR, f"GIZMO_{arm}")
            shutil.move(join(TEST_DIR, "GIZMO"), binaries[arm])
    finally:
        os.chdir(cwd)
    res = {}
    for T in (100.0, 1.0e4):
        for arm in ("on", "off"):
            text, out = _run(binaries[arm], fbc, f"{arm}_T{int(T)}", T)
            ex = _excess(out)
            res[(arm, T)] = (text.count("G17RECOMPUTE"), ex, out)
            print(f"T = {T:g} K, flag {arm}: {res[(arm, T)][0]} recomputes, excess {ex.min():+.4f} .. {ex.max():+.4f} %, end {ex[-1]:+.4f} %")
    for T in (100.0, 1.0e4):
        n_on, ex_on, out_on = res[("on", T)]
        n_off, ex_off, _ = res[("off", T)]
        assert n_on > 0, f"T = {T:g}: no hydro recompute with {FLAG}"
        assert n_off == 0, f"T = {T:g}: {n_off} recomputes without {FLAG} (SN-only behaviour expected)"
        assert np.abs(ex_on).max() <= BAR_PERCENT, f"T = {T:g}: |excess| {np.abs(ex_on).max():.4f} % > {BAR_PERCENT} %"
        rep = keep_and_check_traces(out_on, out_on)
        assert rep.ok, rep.summary()
    gain = max(np.abs(res[("off", T)][1]).max() - np.abs(res[("on", T)][1]).max() for T in (100.0, 1.0e4))
    print(f"largest improvement from {FLAG}: {gain:+.4f} percentage points")
    assert gain >= DIFF_PERCENT, f"{FLAG} changes max|excess| by only {gain:+.4f} percentage points"
