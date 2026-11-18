"""Star-side radiation pressure, anisotropic neighbours (production cooling and chemistry as radp_starside): one 40 Msun star on the face
of a dusty gas half-space (n_H = 300, solar Z, 0.8 Msun cells, no gravity), so the weighted mean neighbour direction has |m| near 0.5.
Every step that books momentum in the radiation pressure channel (FeedbackBudget channel 3, star-side part), the walk must deliver it:
Σ_j |Δp_j| over the gas equals the booked p and Σ_j Δp_j = 0 (no net push, no star recoil), both to round-off (bar 1e-12 relative,
set before running). Kicks along r̂ − m without the Σ|q| normalisation deliver Σ|Δp| ≤ p (1 − |m|²)^{1/2}.
The ledger covers every star, including those that deposit nothing: the first 15 kyr no gas is in reach (TestFBNoGasUntilTime),
and a second run has a 1.9 Msun main-sequence star (back-dated 30 Myr past its pre-main sequence), below the walk's 2 Msun cut.
Neither may book momentum it does not deliver."""

import os
import subprocess
import sys

from gizmo.test import build_gizmo_for_test, mpi_launch_command
from gizmo.conservation_trace import keep_and_check_traces

TEST_NAME = "radp_starside_aniso"
TEST_DIR = f"test/{TEST_NAME}"
BAR = 1e-12


def _run(extra, out):
    extra = dict(extra, OutputDir=out)
    lines = [l for l in open(f"{TEST_NAME}.params").read().splitlines() if not (l.split() and l.split()[0] in extra)]
    open("run.params", "w").write("\n".join(lines + [f"{k} {v}" for k, v in extra.items()]) + "\n")
    subprocess.run(["rm", "-rf", out]); os.makedirs(out)
    res = subprocess.run(mpi_launch_command(2, 1) + ["./GIZMO", "run.params"], stdout=open(f"{out}.log", "w"), stderr=subprocess.STDOUT)
    assert res.returncode == 0, "run failed: " + open(f"{out}.log").read()[-400:]
    rows = [dict(kv.split("=") for kv in l.split()[1:]) for l in open(f"{out}.log") if l.startswith("RADPLEDGER")]
    return [{k: float(v) for k, v in r.items()} for r in rows]


def test_radp_starside_aniso():
    subprocess.run([sys.executable, f"make_{TEST_NAME}_ics.py", f"{TEST_NAME}_ics.hdf5"], cwd=TEST_DIR, check=True)
    build_gizmo_for_test(TEST_NAME, 1)
    repo = os.getcwd(); os.chdir(TEST_DIR)
    try:
        if not os.path.islink("gizmo"):
            os.symlink(repo, "gizmo")  # the parameter file reads the stellar tables under gizmo/
        rows = _run({"TestFBNoGasUntilTime": 1.5e-5}, "output")
        rows_low = _run({"TestStarMass": 1.9, "TestStarInitAgeYr": 3.0e7, "TimeMax": 5.0e-6}, "output_lowmass")
    finally:
        os.chdir(repo)
    assert len(rows) > 0, "no step booked radiation pressure momentum"
    for name, rs in (("40 Msun", rows), ("1.9 Msun", rows_low)):
        bad = [r for r in rs if r["p_book"] <= 0 or abs(r["p_abs"] / r["p_book"] - 1) >= BAR]
        print(f"{name}: {len(rs)} steps with booked or delivered momentum, {len(bad)} where booked != delivered"
              + (f" (first t={bad[0]['t']:.4e}: booked {bad[0]['p_book']:.4e}, delivered {bad[0]['p_abs']:.4e})" if bad else ""))
        assert not bad
    err_vec = max(r["p_vec"] / r["p_book"] for r in rows)
    m2 = sum(r["m2"] for r in rows) / len(rows)
    print(f"mean |m|^2 {m2:.3f}, max |Σdp|/p_book {err_vec:.3e}")
    assert m2 > 0.1, "neighbour distribution not anisotropic"
    assert err_vec < BAR
    for out in ("output", "output_lowmass"):
        rep = keep_and_check_traces(os.path.join(TEST_DIR, out), os.path.join(TEST_DIR, out))
        assert rep.ok, rep.summary()
