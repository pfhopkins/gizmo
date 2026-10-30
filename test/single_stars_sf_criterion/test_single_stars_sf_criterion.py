"""single_stars_sf_criterion test: the star-formation criterion of IMF-sampled single stars (GALSF_SINGLE_STARS_SF_CRITERION,
which implies the gas collection and the star record) and the cool-down after a birth, on a magnetised lattice at n_H ~ 1e4 cm^-3 whose temperatures put the
Jeans mass on either side of the limit. See README.md."""
import os
import shutil
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "single_stars_sf_criterion"
TEST_DIR = Path(__file__).parent
G_CODE = 6.6743e-8 * 1.989e33 * (3.085678e18 / 1e5) ** 2 / 3.085678e18 ** 3  # pc, Msun, km/s


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME)
    return True


def run(tag, ranks=2, **overrides):
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_sf_criterion_ics import make_single_stars_sf_criterion_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    d.mkdir()
    os.chdir(d)
    try:
        make_single_stars_sf_criterion_ics(f"{TEST_NAME}_ics.hdf5")
        lines, seen = [], set()
        for line in Path(TEST_DIR / f"{TEST_NAME}.params").read_text().splitlines():
            key = line.split()[0] if line.split() else ""
            lines.append(f"{key}    {overrides[key]}" if key in overrides else line)
            seen.add(key)
        lines += [f"{k}    {v}" for k, v in overrides.items() if k not in seen]
        Path(f"{TEST_NAME}.params").write_text("\n".join(lines) + "\n")
        if not Path("GIZMO").exists():
            os.symlink(TEST_DIR / "GIZMO", "GIZMO")
        run_test(TEST_NAME, num_mpi_ranks=ranks)
        snap = sorted(Path("output").resolve().glob("snapshot_*.hdf5"))[-1]
        births = Path("output/single_star_births.txt").resolve()
    finally:
        os.chdir("../../../")
    return snap, (np.atleast_2d(np.loadtxt(births)) if births.exists() else np.zeros((0, 12)))


def decisions(snap):
    """SF rate > 0 per cell, and the Jeans ratio M_J / (FacSfThreshMJ N_ngb m) recomputed from the snapshot with the
    physical density and the physical sound speed"""
    with h5py.File(snap, "r") as F:
        g, a = F["PartType0"], float(F["Header"].attrs["Time"]) if F["Header"].attrs.get("ComovingIntegrationOn", 0) else 1.0
        rho, u, m, sfr = g["Density"][()] / a ** 3, g["InternalEnergy"][()], g["Masses"][()], g["StarFormationRate"][()]
    cs = np.sqrt(5 / 3 * 2 / 3 * u)
    return sfr > 0, 2.9155697 * cs ** 3 / (G_CODE ** 1.5 * np.sqrt(rho)) / (1.0 * 32 * m)


def test_decisions_match_criterion(built):
    forms, ratio = decisions(run("decide")[0])
    mismatch = forms != (ratio < 1)
    assert np.all(np.abs(ratio[mismatch] - 1) < 1e-3)
    assert 0.3 < np.mean(forms) < 0.7


def test_density_threshold(built):
    forms, ratio = decisions(run("dense_threshold", CritPhysDensity=1e6)[0])
    assert np.mean(ratio < 1) > 0.3 and not np.any(forms)  # the Jeans test would pass, the density threshold does not


LONG = dict(SfEffPerFreeFall=100, TimeMax=0.002, TimeBetSnapshot=0.002, MaxSizeTimestep=0.0002)


def repeat_intervals(rows, frac):
    """for every cell that formed more than one star: time between its births / (frac x its free-fall time)"""
    out = []
    for i in np.unique(rows[:, 1]):
        r = rows[rows[:, 1] == i]
        r = r[np.argsort(r[:, 0])]
        if len(r) > 1:
            out += list(np.diff(r[:, 0]) / (frac * np.sqrt(3 * np.pi / (32 * G_CODE * r[:-1, 11]))))
    return np.array(out)


def test_cooldown(built):
    """16 steps: a cell's next birth comes at least SFCooldownFreeFallFraction x its free-fall time after the last one"""
    _, rows = run("cool", SFCooldownFreeFallFraction=0.002, **LONG)
    assert np.all(rows[:, 8] == 1) and len(rows) > 300
    q = repeat_intervals(rows, 0.002)
    assert len(q) >= 5 and np.all(q >= 1)  # the cool-down is exercised and holds


def test_cooldown_off_fails_interval(built):
    """negative control: without the cool-down some cell forms its next star sooner than the cool-down would allow"""
    _, rows = run("nocool", SFCooldownFreeFallFraction=0, **LONG)
    q = repeat_intervals(rows, 0.002)
    assert len(q) >= 5 and np.any(q < 1)
