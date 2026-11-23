"""Tall-box gravity by the 2D Ewald table (GRAVITY_PERIODIC_XY, Config_ewald.sh), open in z.

An exact lattice sheet of surface density Sigma at z = Z0 in the 1 x 1 x 8 box: periodic in x/y and open in z, it pulls with
exactly a_z = -sign(h) 2 pi G Sigma at every height h (the 16 x 16 lattice's discreteness falls off as exp(-2 pi 16 h), nil at
h >= 0.2). Probes at h = 0.2, 0.5, 1 (inside the table, |dz| <= 2 L) and 2.5, 3.5 (analytic far field), above and below.
Limits fixed before the first run (stella_dev tallbox_gravity_cr bar, Uli): |a_z / (-sign(h) 2 pi G Sigma) - 1| < 1e-3 at every
probe on 1 and 2 ranks with the tight walk; the sheet's own acceleration below 5e-3 of 2 pi G Sigma; mass to 1e-12 and net
momentum as test_tallbox_gravity."""
import sys
from os import chdir, path

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, parse_params, run_test

sys.path.insert(0, path.dirname(path.abspath(__file__)))
import make_ics  # noqa: E402

TEST_NAME = "tallbox_gravity"
TIGHT = {"ErrTolTheta": 0.15, "ErrTolForceAcc": 1e-6}
G, SIGMA = 1.0, 1.0
HEIGHTS = (0.2, 0.5, 1.0, 2.5, 3.5)
LIMITS = {"sheet_probe": 1e-3, "sheet_self": 5e-3, "momentum": 1e-2, "mass": 1e-12}


@pytest.fixture(scope="module")
def built():
    build_gizmo_for_test(TEST_NAME, 1, config_file="Config_ewald.sh")
    chdir(f"test/{TEST_NAME}/")
    keep = make_ics.SHEET_PROBE_HEIGHTS
    make_ics.SHEET_PROBE_HEIGHTS = HEIGHTS
    try:
        make_ics.make_sheet_ics("tallbox_sheet_ics.hdf5")
    finally:
        make_ics.SHEET_PROBE_HEIGHTS = keep
    yield
    chdir("../../")


def _read(snap):
    with h5py.File(snap, "r") as F:
        g = F["PartType1"]
        order = np.argsort(g["ParticleIDs"][:])
        return {k: g[k][:][order] for k in ("Coordinates", "Velocities", "Masses", "Acceleration")}


@pytest.mark.parametrize("ranks", (1, 2))
def test_ewald_sheet_field_at_any_height(built, ranks):
    params = "tallbox_sheet_ewald"
    outdir = parse_params(f"{params}.params")["OutputDir"] + f"_ewald_r{ranks}"
    run_test(TEST_NAME, ranks, 1, params_name=params, param_overrides=dict(OutputDir=outdir, **TIGHT))
    snaps = sorted(__import__("glob").glob(f"{outdir}/snapshot_*.hdf5"))
    d, l = _read(snaps[0]), _read(snaps[-1])
    n_sheet = make_ics.NG ** 2
    h = d["Coordinates"][n_sheet:, 2] - make_ics.Z0
    expect = -np.sign(h) * 2 * np.pi * G * SIGMA
    got = d["Acceleration"][n_sheet:, 2]
    err = np.abs(got / expect - 1)
    self_acc = np.linalg.norm(d["Acceleration"][:n_sheet], axis=1).max() / (2 * np.pi * G * SIGMA)
    print(f"[ranks={ranks}] h {np.round(h, 3)} a_z/(2 pi G Sigma) {np.round(got / (2 * np.pi * G * SIGMA), 6)}; "
          f"max rel err {err.max():.2e}; max |a_sheet| / 2 pi G Sigma = {self_acc:.2e}")
    assert err.max() < LIMITS["sheet_probe"]
    assert self_acc < LIMITS["sheet_self"]
    assert abs(l["Masses"].sum() / d["Masses"].sum() - 1) < LIMITS["mass"]
    p = (l["Masses"][:, None] * l["Velocities"]).sum(0)
    scale = (l["Masses"] * np.linalg.norm(l["Velocities"], axis=1)).sum()
    assert np.linalg.norm(p) / scale < LIMITS["momentum"]
