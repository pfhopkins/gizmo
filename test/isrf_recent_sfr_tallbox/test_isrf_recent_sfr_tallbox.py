"""ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR for the RT_ISRF_BACKGROUND field (no SGCHEM) in the tall-box configuration
(periodic x/y by image sums, open z), with a pure-gas lattice IC.

A 50 pc box of cold gas (n_H ~ 100 cm^-3) forms stars without feedback while 20 pre-existing stars (ages 0-40 Myr from
InitStellarAge) age through the 30 Myr window. At every star-formation call the printed FactorG0 must equal, to 1e-12,
[M(type-4 stars younger than 30 Myr) / 30 Myr] / [Mgas0 / 4 Gyr] rebuilt from the final snapshot,
the running sum equal the full scan, and the background field ISRF_BACKGROUND_STRENGTH = max(F, 0.324e-2 / 1.7)
(InterstellarRadiationFieldStrength = 1), printed as G0 = 1.7 x that. F must rise with star formation and fall when an
old star leaves the window.
"""
import re
import numpy as np
import h5py
import pytest

from gizmo.test import build_and_run_test, clean_test_outputs, default_mpi_ranks, get_cooling_tables, get_final_snapshot

TEST_NAME = "isrf_recent_sfr_tallbox"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
BOX, T_BEGIN, Z_SHIFT = 0.05, 0.0, 0.075
CODE_TIME_IN_GYR = 3.085678e21 / 1e5 / (1e9 * 365.25 * 86400)
LINE = re.compile(r"RECENT_SFR: (\S+) (\S+) (\S+) (\d+) (\S+) (\S+)")


def make_ics(out):
    """lifted to the middle of the 4-times-taller box: 4096 gas cells at n_H ~ 100 cm^-3, T ~ 100 K, 20 type-4 stars"""
    rng = np.random.default_rng(1)
    g = (np.arange(16) + 0.5) / 16 * BOX
    xg = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + [0, 0, Z_SHIFT]  # in the middle of the tall box, away from the z faces (stars crossing them are removed)
    ng, ns, rho = len(xg), 20, 3.25
    with h5py.File(out, "w") as f:
        p0 = f.create_group("PartType0")
        p0["Coordinates"] = xg; p0["Velocities"] = np.zeros((ng, 3)); p0["ParticleIDs"] = np.arange(1, ng + 1, dtype=np.uint64)
        p0["Masses"] = np.full(ng, rho * BOX**3 / ng); p0["InternalEnergy"] = np.full(ng, 1.0)
        p4 = f.create_group("PartType4")
        p4["Coordinates"] = rng.random((ns, 3)) * BOX + [0, 0, Z_SHIFT]; p4["Velocities"] = np.zeros((ns, 3))
        p4["ParticleIDs"] = np.arange(ng + 1, ng + ns + 1, dtype=np.uint64)
        p4["Masses"] = 2e-5 * (1 + 0.05 * np.arange(ns))
        h = f.create_group("Header"); npart = np.array([ng, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32); h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = T_BEGIN; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = BOX
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Flag_DoublePrecision"] = 1


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
def test_isrf_recent_sfr_tallbox(num_mpi_ranks):
    clean_test_outputs(TEST_NAME)
    get_cooling_tables(TEST_DIR)
    make_ics(IC_FILE)
    build_and_run_test(TEST_NAME, num_mpi_ranks)
    rows = np.array([[float(v) for v in m.groups()] for m in map(LINE.search, open(f"{TEST_DIR}/test_{TEST_NAME}.out")) if m])
    assert len(rows) > 10, "FactorG0 is not being updated"
    t, F, F_scan, g0 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 4]
    with h5py.File(get_final_snapshot(TEST_NAME)) as f:
        tf, ms, t_snap = f["PartType4/StellarFormationTime"][:], f["PartType4/Masses"][:], f["Header"].attrs["Time"]
    k = t < t_snap  # stars formed at the final sync point are written after the last snapshot
    with h5py.File(IC_FILE) as f:
        mgas0 = f["PartType0/Masses"][:].sum()
    age = (t[k, None] - tf[None, :]) * CODE_TIME_IN_GYR
    m_exp = np.sum(ms[None, :] * ((age >= 0) & (age < 0.03)), axis=1)
    np.testing.assert_allclose(F[k], (m_exp / 0.03) / (mgas0 / 4.0), rtol=1e-12, atol=0)
    np.testing.assert_allclose(F, F_scan, rtol=1e-12, atol=0)
    np.testing.assert_allclose(g0, 1.7 * np.maximum(F, 0.324e-2 / 1.7), rtol=1e-14, atol=0)
    d = np.diff(F)
    n_formed = np.sum(tf > T_BEGIN)
    print(f"{len(rows)} calls, {n_formed} stars formed, F {F.min():.6g}-{F.max():.6g}")
    assert n_formed > 0 and np.any(d > 0) and np.any(d < 0), "F should rise with star formation and fall as stars age out"
