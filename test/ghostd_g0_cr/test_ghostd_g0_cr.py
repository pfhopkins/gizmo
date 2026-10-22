"""GHOSTD_G0_CR: FactorG0 is set every star-formation call and follows the 30 Myr star formation rate.

A cold dense gas box forms stars (no feedback, so star masses stay fixed) while pre-existing type-4 stars
(ages drawn by init.cc from InitStellarAge, uniform in 0-40 Myr) age through the 30 Myr window. From the final snapshot we rebuild, for every logged call at time t,
M_young(t) = sum of type-4 masses with 0 <= t - t_form < 30 Myr, and require
FactorG0 = (M_young / 30 Myr) / (Mgas0 / 4 Gyr), with Mgas0 the initial gas mass.
"""

import re
import sys
from os import path

import h5py
import numpy as np
import pytest

from gizmo.test import build_and_run_test, clean_test_outputs, default_mpi_ranks, get_cooling_tables, get_final_snapshot

TEST_NAME = "ghostd_g0_cr"
TEST_DIR = f"test/{TEST_NAME}"
IC_FILE = f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"
BOX, T_BEGIN = 0.05, 0.0  # kpc, code time (kpc/(km/s))
CODE_TIME_IN_GYR = 3.085678e21 / 1e5 / (1e9 * 365.25 * 86400)
LINE = re.compile(r"GHOSTD: Time=(\S+) Mgas0=(\S+) M_young=(\S+) FactorG0=(\S+)")


def make_ics(out):
    """4096 gas cells at n_H ~ 100 cm^-3, T ~ 100 K, plus 20 type-4 stars (ages set at startup from InitStellarAge)."""
    rng = np.random.default_rng(1)
    g = (np.arange(16) + 0.5) / 16 * BOX
    xg = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    ng = len(xg); rho = 3.25  # code density (1e10 Msun/kpc^3), n_H ~ 100 cm^-3
    ns = 20
    with h5py.File(out, "w") as f:
        p0 = f.create_group("PartType0")
        p0["Coordinates"] = xg; p0["Velocities"] = np.zeros((ng, 3)); p0["ParticleIDs"] = np.arange(1, ng + 1, dtype=np.uint64)
        p0["Masses"] = np.full(ng, rho * BOX**3 / ng); p0["InternalEnergy"] = np.full(ng, 1.0)
        p4 = f.create_group("PartType4")
        p4["Coordinates"] = rng.random((ns, 3)) * BOX; p4["Velocities"] = np.zeros((ns, 3))
        p4["ParticleIDs"] = np.arange(ng + 1, ng + ns + 1, dtype=np.uint64)
        p4["Masses"] = 2e-5 * (1 + 0.05 * np.arange(ns))
        h = f.create_group("Header"); npart = np.array([ng, 0, 0, 0, ns, 0])
        h.attrs["NumPart_ThisFile"] = npart.astype(np.int32); h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32); h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = T_BEGIN; h.attrs["Redshift"] = 0.0; h.attrs["BoxSize"] = BOX
        h.attrs["NumFilesPerSnapshot"] = 1; h.attrs["Flag_DoublePrecision"] = 1


def check(logfile, ic_file, final_snapshot):
    rows = np.array([[float(v) for v in m.groups()] for m in map(LINE.search, open(logfile)) if m])
    assert len(rows) > 10, "FactorG0 is not being updated"
    t, mgas0, m_young, factor = rows.T
    with h5py.File(ic_file) as f:
        assert np.allclose(mgas0, f["PartType0/Masses"][:].sum(), rtol=1e-9, atol=0)
    with h5py.File(final_snapshot) as f:
        tf, ms, t_snap = f["PartType4/StellarFormationTime"][:], f["PartType4/Masses"][:], f["Header"].attrs["Time"]
    k = t < t_snap  # stars formed at the final sync point are written after the last snapshot
    age = (t[k, None] - tf[None, :]) * CODE_TIME_IN_GYR
    m_exp = np.sum(ms[None, :] * ((age >= 0) & (age < 0.03)), axis=1)
    assert np.allclose(m_young[k], m_exp, rtol=1e-12, atol=0), "M_young does not match the stars younger than 30 Myr"
    assert np.allclose(factor, (m_young / 0.03) / (mgas0 / 4.0), rtol=1e-12, atol=0)
    n_formed = np.sum(tf > T_BEGIN)
    d = np.diff(factor)
    assert n_formed > 0 and np.any(d > 0) and np.any(d < 0), "FactorG0 should rise with star formation and fall as stars age out"
    return len(rows), n_formed, factor.min(), factor.max()


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(2),))
def test_ghostd_g0_cr(num_mpi_ranks):
    clean_test_outputs(TEST_NAME)
    get_cooling_tables(TEST_DIR)
    make_ics(IC_FILE)
    build_and_run_test(TEST_NAME, num_mpi_ranks)
    check(f"{TEST_DIR}/test_{TEST_NAME}.out", IC_FILE, get_final_snapshot(TEST_NAME))


if __name__ == "__main__":  # standalone: make_ics OUT | check LOG IC SNAP
    if sys.argv[1] == "make_ics":
        make_ics(sys.argv[2])
    else:
        print("calls=%d stars formed=%d FactorG0 range [%g, %g]" % check(*sys.argv[2:5]))
