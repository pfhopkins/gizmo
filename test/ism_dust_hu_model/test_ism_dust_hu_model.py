"""ism_dust_hu_model test: the dust processes of Hu et al. 2019, 2023, 2026 on the fields of the ISM dust model
(GALSF_ISMDUSTCHEM_HU_MODEL) with SGChem: sputtering and growth against the analytic rates, the metal ledger. See README.md."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, get_cooling_tables, mpi_launch_command

TEST_NAME = "ism_dust_hu_model"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "single_stars_metals"))
from make_single_stars_metals_ics import make_single_stars_metals_ics  # noqa: E402

MP, YR, A_EFF = 1.67262192e-24, 3.15576e7, np.sqrt(0.005 * 0.25)
T_UNIT_YR = 3.085678e13 / 3.155e7  # time unit [yr], with the code's year
ATOMS, ATOMIC = np.array([4.0, 1.0, 1.0, 1.0]), np.array([15.99, 24.305, 28.086, 55.845])  # FeMgSiO4: O, Mg, Si, Fe
FRAC = ATOMS * ATOMIC / (ATOMS * ATOMIC).sum()
SOLAR = np.array([0.014, 0.2485, 2.47653512e-3, 7.25454958e-4, 6.00285929e-3, 1.31578971e-3, 7.41243589e-4, 6.96209192e-4, 3.23764213e-4, 6.7169384e-5, 1.35284992e-3])  # FIRE slots, the tables' Z = 0.014
ABHE, Z_SUN = 0.1, 0.014


@pytest.fixture(scope="module")
def exe():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    get_cooling_tables(str(TEST_DIR))
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return TEST_DIR / "GIZMO"


def seed(Z):
    """half of the carbon and 82% of the silicon in dust (stoichiometric silicate): element dust (slot 0 total), four sources
    (fractions, all SNe), species silicate and carbon"""
    sil = 0.82 * Z[7] / FRAC[2]
    d = np.zeros(11)
    d[2] = 0.5 * Z[2]
    d[[4, 6, 7, 10]] = sil * FRAC
    d[0] = d[1:].sum()
    return np.concatenate([d, [0, 0, 1, 0]]), np.array([sil, d[2]])


def run(tag, exe, n_H, T, over, ranks=2, n=8):
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        if d.exists():
            shutil.rmtree(d)
        (d / "output").mkdir(parents=True)
        dust, species = seed(SOLAR)
        L = make_single_stars_metals_ics(str(d / "output" / "snapshot_000.hdf5"), SOLAR, n=n, n_H=n_H, T=T, gas_fields={"DustMetallicity": dust, "DustSpeciesAbundance": species})
        for f in ("TREECOOL", "spcool_tables"):
            if (TEST_DIR / f).exists():
                os.symlink(TEST_DIR / f, d / f)
        p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
        p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ['SGCHEM_TABLES_DIR']}")
        o = {"BoxSize": L, **over}
        p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in o)] + [f"{k:27s} {v}" for k, v in o.items()]) + "\n"
        (d / "run.params").write_text(p)
        with open(d / "run.out", "w") as out:
            subprocess.run(mpi_launch_command(ranks) + [str(exe), "run.params", "2", "0"], cwd=d, stdout=out, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    return sorted((d / "output").glob("snapshot_*.hdf5"))


def snap(f):
    with h5py.File(f, "r") as F:
        g = F["PartType0"]
        m = g["Masses"][()]
        i = np.argsort(g["ParticleIDs"][()])
        o = dict(t=float(F["Header"].attrs["Time"]) * T_UNIT_YR, m=m[i], Z=g["Metallicity"][()][i][:, :11], D=g["DustMetallicity"][()][i], S=g["DustSpeciesAbundance"][()][i])
        for k, n in (("rho", "Density"), ("T", "Temperature"), ("x", "SGChemTracAbund")):
            o[k] = g[n][()][i] if n in g else None
    return o


def Yth(T, c):
    x = np.log10(np.maximum(T, 3e4))
    return np.where(T < 3e4, 0, 10 ** np.polyval(c[::-1], x))


Y_C = [-2.34333937e+02, 1.38485732e+02, -3.39021615e+01, 4.17705353e+00, -2.58281473e-01, 6.38827523e-03]
Y_S = [-2.34790500e+02, 1.33208637e+02, -3.13027448e+01, 3.71345730e+00, -2.21823668e-01, 5.31746427e-03]


def nH(s):
    return s["rho"] * 1.989e33 / 3.085678e18 ** 3 / MP * (1 - s["Z"][:, 0] - s["Z"][:, 1])


def ledger(snaps):
    """the metals do not change, slot by slot; dust inside its element; sources and species add up to the dust"""
    a = [snap(f) for f in snaps]
    mz0 = (a[0]["m"][:, None] * a[0]["Z"]).sum(0)
    for s in a[1:]:
        mz = (s["m"][:, None] * s["Z"]).sum(0)
        assert np.all(np.abs(mz / mz0 - 1) < 1e-12), mz / mz0 - 1
        assert np.all(s["D"][:, 1:11] <= s["Z"][:, 1:11] * (1 + 1e-12))
        assert np.all(np.abs(s["D"][:, 11:15].sum(1) - 1) < 1e-12)  # the snapshots hold the sources as fractions
        assert np.all(np.abs(s["S"].sum(1) / s["D"][:, 0] - 1) < 1e-12)
        assert np.all(np.abs(s["D"][:, 1:11].sum(1) / s["D"][:, 0] - 1) < 1e-12)


def test_thermal_sputtering(exe):
    """hot gas at rest: the dust falls as exp(-3 n_H Y(T) t / a_eff) with the Nozawa+06 fits of Hu+19, a_eff the MRN effective size"""
    snaps = run("hot", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01})
    ledger(snaps)
    for a, b in zip(snaps[1:-1], snaps[2:]):
        s0, s1 = snap(a), snap(b)
        dt = s1["t"] - s0["t"]
        T = 0.5 * (s0["T"] + s1["T"])
        for slot, c, name in ((2, Y_C, "C"), (7, Y_S, "Si")):
            got = np.log(s1["D"][:, slot].sum() / s0["D"][:, slot].sum())
            want = -(3 * nH(s0) * Yth(T, c) / A_EFF * dt).mean()
            print(f"{name}: T = {T.mean():.3g} K, ln(D1/D0) = {got:.5f}, analytic {want:.5f}, ratio {got / want:.4f}")
            assert abs(got / want - 1) < 0.01


def logistic(m0, K, r, dt):
    return K / (1 + (K / m0 - 1) * np.exp(-r * dt))


def test_growth_below_300K(exe):
    """cold dense gas: the logistic of Hu+23/26, rate Z'/(1.5 Gyr) n_H sqrt(T/100) (a_eff/0.035)^-1, pools without the CO-bound C, O"""
    snaps = run("cold", exe, 1.0e3, 50.0, {"TimeMax": 0.15, "TimeBetSnapshot": 0.05, "MaxSizeTimestep": 2.5e-3})
    ledger(snaps)
    x_h = 1 / (1 + 4 * ABHE)
    s0, s1 = snap(snaps[1]), snap(snaps[-1])
    dt = s1["t"] - s0["t"]
    T = 0.5 * (s0["T"] + s1["T"])
    print(f"T {T.mean():.1f} K, x(CO) {s1['x'][:, 2].mean():.3g}")
    assert T.max() < 300
    r0 = nH(s0) * np.sqrt(T / 100) / 1.5e9 * (0.035 / A_EFF)
    Zp = s0["Z"][:, 0] / Z_SUN
    xco = 0.5 * (s0["x"][:, 2] + s1["x"][:, 2])
    P = s0["Z"][:, 2] - 12 * xco * x_h
    C1 = logistic(s0["D"][:, 2], P, Zp * r0 * P / s0["Z"][:, 2], dt)
    pools = np.stack([s0["Z"][:, 4] - 16 * xco * x_h, s0["Z"][:, 6], s0["Z"][:, 7], s0["Z"][:, 10]], 1)
    K, KSi = (pools / FRAC).min(1), pools[:, 2] / FRAC[2]
    S0 = s0["D"][:, 7] / FRAC[2]
    S1 = logistic(S0, K, Zp * r0 * K / KSi, dt)
    want_c, got_c = (C1 - s0["D"][:, 2]).sum(), (s1["D"][:, 2] - s0["D"][:, 2]).sum()
    want_s, got_s = (S1 - S0).sum(), (s1["D"][:, 7] / FRAC[2] - S0).sum()
    print(f"growth C {got_c:.5g} (analytic {want_c:.5g}), silicate {got_s:.5g} (analytic {want_s:.5g})")
    assert got_c > 0 and got_s > 0
    assert abs(got_c / want_c - 1) < 0.02 and abs(got_s / want_s - 1) < 0.02
    assert s1["D"][:, 11].sum() > 0  # gas-dust accretion is booked as a dust source


def test_one_and_four_ranks(exe):
    ref = snap(run("hot", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01})[-1])
    a = snap(run("hot_r4", exe, 1.0, 1e7, {"TimeMax": 0.02, "TimeBetSnapshot": 0.01}, ranks=4)[-1])
    assert abs(a["D"][:, 0].sum() / ref["D"][:, 0].sum() - 1) < 1e-8
