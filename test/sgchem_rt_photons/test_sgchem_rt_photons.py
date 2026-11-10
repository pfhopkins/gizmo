"""sgchem_rt_photons: what the moment solver absorbs in the ionising bins is what the chemistry consumes
(reports/design/design_m1_coupling.md). A 40 Msun star in a uniform box at n_H = 100 cm^-3, 0.25 Msun cells, four ionising
bins, RSOL 0.1 so light outruns the front, over 0.3 recombination times. Rate of new ionisations between the second and
last snapshots, plus recombinations (alpha_B n_e n_p), against the table's photon rate.
  test_atomic: atomic gas, (d N_H+ + N_rec) / (Q dt) within 0.8-1.2.
  test_molecular: all H in H2: H+ forms at more than half the photon rate and the front stays inside the box. The star's
    Lyman-Werner light also dissociates H2 here.
  test_radiation_pressure_once: thick gas, production opacities: outward gas momentum 0.7-1.1 of absorbed energy / c.
  test_h0_only_atomic, test_h2_ledger: production RSOL 3e-4, 1e49 photons/s in the H0 bin alone (RT_STARBENCH_TEST), no background G0:
    (new H+ + H2 lost + recombinations + photons added to the bins) / (Q dt) within 0.8-1.2. H2 with SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK.
  test_h2_production_opacity_reported: the same on H2 with production opacity, reported.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, get_cooling_tables, mpi_launch_command

TEST_NAME = "sgchem_rt_photons"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

N, N_H, M_CELL, M_STAR = 24, 100.0, 0.25, 40.0
MSUN, PC, MP = 1.989e33, 3.085678e18, 1.6726e-24
T_UNIT = PC / 1e5
RHO = N_H * 1.4 * MP * PC ** 3 / MSUN
L = N * (M_CELL / RHO) ** (1.0 / 3.0)
ALPHA_B = 2.6e-13
E_H0 = 17.3845  # mean photon energy of the H0 bin with He bins [eV]
BUILD = []


@pytest.fixture(scope="module")
def tables():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    t = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    return t


def build(*flags):
    if BUILD != [flags]:
        build_gizmo_for_test(TEST_NAME, extra_config_flags=flags)
        BUILD[:] = [flags]


def run(tag, tables, x_h2, fixed_q=None, over={}, n=N, box=L, m_cell=M_CELL, n_h=N_H):
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(0.0142)
    tms = 0.3 * tab._i2(tab.life, lM, lZ)
    age = tab._i2(tab.t_pms, lM, lZ) + tms
    with h5py.File(tables["StellarTablesFile"], "r") as F:
        Q = 10 ** tab._i3(F["log_Q_ion"][()].astype(np.float32).astype(np.float64), lM, lZ, np.log10(tms))
    Q = fixed_q or Q
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)
    (d / "output").mkdir(parents=True)
    ab = np.zeros((n ** 3, 3)); ab[:, 0] = x_h2; ab[:, 1] = 1e-4
    make_treecol_pixels_ics(str(d / "output" / "snapshot_000.hdf5"), box, lattice(n, box), [(M_STAR, age, np.array([0.51, 0.51, 0.51]) * box)],
                            gas_mass=m_cell, hsml=0.3 * box / n * 3, tracabund=ab)
    for f in ("TREECOOL", "spcool_tables"):
        if (TEST_DIR / f).exists():
            os.symlink(TEST_DIR / f, d / f)
    T = tables["StellarTablesFile"]
    p = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    p = p.replace("StellarTablesFile           SET_BY_TEST", f"StellarTablesFile           {T}")
    p = p.replace("SGChemTablesPath            SET_BY_TEST", f"SGChemTablesPath            {os.environ.get('SGCHEM_TABLES_DIR', os.path.dirname(T))}")
    over = dict(over, BoxSize=box)
    p = "\n".join([l for l in p.splitlines() if not (l.split() and l.split()[0] in over)] + [f"{k:27s} {v}" for k, v in over.items()]) + "\n"
    (d / f"{TEST_NAME}.params").write_text(p)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(4) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "2", "0"], cwd=d, stdout=out,
                       stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True)
    n_cell = m_cell * MSUN / (1.4 * MP)
    rows = []
    for k in range(1, 5):
        with h5py.File(d / "output" / f"snapshot_00{k}.hdf5", "r") as F:
            ab = F["PartType0/SGChemTracAbund"][()]
            t = F["Header"].attrs["Time"] * T_UNIT
            r = np.linalg.norm(F["PartType0/Coordinates"][()] - F["PartType4/Coordinates"][0], axis=1)
        rows.append((t, ab[:, 1].sum() * n_cell, ab[:, 0].sum() * n_cell, ALPHA_B * n_h * (ab[:, 1] ** 2).sum() * n_cell, r[ab[:, 1] > 0.5].max(initial=0) / box))
    return np.array(rows), Q


def ledger(rows, Q):
    t, nion, nh2, rec, _ = rows.T
    n_rec = np.trapezoid(rec[1:], t[1:])  # snapshots 2-4
    return (nion[-1] - nion[1], nh2[1] - nh2[-1], n_rec, Q * (t[-1] - t[1]))


THIN = ("RT_SPEEDOFLIGHT_REDUCTION=0.1", "RT_DISABLE_RAD_PRESSURE", "SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK")


def test_atomic(tables):
    build(*THIN)
    rows, Q = run("atomic", tables, 0.0)
    a, _, r, q = ledger(rows, Q)
    print("atomic new H+, recombined, photons:", a, r, q, (a + r) / q, rows[:, 4])
    assert 0.8 < (a + r) / q < 1.2
    assert rows[-1, 4] < 0.45


def test_molecular(tables):
    build(*THIN)
    rows, Q = run("molecular", tables, 0.5)
    a, h2, r, q = ledger(rows, Q)
    print("molecular new H+, H2 lost, recombined, photons:", a, h2, r, q, (a + r) / q, rows[:, 4])
    assert (a + r) / q > 0.5
    assert rows[-1, 4] < 0.45


H0_ONLY = ("RT_SPEEDOFLIGHT_REDUCTION=3.0e-4", "RT_DISABLE_RAD_PRESSURE", "RT_STARBENCH_TEST")
H0_OVER = {"TimeMax": 3.6e-3, "TimeBetSnapshot": 0.9e-3, "MaxSizeTimestep": 1e-4}


H0_ONLY_NET = H0_ONLY + ("SGCHEM_RT_ABSORBERS_FOLLOW_THE_NETWORK",)
RSOL_H0 = 3e-4


def bins_photons(tag, snap, e_ev):
    """photons still in the bins, in the source's count: at reduced speed most light of a young source is in flight"""
    with h5py.File(TEST_DIR / f"run_{tag}" / "output" / f"snapshot_00{snap}.hdf5", "r") as F:
        return F["PartType0/PhotonEnergy"][:, 0].sum() * (MSUN * 1e10 / 1.602176634e-12) / e_ev / RSOL_H0


def ledger_h0(tag, rows, Q, h2_counts):
    a, h2, r, q = ledger(rows, Q)
    in_flight = bins_photons(tag, 4, E_H0) - bins_photons(tag, 2, E_H0)
    return (a + (h2 if h2_counts else 0) + r + in_flight) / q


def test_h0_only_atomic(tables):
    """no background G0 and no FUV: the ionising rates still reach the network"""
    build(*H0_ONLY)
    rows, Q = run("h0_atomic", tables, 0.0, fixed_q=1e49, over=H0_OVER)
    x = ledger_h0("h0_atomic", rows, Q, False)
    print("H0 only atomic ledger with photons in flight:", x)
    assert 0.8 < x < 1.2


def test_h2_ledger(tables):
    build(*H0_ONLY_NET)
    rows, Q = run("h2_ledger", tables, 0.5, fixed_q=1e49, over=H0_OVER)
    x = ledger_h0("h2_ledger", rows, Q, True)
    print("H2 ledger with photons in flight:", x)
    assert 0.8 < x < 1.2


def test_h2_production_opacity_reported(tables):
    """production: rt_kappa does not see H2, the chemistry still takes rt_phot_H2. Reported for the decision on the opt-in"""
    build(*H0_ONLY)
    rows, Q = run("h2_sd", tables, 0.5, fixed_q=1e49, over=H0_OVER)
    print("H2, ledger with photons in flight:", ledger_h0("h2_sd", rows, Q, True), "front", rows[-1, 4])


def radial_momentum(tag, box):
    with h5py.File(TEST_DIR / f"run_{tag}" / "output" / "snapshot_004.hdf5", "r") as F:
        g = F["PartType0"]; c = F["PartType4/Coordinates"][0]
        r = g["Coordinates"][()] - c; r -= box * np.round(r / box)
        return (g["Masses"][()] * (g["Velocities"][()] * r).sum(axis=1) / np.linalg.norm(r, axis=1)).sum(), g["PhotonEnergy"][()].sum(), F["Header"].attrs["Time"]


def test_radiation_pressure_once(tables):
    """no infrared bin (re-absorbed IR adds momentum by design). Star in gas thick in every dust bin (pure absorption): the outward momentum the moment forces add, force build minus
    the same run without them (gas pressure of the heated cells cancels), over absorbed energy / c, 0.7-1.1;
    absorbed = injected - left in the bins"""
    rsol, t_end, box = 1e-2, 2e-3, 1.0
    over = {"TimeMax": t_end, "TimeBetSnapshot": t_end / 4, "MaxSizeTimestep": 5e-5}
    kw = dict(n=16, box=box, m_cell=0.04, n_h=1e5)
    build(f"RT_SPEEDOFLIGHT_REDUCTION={rsol}", "RT_DISABLE_RAD_PRESSURE")
    run("rp_off", tables, 0.0, over=over, **kw)
    build(f"RT_SPEEDOFLIGHT_REDUCTION={rsol}")
    run("rp", tables, 0.0, over=over, **kw)
    p_on, e_bins, t = radial_momentum("rp", box)
    p_off = radial_momentum("rp_off", box)[0]
    tab = Tables(get_stellar_evolution_tables()["StellarTablesFile"])
    lM, lZ = np.log10(M_STAR), np.log10(0.0142)
    la = np.log10(0.3 * tab._i2(tab.life, lM, lZ))
    with h5py.File(tab.fname, "r") as F:
        L = 10 ** tab._i3(F["log_L_bol"][()].astype(np.float32).astype(np.float64), lM, lZ, la)
    L_code = L / (MSUN * 1e10 / T_UNIT)
    e_abs = L_code * t * rsol - e_bins
    want = e_abs / (rsol * 299792.458)
    print("force momentum / (absorbed / c):", (p_on - p_off) / want, p_on, p_off, "absorbed fraction", e_abs / (L_code * t * rsol))
    assert 0.7 < (p_on - p_off) / want < 1.1
