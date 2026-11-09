"""sgchem_cr_fluid_energy: energy exchange between the CR fluid and the SGChem chemistry, as stella_dev (sgchem.cc and
cosmic_ray_utilities.cc): the ionisation heating the chemistry solve applies (the share of zeta from the CR fluid) comes
out of the CR energy, and the CR fluid's own loss is hadronic plus Coulomb on free electrons only (no ionisation term,
which the chemistry already heats with). A uniform periodic box of 8^3 cells of 1 Msun (n_H ~ 30 cm^-3, x_H2 = 0.4) with
2 eV cm^-3 of CRs, no sources, 2 ranks, 0.29 Myr; ledger rows at 7 snapshots.
  test_cr_energy_closes: the ionisation energy taken from the fluid is the fluid's share (> 0.5, <= 1) of the
    chemistry's CR ionisation heating and at least 1e-3 of E_CR,0; E_CR - E_CR,0 = -(fluid losses + ionisation energy
    taken by the chemistry) up to the CR-gas exchange at the kicks (not booked; reported, below 1e-3 of E_CR,0).
  test_cr_loss_rate_as_stella_dev: the CR fluid losses between rows = (0.87 x 6.37e-16 + 0.53 x 3.09e-16 n_e X_H)
    (rho / m_p) integral(E_CR dt) (stella_dev's rate and density argument, n_e from the snapshots, trapezoid; 2 %).
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

TEST_NAME = "sgchem_cr_fluid_energy"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402

L, N_SIDE, E_CR_CELL = 8.0, 8, 4.7  # 4.7 code energy per 1 pc^3 cell = 2.0 eV cm^-3
UNIT_E, UNIT_T = 1.989e33 * 1e10, 3.085678e18 / 1e5  # erg, s
X_H, M_P = 0.7, 1.6726e-24


@pytest.fixture(scope="module")
def run():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    d = TEST_DIR / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    n = N_SIDE ** 3
    ab = np.tile([0.4, 1e-4, 0.0], (n, 1))
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), [], hsml=2.0, tracabund=ab,
                            e_cr=np.full(n, E_CR_CELL))
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text().replace(
        "SET_BY_TEST", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    lines = (d / "ledger.txt").read_text().splitlines()
    head = [t for t in lines[0].split()[2:] if t.endswith(("_Msun", "_erg"))]
    rows = np.atleast_2d(np.loadtxt(d / "ledger.txt"))
    assert rows.shape[1] == len(head) + 1
    return {k: rows[:, j + 1] for j, k in enumerate(head)} | {"time": rows[:, 0]}, d


def test_cr_energy_closes(run):
    b, d = run
    need = ("E_CR_erg", "E_cr_losses_erg", "E_cr_ionisation_from_fluid_erg", "E_chem_cr_ionisation_heating_erg")
    assert all(k in b for k in need), f"ledger columns: {sorted(b)}"
    E0 = N_SIDE ** 3 * E_CR_CELL * UNIT_E
    res = (b["E_CR_erg"] - E0) + b["E_cr_losses_erg"] + b["E_cr_ionisation_from_fluid_erg"]
    print(f"E_CR {b['E_CR_erg'][-1] / E0:.4f} of E_CR,0; losses {b['E_cr_losses_erg'][-1] / E0:.4e}, ionisation "
          f"{b['E_cr_ionisation_from_fluid_erg'][-1] / E0:.4e}; residual / E_CR,0 max {np.max(np.abs(res)) / E0:.2e}")
    assert np.max(np.abs(res)) <= 1e-3 * E0
    assert b["E_cr_losses_erg"][-1] > 0.01 * E0 and b["E_cr_ionisation_from_fluid_erg"][-1] > 1e-3 * E0
    share = b["E_cr_ionisation_from_fluid_erg"][1:] / b["E_chem_cr_ionisation_heating_erg"][1:]
    print(f"fluid share of the CR ionisation heating: {share.min():.6f}-{share.max():.6f}")
    assert np.all((share > 0.5) & (share <= 1.0))


def test_cr_loss_rate_as_stella_dev(run):
    b, d = run
    snaps = sorted(d.glob("snapshot_*.hdf5"))[1:]  # the outputs (snapshot_000 is the initial conditions)
    assert len(snaps) == len(b["time"])
    ne, nh = [], []
    for s in snaps:
        with h5py.File(s) as f:
            g = f["PartType0"]
            ne.append(np.mean(g["ElectronAbundance"][:]))
            nh.append(np.mean(g["Density"][:]) * 1.989e33 / 3.085678e18 ** 3 / M_P)  # rho / m_p, as the call passes it
    ne, nh, t = np.array(ne), np.array(nh), b["time"] * UNIT_T
    rate = (0.87 * 6.37e-16 + 0.53 * 3.09e-16 * ne * X_H) * nh
    for k in range(1, len(t)):
        want = 0.5 * (rate[k] * b["E_CR_erg"][k] + rate[k - 1] * b["E_CR_erg"][k - 1]) * (t[k] - t[k - 1])
        got = b["E_cr_losses_erg"][k] - b["E_cr_losses_erg"][k - 1]
        print(f"t = {b['time'][k]:.2f}: n_e {ne[k]:.3e}, losses {got:.4e} erg, stella_dev rate {want:.4e} erg, ratio {got / want:.4f}")
        assert abs(got / want - 1) < 0.02
