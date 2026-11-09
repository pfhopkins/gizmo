"""ledger_mass_metals: mass and metal rows of the ledger (OUTPUT_MASS_METAL_ENERGY_LEDGER) in a closed periodic box of
8^3 cells of 1 Msun with SGChem and single stars from the tables, no metal diffusion: two births at the first step
(drawn mass collected from the gas), a young 8 Msun star losing wind mass, a 3 Msun star dying on the AGB (white-dwarf
remnant); 2 ranks, 0.29 Myr, a ledger row at each of 6 snapshots.
  test_ledger_mass_metals: at every row the books close against the state rows (sums over the particles at output),
    relative to the total mass and the total metal mass (1e-12):
      (,0: the initial conditions)  M_stars - M_stars,0 = formed - ejecta - remnant reset;   M_gas - M_gas,0 = ejecta delivered to gas - formed;
      ejecta delivered to gas = ejecta taken from the stars, for mass and metals;
      MZ_stars - MZ_stars,0 = MZ formed - MZ lost by stars; MZ_gas - MZ_gas,0 = MZ delivered to gas - MZ formed;
    and the state rows equal the snapshot sums (1e-12).
  test_ledger_books_what_reached_the_gas: in hot gas the heat and kinetic rows equal what the gas gained, from snapshots (2 %).
  test_ledger_feedback_energy: the ejecta energy booked from the tables (1/2 dm v_ej^2) against the energy the feedback
    loop gives the gas (heat now, heat to the deferred branch, kinetic); the residual is reported per row (bar 5 %).
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

TEST_NAME = "ledger_mass_metals"
TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(TEST_DIR.parent / "treecol_pixels"))
from make_treecol_pixels_ics import make_treecol_pixels_ics, lattice  # noqa: E402
from stella_table_bands import Tables  # noqa: E402

L, N_SIDE = 8.0, 8
BIRTHS = "100:0.5,300:3.0"  # cell ID: drawn mass [Msun]


def star_age(tab, M, f):
    lM, lZ = np.log10(M), np.log10(0.0142)
    return tab._i2(tab.t_pms, lM, lZ) + f * tab._i2(tab.life, lM, lZ)


@pytest.fixture(scope="module")
def ledger():
    if not os.environ.get("SUNDIALS_DIR"):
        pytest.skip("set SUNDIALS_DIR (SGCHEM builds against SUNDIALS 6.x or 7.x)")
    tables = get_stellar_evolution_tables()
    get_cooling_tables(str(TEST_DIR))
    build_gizmo_for_test(TEST_NAME)
    tab = Tables(tables["StellarTablesFile"])
    lM, lZ = np.log10(3.0), np.log10(0.0142)
    death3 = tab._i2(tab.t_pms, lM, lZ) + tab._i2(tab.life, lM, lZ)
    stars = [(8.0, star_age(tab, 8.0, 0.4), np.full(3, 0.5 * L)), (3.0, death3 - 0.1e6, np.array([0.25, 0.25, 0.75]) * L)]
    d = TEST_DIR / "output"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    make_treecol_pixels_ics(str(d / "snapshot_000.hdf5"), L, lattice(N_SIDE, L), stars, hsml=2.0)
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables["StellarTablesFile"]), ("TestSingleStarBirths", BIRTHS),
                 ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables["StellarTablesFile"])))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    lines = (d / "ledger.txt").read_text().splitlines()
    assert lines[0].startswith("# time"), "ledger.txt needs its header line"
    head = [t for t in lines[0].split()[2:] if t.endswith(("_Msun", "_erg"))]
    rows = np.atleast_2d(np.loadtxt(d / "ledger.txt"))
    assert rows.shape[1] == len(head) + 1
    with h5py.File(d / "snapshot_000.hdf5") as f:  # the initial conditions
        m0 = {k: (f[f"{g}/Masses"][:].sum(), (f[f"{g}/Masses"][:] * f[f"{g}/Metallicity"][:, 0]).sum()) for k, g in (("gas", "PartType0"), ("stars", "PartType4"))}
    return {k: rows[:, j + 1] for j, k in enumerate(head)} | {"time": rows[:, 0]}, m0, d


def test_ledger_mass_metals(ledger):
    b, m0, d = ledger
    need = ("M_gas_Msun", "M_stars_Msun", "MZ_gas_Msun", "MZ_stars_Msun", "M_stars_formed_Msun", "MZ_stars_formed_Msun",
            "M_fb_to_gas_Msun", "MZ_fb_to_gas_Msun", "M_single_star_ejecta_Msun", "MZ_single_star_ejecta_Msun",
            "M_remnant_reset_Msun", "MZ_stars_lost_Msun")
    assert all(k in b for k in need), f"ledger columns: {sorted(b)}"
    assert len(b["time"]) >= 6
    M_tot, Z_tot = m0["gas"][0] + m0["stars"][0], m0["gas"][1] + m0["stars"][1]
    res = {  # changes since the initial conditions against the bookings
        "stars": b["M_stars_Msun"] - m0["stars"][0] - (b["M_stars_formed_Msun"] - b["M_single_star_ejecta_Msun"] - b["M_remnant_reset_Msun"]),
        "gas": b["M_gas_Msun"] - m0["gas"][0] - (b["M_fb_to_gas_Msun"] - b["M_stars_formed_Msun"]),
        "ejecta": b["M_fb_to_gas_Msun"] - b["M_single_star_ejecta_Msun"],
    }
    resZ = {
        "stars": b["MZ_stars_Msun"] - m0["stars"][1] - (b["MZ_stars_formed_Msun"] - b["MZ_stars_lost_Msun"]),
        "gas": b["MZ_gas_Msun"] - m0["gas"][1] - (b["MZ_fb_to_gas_Msun"] - b["MZ_stars_formed_Msun"]),
        "ejecta": b["MZ_fb_to_gas_Msun"] - b["MZ_single_star_ejecta_Msun"],
    }
    print("formed", b["M_stars_formed_Msun"][-1], "ejecta", b["M_single_star_ejecta_Msun"][-1], "remnant reset",
          b["M_remnant_reset_Msun"][-1], "MZ ejecta", b["MZ_single_star_ejecta_Msun"][-1])
    print("mass residuals / M_tot:", {k: float(np.max(np.abs(v)) / M_tot) for k, v in res.items()})
    print("metal residuals / MZ_tot:", {k: float(np.max(np.abs(v)) / Z_tot) for k, v in resZ.items()})
    assert b["M_stars_formed_Msun"][-1] > 3.0 and b["M_single_star_ejecta_Msun"][-1] > 2.0  # births; winds and the AGB death
    for v in res.values():
        assert np.max(np.abs(v)) <= 1e-12 * M_tot
    for v in resZ.values():
        assert np.max(np.abs(v)) <= 1e-12 * Z_tot
    with h5py.File(sorted(d.glob("snapshot_*.hdf5"))[-1]) as f:  # the state rows are the snapshot sums
        mg, ms = f["PartType0/Masses"][:], f["PartType4/Masses"][:]
        zg, zs = f["PartType0/Metallicity"][:, 0], f["PartType4/Metallicity"][:, 0]
    np.testing.assert_allclose([b["M_gas_Msun"][-1], b["M_stars_Msun"][-1], b["MZ_gas_Msun"][-1], b["MZ_stars_Msun"][-1]],
                               [mg.sum(), ms.sum(), (mg * zg).sum(), (ms * zs).sum()], rtol=1e-12)


def hot_wind(tag, tables, last_wind_log_age):
    """a 20 Msun star 1 Myr old with a steady wind in 1e7 K ionised gas, n_H = 10 (L = 22 pc, 12^3 cells of 6 Msun)"""
    d = TEST_DIR / f"output_{tag}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir()
    n, L_box, ic = 12, 22.0, d / "snapshot_000.hdf5"
    make_treecol_pixels_ics(str(ic), L_box, lattice(n, L_box), [(20.0, 1.0e6, np.array([0.31, 0.31, 0.31]) * L_box)], gas_mass=6.0, T=1.0e7, hsml=4.0,
                            tracabund=np.tile([0.0, 1.0, 0.0], (n ** 3, 1)))
    with h5py.File(ic, "a") as f:
        f["PartType4/SingleStarLastWindLogAge"] = np.array([last_wind_log_age])
    params = (TEST_DIR / f"{TEST_NAME}.params").read_text()
    for k, v in (("StellarTablesFile", tables), ("TestSingleStarBirths", "1:0.5"), ("SGChemTablesPath", os.environ.get("SGCHEM_TABLES_DIR", os.path.dirname(tables)))):
        params = params.replace(f"{k:35s}SET_BY_TEST", f"{k:35s}{v}")
    for k, v in (("OutputDir", d.name), ("BoxSize", L_box), ("TimeMax", 0.012), ("TimeBetSnapshot", 0.004), ("TimeBetStatistics", 0.004), ("MaxSizeTimestep", 0.001)):
        params = "\n".join(f"{k:35s}{v}" if ln.split()[:1] == [k] else ln for ln in params.splitlines()) + "\n"
    (d / "run.params").write_text(params)
    with open(d / "run.out", "w") as out:
        subprocess.run(mpi_launch_command(2, 1) + [str(TEST_DIR / "GIZMO"), str(d / "run.params"), "2", "0"], cwd=TEST_DIR, stdout=out, stderr=subprocess.STDOUT, check=True)
    return d


def test_ledger_books_what_reached_the_gas(ledger):
    """heat row + kinetic row = what the gas gained at the first output, from snapshots, run minus a twin with no wind owed
    (cooling cancels); heat in the deferred rate is not delivered and is not in those rows"""
    tables = get_stellar_evolution_tables()["StellarTablesFile"]
    tab = Tables(tables)
    run, twin = hot_wind("wind", tables, np.log10(1.0e6 - tab._i2(tab.t_pms, np.log10(20.0), np.log10(0.0142)))), hot_wind("twin", tables, 9.9)

    def energy(d):
        with h5py.File(d / "snapshot_001.hdf5") as f:
            g, s = f["PartType0"], f["PartType4"]
            return (g["Masses"][:] * (g["InternalEnergy"][:] + 0.5 * (g["Velocities"][:] ** 2).sum(1))).sum() + (0.5 * s["Masses"][:] * (s["Velocities"][:] ** 2).sum(1)).sum()
    lines = (run / "ledger.txt").read_text().splitlines()
    rows = np.atleast_2d(np.loadtxt(run / "ledger.txt"))
    head = [t for t in lines[0].split()[2:] if t.endswith(("_Msun", "_erg"))]
    b = {k: rows[0, j + 1] for j, k in enumerate(head)}
    gain = (energy(run) - energy(twin)) * 1.989e33 * 1.0e10
    print(f"gain {gain:.4e} erg, heat {b['E_fb_thermal_now_erg']:.4e}, kinetic {b['E_fb_kinetic_erg']:.4e}, ejecta {b['E_single_star_ejecta_erg']:.4e}")
    assert abs(b["E_fb_thermal_now_erg"] + b["E_fb_kinetic_erg"] - gain) < 0.02 * b["E_single_star_ejecta_erg"]  # of what was offered: the gain can be near zero


def test_ledger_feedback_energy(ledger):
    """the energy the single stars' ejecta carry (1/2 dm v_ej^2 per coupled event, as the loop is asked to couple)
    against what the feedback loop gives the gas: heat (always injected) and kinetic"""
    b, _, _ = ledger
    need = ("E_single_star_ejecta_erg", "E_fb_thermal_now_erg", "E_fb_kinetic_erg")
    assert all(k in b for k in need), f"ledger columns: {sorted(b)}"
    booked = b["E_single_star_ejecta_erg"]
    coupled = b["E_fb_thermal_now_erg"] + b["E_fb_kinetic_erg"]
    for k in range(len(booked)):
        print(f"t = {b['time'][k]:.2f}: ejecta {booked[k]:.6e} erg; to gas: heat now {b['E_fb_thermal_now_erg'][k]:.4e}, "
              f"kinetic {b['E_fb_kinetic_erg'][k]:.4e}; "
              f"residual (gas - ejecta) / ejecta {coupled[k] / booked[k] - 1:+.3e}")
    assert booked[-1] > 0 and np.all(np.diff(booked) >= 0)
    assert np.all(np.abs(coupled / booked - 1) < 0.05)  # the loop's relative-velocity and momentum terms, not a gap
