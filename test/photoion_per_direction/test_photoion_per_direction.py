"""photoion_per_direction test: per-direction photoionisation around single stars (GALSF_FB_PHOTOIONIZATION_PER_DIRECTION).
See README.md."""
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from gizmo.test import build_gizmo_for_test, get_stellar_evolution_tables, mpi_launch_command

TEST_NAME = "photoion_per_direction"
TEST_DIR = Path(__file__).parent
PC_KMS_IN_YR = 3.085678e13 / 3.155e7
X_H = 0.76
ALPHA_B = 2.6e-13
PC = 3.085678e18
CENTRE = [(35.0, 1.0e6, (0.5, 0.5, 0.5))]  # a 35 Msun star 1 Myr old


@pytest.fixture(scope="module")
def built():
    tables = get_stellar_evolution_tables()
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
    return tables


def run(tag, tables, stars=CENTRE, ranks=1, threads=1, seed=None, overrides=None, exe="GIZMO", extra="", n_metals=1):
    """the run directory and the lattice (cell size, n_H)"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_photoion_per_direction_ics import make_photoion_per_direction_ics
    d = TEST_DIR / f"run_{tag}"
    if d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs
    (d / "output").mkdir(parents=True)
    L, dx, n_H = make_photoion_per_direction_ics(str(d / "output" / "snapshot_000.hdf5"), stars, seed=seed, n_metals=n_metals)
    params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", tables["StellarTablesFile"])
    params = params.replace("BoxSize                            45.0", f"BoxSize                            {L}")
    for k, v in (overrides or {}).items():
        params = "\n".join(f"{k:34s} {v}" if l.split() and l.split()[0] == k else l for l in params.splitlines()) + "\n"
    (d / f"{TEST_NAME}.params").write_text(params + extra)
    with open(d / f"test_{TEST_NAME}.out", "w") as out:
        subprocess.run(mpi_launch_command(ranks, threads) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"],
                       cwd=d, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "OMP_NUM_THREADS": str(threads)}, check=True)
    return d, dx, n_H


def log(d):
    """the photoionisation log rows (HIIheating.txt)"""
    return np.atleast_2d(np.loadtxt(d / "output" / "HIIheating.txt"))


def timebins(out):
    """per step: [(non-cells, cells, dt)] of the occupied bins, from the run's timebin tables"""
    steps, cur = [], None
    for l in Path(out).read_text().splitlines():
        if l.startswith("Occupied timebins"):
            cur = []; steps.append(cur)
        elif cur is not None and (m := re.match(r"\s*X?\s*bin=\s*\d+\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)", l)):
            cur.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
        else:
            cur = None
    return steps


def test_ionising_stars_in_the_photoionisation_bin(built):
    """an ionising star steps at most PhotoionMaxTimestep (1 kyr here); a 35 Msun star 1000 yr old steps onto its onset
    (15 kyr) and from then on keeps to that bin too; the gas keeps its own coarse step"""
    stars = [(35.0, 1.0e6, (0.25, 0.25, 0.25)), (35.0, 1.0e3, (0.75, 0.75, 0.75))]
    d, _, _ = run("cadence", built, stars=stars, overrides={"MaxSizeTimestep": "0.02", "TimeMax": "0.06", "TimeBetSnapshot": "0.06",
                                                          "TimeBetStatistics": "0.06", "PhotoionMaxTimestep": "1000"})
    cap = 1000.0 / PC_KMS_IN_YR
    steps = [s for s in timebins(d / f"test_{TEST_NAME}.out") if not any(dt == 0 for _, _, dt in s)]
    assert len(steps) > 10
    for s in steps:
        assert sum(n for n, _, dt in s if 0 < dt <= cap) >= 1, s  # the ionising star
        assert all(dt > 10 * cap for _, c, dt in s if c > 0), s  # the gas: its own, coarse step
    assert sum(n for n, _, dt in steps[-1] if dt <= cap) == 2, steps[-1]  # both, after the onset


def stromgren_pc(Q, n_H):
    return (3 * Q / (4 * np.pi * ALPHA_B * n_H ** 2)) ** (1 / 3) / PC


def test_stromgren_sphere(built):
    """one source: every pixel front lies within one cell of the Stromgren radius"""
    d, dx, n_H = run("single", built)
    rows = log(d)
    assert len(rows) >= 3 and np.all(rows[:, 1] == 1)
    for r in rows:
        R_S = stromgren_pc(r[2] * X_H, n_H)
        assert abs(r[9] * 1e3 - R_S) < dx and abs(r[10] * 1e3 - R_S) < dx, (r[9] * 1e3, r[10] * 1e3, R_S)
        assert 0.98 < r[6] < 1.02, r[6]  # recombinations of the ionised cells / Q
        assert abs((3 * r[3] * dx ** 3 / (4 * np.pi)) ** (1 / 3) - R_S) < dx  # their volume radius
        assert abs(r[3] / rows[0, 3] - 1) < 2e-3  # static gas: the same cells at every solve, up to the slow rise of Q


def test_ranks_and_order(built):
    """the fronts do not depend on the rank count or the particle order (integer sums)"""
    ref = log(run("single_r1", built)[0])
    for tag, kw in [("single_r2", dict(ranks=2)), ("single_r4", dict(ranks=4)), ("single_perm", dict(ranks=2, seed=7))]:
        rows = log(run(tag, built, **kw)[0])
        assert rows.shape == ref.shape, tag
        assert np.array_equal(rows[:, [1, 3, 8]], ref[:, [1, 3, 8]]), tag  # sources, ionised cells, passes
        assert np.allclose(rows[:, 6], ref[:, 6], rtol=1e-9), tag  # ledger
        assert np.allclose(rows[:, [9, 10]], ref[:, [9, 10]], rtol=1e-6, atol=0), tag  # fronts


def first_row(d):
    return log(d)[0]


def test_overlapping_sources_share(built):
    """two identical stars 0.5 Stromgren radii apart ionise twice the single star's cells (photon-conserving overlap),
    with the recombinations of the union matching their summed photon rate; the same on 1, 2 and 4 ranks"""
    d1, dx, n_H = run("ovl_single", built)
    one = first_row(d1)
    R_S = stromgren_pc(one[2] * X_H, n_H)
    L = 32 * dx
    off = 0.25 * R_S / L
    pair = [(35.0, 1.0e6, (0.5 - off, 0.5, 0.5)), (35.0, 1.0e6, (0.5 + off, 0.5, 0.5))]
    rows = {}
    for tag, kw in [("ovl_pair_r1", {}), ("ovl_pair_r2", dict(ranks=2)), ("ovl_pair_r4", dict(ranks=4))]:
        rows[tag] = first_row(run(tag, built, stars=pair, **kw)[0])
    two = rows["ovl_pair_r1"]
    assert two[1] == 2
    assert abs(two[3] / one[3] - 2.0) < 0.05, two[3] / one[3]  # union / single
    assert 0.98 < two[6] < 1.02, two[6]  # recombinations of the union / summed Q
    for tag in ("ovl_pair_r2", "ovl_pair_r4"):
        assert rows[tag][3] == two[3] and rows[tag][7] == two[7], tag  # cells, rounds


def test_bright_and_faint_source(built):
    """a bright and a faint star (Q ratio 7) 0.9 pc apart: the union recombines at their summed photon rate, and in
    uniform gas its cells grow with the summed rate"""
    d1, _, _ = run("asym_single", built)
    one = first_row(d1)
    stars = [(35.0, 1.0e6, (0.5, 0.5, 0.5)), (20.0, 1.0e6, (0.52, 0.5, 0.5))]
    two = first_row(run("asym_pair", built, stars=stars)[0])
    assert two[1] == 2 and 0.98 < two[6] < 1.02, two[6]
    assert abs(two[3] / one[3] - two[2] / one[2]) < 0.02, (two[3] / one[3], two[2] / one[2])


SGCHEM = ("COOLING", "COOL_METAL_LINES_BY_SPECIES", "SGCHEM", "CHEMISTRYNETWORK=5", "OUTPUT_MASS_METAL_ENERGY_LEDGER")


@pytest.fixture(scope="module")
def built_sgchem(built):
    """the same build with SGChem (network 5) as the cooling; its parameters"""
    import sys
    sys.path.insert(0, str(TEST_DIR.parent / "sgchem_box"))
    from test_sgchem_box import chemistry_tables_dir
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME, extra_config_flags=SGCHEM)
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_sgchem")
        build_gizmo_for_test(TEST_NAME)
    return built, "".join(f"{k:34s} {v}\n" for k, v in [("SGChemTablesPath", chemistry_tables_dir()), ("SGChemCosmicRayIonRate", 2.0e-16),
            ("SGChemDustToGasRatio", 1.0), ("SGChemDeuteriumAbundance", 2.6e-5), ("SGChemG0", 0), ("SGChemG0_LW", 0), ("SGChemBackgroundG0", 0)])


def gas(snap):
    import h5py
    with h5py.File(snap, "r") as F:
        g = F["PartType0"]
        c = F["PartType4"]["Coordinates"][0]
        x = g["SGChemTracAbund"][()][:, 1] if "SGChemTracAbund" in g else np.zeros(len(g["Masses"]))
        return dict(r=np.linalg.norm(g["Coordinates"][()] - c, axis=1), x=x, u=g["InternalEnergy"][()], m=g["Masses"][()],
                    id=g["ParticleIDs"][()])


def test_sgchem_reads_the_ionised_cells(built_sgchem):
    """with SGChem as the cooling, cells inside the fronts are ionised (H+ 0.9998) at 1e4 K or above, cells outside are
    untouched by the photons; the heating to 1e4 K is booked in the ledger"""
    tables, extra = built_sgchem
    d, dx, n_H = run("sgchem", tables, exe="GIZMO_sgchem", extra=extra, n_metals=11)
    rows = log(d)
    x_sgchem = 0.7065  # HYDROGEN_MASSFRAC of SGChem builds
    R_S = stromgren_pc(rows[0, 2] * x_sgchem, n_H * x_sgchem / X_H)
    a, b = gas(sorted((d / "output").glob("snapshot_*.hdf5"))[0]), gas(sorted((d / "output").glob("snapshot_*.hdf5"))[-1])
    o = np.argsort(a["id"]); a = {k: v[o] for k, v in a.items()}; o = np.argsort(b["id"]); b = {k: v[o] for k, v in b.items()}
    inner, outer = b["r"] < R_S - dx, b["r"] > R_S + 2 * dx
    u_hii = 1.5 * 1.380649e-16 * 1e4 / (0.6 * 1.6726e-24) / 1e10  # 1e4 K ionised gas [km^2/s^2], roughly
    assert np.all(b["x"][inner] > 0.999) and np.all(b["u"][inner] > 0.9 * u_hii)
    assert np.all(b["x"][outer] < 0.01) and np.all(b["u"][outer] < 0.05 * u_hii)
    ion = b["x"] > 0.99
    led = np.atleast_2d(np.loadtxt(d / "output" / "ledger.txt"))
    E_pi = led[-1, 3]  # erg
    dE = np.sum(b["m"][ion] * (b["u"][ion] - a["u"][ion])) * 1.989e33 * 1e10
    assert E_pi > 0 and abs(E_pi / dE - 1) < 0.05, (E_pi, dE)
    assert led[-1, 1] < 0.01 * E_pi, (led[-1, 1], E_pi)  # chemistry heating: the network only, not the photoionisation


def test_quiet_steps(built):
    """a 35 Msun star 10 kyr old turns on at its onset (15.3 kyr, the pre-main sequence of the tables). Before, the
    photoionisation does no work and no collective on its ~5 steps, except one count at start-up (unknown sources);
    it solves from the first photoionisation step after the onset. Log column 12: the module's reductions and
    neighbour-loop calls since the previous row; a solve with one source and p search passes makes 7 + 2 p."""
    import h5py
    d, _, _ = run("quiet", built, stars=[(35.0, 1.0e4, (0.5, 0.5, 0.5))],
                  overrides={"MaxSizeTimestep": "1e-3", "PhotoionMaxTimestep": "1000", "TimeMax": "0.03", "TimeBetSnapshot": "0.03", "TimeBetStatistics": "0.03"})
    rows = log(d)
    with h5py.File(built["StellarTablesFile"], "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        onset = F["t_PMS"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - 35.0)))]
    t_first = 1.0e4 + rows[0, 0] * PC_KMS_IN_YR
    assert onset <= t_first < onset + 1000.0, (t_first, onset)
    assert rows[0, 11] == 7 + 2 * rows[0, 8] + 1, rows[0]  # this solve, and the one start-up count
    assert np.all(rows[1:, 11] == 7 + 2 * rows[1:, 8]), rows[1:, [8, 11]]
