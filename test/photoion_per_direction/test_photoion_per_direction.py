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
        if not any(l.split() and l.split()[0] == k for l in params.splitlines()):
            params += f"{k:34s} {v}\n"
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
    photoionisation does no work and no collective on its ~5 steps; it solves from the first active step after the
    onset. Log column 12: the module's reductions and neighbour-loop calls since the previous row; a solve with one
    source and p search passes makes 8 + 5 p (9 + 5 p off the cap step), and nothing else counts: no collective on
    the steps between."""
    import h5py
    d, _, _ = run("quiet", built, stars=[(35.0, 1.0e4, (0.5, 0.5, 0.5))],
                  overrides={"MaxSizeTimestep": "1e-3", "PhotoionMaxTimestep": "1000", "TimeMax": "0.03", "TimeBetSnapshot": "0.03", "TimeBetStatistics": "0.03"})
    rows = log(d)
    with h5py.File(built["StellarTablesFile"], "r") as F:
        Zt, Mt = F["Z"][()], F["M_init"][()]
        onset = F["t_PMS"][()].reshape(len(Zt), len(Mt))[int(np.argmin(np.abs(Zt - 0.014))), int(np.argmin(np.abs(Mt - 35.0)))]
    t_first = 1.0e4 + rows[0, 0] * PC_KMS_IN_YR
    assert onset <= t_first < onset + 1000.0, (t_first, onset)
    cap = rows[-1, 0] - rows[-2, 0]  # the star is on the cap step by the end
    off_cap = np.abs(rows[:, 0] / cap - np.round(rows[:, 0] / cap)) > 1e-6  # one more: the partner search
    assert np.all(rows[:, 11] == 8 + 5 * rows[:, 8] + off_cap), rows[:, [0, 8, 11]]
    assert rows.shape[1] == 15 and np.all(rows[:, 12] > 0) and np.all(rows[:, 12] < 10), rows[:, 12]  # wall time of each solve [s]


N_CELLS = 32 ** 3
STEPS = {"MaxSizeTimestep": "0.02", "TimeMax": "0.06", "TimeBetSnapshot": "0.0123", "TimeBetStatistics": "0.06", "PhotoionMaxTimestep": "1e4",
         "TestPhotoionQuarterStepStarID": str(N_CELLS + 1)}  # the first star: PhotoionMaxTimestep / 4


def ionised_snapshots(d):
    """per snapshot after the start: time, cell positions, ionised flag, star positions, box"""
    import h5py
    out = []
    for s in sorted((d / "output").glob("snapshot_*.hdf5")):
        with h5py.File(s, "r") as F:
            t = F["Header"].attrs["Time"]
            if t > 0:
                g = F["PartType0"]
                c = F["PartType4"]["Coordinates"][()][np.argsort(F["PartType4"]["ParticleIDs"][()])]  # by ID
                out.append((t, g["Coordinates"][()], g["PhotoionizedUntilTime"][()] > t, c, F["Header"].attrs["BoxSize"]))
    return out


def distance(x, c, L):
    d = np.abs(x - c)
    return np.linalg.norm(np.minimum(d, L - d), axis=1)


def check_two_cadences(rows):
    """sources solve together on the long step, the short-step star alone on the three steps between"""
    both = rows[rows[:, 1] == 2, 0]
    assert len(both) >= 3 and np.any(rows[:, 1] == 1), rows[:, :2]
    dt_long = np.diff(both)
    assert np.allclose(dt_long, dt_long[0], rtol=1e-9), dt_long
    assert np.allclose(np.diff(rows[:, 0]), dt_long[0] / 4, rtol=1e-9), np.diff(rows[:, 0])  # no row skipped, none extra
    return dt_long[0]


@pytest.mark.parametrize("ranks", [1, 2])
def test_each_source_solves_on_its_own_step(built, ranks):
    """a star on a quarter of the long step solves on each of its steps, alone; the inactive star's cells stay ionised
    through those solves, and every source pays for just its own cells"""
    far = [(35.0, 1.0e6, (0.25, 0.25, 0.25)), (35.0, 1.0e6, (0.75, 0.75, 0.75))]
    one = first_row(run(f"own_single_r{ranks}", built, ranks=ranks)[0])
    d, dx, n_H = run(f"own_step_r{ranks}", built, stars=far, ranks=ranks, overrides=STEPS)
    rows = log(d)
    check_two_cadences(rows)
    assert np.all(rows[:, 13] > 0.98) and np.all(rows[:, 14] < 1.02), rows[:, 13:15]  # recombinations / Q of each source
    R_S = stromgren_pc(one[2] * X_H, n_H)
    snaps = ionised_snapshots(d)
    assert len(snaps) >= 3
    for t, x, ion, c, L in snaps:
        for k in (0, 1):  # both regions, whichever star solved last
            assert np.all(ion[distance(x, c[k], L) < R_S - dx]), (t, k)
        assert abs(ion.sum() / (2 * one[3]) - 1) < 0.02, (t, ion.sum(), one[3])


@pytest.mark.parametrize("kw", [dict(), dict(ranks=2), dict(ranks=2, seed=7)], ids=["r1", "r2", "r2_permuted"])
def test_overlapping_sources_on_different_steps(built, kw):
    """the overlapping pair of test_overlapping_sources_share with the two stars on different steps: when one solves,
    the other's fronts count as claims. The pair ionises 2.0 x one star's cells at every solve, each source pays Q, and
    ranks and particle order give the same cells and rounds"""
    d1, dx, n_H = run("ovl2_single", built)
    one = first_row(d1)
    R_S = stromgren_pc(one[2] * X_H, n_H)
    off = 0.25 * R_S / (32 * dx)
    pair = [(35.0, 1.0e6, (0.5 - off, 0.5, 0.5)), (35.0, 1.0e6, (0.5 + off, 0.5, 0.5))]
    tag = "_".join(f"{k}{v}" for k, v in kw.items())
    rows = log(run(f"ovl2_pair_{tag}", built, stars=pair, overrides=STEPS, **kw)[0])
    check_two_cadences(rows)
    alone, both = rows[rows[:, 1] == 1], rows[rows[:, 1] == 2]
    assert np.all(np.abs(both[:, 3] / one[3] - 2.0) < 0.05), both[:, 3] / one[3]
    assert np.all(np.abs(alone[:, 3] / one[3] - 1.0) < 0.05), alone[:, 3] / one[3]  # its share of the union
    assert np.all(rows[:, 13] > 0.98) and np.all(rows[:, 14] < 1.02), rows[:, 13:15]
    if kw:
        ref = log(run("ovl2_pair_ref", built, stars=pair, overrides=STEPS)[0])
        assert np.array_equal(rows[:, [1, 3, 7]], ref[:, [1, 3, 7]])


def test_source_releases_cells_it_left(built):
    """a star moving at 1000 km/s on a quarter of the long step: at its own solve it drops the cells it left, so no
    cell lies further from it than the front plus what it moved in one short step; the region of the star at rest stays"""
    v = 1000.0
    stars = [(35.0, 1.0e6, (0.25, 0.5, 0.5), (v, 0.0, 0.0)), (35.0, 1.0e6, (0.75, 0.0, 0.0))]
    params = dict(STEPS, MaxRMSDisplacementFac="1e6", TimeMax="0.03", TimeBetStatistics="0.03", TimeBetSnapshot="0.0071")
    one = first_row(run("rel_single", built)[0])
    d, dx, n_H = run("release", built, stars=stars, overrides=params)
    rows = log(d)
    dt_long = check_two_cadences(rows)
    R_S = stromgren_pc(one[2] * X_H, n_H)
    assert v * dt_long > 3 * dx  # the star leaves its region within a long step
    for t, x, ion, c, L in [s for s in ionised_snapshots(d) if s[0] > dt_long]:  # after the first solve of both
        near = distance(x, c[0], L) < distance(x, c[1], L)
        assert distance(x[near & ion], c[0], L).max() < R_S + 2 * dx + v * dt_long / 4, t
        assert np.all(ion[distance(x, c[1], L) < R_S - dx]), t


def test_restart_keeps_the_fronts(built):
    """a pair on different steps stopped halfway and continued from its restart files, with TimeMax raised so the integer
    timeline is halved: the short-step star still shares with its partner's fronts read back from the restart files
    (rounds > 0 and its half of the union), and the cells keep expiring at the right time"""
    d1, dx, n_H = run("rst_single", built)
    one = first_row(d1)
    off = 0.25 * stromgren_pc(one[2] * X_H, n_H) / (32 * dx)
    pair = [(35.0, 1.0e6, (0.5 - off, 0.5, 0.5)), (35.0, 1.0e6, (0.5 + off, 0.5, 0.5))]
    d, _, _ = run("rst_pair", built, stars=pair, overrides=dict(STEPS, TimeMax="0.03", TimeBetStatistics="0.03", TimeBetSnapshot="0.0123"))
    n_before = len(log(d))
    p = d / f"{TEST_NAME}.params"
    p.write_text("".join(f"TimeMax {0.06}\n" if l.split() and l.split()[0] == "TimeMax" else l + "\n" for l in p.read_text().splitlines()))
    with open(d / f"test_{TEST_NAME}.out", "a") as out:
        subprocess.run(mpi_launch_command(1, 1) + [str(TEST_DIR / "GIZMO"), f"{TEST_NAME}.params", "1"], cwd=d, stdout=out, stderr=subprocess.STDOUT, check=True)
    rows = log(d)[n_before:]
    alone = rows[rows[:, 1] == 1]
    assert len(alone) >= 3 and np.all(alone[:, 7] > 0), alone[:, [0, 7]]
    assert np.all(np.abs(alone[:, 3] / one[3] - 1.0) < 0.05), alone[:, 3] / one[3]
    for t, x, ion, c, L in ionised_snapshots(d):
        assert abs(ion.sum() / (2 * one[3]) - 1) < 0.05, (t, ion.sum(), one[3])  # no stale stamps kept cells from expiring


def test_compact_exchange(built):
    """the histograms do not travel in the neighbour-loop reply records: four separate stars on 4 ranks with a 0.02 MB
    communication chunk; no source's export set exceeds a round (the runner's 'cap ineffective' warning, which a 12 kB
    histogram record per peer triggered), and cells and fronts are those of 1 rank"""
    four = [(35.0, 1.0e6, x) for x in ((0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75))]
    ref = log(run("chunk_r1", built, stars=four)[0])
    d = run("chunk_r4", built, stars=four, ranks=4, overrides={"CommChunkSize": "0.02"})[0]
    rows = log(d)
    assert "cap ineffective" not in (d / f"test_{TEST_NAME}.out").read_text()
    assert np.array_equal(rows[:, [1, 3, 8]], ref[:, [1, 3, 8]]) and np.allclose(rows[:, [9, 10]], ref[:, [9, 10]], rtol=1e-6, atol=0)
