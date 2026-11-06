"""single_stars_collect_gas test: new single stars collect their drawn IMF mass from the gas around them
(GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS), with stars placed by TEST_SINGLE_STARS_BIRTH_AT_GIVEN_CELLS on a regular
lattice of 1 Msun cells (uniform density and temperature; metallicity varying across the box; bulk velocity plus a shear
v_y(x)). The main build switches merging of light cells off (MERGESPLIT_HARDCODE_MIN_MASS), so the snapshot after the
first step shows the collection itself; one variant keeps the default merging. See README.md.
"""
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
import h5py
import pytest
from gizmo.test import build_gizmo_for_test, mpi_launch_command

TEST_NAME = "single_stars_collect_gas"
TEST_DIR = Path(__file__).parent
NO_MERGING = ("MERGESPLIT_HARDCODE_MIN_MASS=(1.e-6)",)
TWO = [((12, 12, 12), 8.0), ((12, 12, 13), 40.0)]
FIVE = [((12, 12, 12), 8.0), ((12, 12, 13), 40.0), ((12, 13, 12), 15.0), ((13, 12, 12), 60.0), ((11, 11, 11), 25.0)]
EDGES = [((0, 0, 0), 30.0), ((0, 12, 23), 20.0), ((23, 23, 0), 12.0)]  # at the box edges: donors across the periodic wrap


@pytest.fixture(scope="module")
def built():
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        build_gizmo_for_test(TEST_NAME)
        os.replace(TEST_DIR / "GIZMO", TEST_DIR / "GIZMO_default_merging")
        build_gizmo_for_test(TEST_NAME, extra_config_flags=NO_MERGING)
    return True


def run(tag, stars, ranks=2, threads=1, n_side=None, shuffle=False, exe="GIZMO"):
    """one run in its own directory, started from the lattice written as snapshot 0 (start flag 2, which reads the
    metallicity); returns (initial snapshot, final snapshot, birth log rows)"""
    import sys
    sys.path.insert(0, str(TEST_DIR))
    from make_single_stars_collect_gas_ics import make_single_stars_collect_gas_ics, cell_id, N_SIDE
    n = n_side or N_SIDE
    d = TEST_DIR / f"run_{tag}"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN") and d.exists():
        shutil.rmtree(d)  # this run's own directory: no stale logs or snapshots
    (d / "output").mkdir(parents=True, exist_ok=True)
    ic = d / "output" / "snapshot_000.hdf5"
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        L = make_single_stars_collect_gas_ics(str(ic), n=n, shuffle=shuffle)
        births = ",".join(f"{cell_id(*c, n=n)}:{m}" for c, m in stars)
        params = Path(TEST_DIR / f"{TEST_NAME}.params").read_text().replace("SET_BY_TEST", births)
        params = params.replace("BoxSize                            15.84", f"BoxSize                            {L}")
        (d / f"{TEST_NAME}.params").write_text(params)
        cmd = mpi_launch_command(ranks, threads) + [str(TEST_DIR / exe), f"{TEST_NAME}.params", "2", "0"]
        env = {**os.environ, "OMP_NUM_THREADS": str(max(threads, 1))}
        with open(d / f"test_{TEST_NAME}.out", "w") as out:
            subprocess.run(cmd, cwd=d, stdout=out, stderr=subprocess.STDOUT, env=env, check=True)
        assert "Final time" in (d / f"test_{TEST_NAME}.out").read_text() or (d / "output" / "snapshot_001.hdf5").exists()
    snaps = sorted((d / "output").glob("snapshot_*.hdf5"))
    rows = np.atleast_2d(np.loadtxt(d / "output" / "single_star_births.txt"))
    return ic, snaps[-1], rows


def particles(fname):
    out = {}
    with h5py.File(fname, "r") as F:
        for t in ("PartType0", "PartType4"):
            if t in F:
                out[t] = {k: F[t][k][()] for k in ("ParticleIDs", "Masses", "Metallicity", "Velocities", "SingleStarDrawnMass") if k in F[t]}
                if "Metallicity" in out[t]:
                    out[t]["Metallicity"] = out[t]["Metallicity"].reshape(len(out[t]["Masses"]), -1)
    return out


def exact_sum(a):
    return sum((Fraction(float(x)) for x in np.ravel(a)), Fraction(0))


def exact_dot(m, z):
    return sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(np.ravel(m), np.ravel(z))), Fraction(0))


def by_id(d):
    o = np.argsort(d["ParticleIDs"], kind="stable")
    return {k: v[o] for k, v in d.items()}


def check_conservation(ic, snap, rows, stars, exact=True):
    p0, p1 = particles(ic), particles(snap)
    g0, g1, s1 = p0["PartType0"], p1["PartType0"], p1["PartType4"]
    m_star = s1["Masses"]
    # every star within 1% of its drawn mass, which its record carries
    assert len(m_star) == len(stars) and np.all(rows[:, 8] == 1)
    assert sorted(s1["SingleStarDrawnMass"].tolist()) == sorted(m for _, m in stars)
    assert np.all(np.abs(rows[:, 5] - rows[:, 4]) <= 0.01 * rows[:, 4])
    M0, M1 = exact_sum(g0["Masses"]), exact_sum(g1["Masses"]) + exact_sum(m_star)
    Z0 = exact_dot(g0["Masses"], g0["Metallicity"][:, 0])
    Z1 = exact_dot(g1["Masses"], g1["Metallicity"][:, 0]) + exact_dot(m_star, s1["Metallicity"][:, 0])
    if exact:
        # mass: exactly conserved; donors keep their metallicity bitwise
        assert M1 == M0
        a, b = by_id(g0), by_id(g1)
        assert np.array_equal(a["ParticleIDs"], b["ParticleIDs"]) and np.array_equal(a["Metallicity"], b["Metallicity"])
        # metals: the only rounding is the star's stored metallicity fraction (half an ulp of Z per star, times its mass)
        bound = sum(float(m) * np.spacing(float(z)) for m, z in zip(m_star, s1["Metallicity"][:, 0]))
        assert abs(float(Z1 - Z0)) <= bound, (float(Z1 - Z0), bound)
    else:
        # with merging of drained donors the merge itself rounds
        assert abs(float(M1 - M0)) <= 1e-13 * float(M0)
        assert abs(float(Z1 - Z0)) <= 1e-12 * float(Z0)
    # the star's metallicity is the mass-weighted mean over its donors: inside the range the gas spans
    assert np.all(s1["Metallicity"][:, 0] >= g0["Metallicity"][:, 0].min()) and np.all(s1["Metallicity"][:, 0] <= g0["Metallicity"][:, 0].max())
    assert np.min(g1["Masses"]) > 0  # donors give at most half their mass per pass
    # momentum: the collection carries it in fixed point; the hydro step conserves it to rounding
    q0 = np.sum(g0["Masses"][:, None] * g0["Velocities"], axis=0)
    q1 = np.sum(g1["Masses"][:, None] * g1["Velocities"], axis=0) + np.sum(m_star[:, None] * s1["Velocities"], axis=0)
    assert np.all(np.abs(q1 - q0) <= 1e-12 * np.sum(g0["Masses"]) * np.max(np.abs(g0["Velocities"])))


def test_shared_donors_two(built):
    ic, snap, rows = run("two", TWO)
    check_conservation(ic, snap, rows, TWO)
    assert np.sum(rows[:, 9]) > 0  # some donors were oversubscribed and split


def test_shared_donors_five(built):
    ic, snap, rows = run("five", FIVE)
    check_conservation(ic, snap, rows, FIVE)
    assert np.sum(particles(snap)["PartType0"]["Masses"] < 1.0) > 60  # radii overlapped: many donors gave mass


def test_unreachable_draw_returns_gas(built):
    ic, snap, rows = run("fail", [((2, 2, 2), 100.0)], n_side=4)
    assert rows.shape[0] == 1 and rows[0, 8] == 0
    p0, p1 = particles(ic), particles(snap)
    assert "PartType4" not in p1 or len(p1["PartType4"]["Masses"]) == 0
    a, b = by_id(p0["PartType0"]), by_id(p1["PartType0"])
    for k in ("ParticleIDs", "Masses", "Metallicity"):
        assert np.array_equal(a[k], b[k]), k


def test_unreachable_and_reachable(built):
    stars = [((2, 2, 2), 100.0), ((0, 0, 0), 5.0)]
    ic, snap, rows = run("failok", stars, n_side=4)
    assert sorted(rows[:, 8].tolist()) == [0.0, 1.0]
    p0, p1 = particles(ic), particles(snap)
    assert len(p1["PartType4"]["Masses"]) == 1
    assert exact_sum(p1["PartType0"]["Masses"]) + exact_sum(p1["PartType4"]["Masses"]) == exact_sum(p0["PartType0"]["Masses"])
    Z0 = exact_dot(p0["PartType0"]["Masses"], p0["PartType0"]["Metallicity"][:, 0])
    Z1 = exact_dot(p1["PartType0"]["Masses"], p1["PartType0"]["Metallicity"][:, 0]) + exact_dot(p1["PartType4"]["Masses"], p1["PartType4"]["Metallicity"][:, 0])
    assert abs(float(Z1 - Z0)) <= float(p1["PartType4"]["Masses"][0]) * np.spacing(float(p1["PartType4"]["Metallicity"][0, 0]))


def final_state(snap):
    p = particles(snap)
    assert "PartType4" in p
    return {t: by_id(d) for t, d in p.items()}


def assert_same(ref, other, tag):
    for t in ref:
        for k in ("ParticleIDs", "Masses", "Metallicity", "Velocities"):
            assert ref[t][k].tobytes() == other[t][k].tobytes(), (tag, t, k)


def test_identical_across_ranks_threads_and_order(built):
    ref = final_state(run("five", FIVE)[1])
    for tag, kw in [("r1", dict(ranks=1)), ("r4", dict(ranks=4)), ("r2t2", dict(ranks=2, threads=2)), ("perm", dict(shuffle=True))]:
        assert_same(ref, final_state(run(f"five_{tag}", FIVE, **kw)[1]), tag)


def test_donors_on_other_ranks(built):
    """stars at the box edges collect across the periodic wrap: on 4 ranks many of their grants come from donors on another
    rank than the star (counted in the birth log); the result is bitwise the 1-rank result"""
    ic, snap4, rows4 = run("edges_r4", EDGES, ranks=4)
    check_conservation(ic, snap4, rows4, EDGES)
    assert np.all(rows4[:, 10] > 0), rows4[:, 10]  # every star took mass from another rank
    _, snap1, rows1 = run("edges_r1", EDGES, ranks=1)
    assert np.sum(rows1[:, 10]) == 0
    assert_same(final_state(snap1), final_state(snap4), "edges")


def test_default_merging(built):
    """with the default merging of light cells, drained donors may merge afterwards: stars still form, the totals hold
    to the merge's rounding"""
    ic, snap, rows = run("five_merging", FIVE, exe="GIZMO_default_merging")
    check_conservation(ic, snap, rows, FIVE, exact=False)
