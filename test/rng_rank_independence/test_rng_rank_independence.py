"""Keyed random draws (declarations/gpu_rng.h): stochastic outcomes depend on the particles, not on the decomposition.

One small IC (gas, pre-existing stars, sinks) is run on 1, 2 and 4 ranks and on 1 or 2 OpenMP threads. The draws at
initialization (stellar ages, sink spins) must agree bit for bit, and the runtime draws must pick nearly the same gas
cells to form stars over a few steps (the sink accretion draws run every step too, so a shared or per-rank stream
shifts every later draw). With one random stream per rank all of these change with the rank count. Also checked:
draws from different call sites are uncorrelated, and duplicate IDs (a snapshot reused as an IC) or missing IDs
(ASSIGN_NEW_IDS) do not collapse every particle onto one draw.
"""

import os
import shutil
from os.path import abspath, dirname, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, run_test

TEST_NAME = "rng_rank_independence"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NGAS, NSTAR, NSINK = 4096, 2000, 64
ID0_STAR, ID0_SINK = 10001, 20001
INIT_STELLAR_AGE_CODE = 3.155e16 / 3.085678e16  # InitStellarAge = 1 Gyr in code time units (kpc/(km/s); GIZMO year = 3.155e7 s)

# name: (IC variant, binary, MPI ranks, OpenMP threads)
RUNS = {
    "np1": ("unique", "keyed", 1, 1),
    "np2": ("unique", "keyed", 2, 1),
    "np4": ("unique", "keyed", 4, 1),
    "np1_omp2": ("unique", "keyed", 1, 2),
    "np2_omp2": ("unique", "keyed", 2, 2),
    "dupids": ("dupids", "keyed", 1, 1),
    "newids": ("noids", "newids", 1, 1),
}
SAME_AS_NP1 = ("np2", "np4", "np1_omp2", "np2_omp2")


def write_ics(variant, fname):
    """unique: all IDs distinct, sink IDs consecutive. dupids: star IDs take 4 values. noids: no ParticleIDs dataset."""
    rng = np.random.default_rng(1234)

    def ball(n, r):
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1)[:, None]
        return x * r * rng.random(n)[:, None] ** (1.0 / 3.0) + 100.0

    pos = {0: ball(NGAS, 1.0), 4: ball(NSTAR, 1.0), 5: ball(NSINK, 1.0)}
    mass = {0: np.full(NGAS, 1e-2 / NGAS), 4: np.full(NSTAR, 1e-3 / NSTAR), 5: np.full(NSINK, 1e-6)}
    ids = {0: np.arange(1, NGAS + 1), 4: np.arange(ID0_STAR, ID0_STAR + NSTAR), 5: np.arange(ID0_SINK, ID0_SINK + NSINK)}
    if variant == "dupids":
        ids[4] = ID0_STAR + (np.arange(NSTAR) % 4)
    with h5py.File(fname, "w") as f:
        h = f.create_group("Header")
        npart = np.array([len(pos.get(t, [])) for t in range(6)], dtype=np.int32)
        h.attrs["NumPart_ThisFile"] = npart
        h.attrs["NumPart_Total"] = npart.astype(np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["Redshift"] = 0.0
        h.attrs["BoxSize"] = 200.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"] = 0.0
        h.attrs["OmegaLambda"] = 0.0
        h.attrs["HubbleParam"] = 1.0
        h.attrs["Flag_Sfr"] = 1
        h.attrs["Flag_Cooling"] = 0
        h.attrs["Flag_StellarAge"] = 1
        h.attrs["Flag_Metals"] = 1
        h.attrs["Flag_Feedback"] = 0
        h.attrs["Flag_DoublePrecision"] = 0
        h.attrs["Flag_IC_Info"] = 0
        for t in pos:
            g = f.create_group(f"PartType{t}")
            g["Coordinates"] = pos[t].astype(np.float32)
            g["Velocities"] = np.zeros_like(pos[t], dtype=np.float32)
            g["Masses"] = mass[t].astype(np.float32)
            if variant != "noids":
                g["ParticleIDs"] = ids[t].astype(np.uint64)
            if t == 0:
                g["InternalEnergy"] = np.full(NGAS, 100.0, dtype=np.float32)


def load(run, snap):
    """Per type: datasets sorted by (ID, child number)."""
    out = {}
    with h5py.File(join(TEST_DIR, f"output_{run}", f"snapshot_{snap:03d}.hdf5"), "r") as f:
        for t in (0, 4, 5):
            if f"PartType{t}" not in f:
                out[t] = {"ParticleIDs": np.zeros(0, dtype=np.uint64)}
                continue
            g = f[f"PartType{t}"]
            child = g["ParticleChildIDsNumber"][:] if "ParticleChildIDsNumber" in g else np.zeros(len(g["ParticleIDs"]), dtype=np.uint64)
            o = np.lexsort((child, g["ParticleIDs"][:]))
            out[t] = {k: g[k][:][o] for k in g.keys()}
    return out


def sink_spin_angles(d):
    L = d[5]["Sink_Specific_AngMom"]
    return L[:, 2] / np.linalg.norm(L, axis=1), np.mod(np.arctan2(L[:, 1], L[:, 0]), 2 * np.pi)


@pytest.fixture(scope="module")
def runs():
    if not os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        cwd = os.getcwd()
        os.chdir(REPO)
        try:
            os.makedirs(join(TEST_DIR, "output_bin"), exist_ok=True)
            build_gizmo_for_test(TEST_NAME, num_openmp_threads=2)
            shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "output_bin", "GIZMO_keyed"))
            build_gizmo_for_test(TEST_NAME, extra_config_flags=("ASSIGN_NEW_IDS",))
            shutil.move(join(TEST_DIR, "GIZMO"), join(TEST_DIR, "output_bin", "GIZMO_newids"))
        finally:
            os.chdir(cwd)
        os.chdir(TEST_DIR)
        try:
            for v in ("unique", "dupids", "noids"):
                write_ics(v, f"{v}_ics.hdf5")
            for name, (ic, binary, ranks, threads) in RUNS.items():
                shutil.rmtree(f"output_{name}", ignore_errors=True)
                shutil.copy(join("output_bin", f"GIZMO_{binary}"), "GIZMO")
                run_test(TEST_NAME, ranks, threads, param_overrides={"InitCondFile": f"{ic}_ics", "OutputDir": f"output_{name}"})
        finally:
            os.chdir(cwd)
    return RUNS


def test_initial_draws_independent_of_ranks_and_threads(runs):
    ref = load("np1", 0)
    for run in SAME_AS_NP1:
        d = load(run, 0)
        for t in (4, 5):
            assert np.array_equal(ref[t]["ParticleIDs"], d[t]["ParticleIDs"])
            ndiff = int(np.sum(ref[t]["StellarFormationTime"] != d[t]["StellarFormationTime"]))
            assert ndiff == 0, f"{run}: initial age of {ndiff}/{len(d[t]['ParticleIDs'])} type-{t} particles differs from np1"
        ndiff = int(np.sum(np.any(ref[5]["Sink_Specific_AngMom"] != d[5]["Sink_Specific_AngMom"], axis=1)))
        assert ndiff == 0, f"{run}: initial spin of {ndiff}/{NSINK} sinks differs from np1"


def test_runtime_draws_independent_of_ranks_and_threads(runs):
    """Which gas cells form stars over the run, on any rank or thread count. The tree force depends on the
    decomposition, so states drift apart and a particle's single-precision position (part of its key) eventually
    flips; the sets must still nearly coincide. With a per-rank stream the overlap is chance (~2%)."""
    for snap, min_common in ((1, 0.95), (2, 0.85)):
        ref = load("np1", snap)
        formed = ref[4]["ParticleIDs"][ref[4]["ParticleIDs"] <= NGAS]
        assert len(formed) >= 20, f"only {len(formed)} stars formed by snapshot {snap}: too few to test the star-formation draws"
        for run in SAME_AS_NP1:
            d = load(run, snap)
            f = d[4]["ParticleIDs"][d[4]["ParticleIDs"] <= NGAS]
            common = len(np.intersect1d(formed, f)) / len(np.union1d(formed, f))
            assert common >= min_common, (f"snapshot {snap}, {run}: only {100*common:.1f}% of the cells converted to stars "
                                          f"on either run are common to np1 and {run} ({len(formed)} vs {len(f)})")


def test_draw_sites_are_uncorrelated(runs):
    d = load("np1", 0)
    age5 = d[5]["StellarFormationTime"]
    mu, phi = sink_spin_angles(d)
    assert np.all(np.diff(d[5]["ParticleIDs"].astype(np.int64)) == 1)
    # 64 sinks: 1 sigma ~ 0.125. Keys built as ID + small offsets give exactly -1 and +1 here.
    assert abs(np.corrcoef(age5, mu)[0, 1]) < 0.5, "sink age and sink spin come from correlated draws"
    assert abs(np.corrcoef(phi[:-1], mu[1:])[0, 1]) < 0.5, "spin phi of one sink and spin mu of the next are correlated"
    assert abs(np.corrcoef(mu, phi)[0, 1]) < 0.5, "the two spin angles of a sink are correlated"
    age4 = d[4]["StellarFormationTime"]
    lo, hi = -2 * INIT_STELLAR_AGE_CODE, 0.0
    assert np.all((age4 >= lo * 1.0001) & (age4 <= hi)), "initial stellar ages outside [-2 InitStellarAge, 0]"
    sigma = (hi - lo) / np.sqrt(12 * NSTAR)
    assert abs(np.mean(age4) - 0.5 * (lo + hi)) < 5 * sigma, "initial stellar ages are not uniform"


def test_duplicate_ids_do_not_collapse_draws(runs):
    d = load("dupids", 0)
    assert len(np.unique(d[4]["ParticleIDs"])) == 4
    n = len(np.unique(d[4]["StellarFormationTime"]))
    assert n == NSTAR, f"{NSTAR} stars sharing 4 IDs drew only {n} distinct initial ages"


def test_missing_ids_do_not_collapse_draws(runs):
    d = load("newids", 0)
    n4 = len(np.unique(d[4]["StellarFormationTime"]))
    mu, _ = sink_spin_angles(d)
    n5 = len(np.unique(mu))
    assert n4 == NSTAR, f"IC without IDs (ASSIGN_NEW_IDS): {NSTAR} stars drew only {n4} distinct initial ages"
    assert n5 == NSINK, f"IC without IDs (ASSIGN_NEW_IDS): {NSINK} sinks drew only {n5} distinct spins"
