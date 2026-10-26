"""OpenMP thread-count invariance.

Runs one problem with 1 and with N OpenMP threads from the same binary and compares the final
snapshots. With MFM every threaded loop this configuration exercises works per particle with a
fixed neighbour order, so the thread count must not change a single bit; a data race does. The build is unoptimized (BUILD_UNOPTIMIZED) because at -O1 and above a racing shared
loop counter lives in a register and the race never reaches the binary. See README.md.

MFV exchanges mass between cells with omp atomic updates, whose order legitimately changes the last
bits, so its variant compares the total mass instead: a lost update moves it by the exchanged mass.
"""

import glob
import importlib.util
import os
import shutil

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, clean_test_outputs, run_test

TEST_NAME = "omp_thread_invariance"
TEST_DIR = f"test/{TEST_NAME}"
N_METAL_SPECIES = 14  # 1 + STARFORGE_FEEDBACK_TRACERS in Config.sh
MAX_THREADS = 8
SCHEMES = {"mfm": (), "mfv": ("HYDRO_MESHLESS_FINITE_VOLUME",)}  # Config.sh leaves the default, MFM
MASS_RTOL = 1e-12


def _make_ic():
    """Regenerated every run: the IC file is gitignored, so a stale one from an older generator would persist."""
    spec = importlib.util.spec_from_file_location("make_ics", f"{TEST_DIR}/make_{TEST_NAME}_ics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.make_IC(n_species=N_METAL_SPECIES, fname=f"{TEST_DIR}/{TEST_NAME}_ics.hdf5")


def _final_snapshot(outdir):
    snaps = sorted(glob.glob(f"{outdir}/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError(f"no snapshots in {outdir}")
    return snaps[-1]


def _differences(snap_a, snap_b):
    """Fields of PartType0 that differ at all, with the number of particles affected."""
    out = {}
    with h5py.File(snap_a, "r") as A, h5py.File(snap_b, "r") as B:
        assert A["Header"].attrs["Time"] == B["Header"].attrs["Time"], "runs ended at different times"
        ia = np.argsort(A["PartType0/ParticleIDs"][:])
        ib = np.argsort(B["PartType0/ParticleIDs"][:])
        assert len(ia) == len(ib), "runs ended with different particle counts"
        for key in A["PartType0"]:
            if key not in B["PartType0"]:
                continue
            a = A[f"PartType0/{key}"][:][ia]
            b = B[f"PartType0/{key}"][:][ib]
            if not np.array_equal(a, b):
                out[key] = int(np.any((a != b).reshape(len(a), -1), axis=1).sum())
    return out


def _diverged_twins(snap):
    """Species pairs (s, s + N/2) start identical (see the IC generator) and get identical physics."""
    with h5py.File(snap, "r") as F:
        Z = F["PartType0/Metallicity"][:]
    half = Z.shape[1] // 2
    return [s for s in range(half) if not np.array_equal(Z[:, s], Z[:, s + half])]


def _metals_evolved(ic, snap):
    """Fraction of surviving cells whose metallicity changed: guards against metal exchange being a no-op."""
    with h5py.File(ic, "r") as A, h5py.File(snap, "r") as B:
        ia, za = A["PartType0/ParticleIDs"][:], A["PartType0/Metallicity"][:]
        ib, zb = B["PartType0/ParticleIDs"][:], B["PartType0/Metallicity"][:]
    _, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    return np.mean(np.any(za[ka] != zb[kb], axis=1))


def _total_mass(snap):
    with h5py.File(snap, "r") as F:
        return F["PartType0/Masses"][:].sum()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_omp_thread_invariance(scheme):
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("compares two runs it makes itself; nothing to validate from external snapshots")
    nthreads = min(MAX_THREADS, len(os.sched_getaffinity(0)))
    if nthreads < 2:
        pytest.skip("needs at least 2 cores")
    flags = SCHEMES[scheme]
    clean_test_outputs(TEST_NAME, flags)
    outdirs = {nt: f"output_{scheme}_{nt}threads" for nt in (1, nthreads)}
    for d in outdirs.values():
        shutil.rmtree(f"{TEST_DIR}/{d}", ignore_errors=True)
    build_gizmo_for_test(TEST_NAME, num_openmp_threads=nthreads, extra_config_flags=flags)
    _make_ic()

    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        for nt, d in outdirs.items():
            # restart flag 2 keeps the IC's metallicities; a flag-0 start resets them to uniform
            run_test(TEST_NAME, 1, nt, param_overrides={"OutputDir": d}, restart_flag=2)
            shutil.copyfile(f"test_{TEST_NAME}.out", f"{d}/test_{TEST_NAME}.out")
    finally:
        os.chdir(cwd)

    with open(f"{TEST_DIR}/{outdirs[1]}/test_{TEST_NAME}.out") as f:
        assert "BUILD_UNOPTIMIZED" in f.read(), "binary was not built unoptimized; the test would be insensitive"

    snaps = [_final_snapshot(f"{TEST_DIR}/{d}") for d in outdirs.values()]
    evolved = _metals_evolved(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5", snaps[0])
    assert evolved > 0.5, f"metallicity changed in only {evolved:.0%} of cells; metal exchange is not being exercised"
    for nt, snap in zip(outdirs, snaps):
        twins = _diverged_twins(snap)
        assert not twins, f"{nt} thread(s): species {twins} diverged from their identical twins"
    if scheme == "mfv":
        m1, mn = (_total_mass(s) for s in snaps)
        assert abs(mn / m1 - 1) < MASS_RTOL, f"total mass with 1 vs {nthreads} threads: {m1!r} vs {mn!r}"
    else:
        diff = _differences(*snaps)
        assert not diff, f"1 vs {nthreads} OpenMP threads differ (field: particles affected): {diff}"


DECOMP_RANKS = 2
N_DECOMPOSITIONS = 5  # correct code agrees for ~10; a race in the cost sums shows in the first, at startup


def _decompositions(log, n):
    with open(log) as f:
        return [l for l in f if l.startswith(("Balance:", "Domain adaptive weights:"))][: 2 * n]


def test_omp_domain_decomposition():
    """With 2 MPI ranks, the first decompositions must agree line for line between 1 and N threads.

    domain_sumCost sums each top-tree leaf's cost and particle count under OpenMP; a leaf index shared between
    threads attributes particles to the wrong leaves and changes the decomposition from the first one, before any
    physics. Snapshots are not compared: multi-rank runs are reproducible at a fixed thread count but not bit-identical
    across thread counts, and drift apart after ~10 decompositions.
    """
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("compares two runs it makes itself; nothing to validate from external snapshots")
    nthreads = min(4, len(os.sched_getaffinity(0)) // DECOMP_RANKS)
    if nthreads < 2:
        pytest.skip(f"needs at least {2 * DECOMP_RANKS} cores")
    clean_test_outputs(TEST_NAME)
    build_gizmo_for_test(TEST_NAME, num_openmp_threads=nthreads)
    _make_ic()
    logs = {}
    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        for nt in (1, nthreads):
            d = f"output_decomp_{nt}threads"
            shutil.rmtree(d, ignore_errors=True)
            run_test(TEST_NAME, DECOMP_RANKS, nt, param_overrides={"OutputDir": d, "TimeMax": 0.02, "TimeBetSnapshot": 0.02},
                     restart_flag=2)
            shutil.copyfile(f"test_{TEST_NAME}.out", f"{d}/test_{TEST_NAME}.out")
            logs[nt] = _decompositions(f"{d}/test_{TEST_NAME}.out", N_DECOMPOSITIONS)
    finally:
        os.chdir(cwd)
    assert len(logs[1]) == 2 * N_DECOMPOSITIONS, f"expected {N_DECOMPOSITIONS} decompositions, got {len(logs[1]) // 2}"
    diff = [(a.strip(), b.strip()) for a, b in zip(logs[1], logs[nthreads]) if a != b]
    assert not diff, f"decomposition with 1 vs {nthreads} threads, first difference:\n  {diff[0][0]}\n  {diff[0][1]}"
