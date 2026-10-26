"""OpenMP thread-count invariance.

Runs one problem with 1 and with N OpenMP threads from the same binary and requires
bit-identical final snapshots. Every threaded loop this configuration exercises works per
particle with a fixed neighbour order, so the thread count must not change a single bit; a data
race does. The build is unoptimized (BUILD_UNOPTIMIZED) because at -O1 and above a racing shared
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


def _ensure_ic():
    if os.path.isfile(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5"):
        return
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


def _total_mass(snap):
    with h5py.File(snap, "r") as F:
        return F["PartType0/Masses"][:].sum()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_omp_thread_invariance(scheme):
    nthreads = min(MAX_THREADS, os.cpu_count() or 1)
    if nthreads < 2:
        pytest.skip("needs at least 2 cores")
    flags = SCHEMES[scheme]
    clean_test_outputs(TEST_NAME, flags)
    outdirs = {nt: f"output_{scheme}_{nt}threads" for nt in (1, nthreads)}
    for d in outdirs.values():
        shutil.rmtree(f"{TEST_DIR}/{d}", ignore_errors=True)
    build_gizmo_for_test(TEST_NAME, num_openmp_threads=nthreads, extra_config_flags=flags)
    _ensure_ic()

    cwd = os.getcwd()
    try:
        os.chdir(TEST_DIR)
        for nt, d in outdirs.items():
            run_test(TEST_NAME, 1, nt, param_overrides={"OutputDir": d})
            shutil.copyfile(f"test_{TEST_NAME}.out", f"{d}/test_{TEST_NAME}.out")
    finally:
        os.chdir(cwd)

    with open(f"{TEST_DIR}/{outdirs[1]}/test_{TEST_NAME}.out") as f:
        assert "BUILD_UNOPTIMIZED" in f.read(), "binary was not built unoptimized; the test would be insensitive"

    snaps = [_final_snapshot(f"{TEST_DIR}/{d}") for d in outdirs.values()]
    if scheme == "mfv":
        m1, mn = (_total_mass(s) for s in snaps)
        assert abs(mn / m1 - 1) < MASS_RTOL, f"total mass with 1 vs {nthreads} threads: {m1!r} vs {mn!r}"
    else:
        diff = _differences(*snaps)
        assert not diff, f"1 vs {nthreads} OpenMP threads differ (field: particles affected): {diff}"
