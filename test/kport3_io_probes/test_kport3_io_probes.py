"""Scratch probes for I/O and parameter fixes (to be stripped at integration).

One small binary-IC gas box (as in test/read_ic_binary), each case in its own run directory under output/:
  readsnap_binary         scripts/readsnap.py and gizmo.readsnap read a SnapFormat 1 snapshot
  numfiles_not_pow2       NumFilesWrittenInParallel 3 on 4 ranks runs and writes every restart file
  numfiles_default        omitted, NumFilesWrittenInParallel defaults to the number of ranks
  buffersize_fraction     BufferSize 0.5 (MB) runs, rather than truncating to 0 bytes
  hdf5_snapshot_roundtrip a snapshot from this build (NumPart_ThisFile at 64 bits) reads back as an IC
  hdf5_count_above_int32  an HDF5 IC header holding 3e9 cells of one type is read at 64 bits
  binary_record_fit       SnapFormat 1/2 refuse a file whose block exceeds the 32-bit record marker
"""

import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from os.path import abspath, dirname, isfile, join

import h5py
import numpy as np
import pytest

from gizmo.test import build_gizmo_for_test, link_test_driver, mpi_launch_command

TEST_NAME = "kport3_io_probes"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
RUNS = join(TEST_DIR, "output")
DRIVER = join(TEST_DIR, "output", "record_fit_driver")
N_GAS = 512

sys.path.insert(0, join(REPO, "test", "read_ic_binary"))
from make_binary_ic import make_binary_ic  # noqa: E402

_driver_link_error = None
_readsnap_run_done = []


@pytest.fixture(scope="module")
def gizmo():
    def _link():
        global _driver_link_error
        try:
            link_test_driver(join(TEST_DIR, "record_fit_driver.cc"), DRIVER)
        except AssertionError as e:  # the record check may not exist in the build under test
            _driver_link_error = str(e)[-2000:]

    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, 0, (), post_build=_link)
    finally:
        os.chdir(cwd)
    return join(TEST_DIR, "GIZMO")


def _params(overrides=None, drop=()):
    overrides = dict(overrides or {})
    out = []
    with open(join(TEST_DIR, f"{TEST_NAME}.params")) as f:
        for line in f:
            key = line.split("%")[0].split()
            if key and key[0] in drop:
                continue
            if key and key[0] in overrides:
                out.append(f"{key[0]}    {overrides.pop(key[0])}\n")
            else:
                out.append(line)
    out += [f"{k}    {v}\n" for k, v in overrides.items()]
    return "".join(out)


def _fresh(tag):
    rundir = join(RUNS, tag)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    return rundir


def _launch(gizmo, rundir, params_text, nranks=2, timeout=300):
    """Run GIZMO in rundir; returns (exit status, or None on timeout; stdout+stderr)."""
    with open(join(rundir, "run.params"), "w") as f:
        f.write(params_text)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    cmd = mpi_launch_command(nranks) + [gizmo, "run.params", "0"]
    with open(join(rundir, "log.txt"), "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=rundir, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = None
    return rc, open(join(rundir, "log.txt"), errors="replace").read()


def _run(gizmo, tag, nranks=2, overrides=None, drop=(), timeout=300):
    """The binary-IC box in a fresh output/<tag>; returns (exit status or None, log, run directory)."""
    rundir = _fresh(tag)
    make_binary_ic(join(rundir, f"{TEST_NAME}_ics"), ngas=N_GAS)
    rc, log = _launch(gizmo, rundir, _params(overrides, drop), nranks, timeout)
    return rc, log, rundir


def _ic_arrays():
    """The IC arrays make_binary_ic writes, in ID order."""
    rng = np.random.default_rng(42)
    pos = rng.uniform(0.0, 1.0, size=(N_GAS, 3)).astype(np.float32)
    return pos, np.arange(1, N_GAS + 1)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("reader", ["scripts/readsnap.py", "python_src/gizmo/readsnap.py"])
def test_readsnap_binary(gizmo, reader):
    rundir = join(RUNS, "readsnap_binary")
    if not _readsnap_run_done:
        rc, log, rundir = _run(gizmo, "readsnap_binary", overrides={"SnapFormat": 1})
        assert rc == 0, log[-3000:]
        _readsnap_run_done.append(rundir)
    mod = _load_module(join(REPO, reader), "readsnap_probe")
    P = mod.readsnap(join(rundir, "output"), 0, 0)
    assert P["k"] == 1, P
    pos_ic, ids_ic = _ic_arrays()
    order = np.argsort(P["id"])
    assert np.array_equal(P["id"][order], ids_ic)
    assert np.array_equal(P["p"][order], pos_ic.astype(np.float64))
    assert np.allclose(P["m"], 1.0 / N_GAS, rtol=1e-6, atol=0)
    assert np.allclose(P["u"], 0.05, rtol=1e-6, atol=0)
    assert np.all(P["h"] > 0) and np.all(P["rho"] > 0)


def test_numfiles_not_pow2(gizmo):
    rc, log, rundir = _run(gizmo, "numfiles_not_pow2", nranks=4, overrides={"NumFilesWrittenInParallel": 3})
    assert rc == 0, log[-3000:]
    assert isfile(join(rundir, "output", "snapshot_002.hdf5"))
    restarts = sorted(glob.glob(join(rundir, "output", "restartfiles", "restart.[0-9]")))
    assert len(restarts) == 4, restarts


def test_numfiles_default(gizmo):
    rc, log, rundir = _run(gizmo, "numfiles_default", nranks=4, drop=("NumFilesWrittenInParallel",))
    assert rc == 0, log[-3000:]
    lines = [l for l in log.splitlines() if l.startswith("Tag NumFilesWrittenInParallel")]
    assert len(lines) == 1, lines
    assert lines[0].rstrip().endswith("(=4)"), lines[0]


def test_buffersize_fraction(gizmo):
    rc, log, rundir = _run(gizmo, "buffersize_fraction", overrides={"BufferSize": 0.5}, timeout=240)
    assert rc == 0, f"exit {rc}:\n{log[-3000:]}"
    assert isfile(join(rundir, "output", "snapshot_002.hdf5"))


def test_hdf5_snapshot_roundtrip(gizmo):
    """A snapshot from this build reads back as an IC; its per-file count is written at 64 bits."""
    rc, log, rundir = _run(gizmo, "hdf5_write")
    assert rc == 0, log[-3000:]
    with h5py.File(join(rundir, "output", "snapshot_000.hdf5"), "r") as F:
        npart = F["Header"].attrs["NumPart_ThisFile"]
        print("NumPart_ThisFile dtype:", npart.dtype)
        assert npart[0] == N_GAS
        assert npart.dtype.itemsize == 8, npart.dtype
    rdir = _fresh("hdf5_read")
    shutil.copy(join(rundir, "output", "snapshot_000.hdf5"), join(rdir, "snap_ics.hdf5"))
    rc, log = _launch(gizmo, rdir, _params({"ICFormat": 3, "Initial_Conditions_File": "snap_ics"}))
    assert rc == 0, log[-3000:]
    with h5py.File(join(rdir, "output", "snapshot_002.hdf5"), "r") as F:
        assert F["Header"].attrs["NumPart_Total"][0] == N_GAS



def test_hdf5_count_above_int32(gizmo):
    """A header claiming more cells of one type than an int holds: the total must come through whole."""
    n_big = 3_000_000_000
    bdir = _fresh("hdf5_count_above_int32")
    with h5py.File(join(bdir, "big_ics.hdf5"), "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = np.array([n_big, 0, 0, 0, 0, 0], dtype=np.int64)
        h.attrs["NumPart_Total"] = np.array([n_big & 0xFFFFFFFF, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["NumPart_Total_HighWord"] = np.array([n_big >> 32, 0, 0, 0, 0, 0], dtype=np.uint32)
        h.attrs["MassTable"] = np.zeros(6)
        h.attrs["Time"] = 0.0
        h.attrs["NumFilesPerSnapshot"] = np.int32(1)
        h.attrs["Flag_DoublePrecision"] = np.int32(0)
        F.create_group("PartType0")
    rc, log = _launch(gizmo, bdir, _params({"ICFormat": 3, "Initial_Conditions_File": "big_ics"}))
    print(f"exit {rc}; tail:\n{log[-1500:]}")
    assert f"{n_big} particles" in log, log[-3000:]


def test_binary_record_fit(gizmo):
    assert _driver_link_error is None, _driver_link_error
    r = subprocess.run([DRIVER], capture_output=True, text=True, cwd=TEST_DIR, timeout=120)
    print(r.stdout)
    assert "DONE" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    res = {}
    for l in r.stdout.splitlines():
        if l.startswith("FIT"):
            kv = dict(t.split("=") for t in l.split()[1:])
            res[(int(kv["fmt"]), kv["case"])] = int(kv["fits"])
    for fmt in (1, 2):
        assert res[(fmt, "small")] == 1
        assert res[(fmt, "at_limit")] == 1
        assert res[(fmt, "above_limit")] == 0
        assert res[(fmt, "two_types_above")] == 0
        assert res[(fmt, "above_int32")] == 0
