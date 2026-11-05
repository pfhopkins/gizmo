"""TURB_DIFF_METALS conservation with individual timesteps.

A periodic box with a metallicity step (0.1 Zsun cold | 60 Zsun hot) puts neighbouring cells in different
timebins, where a one-sided or timestep-mismatched diffusive exchange creates or destroys metals.
Each case builds with TURB_DIFF_METALS_EXACT_CONSERVATION (added here, not in Config.sh, so one case can run the
default exchange) and checks, from the OUTPUT_TURB_DIFF_METALS_BUDGET line, that sum(M*Z) + sum(pending exchange)
stays at its initial value to round-off, and that sum(M*Z) alone is back at that value in the final snapshot.
The IC is read as a snapshot restart (RestartFlag 2) so that its per-cell metallicities are kept.
"""

import glob
import os
import re
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.test import build_gizmo_for_test, mpi_launch_command, write_params_with_overrides
from make_turb_diff_metals_ics import make_ics

TEST_NAME = "turb_diff_metals"
TEST_DIR = f"test/{TEST_NAME}"
RTOL_CONSERVATION = 1e-12
EXACT = ("TURB_DIFF_METALS_EXACT_CONSERVATION",)

_last_build = None


def _build(num_openmp_threads=0, extra_config_flags=()):
    """Builds only when the configuration differs from the binary already in the test directory."""
    global _last_build
    key = (num_openmp_threads, tuple(extra_config_flags))
    if _last_build != key or not os.path.isfile(f"{TEST_DIR}/GIZMO"):
        build_gizmo_for_test(TEST_NAME, num_openmp_threads, tuple(extra_config_flags))
        _last_build = key


def run_case(tag, num_mpi_ranks=2, num_openmp_threads=0, extra_config_flags=(), ic_kwargs=None, param_overrides=None, exact=True):
    """Builds, writes the IC, runs from it as a snapshot restart; returns (initial metal mass, budget rows, output dir)."""
    _build(num_openmp_threads, (EXACT if exact else ()) + tuple(extra_config_flags))
    cwd = os.getcwd()
    os.chdir(TEST_DIR)
    try:
        outdir = f"output_{tag}"
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)
        ic = f"ics_{tag}"
        MZ0 = make_ics(f"{ic}.hdf5", **(ic_kwargs or {}))
        params = write_params_with_overrides(f"{TEST_NAME}.params", {"InitCondFile": ic, "OutputDir": outdir, **(param_overrides or {})})
        os.environ["OMP_NUM_THREADS"] = str(max(num_openmp_threads, 1))
        cmd = mpi_launch_command(num_mpi_ranks, num_openmp_threads) + ["./GIZMO", params, "2"]
        with open(f"{outdir}/gizmo.out", "w") as out, open(f"{outdir}/gizmo.err", "w") as err:
            rc = subprocess.run(cmd, stdout=out, stderr=err, check=False).returncode
        assert rc == 0, f"GIZMO exited with {rc} (see {TEST_DIR}/{outdir}/gizmo.err)"
        budget = read_budget(f"{outdir}/gizmo.out")
    finally:
        os.chdir(cwd)
    return MZ0, budget, f"{TEST_DIR}/{outdir}"


def read_budget(logfile):
    """Rows of (time, sum M*Z, sum pending) from the per-step budget lines."""
    pat = re.compile(r"TURB_DIFF_METALS_BUDGET time= (\S+) MZ= (\S+) pending= (\S+)")
    with open(logfile) as f:
        rows = [tuple(map(float, m.groups())) for m in map(pat.search, f) if m]
    assert rows, f"no TURB_DIFF_METALS_BUDGET lines in {logfile}"
    return np.array(rows)


def snapshots(outdir):
    return sorted(glob.glob(f"{outdir}/snapshot_*.hdf5"))


def metal_mass(snapshot):
    with h5py.File(snapshot, "r") as f:
        p = f["PartType0"]
        Z = p["Metallicity"][:]
        Z = Z[:, 0] if Z.ndim == 2 else Z
        return (p["Masses"][:] * Z).sum(), Z.min()


def max_timebins(outdir):
    """Largest number of timebins holding gas cells at once, from GIZMO's per-step 'Occupied timebins' table."""
    n, best = None, 0
    with open(f"{outdir}/gizmo.out") as f:
        for line in f:
            if line.startswith("Occupied timebins"):
                n = 0
            elif n is not None and "bin=" in line:
                n += int(line.split("bin=")[1].split()[2]) > 0  # columns after 'bin=': bin, non-cells, cells
                best = max(best, n)
            else:
                n = None
    return best


def assert_conserved(MZ0, budget, outdir, min_timebins=2):
    """The checks every case shares."""
    total = budget[:, 1] + budget[:, 2]
    assert abs(total[0] - MZ0) / MZ0 < 1e-6, "first-step metal mass does not match the IC"
    MZ0 = total[0]
    drift = np.abs(total - MZ0) / MZ0
    assert drift.max() < RTOL_CONSERVATION, f"sum(M*Z)+pending drifts by {drift.max():.2e} (step {drift.argmax()}, t={budget[drift.argmax(), 0]:.4g})"
    assert max_timebins(outdir) >= min_timebins, f"fewer than {min_timebins} timebins occupied: no cross-timebin exchange tested"
    assert np.abs(budget[:, 2]).max() > 1e-6 * RTOL_CONSERVATION * MZ0, "no exchange was ever pending: cross-timebin path not exercised"
    MZ_final, Zmin = metal_mass(snapshots(outdir)[-1])
    assert abs(MZ_final - MZ0) / MZ0 < RTOL_CONSERVATION, f"final synchronized snapshot: sum(M*Z) off by {abs(MZ_final - MZ0) / MZ0:.2e}"
    assert Zmin >= 0, "negative metallicity"


@pytest.mark.parametrize("num_mpi_ranks", [1, 4])
def test_turb_diff_metals_conservation(num_mpi_ranks):
    """Exact conservation at every step, independent of the domain decomposition."""
    MZ0, budget, outdir = run_case(f"mpi{num_mpi_ranks}", num_mpi_ranks)
    assert_conserved(MZ0, budget, outdir)


def test_turb_diff_metals_openmp():
    """The j-side write to the pending buffer is shared between threads."""
    MZ0, budget, outdir = run_case("omp", num_mpi_ranks=2, num_openmp_threads=2)
    assert_conserved(MZ0, budget, outdir)


def test_turb_diff_metals_deep_hierarchy():
    """10^4 temperature contrast: a deeper timestep hierarchy (at least 4 occupied timebins)."""
    MZ0, budget, outdir = run_case("deep", ic_kwargs=dict(T_cold=1e3, T_hot=1e7), param_overrides={"TimeMax": 0.02})
    assert_conserved(MZ0, budget, outdir, min_timebins=4)


def test_turb_diff_metals_default_exchange_drifts():
    """The default exchange changes the metal mass with individual timesteps; guards against the two code paths
    silently becoming the same."""
    MZ0, budget, outdir = run_case("default", exact=False)
    assert max_timebins(outdir) >= 2
    MZ_final, _ = metal_mass(snapshots(outdir)[-1])
    drift = abs(MZ_final - MZ0) / MZ0
    assert drift > 1e-9, f"default exchange drifted by only {drift:.2e}: the flag no longer changes anything"
