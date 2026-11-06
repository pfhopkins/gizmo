"""GMC cooling and chemistry test"""

from gizmo.test import (build_and_run_test, build_gizmo_for_test, download_test_files, get_cooling_tables, assert_final_time,
                        default_omp_threads, default_mpi_ranks, get_final_snapshot, mpi_launch_command,
                        write_params_with_overrides)

from matplotlib import pyplot as plt
from os import chdir, environ, getcwd, path
from pathlib import Path
import re
import shutil
import subprocess
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np
import pytest
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


# Reference solution for this branch, published next to the ICs. gmc_cooling_exact.hdf5 was made while the shipped
# TREECOOL's comment header silently disabled the UV background (and with it the tabulated metal-line cooling); this
# one has both. A starforge_dev reference of the same physics is gmc_cooling_exact_uvb.hdf5; the two codes differ.
BENCHMARK = "gmc_cooling_exact_uvb_kokkos.hdf5"
BENCHMARK_URL = f"https://users.flatironinstitute.org/~mgrudic/gizmo_tests/gmc_cooling/{BENCHMARK}"


def fetch_benchmark():
    dest = f"test/gmc_cooling/{BENCHMARK}"
    if not path.isfile(dest):
        try:
            urlretrieve(BENCHMARK_URL, dest)
        except (HTTPError, URLError) as err:
            pytest.fail(f"could not fetch the reference {BENCHMARK_URL}: {err}")


def compute_test_statistic(f, save_reference_solution=False, plot=False):
    """Returns the test statistic to be compared with the reference solution. Optionally, saves the input snapshot data as the reference solution."""

    # get data required to plot n_H vs. T
    with h5py.File(f, "r") as F:
        Z = F["PartType0/Metallicity"][:]
        XH = 1 - F["PartType0/Metallicity"][:, 0] - F["PartType0/Metallicity"][:, 1]
        rho_to_nH = XH * (u.Msun / u.pc**3).to(c.m_p / u.cm**3)
        rho = F["PartType0/Density"][:]
        nH = F["PartType0/Density"][:] * rho_to_nH
        T = F["PartType0/Temperature"][:]

    if save_reference_solution:
        # this option will save the current test solution as the reference solution
        with h5py.File(f"test/gmc_cooling/{BENCHMARK}", "w") as F:
            F.create_dataset("PartType0/Metallicity", data=Z)
            F.create_dataset("PartType0/Density", data=rho)
            F.create_dataset("PartType0/Temperature", data=T)

    if plot:  # generate a plot of n_H vs T for the test and benchmark solutions
        with h5py.File(f"test/gmc_cooling/{BENCHMARK}", "r") as F:
            nH_ref = F["PartType0/Density"][:] * rho_to_nH
            T_ref = F["PartType0/Temperature"][:]
        plt.loglog(nH_ref, T_ref, ".", markersize=0.3, color="red", label="Benchmark")
        plt.loglog(nH, T, ".", markersize=0.3, color="black", label="Test")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling/nH_vs_T.png", bbox_inches="tight")
        plt.close()

    # set logarithmic bins for n_H in which to measure the median temperature
    nH_bins = np.logspace(1, 3, 10)

    # return nH-binned median temperature
    return binned_statistic(nH, T, "median", nH_bins)[0]


@pytest.mark.parametrize("num_mpi_ranks", (2,))  # reference generated at 2 ranks; stochastic cooling is rank-count sensitive
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_gmc_cooling(num_mpi_ranks, num_omp_threads):
    # specify the test name
    test_name = "gmc_cooling"
    test_directory = "test/gmc_cooling"

    # download necessary cooling tables (needed for tests with COOLING flag)
    get_cooling_tables(test_directory)

    # build GIZMO and run the test
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads)
    fetch_benchmark()

    # Check that the simulation produced output and the final time is correct
    final_snap = get_final_snapshot(test_name)
    assert_final_time(final_snap, test_name)

    # Compute a test statistic from the output
    test_stats = compute_test_statistic("test/gmc_cooling/output/snapshot_010.hdf5", plot=True)

    # compute that same test statistic from a benchmark snapshot
    benchmark_stats = compute_test_statistic(f"test/gmc_cooling/{BENCHMARK}")

    # check that the test and benchmark agree within 10%
    assert test_stats == pytest.approx(benchmark_stats, rel=0.1)


# The UV background table data/cooling/TREECOOL starts with a '##' comment header. A reader that cannot skip it loads
# no rows and runs on with no UV background, saying so only in a log line; an unusable table must stop the run instead.
UVB_LOADED = re.compile(r"read ionization table \[TREECOOL\] with (\d+) non-zero UVB entries")
REPO_TREECOOL = Path(__file__).resolve().parents[2] / "data" / "cooling" / "TREECOOL"


def treecool_uvb_rows(text):
    """Leading data rows with a non-zero HI photoionization rate: the number of entries GIZMO should load."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if float(s.split()[1]) == 0:
            break
        n += 1
    return n


def run_with_treecool(case, treecool_text, num_mpi_ranks, num_omp_threads):
    """Run the gmc_cooling binary for a few steps in its own directory with the given TREECOOL; return (exit code,
    stdout+stderr). The timeout turns a rank left waiting on a stopped one into a failure."""
    src = Path("test/gmc_cooling").resolve()
    run_dir = src / "output_uvb_table" / case
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    for name in ("GIZMO", "gmc_cooling_ics.hdf5", "spcool_tables"):
        (run_dir / name).symlink_to(src / name)
    shutil.copy(src / "gmc_cooling.params", run_dir / "gmc_cooling.params")
    # TimeMax stays above MaxSizeTimestep: a shorter timeline cannot represent the first step
    params = write_params_with_overrides(str(run_dir / "gmc_cooling.params"), {"TimeMax": "0.05", "TimeBetSnapshot": "0.05"})
    (run_dir / "TREECOOL").write_text(treecool_text)
    cmd = mpi_launch_command(num_mpi_ranks, num_omp_threads) + ["./GIZMO", Path(params).name, "0"]
    env = dict(environ, OMP_NUM_THREADS=str(num_omp_threads))
    proc = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True, timeout=900)
    log = proc.stdout + proc.stderr
    (run_dir / "run.log").write_text(log)
    return proc.returncode, log


@pytest.mark.parametrize("num_mpi_ranks", (2,))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_gmc_cooling_uvb_table(num_mpi_ranks, num_omp_threads):
    test_name = "gmc_cooling"
    get_cooling_tables(f"test/{test_name}")
    build_gizmo_for_test(test_name, num_omp_threads)
    if not Path(f"test/{test_name}/{test_name}_ics.hdf5").exists():
        cwd = getcwd()
        try:
            chdir(f"test/{test_name}")
            download_test_files(test_name)
        finally:
            chdir(cwd)

    shipped = REPO_TREECOOL.read_text()
    expected = treecool_uvb_rows(shipped)
    assert expected > 0 and shipped.lstrip().startswith("#"), "the shipped TREECOOL no longer exercises a comment header"
    code, log = run_with_treecool("shipped", shipped, num_mpi_ranks, num_omp_threads)
    m = UVB_LOADED.search(log)
    assert code == 0 and m, f"run with the shipped TREECOOL failed (exit {code}):\n{log[-3000:]}"
    assert int(m.group(1)) == expected, f"GIZMO loaded {m.group(1)} UVB entries from the shipped TREECOOL, expected {expected}"

    header = "".join(l for l in shipped.splitlines(keepends=True) if l.lstrip().startswith("#"))
    code, log = run_with_treecool("comments_only", header + "\n", num_mpi_ranks, num_omp_threads)
    assert code != 0 and "has no rows with a non-zero HI photoionization rate" in log, \
        f"a TREECOOL with no data rows did not stop the run (exit {code}):\n{log[-3000:]}"

    first_row = next(l for l in shipped.splitlines(keepends=True) if l.strip() and not l.lstrip().startswith("#"))
    bad_line = len(header.splitlines()) + 2
    code, log = run_with_treecool("malformed_row", header + first_row + "0.005 3.8e-14 not-a-number\n" + first_row,
                                  num_mpi_ranks, num_omp_threads)
    assert code != 0 and f"line {bad_line} is not blank, a '#' comment, or 7 numbers" in log, \
        f"a malformed TREECOOL row did not stop the run with its line number (exit {code}):\n{log[-3000:]}"
