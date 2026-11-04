"""GMC cooling and chemistry test"""

from gizmo.test import (build_and_run_test, build_gizmo_for_test, download_test_files, get_cooling_tables, assert_final_time,
                        default_omp_threads, default_mpi_ranks, get_final_snapshot, mpi_launch_command, variant_output_dir,
                        write_params_with_overrides)

from matplotlib import pyplot as plt
from os import chdir, environ, getcwd
from pathlib import Path
import re
import shutil
import subprocess
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np
import pytest


# Reference solution, published next to the ICs. gmc_cooling_exact.hdf5 was made while the shipped TREECOOL's comment
# header silently disabled the UV background (and with it the tabulated metal-line cooling); this one has both.
BENCHMARK = "gmc_cooling_exact_uvb.hdf5"

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
        nH_plot_bins = np.logspace(np.log10(nH.min()+1e-30), np.log10(nH.max()), 30)
        centers = np.sqrt(nH_plot_bins[1:] * nH_plot_bins[:-1])

        with h5py.File(f"test/gmc_cooling/{BENCHMARK}", "r") as F:
            nH_ref = F["PartType0/Density"][:] * rho_to_nH
            T_ref = F["PartType0/Temperature"][:]
        q_ref = [binned_statistic(nH_ref, T_ref, lambda x: np.percentile(x, q), nH_plot_bins)[0] for q in (16, 50, 84)]
        q_test = [binned_statistic(nH, T, lambda x: np.percentile(x, q), nH_plot_bins)[0] for q in (16, 50, 84)]

        plt.loglog(centers, q_ref[1], color="red", label="Benchmark")
        plt.fill_between(centers, q_ref[0], q_ref[2], color="red", alpha=0.2)
        plt.loglog(centers, q_test[1], color="black", label="Test")
        plt.fill_between(centers, q_test[0], q_test[2], color="black", alpha=0.2)
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling/nH_vs_T.png", bbox_inches="tight")
        plt.close()

    # set logarithmic bins for n_H in which to measure the median temperature
    nH_bins = np.logspace(1, 3, 10)

    # return nH-binned median temperature
    return binned_statistic(nH, T, "median", nH_bins)[0]


@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
# JACO_MODEL_STARFORGE is not exercised here: it has never built in this suite, so the variant
# only ever reported a build failure that says nothing about gmc_cooling.
# COOLING_COMPACT_CROSSCHECK aborts the run if the compact-array cooling chain reads the global
# particle arrays instead of its arguments, so reaching TimeMax is that variant's pass condition.
@pytest.mark.parametrize("extra_config_flags", [(), ("COOLING_COMPACT_CROSSCHECK",)], ids=["baseline", "compact_crosscheck"])
def test_gmc_cooling(num_mpi_ranks, num_omp_threads, extra_config_flags):
    test_name = "gmc_cooling"

    get_cooling_tables(f"test/{test_name}")
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, extra_config_flags, exact_files=(BENCHMARK,))

    final_snap = get_final_snapshot(test_name, extra_config_flags)
    assert_final_time(final_snap, test_name)

    if extra_config_flags:
        return  # no benchmark yet for variant runs

    output_dir = variant_output_dir(test_name, extra_config_flags)
    test_stats = compute_test_statistic(f"{output_dir}/snapshot_010.hdf5", plot=True)
    benchmark_stats = compute_test_statistic(f"test/gmc_cooling/{BENCHMARK}")
    assert test_stats == pytest.approx(benchmark_stats, rel=0.1)


# OUTPUT_COOLRATE writes the net cooling rate, -du/dt: positive where the gas cools, negative where it is net
# heating. By the end of the run about a quarter of the cloud sits just below thermal balance, so both signs are
# present; a rate derived from the cooling time came out as exactly zero for every heated cell.
@pytest.mark.parametrize("num_mpi_ranks", (default_mpi_ranks(),))
@pytest.mark.parametrize("num_omp_threads", (default_omp_threads(),))
def test_gmc_cooling_coolrate_sign(num_mpi_ranks, num_omp_threads):
    test_name = "gmc_cooling"
    flags = ("OUTPUT_COOLRATE",)

    get_cooling_tables(f"test/{test_name}")
    build_and_run_test(test_name, num_mpi_ranks, num_omp_threads, flags)
    final_snap = get_final_snapshot(test_name, flags)
    assert_final_time(final_snap, test_name)

    with h5py.File(final_snap, "r") as F:
        rate = F["PartType0/CoolingRate"][:]
    n_neg, n_zero, n_pos = int((rate < 0).sum()), int((rate == 0).sum()), int((rate > 0).sum())
    print(f"CoolingRate: {n_neg} negative, {n_zero} zero, {n_pos} positive of {rate.size}")
    assert n_pos > 0, "no cell is reported as cooling"
    assert n_neg > 0, "no cell is reported as net heating"


# The UV background table cooling/TREECOOL starts with a '##' comment header. A reader that cannot skip it loads no
# rows and runs on with no UV background, saying so only in a log line; an unusable table must stop the run instead.
UVB_LOADED = re.compile(r"read ionization table \[TREECOOL\] with (\d+) non-zero UVB entries")
REPO_TREECOOL = Path(__file__).resolve().parents[2] / "cooling" / "TREECOOL"


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
    stdout+stderr). The timeout turns a rank left waiting on an aborted one into a failure."""
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
