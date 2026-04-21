"""General routines to build gizmo for a test and obtain ICs and params files"""

from os import system, environ, path, chdir, cpu_count, remove
from urllib.request import urlretrieve, HTTPError
from shutil import move, rmtree
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py


def flush_colorbar(mappable, ax=None, label=None, **kwargs):
    """Add a colorbar that is flush with the axes and lines up with its edges."""
    if ax is None:
        ax = mappable.axes
    fig = ax.get_figure()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, label=label, **kwargs)


def variant_suffix(extra_config_flags=()):
    """Return a filename-safe suffix encoding extra_config_flags (empty for no flags)."""
    if not extra_config_flags:
        return ""
    sanitized = ["".join(c if c.isalnum() else "_" for c in f) for f in extra_config_flags]
    return "_" + "__".join(sanitized)


def variant_output_dir(test_name: str, extra_config_flags=()) -> str:
    """Return the output directory used for a given (test, flag combination)."""
    return f"test/{test_name}/output{variant_suffix(extra_config_flags)}"


def clean_test_outputs(test_name: str, extra_config_flags=()):
    """Remove this variant's output directory, plot PNGs, and log files from a previous test run.
    Other variants' output directories (including the baseline plain "output") are left untouched.
    No-op when GIZMO_TEST_SKIP_BUILD_RUN is set (we're validating externally produced snapshots)."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    test_dir = f"test/{test_name}"
    output_dir = variant_output_dir(test_name, extra_config_flags)
    if path.isdir(output_dir):
        rmtree(output_dir)
    for f in glob(path.join(test_dir, f"test_{test_name}.out")):
        remove(f)
    for f in glob(path.join(test_dir, f"test_{test_name}.err")):
        remove(f)


def default_omp_threads():
    """Return the default number of OpenMP threads for tests."""
    return 2


def default_mpi_ranks(max_ranks=None):
    """Return the number of MPI ranks to use, defaulting to half the available CPU count
    (to leave room for OpenMP threads). Optionally cap at max_ranks."""
    n = cpu_count() // 2
    if max_ranks is not None:
        n = min(n, max_ranks)
    return max(n, 1)


_KOKKOS_SYSTYPES = ("MacBookCellar_Kokkos", "Vista")


def _current_systype():
    """Read active SYSTYPE from Makefile.systype (first uncommented SYSTYPE=... line)."""
    try:
        with open("Makefile.systype") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SYSTYPE=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ""


def build_gizmo_for_test(test_name: str, num_openmp_threads: int = 0, extra_config_flags: tuple = ()):
    """Sets environment variables and runs a script for building gizmo for a given test.
    If num_openmp_threads > 0, appends OPENMP=<num_openmp_threads> to Config.sh before building.
    extra_config_flags is a tuple of strings to append to Config.sh (e.g. ("TRANSPORT_SUBCYCLE=10",)).
    On Kokkos systypes (MacBookCellar_Kokkos, Vista) auto-appends GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY
    if absent, so the Kokkos neighbor-list code path is actually exercised (the non-flag legacy tree
    walk is retained only for backward compat)."""
    system("rm -f GIZMO test/*/GIZMO")
    system(f"cp test/{test_name}/Config.sh .")
    if num_openmp_threads > 0:
        with open("Config.sh", "a") as f:
            f.write(f"\nOPENMP={num_openmp_threads}\n")
    if extra_config_flags:
        with open("Config.sh", "a") as f:
            for flag in extra_config_flags:
                f.write(f"\n{flag}\n")
    if _current_systype() in _KOKKOS_SYSTYPES:
        with open("Config.sh") as f:
            cfg = f.read()
        needs_flag = not any(
            line.strip().startswith("GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY")
            and not line.strip().startswith("#")
            for line in cfg.splitlines()
        )
        if needs_flag:
            with open("Config.sh", "a") as f:
                f.write("\nGIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY\n")
    system("make clean && make -j8")
    if not path.isfile("GIZMO"):
        raise FileNotFoundError("Did not successfully build GIZMO")
    move("GIZMO", f"test/{test_name}/GIZMO")
    system(f"chmod +x test/{test_name}/GIZMO")


def download_test_files(test_name: str):
    """Downloads the ICs and parameter files for a test of a given name"""

    website_path = "http://www.tapir.caltech.edu/~phopkins/sims/"
    website_path2 = f"https://users.flatironinstitute.org/~mgrudic/gizmo_tests/{test_name}/"

    # Note: we are assuming a convention for the test ICs, params, and exact values
    icfile = f"{test_name}_ics.hdf5"
    exactfile = f"{test_name}_exact.txt"  # exact solution (might not exist!)
    exactfile2 = f"{test_name}_exact.hdf5"  # exact solution (might not exist!)

    for f in icfile, exactfile, exactfile2:
        try:
            urlretrieve(website_path + f, f)
        except HTTPError as err:
            try:
                urlretrieve(website_path2 + f, f)
            except HTTPError as err:
                print(f"Could not find {f} at {website_path} or {website_path2}")

    if not path.isfile(icfile):
        raise (FileNotFoundError(f"Could not find ICs and params for test {test_name}"))


def run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0):
    """Runs the test. If num_openmp_threads > 0, sets OMP_NUM_THREADS for the run."""
    if num_openmp_threads > 0:
        environ["OMP_NUM_THREADS"] = str(num_openmp_threads)
    # Pin BLAS to single-threaded so transitive uses (e.g. via Hypre's BoomerAMG
    # in MHD_MODIFIED_GRADIENT) don't introduce nondeterministic/non-reproducible
    # results that get amplified by the divergence-cleaning feedback loop.
    environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    environ.setdefault("MKL_NUM_THREADS", "1")
    # Silence Kokkos "OMP_PROC_BIND not set" warnings.  `false` is the documented
    # unit-testing value; for production runs on a managed cluster the launcher
    # (e.g. ibrun on TACC) pins ranks to cpusets, so binding is already handled.
    environ.setdefault("OMP_PROC_BIND", "false")
    environ.setdefault("OMP_PLACES", "threads")
    paramsfile = f"{test_name}.params"
    bind_opts = "--bind-to none" if num_openmp_threads > 0 else ""
    system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus {bind_opts} ./GIZMO {paramsfile} 0 1>test_{test_name}.out 2>test_{test_name}.err")


def get_cooling_tables(test_directory="."):
    """Downloads spcool_tables.tar.gz and copies TREECOOL to test directory.
    Idempotent: skips steps whose targets already exist (supports symlinks as well as
    real files/dirs), so tests that share cooling data with sibling tests via symlinks
    don't trigger unnecessary downloads."""

    spcool_dir = f"{test_directory}/spcool_tables"
    if not (path.isdir(spcool_dir) or path.islink(spcool_dir)):
        url = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/spcool_tables.tgz"
        urlretrieve(url, f"{test_directory}/spcool_tables.tgz")
        system(f"tar -xvf {test_directory}/spcool_tables.tgz -C {test_directory}/; rm spcool_tables.tgz")
    treecool_dst = f"{test_directory}/TREECOOL"
    if not (path.isfile(treecool_dst) or path.islink(treecool_dst)):
        system(f"cp cooling/TREECOOL {test_directory}")


_BASELINE_STASH = "__output_baseline_stash__"


def stash_baseline_output(test_name: str, extra_config_flags=()):
    """If running a non-baseline variant, move an existing output/ aside so the variant
    run doesn't clobber it. Returns True if a stash was made."""
    if not variant_suffix(extra_config_flags):
        return False
    plain = f"test/{test_name}/output"
    stash = f"test/{test_name}/{_BASELINE_STASH}"
    if path.isdir(plain):
        if path.isdir(stash):
            rmtree(stash)
        move(plain, stash)
        return True
    return False


def finalize_variant_output(test_name: str, extra_config_flags=()):
    """After a non-baseline run, rename output/ → variant dir, then restore the
    baseline stash (if any). Idempotent and safe to call in a finally block."""
    if not variant_suffix(extra_config_flags):
        return
    plain = f"test/{test_name}/output"
    stash = f"test/{test_name}/{_BASELINE_STASH}"
    dst = variant_output_dir(test_name, extra_config_flags)
    if path.isdir(plain):
        if path.isdir(dst):
            rmtree(dst)
        move(plain, dst)
    if path.isdir(stash):
        if path.isdir(plain):
            rmtree(plain)
        move(stash, plain)


def build_and_run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0, extra_config_flags: tuple = ()):
    """Top-level routine that does all necessary building, downloading, and running of the test.
    When extra_config_flags is non-empty, the resulting output/ directory is renamed to a
    variant-specific name so that multiple flag combinations can coexist on disk. The baseline
    output/ (if any) is temporarily stashed aside so it isn't overwritten by the variant run.

    Set env var GIZMO_TEST_SKIP_BUILD_RUN=1 to reuse snapshots already present in
    test/<name>/output/ — useful for validating a remote (e.g. Vista GPU) run against
    the pytest assertions + plots without rebuilding/rerunning locally."""
    if environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        return
    clean_test_outputs(test_name, extra_config_flags)
    build_gizmo_for_test(test_name, num_openmp_threads, extra_config_flags)
    stash_baseline_output(test_name, extra_config_flags)
    try:
        chdir(f"test/{test_name}/")
        download_test_files(test_name)
        run_test(test_name, num_mpi_ranks, num_openmp_threads)
        chdir("../../")
    finally:
        finalize_variant_output(test_name, extra_config_flags)


def parse_params(params_file: str) -> dict:
    """Parse a GIZMO parameter file and return a dict of key-value pairs."""
    params = {}
    with open(params_file) as f:
        for line in f:
            line = line.split("%")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                params[parts[0]] = parts[1]
    return params


def get_final_snapshot(test_name: str, extra_config_flags=()) -> str:
    """Return the path to the last snapshot produced by a test (variant-aware)."""
    output_dir = variant_output_dir(test_name, extra_config_flags)
    snaps = sorted(glob(f"{output_dir}/snapshot_*.hdf5"))
    if not snaps:
        raise RuntimeError(f"No snapshots found for test {test_name} in {output_dir}")
    return snaps[-1]


def assert_final_time(snapshot_file: str, test_name: str, rtol: float = 1e-6):
    """Assert that the snapshot time matches TimeMax from the test's parameter file."""
    params_file = f"test/{test_name}/{test_name}.params"
    params = parse_params(params_file)
    time_max = float(params["TimeMax"])
    with h5py.File(snapshot_file, "r") as F:
        time = float(F["Header"].attrs["Time"])
    assert abs(time - time_max) < rtol * abs(
        time_max
    ), f"Snapshot time {time} does not match TimeMax {time_max} (rtol={rtol})"


def assert_snapshots_are_close(
    snapshot1: str,
    snapshot2: str,
    fields_to_compare: tuple = ("Density", "Velocities", "InternalEnergy"),
    rtol: float = 1e-2,
    atol: float = 0,
):
    """Test-assert that the specified gas data fields in two snapshots are within specified tolerance"""
    fields_to_read = ("ParticleIDs",) + fields_to_compare

    datafields = {snapshot1: {}, snapshot2: {}}
    for s in snapshot1, snapshot2:
        with h5py.File(s, "r") as F:  # read
            for f in fields_to_read:
                datafields[s][f] = F["PartType0/" + f][:]

        id_order = datafields[s]["ParticleIDs"].argsort()
        for f in fields_to_read:  # sort by ID
            datafields[s][f] = datafields[s][f][id_order]

    for f in fields_to_compare:
        np.testing.assert_allclose(
            datafields[snapshot1][f], datafields[snapshot2][f], rtol=rtol, atol=atol,
            err_msg=f"Field {f} differs between {snapshot1} and {snapshot2}",
        )


def plot_1D_snapshot_comparison(
    snapshot1: str,
    snapshot2: str,
    fields_to_plot: tuple = ("Density", "Velocities", "InternalEnergy"),
    output_dir: str = ".",
):
    """Plot 1D comparison of gas data fields between two snapshots, sorted by particle ID."""
    fields_to_read = ("ParticleIDs", "Coordinates") + fields_to_plot

    datafields = {snapshot1: {}, snapshot2: {}}
    for s in snapshot1, snapshot2:
        with h5py.File(s, "r") as F:
            for f in fields_to_read:
                datafields[s][f] = F["PartType0/" + f][:]

        id_order = datafields[s]["ParticleIDs"].argsort()
        for f in fields_to_read:
            datafields[s][f] = datafields[s][f][id_order]

    for f in fields_to_plot:
        plt.plot(datafields[snapshot1]["Coordinates"][:, 0], datafields[snapshot1][f], ".", label="Initial")
        plt.plot(datafields[snapshot2]["Coordinates"][:, 0], datafields[snapshot2][f], ".", label="Final")
        plt.legend()
        plt.ylabel(f)
        plt.xlabel("x")
        plt.savefig(path.join(output_dir, f"{f}.png"))
        plt.close()
