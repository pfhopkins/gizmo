"""General routines to build gizmo for a test and obtain ICs and params files"""

from os import system, environ, path, chdir, cpu_count, remove
from urllib.request import urlretrieve, HTTPError
from shutil import move, rmtree
from glob import glob
import pytest
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


def clean_test_outputs(test_name: str):
    """Remove output directory, plot PNGs, and log files from a previous test run."""
    test_dir = f"test/{test_name}"
    output_dir = path.join(test_dir, "output")
    if path.isdir(output_dir):
        rmtree(output_dir)
    for f in glob(path.join(test_dir, "*.png")):
        remove(f)
    for f in glob(path.join(test_dir, f"test_{test_name}.out")):
        remove(f)
    for f in glob(path.join(test_dir, f"test_{test_name}.err")):
        remove(f)


def default_mpi_ranks(max_ranks=None):
    """Return the number of MPI ranks to use, defaulting to the available CPU count.
    Optionally cap at max_ranks (useful for tests with very few particles)."""
    n = cpu_count()
    if max_ranks is not None:
        n = min(n, max_ranks)
    return n


def build_gizmo_for_test(test_name: str):
    """Sets environment variables and runs a script for building gizmo for a given test"""
    environ["TEST_NAME"] = test_name
    system("bash test/build_gizmo_for_test.sh")
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


def run_test(test_name: str, num_mpi_ranks: int = 1):  # , num_openmp_threads: int=0):
    """Runs the test"""
    paramsfile = f"{test_name}.params"
    system(f"mpirun -np {num_mpi_ranks} --use-hwthread-cpus ./GIZMO {paramsfile} 0 1>test_{test_name}.out 2>test_{test_name}.err")


def get_cooling_tables(test_directory="."):
    """Downloads spcool_tables.tar.gz and copies TREECOOL to test directory"""

    url = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/spcool_tables.tgz"
    urlretrieve(url, f"{test_directory}/spcool_tables.tgz")
    system(f"tar -xvf {test_directory}/spcool_tables.tgz -C {test_directory}/; rm spcool_tables.tgz")
    system(f"cp cooling/TREECOOL {test_directory}")


def build_and_run_test(test_name: str, num_mpi_ranks: int = 1, num_openmp_threads: int = 0):
    """Top-level routine that does all necessary building, downloading, and running of the test"""
    clean_test_outputs(test_name)
    if num_openmp_threads > 0:
        environ["OMP_NUM_THREADS"] = num_openmp_threads
    build_gizmo_for_test(test_name)
    chdir(f"test/{test_name}/")
    download_test_files(test_name)
    run_test(test_name, num_mpi_ranks)
    chdir("../../")


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
        pytest.approx((datafields[snapshot1][f], datafields[snapshot2][f]), rel=rtol, abs=atol)


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
