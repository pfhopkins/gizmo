"""General routines to build gizmo for a test and obtain ICs and params files"""

from os import system, environ, path
from urllib.request import urlretrieve, HTTPError


def build_gizmo_for_test(test_name: str):
    """Sets environment variables and runs a script for building gizmo for a given test"""
    environ["TEST_NAME"] = test_name
    system("bash test/build_gizmo_for_test.sh")
    if not path.isfile("GIZMO"):
        raise FileNotFoundError("Did not successfully build GIZMO")


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


def run_test(test_name: str, num_mpi_ranks: int=1):#, num_openmp_threads: int=0):
    """Runs the test"""
    paramsfile = f"test/{test_name}/{test_name}.params"
    system(f"mpirun -np {num_mpi_ranks} ./GIZMO {paramsfile} 0 1>test_{test_name}.out 2>test_{test_name}.err")


def get_cooling_tables():
    """Downloads spcool_tables.tar.gz and copies TREECOOL to test directory"""

    url = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/spcool_tables.tgz"
    urlretrieve(url, "./spcool_tables.tgz")
    system("tar -xvf spcool_tables.tgz")
    system("cp cooling/TREECOOL .")


def build_and_run_test(test_name: str, num_mpi_ranks: int=1, num_openmp_threads: int=0):
    """Top-level routine that does all necessary building, downloading, and running of the test"""
    if num_openmp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = num_openmp_threads
    build_gizmo_for_test(test_name)
    download_test_files(test_name)
    run_test(test_name, num_mpi_ranks)
