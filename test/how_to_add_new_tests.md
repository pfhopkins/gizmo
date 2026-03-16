# How to add new tests

Whenever you add a new feature to `gizmo` or fix a bug, you are probably running some kind of test to make sure that it works already. With a little extra work, we can package that test into something that can be run automatically by anyone, which can save a lot of work in the long run.

## Anatomy of a test

Every test has:
1. A name, e.g. `my_test_name`. This is must be a unique identifier.
2. A subdirectory of `gizmo/test` to store the files associated uniquely with the test, e.g. `gizmo/test/my_test_name`.
3. An initial conditions file. This follows the filename convention `my_test_name_ics.hdf5`. These can be too big for the git repo, so are stored remotely. gizmo will currently try to find it in one of two places: `http://www.tapir.caltech.edu/~phopkins/sims`, or `https://users.flatironinstitute.org/~mgrudic/gizmo_tests/my_test_name/`.
4. The `Config.sh` of option flags with which to build GIZMO for that test. This should be tracked by git and go in `gizmo/test/my_test_name/Config.sh`.
5. The parameters file of runtime parameters passed to GIZMO: `gizmo/test/my_test_name/my_test_name.params`.
6. The python source file containing the actual implementation of the pytest test: `gizmo/test/my_test_name/my_test_name.py`. See [Test implementation](#test-implementation) for what this should look like.
7. The test readme `gizmo/test/my_test_name/README.md` that contains a description of the test.
8. **OPTIONAL**: A reference plaintext and/or HDF5 data file containing data describing the exact or benchmark solution, named `my_test_name_exact.txt` and/or `my_test_name_exact.hdf5`. These should live in the same place as the IC file, and will be downloaded on-the-fly when the test is run.

## Test implementation

The actual pass/fail logic of your test should be implemented in a function `test_my_test_name()` in `gizmo/test/my_test_name/my_test_name.py`. This will generally vary from test to test, but the basic idea is to run the simulation, and have some test statistic that you compute from its output to compare with a reference value. This comparison is done via `assert` statement(s) in the `test_my_test_name()` routine. The function may also be responsible for downloading any other special files needed to run GIZMO or perform the test. 

### Example: `gmc_cooling`

This is the implementation of the [gmc_cooling](https://github.com/mikegrudic/gizmo_imf/tree/master/test/gmc_cooling) test, which runs an idealized giant molecular cloud with the microphysics enabled by `COOLING` for a set time, and compares the resulting density vs. temperature relation determined by the cooling physics to a reference solution.

```python
"""GMC cooling and chemistry test"""

from gizmo.test import build_and_run_test, get_cooling_tables
from os import path
from matplotlib import pyplot as plt
import h5py
from astropy import units as u, constants as c
from scipy.stats import binned_statistic
import numpy as np


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
        with h5py.File("gmc_cooling_exact.hdf5", "w") as F:
            F.create_dataset("PartType0/Metallicity", data=Z)
            F.create_dataset("PartType0/Density", data=rho)
            F.create_dataset("PartType0/Temperature", data=T)

    if plot:  # generate a plot of n_H vs T for the test and benchmark solutions
        plt.loglog(nH, T, ".", markersize=1, color="black", label="Test")
        with h5py.File("gmc_cooling_exact.hdf5", "r") as F:
            nH_ref = F["PartType0/Density"][:] * rho_to_nH
            T_ref = F["PartType0/Temperature"][:]
        plt.loglog(nH_ref, T_ref, ".", markersize=1, color="red", label="Benchmark")
        plt.xlabel(r"$n_{\rm H}\,\rm\left(\rm cm^{-3}\right)$")
        plt.ylabel(r"$T (\rm K)$")
        plt.legend(loc=3)
        plt.savefig("test/gmc_cooling/nH_vs_T.png", bbox_inches="tight")
        plt.close()

    # set logarithmic bins for n_H in which to measure the median temperature
    nH_bins = np.logspace(1, 3, 10)

    # return nH-binned median temperature
    return binned_statistic(nH, T, "median", nH_bins)[0]


def test_gmc_cooling():
    # specify the test name
    test_name = "gmc_cooling"

    # download necessary cooling tables (needed for tests with COOLING flag)
    get_cooling_tables()

    # build GIZMO and run the test
    build_and_run_test(test_name)

    # Check that the specific required output file exists:
    if not path.isfile("test/gmc_cooling/output/snapshot_010.hdf5"):
        raise (RuntimeError("GIZMO did not run successfully."))

    # Compute a test statistic from the output
    test_stats = compute_test_statistic("test/gmc_cooling/output/snapshot_010.hdf5", plot=True)

    # compute that same test statistic from a benchmark snapshot
    benchmark_stats = compute_test_statistic("gmc_cooling_exact.hdf5")

    # check that the test and benchmark agree within 10%
    assert np.all(np.isclose(test_stats, benchmark_stats, rtol=0.1))
```