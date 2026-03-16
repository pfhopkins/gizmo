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
8. **OPTIONAL**: A reference plaintext and/or HDF5 data file containing data describing the exact or benchmark solution, named `my_test_name_exact.txt` and/or `my_test_name_exact.hdf5`. These should live in the same place as the IC file, and will be downloaded on-the-fly whent he

## Test implementation

The actual pass/fail logic of your test should be implemented in a function `test_my_test_name()` in `gizmo/test/my_test_name/my_test_name.py`. This will generally vary from test to test, but the basic idea is to run the simulation, and have some test statistic that you compute from its output to compare with a reference value. This comparison is done via `assert` statement(s) in the `test_my_test_name()` routine. The function may also be responsible for downloading any other special files needed to run GIZMO or perform the test. 

### Example: gmc_cooling

This is the implementation of the [gmc_cooling](https://github.com/mikegrudic/gizmo_imf/tree/master/test/gmc_cooling) test, which runs an idealized giant molecular cloud with the microphysics enabled by `COOLING` for a set time, and compares the resulting density vs. temperature relation determined by the cooling physics to a reference solution.

