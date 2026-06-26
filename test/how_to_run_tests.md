# How to run and interpret tests

## Step 1: install gizmo as a python package

Run `pip install .` from the gizmo code directory. This will install a python package `gizmo` in your python environment (optional step 0: set up a python virtual environment first if you don't want to mess with your regular environment). 

The `gizmo` package includes the python code in `gizmo/python_src`. In addition to conveniently packaging some of the scripts included in `gizmo/scripts` for convenient use (e.g. `load_from_snapshot`), this package has a submodule `test` that implements various routines for downloading ICs, building gizmo, and running tests.

## Step 2: specify your build environment in `Makefile.systype`

If you are not working in an environment that works with the default build environment, uncomment the build environment you normally use in `Makefile.systype`, e.g. `SYSTYPE=MacBookCellar` should work for a typical macbook environment with openmpi, hdf5, and gsl installed with homebrew.

## Step 3: run pytest

Run `pytest` or `python -m pytest` from the gizmo source code directory. This will run all tests found in the subdirectories of `gizmo/test` and let you know which tests have passed or failed. Optionally, you can pass a pattern to pytest to the test subdirectories you want to run, e.g. `pytest test/my_test_name*`.

## Step 4: Interpret the results
A test can fail for several reasons: failure to build `GIZMO` for the provided `Config.sh` flags, failure to run `GIZMO` due to a runtime error, or a failure of the code output to pass the actual test. A failure to build GIZMO will raise an explicit error and may be accompanied with compiler messages hinting at the problem. A failure to pass the test on the output will result in a pytest failure. Runtime failures may be diagnosed by examining `GIZMO`'s standard output files: `test_my_test_name.out` and `test_my_test_name.err`. 

The actual simulation output directory for the test may be found in `test/my_test_name/output` for direct inspection. Optionally, the test may generate informative diagnostic plots that should be written to `test/my_test_name`.

## Per-test timeout (opt-in)

By default no timeout is applied — each test runs to completion regardless of wall time. This avoids spurious skip/fail noise on the long-running setups (e.g. the binary-cluster tests). When you do want a bound, pick one of:

**GIZMO subprocess timeout** (kills the simulation and marks the test **skipped** rather than failed):

- Per pytest invocation: `GIZMO_TEST_TIMEOUT=120 pytest`
- Per-call from test code: `build_and_run_test("my_test", ..., timeout=120)`
- Globally: edit `DEFAULT_TEST_TIMEOUT` in `python_src/gizmo/test.py` (currently `None`)

**Outer pytest-timeout** (kills the whole pytest worker and marks the test **failed**; useful for non-GIZMO test code that hangs):

- Per test: decorate with `@pytest.mark.timeout(seconds)`
- Globally: set `timeout = <seconds>` under `[tool.pytest.ini_options]` in `pyproject.toml` (currently `0` = disabled)

If you set both, keep `GIZMO_TEST_TIMEOUT` below the pytest-timeout so the inner one fires first and you get a skip, not a failure.