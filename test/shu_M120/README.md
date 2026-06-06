# shu_M120 — reference data

This test compares a run against the reference solution `shu_M120_exact.hdf5`.
Like all GIZMO test reference data and ICs, that file is **not** stored in git
(binary HDF5; ignored via `test/*/*.hdf5`). It is fetched automatically by the
harness `download_test_files()` when the test runs, from the canonical GIZMO
test-data hosts:

- `http://www.tapir.caltech.edu/~phopkins/sims/shu_M120_exact.hdf5`
- `https://users.flatironinstitute.org/~mgrudic/gizmo_tests/shu_M120/shu_M120_exact.hdf5`

If the primary (tapir) host is unavailable, download from the Flatiron mirror,
or symlink a local copy. The file must sit in this directory
(`test/shu_M120/shu_M120_exact.hdf5`).

Note: `convergence_test.py` reads `shu_M120_exact.hdf5` at import time, so it
will error during pytest *collection* if the file is not yet present locally;
this is expected when the reference data has not been downloaded.
