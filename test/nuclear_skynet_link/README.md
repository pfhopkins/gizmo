# nuclear_skynet_link

`NUCLEAR_NETWORK_SOLVER=1` builds the SkyNet wrapper (`nuclear/nuclear_skynet.cc`). It calls `nuclear_compute_ye_abar`, declared in `nuclear.h` but defined only inline in `nuclear_physics_functions.h`; the wrapper did not include that header, so the link failed with an undefined reference. The test builds the `nuclear_detonation` setup with solver 1.

Run: `SKYNET_DIR=<SkyNet prefix> pytest test/nuclear_skynet_link`, with SkyNet's headers and libraries (and HDF5 C++, Boost, GSL, LAPACK) on CPATH, LIBRARY_PATH and LD_LIBRARY_PATH and in the link flags. Skipped without `SKYNET_DIR`. Solver 2 (`nuclear_torch.cc`) compiles with the include but is not linked by the test: it needs `-ltorch_nuclear`, which has no public source, and `nuclear_torch.cc` is a stub that names XNet and calls no library.
