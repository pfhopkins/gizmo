# makefile_systype_orion

`SYSTYPE="Orion_Kokkos"` in `src/Makefile` (also selectable from `Makefile.systype` or `~/.gizmo`): the Orion (MPCDF) build of the
Kokkos port with the OpenMP backend, as the Stella runs on Orion use it: `mpicxx -std=c++20`, `-O3 -funroll-loops -march=native
-fopenmp`, Kokkos 5.0.0 at `KOKKOS_HOME_ORION` (default `/u/uli/sw/kokkos/5.0.0-openmp-gcc13`, override from the environment), modules
`gcc openmpi gsl hwloc fftw-mpi hdf5-mpi`. An unknown SYSTYPE falls through to the compiler defaults without a message (plain `mpiCC`,
no Kokkos).

Check: `make -n` with that SYSTYPE and a one-line Config.sh shows the Orion compile and link lines. Nothing is built.
