# nuclear_solver_sources

`GIZMO_config.h` writes `#define NUCLEAR_NETWORK_SOLVER 1` (a space), and the Makefile looked for `NUCLEAR_NETWORK_SOLVER=1`, so `nuclear/nuclear_skynet.cc` (solver 1) and `nuclear/nuclear_torch.cc` (solver 2) were never compiled. The test reads the commands `make -n` would run for each solver. It builds nothing and needs no SkyNet or torch_nuclear.

The test removes the generated `src/GIZMO_config.h` and `src/compile_time_info.cc` when it ends.
