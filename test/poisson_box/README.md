# poisson_box — hydro + MHD + cooling on Poisson-random cell positions

A periodic 100 pc box of magnetized, cooling gas (`MAGNETIC`, `MHD_B_SET_IN_PARAMS`, `COOLING`,
`METALS`, `COOL_METAL_LINES_BY_SPECIES`) at T ~ 1e4 K, whose cells sit at uniformly random
positions -- no lattice, no glass. A random point set is the least regular arrangement the
neighbour search, the kernel-radius iteration and the gradient and MHD estimators can be handed:
local cell counts fluctuate by order unity on the kernel scale, so every kernel radius has to
converge from a poor initial guess and every cell sees a different neighbour geometry. Gravity is
in the code path but negligible (`GravityConstantInternal = 1e-100`).

## Test

`pytest test/poisson_box` generates the 50^3 IC if it is missing, runs a few all-active steps on
2 ranks, and checks:

- the run reaches its final time;
- total gas mass is conserved to round-off;
- every cell has a finite, positive density and kernel radius;
- the magnetic energy of the uniform seed field stays constant to 3e-4 (measured: 3e-5).

The run takes two steps and about a minute on 2 cores.

## Initial conditions and scaling runs

`make_poisson_box_ics.py` writes `poisson_box_<N>_ics.hdf5` for any number of cells per side and
any mean density:

```bash
python make_poisson_box_ics.py                 # 50^3 = 125k cells, n_H = 1 cm^-3 (the test IC)
python make_poisson_box_ics.py 30 50 80 100    # a resolution ladder for scaling runs
python make_poisson_box_ics.py 80 --nH 100     # GMC-like mean density
```

Point `InitCondFile` in a copy of `poisson_box.params` at the IC you want. Because the problem is
homogeneous on average and all cells are active, the cost per step scales cleanly with N, which
makes it a convenient benchmark of the hydro, MHD and cooling pipeline (it is one of the problems
in the CPU-GPU performance table of the User Guide). Set `TimeMax` to a few steps, and compare
runs at matched physical time.
