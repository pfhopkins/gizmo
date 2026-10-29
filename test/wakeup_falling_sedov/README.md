# wakeup_falling_sedov

A point blast (E = 1 in one cell) in cold uniform gas on a 32^3 lattice, in a periodic box that falls in the
uniform field of `GRAVITY_TESTPROBLEM_RT` (g = -0.5 in y). The blast wakes cold cells sitting on long
timesteps (WAKEUP).

Uniform field, periodic box and pairwise hydro forces mean the centre-of-mass velocity is exactly
v_cm,y = -0.5 t. The test reads the total gas momentum from the `ENERGY_BUDGET_DIAGNOSTIC` line at every
full synchronisation (double precision) and requires |v_cm,y + 0.5 t| / (0.5 t) < 1e-3.

A woken cell that loses part of its kick shows up directly here. Energy is the wrong check: the blast's own
integration error (~1e-3) is larger than the missing kick energy (~1e-5), so `sedov` cannot see it.

Reference values (4 MPI ranks, about 3 min):
- before the WAKEUP midpoint re-kick fix: 2.7e-2 (predicted 2.7e-2 from the sum of the missed kick intervals);
- with the fix: 1.6e-4;
- the same with g = 0: 1.1e-4. This is momentum exchange between cells on different timesteps, not the wakeup.
