# sn_driven_box

Energy, mass and metal closure over many supernovae landing in driven turbulence, Stella cloud model.

Periodic box, all mass inside, production Stella cloud flags plus `TURB_DRIVING`, star formation off by parameter.
Stage 1 drives gas only. Stage 2 adds pre-placed stars with table masses and ages to the driven snapshot, so the
real SN path fires; winds and radiation of those stars are on and booked as their own channel.

`TEST_GAS_ENERGY_LEDGER_OUTPUT` (test only, off by default) writes in `OutputDir`:

- `energy_ledger.txt`: at full syncs E_kin, E_th, E_B, every channel, residual. Kick terms are booked per cell, source
  terms from the sum over cells before and after the call, so the feedback channels are the actual change of gas state.
- `ledger_sn_events.txt`: for each step with an SN, the actual change in gas, what the star offered, per element.
- `ledger_mass.txt`: at full syncs gas and star masses, gas element masses, element masses offered so far.
- `ledger_events_<task>.txt`: cells with a single kick above 0.01e51 erg.

The environment variable `LEDGER_WAKEUP_OFF` drops all wake-up requests in such a build, to show the test can see the
hydro energy gain after an SN.

Energy: `test_energy_conserved_over_the_events` holds the gas gain over the 12 events to the explosion energy within 2 %
(driving, gravity, winds and cooling booked out). `test_single_sn_energy` puts one 20 Msun star in uniform gas at n_H = 100,
no driving, at 10, 1 and 0.1 Msun cells, dying 10 kyr before the 50 kyr step boundary: from the first full sync after the
explosion the gas holds the explosion energy within 10 %. Without the causal wake (`SNE_BROADCAST_TIMESTEP`) the long gas
steps around the explosion make the hydro create or remove energy: 2.27, 1.75 and 0.81 at the three resolutions, G_cum 0.96.

Run: `pytest test/sn_driven_box`, ranks from `GIZMO_TEST_SN_BOX_RANKS` (default 32).
