# hot_gas_timestep_sync

`HOT_GAS_TIMESTEP_SYNC`: while gas above 1e6 K exists, particles near it step at no more than a quarter of the hot gas's Courant step. Hydro fluxes are applied one-sided, so a pair conserves energy only when both partners step together. Adiabatic blast from an ejecta cell in 1 Msun gas; adiabatic so that total energy is the bar. Bars are in the test docstring.
