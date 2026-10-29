# turb_diff_metals

Conservation test for the turbulent metal diffusion (`TURB_DIFF_METALS`) with individual timesteps, built with the opt-in `TURB_DIFF_METALS_EXACT_CONSERVATION`. Without that flag the default exchange conserves metal mass to integration accuracy only, and the 1e-12 checks below do not apply.

A periodic MFM box of 20^3 cells carries a metallicity step: 0.1 Zsun at 1e4 K for x < L/2 and 60 Zsun at 1e6 K for x > L/2, at equal density. The temperature contrast puts the two sides in different timebins. The pressure jump drives the shear that sets the Smagorinsky diffusivity. The IC is generated locally (`make_turb_diff_metals_ics.py`) and read as a snapshot restart, so its metallicities are not reset to `InitMetallicity`.

Diffusive exchange that a cell receives while inactive is held in its `Dyield_pending` buffer until its next active step. With `OUTPUT_TURB_DIFF_METALS_BUDGET`, GIZMO prints `sum(M*Z)` and `sum(pending)` after every hydro step. Their sum must equal the initial metal mass to round-off at every step; `sum(M*Z)` alone must equal it once all cells are synchronized (the final snapshot).

| test | what it covers |
|---|---|
| `conservation[1,4,8]` | exact budget at every step, on 1, 4 and 8 MPI ranks |
| `openmp` | 2 ranks x 2 threads: the shared write to the pending buffer |
| `deep_hierarchy` | 1e3 K / 1e7 K, at least 5 occupied timebins |
| `merge_split` | cell masses spread by +-60%, merge/split every timestep |
| `rate_matches_equal_timesteps` | the metal mass gained by the cold side and lost by the hot side both agree to 2% with a `FORCE_EQUAL_TIMESTEPS` run at 10 Myr (later the two runs' flows diverge, and with them the diffusivity) |
| `default_rate_matches_equal_timesteps` | the same, for the default exchange: a cell on a shorter step than its neighbour takes its own step's share of the pair-step exchange |
| `mhd_default_exchange_conserves_at_equal_timesteps` | `MAGNETIC` (uniform weak B_z) with `FORCE_EQUAL_TIMESTEPS`, default exchange: both evaluations of a pair compute the same exchange, so `sum(M*Z)` is constant to 1e-12 at every step |
| `comoving_matches_physical` | the box in comoving coordinates (Einstein-de Sitter from a0 = 0.163, where its mass is the critical density) moves the same metal mass across the interface in 10 Myr as the physical run, to 10% (the diffusion weight is a physical density) |

Each conservation case also checks that more than one timebin was occupied, that some exchange was pending at some step (so the cross-timebin path was exercised), and that no metallicity went negative.
