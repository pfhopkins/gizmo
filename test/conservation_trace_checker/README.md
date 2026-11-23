# conservation_trace_checker

`gizmo.conservation_trace` (test/harness/gizmo; from stella_dev with the fix/trace-imf-2026-11-17 changes) on synthetic
`mass_balance_steps.txt` (the port's OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS layout) and `energy_step_trace.txt`
(OUTPUT_ENERGY_STEP_TRACE). No build. Bars: mass and metal ledger 1e-12 per checkpoint; every STAGE line; integration
residual 1e-2 of the first sync's scale (or of the SN + Ia energy booked so far); cooling energy RHS vs its reported net cooling at every RHS evaluation, 1e-12 of the gross rates (a declared trace with no
covered evaluation is a finding).
