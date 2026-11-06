# ledger_step_breakdown

`OUTPUT_MASS_BALANCE_BREAKDOWN` on the box of `test/ledger_step_checkpoints` with `TURB_DIFF_METALS`: checkpoints at the feedback
call, gas collection, the diffusion apply and merge/split; `M_Z_ledger` holds the exchange not yet applied (column `M_Z_inflight`);
`diff_unpack_breakdown.txt` gives per hydro step the pair-flux sum and what the floor at zero added; `MCBAL` lines on stdout give
the gas mass per element. Bars in the test docstring.
