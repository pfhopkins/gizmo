# stella_fb_wakeup

Stella feedback (SNe, winds, Type Ia, AGB, radiation-pressure kicks) flags the gas cells it heats or kicks for wake-up, so
that a cell on a long step is moved to a short one at the next sync point. `process_wake_ups` visits only the cells in
the pending wake-up list (plus cells created since), so every flag must also be recorded with `wakeup_list_note`.

## Setup

- Stella production model (`Config_stella_default.sh`) with `GALSF_RESOLVEDISM_ISOLATED_FB_TEST`; gas merge/split on.
- Uniform 20 pc periodic box, 1 Msun cells, n_H = 10 cm^-3, 1e4 K, 0.1 Zsun; the cell nearest the centre becomes a
  12 Msun star, which has winds and is forced to explode at 2.1e-6 (~2 kyr).
- Gas held on long steps (`MaxSizeTimestep` 1e-6), the star on short ones (`MaxStellarTimestep` 100 yr), so the
  feedback reaches gas between its own sync points.
- Built with `WAKEUP_GATE_AUDIT`, which at every sync point checks that each pending request is in the list.

## What is tested

- The run reaches `TimeMax` and the SN is delivered (SNinfo status 0).
- No `WAKEUP-LIST-AUDIT` line (a flagged cell missing from the list) and no `GATE-AUDIT` line (a request left behind
  a shut gate).
