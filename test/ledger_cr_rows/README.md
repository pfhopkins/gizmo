# ledger_cr_rows

The magnetised lattice blast of `crfluid_exchange_pdv_only` with `OUTPUT_MASS_METAL_ENERGY_LEDGER`. Every change of the CR energy has a
row (`E_cr_exchange`, `E_cr_transport`, `E_cr_streaming`, `E_cr_clip`, and injected, shock, collected, losses, ionisation).
Bar, set before the first run: at every ledger line the CR state row equals its start value plus the signed rows, to 1e-5 of the start value.
