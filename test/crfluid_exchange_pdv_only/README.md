# crfluid_exchange_pdv_only

The lattice blast of `make_lattice_blast_ics.py` (16^3 periodic cells, total blast energy 1, 8 hot cells) with a uniform CR energy of 2
(CR pressure about the gas pressure of the cold gas) and a 2 microG field along z, `COSMIC_RAY_FLUID` with a 300 km/s CR speed,
`MAGNETIC`, `COOLING_OPERATOR_SPLIT` and `CosmicRayDiffusionCoeff` 1, run to t = 0.01 from a snapshot (start flag 2, which keeps the
CR energy). Two builds: the default and `CRFLUID_EXCHANGE_PDV_ONLY`.

Bars: CR + gas energy conserved to 1e-3 in both builds; final CR energy with the flag at most 0.99 of the default; against the reference implementation
(`reference_values.txt`, same ICs): CR gain within 10 % (default), CR energy within 1e-3 (pdV only), gas energies within 1 %.

Without `COOLING_OPERATOR_SPLIT` the gas pays the exchange and the CRs never receive it (no credit in that path of the CR kick, on
the reference implementation either), so the test runs with `COOLING_OPERATOR_SPLIT`.
