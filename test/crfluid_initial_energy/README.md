# crfluid_initial_energy

A 16^3 periodic lattice at rest (`../crfluid_exchange_pdv_only/make_lattice_blast_ics.py`), left half ten times denser, fresh start
(start flag 0) with `COSMIC_RAY_FLUID`, run to t = 0.001. Parameters: ambient 1 eV cm^-3, cloud 0.1 eV cm^-3, threshold 1000 cm^-3 (n_H is about 285 and 2850 cm^-3 in kpc, 1e10 Msun, km/s units).

Check: the CR energy density (cell CR energy / cell volume) in the first snapshot equals the value of its side of the threshold to 1% in
every cell. Without the parameters a fresh start has no CRs.
