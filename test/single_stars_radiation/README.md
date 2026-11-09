# single_stars_radiation

IMF-sampled single stars as sources of the HII-region loop (`GALSF_FB_FIRE_RT_HIIHEATING`), with the ionising photon rate
of the stellar-evolution tables (handed to the loop in its own units of 3.05e10 photons per erg). The local radiation
pressure, radiation transport and long-range bands stay closed to single stars (compile-time error): on this branch the
local radiation pressure needs the gravity-tree bands, which single stars do not feed yet; that comes with the band-flux
extension.

Problem: a 20^3 lattice of 1 Msun cells at n_H = 100 cm^-3, 100 K, no gravity, with cooling (the HII loop needs it), with
a 35 Msun star 1 Myr old, a 20 Msun supernova remnant (which emits nothing) and a 35 Msun star 1000 yr old (not yet
ionising: stars ionise from the end of the pre-main sequence and of the Kelvin-Helmholtz time, as the reference
implementation); a few steps.

Checks: in every step exactly one source; the photon rate in the loop's log equals the table's (log luminosity linear in
log age at the star's grid node, read with h5py) to the 6 digits the log prints.

Ionising onset (test_ionising_onset): eight 10 Msun stars, two per case, at 0.995 and 1.005 of the onset age that
stella_dev's rule gives (no ionising photons before min(t_KH, t_ff at the birth density), at most 1 Myr, nor during the
pre-main sequence): no birth density and a dense birth (the pre-main sequence decides), a thin birth (t_KH decides) and a
birth density whose free-fall time lies between the pre-main-sequence time and t_KH (t_ff decides). Exactly the four later
stars ionise. Without the birth density in the record, only one does.
