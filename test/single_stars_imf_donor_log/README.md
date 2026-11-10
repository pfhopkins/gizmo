# single_stars_imf_donor_log

Per-donor log of the gas that new single stars collect (`GALSF_SINGLE_STARS_EVENT_LOGS`), `IMFinfo.txt`: one row per
(star, donor cell) in each step that spawned stars: time, star ID, donor gas ID, mass taken [Msun]. The rows come from the
grant records of the gas collection (`SscGrantRecord`, merged over the passes; failed draws are not logged), gathered on rank 0
and sorted by star and donor, so the file does not depend on the rank count. Steps without a new star do nothing.

Problem: lattice and births of `single_stars_collect_gas` (8 and 40 Msun at neighbouring cells), 2 and 1 ranks.

Checks: per star the donor masses add up to the collected mass of the birth log (1e-12); each pair once; donor IDs are lattice
cells; identical rows on 1 and 2 ranks (1e-14).

Reference: the IMFinfo column definitions.
