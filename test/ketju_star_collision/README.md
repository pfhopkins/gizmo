# ketju_star_collision

Type-4 stars inside a KETJU region merge at contact, R1 + R2, as NBODY6 and BIFROST do (FROST-CLUSTERS I, App. B3).
MSTAR checks every accepted step: a separation below R1 + R2, or a pericentre passed since the last check whose
two-body pericentre distance is below R1 + R2 (so a passage inside one GBS step is not missed). The product sits at the
pair's centre of mass with its summed mass and momentum; the heavier star survives.

Cases: head-on, grazing at 0.8 (R1 + R2) with a bound third star, and a miss at 1.25 (R1 + R2). Pure N-body: the radius
is the main-sequence fallback R/Rsun = M^0.8 (M < 1 Msun), M^0.57 (above). Runs take a few seconds on one rank.
