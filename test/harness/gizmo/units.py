"""Physical constants and code-unit conversions, mirroring GIZMO's own values.

WHY NOT astropy. A test that reconstructs what the code computed -- orbital elements from
vis-viva, total energy from positions and velocities, a temperature from an internal energy --
has to use the constants the CODE integrated with, not the currently accepted ones. GIZMO's
declarations/constants.h carries values that are in places decades old:

    GRAVITY_G_CGS   6.672e-8      CODATA 2018 is 6.67430e-8   (3.3e-4 high)
    SOLAR_MASS_CGS  1.989e33      IAU nominal is 1.98841e33   (3.0e-4 high)

Those two partly cancel in the pc - km/s - Msun unit system, leaving G_code differing from the
astropy value by 4.8e-5. Small, but not nothing: in test/binary that mismatch injects a spurious
|dE/E| oscillation of 9.1e-4 -- an order of magnitude above the 7.8e-5 the test actually
measures -- because the reconstructed energy then carries a term proportional to
(G_test - G_code)/r, which sweeps with the orbit. The per-orbit envelope suppresses it (the run
starts at apocentre, so the artifact returns to its t=0 value once per orbit) but the
instantaneous values do not.

So: dynamics comes from here. Pure unit conversions with no dynamical role -- AU per pc for
reporting, days per code time for sampling a period distribution -- may come from astropy, since
nothing in the code depends on them.

If constants.h changes, change it here too. There is no way to make that automatic that is worth
the fragility of parsing a C header at test time.
"""

# --- mirrors declarations/constants.h -----------------------------------------------------
GRAVITY_G_CGS = 6.672e-8        # constants.h:83
SOLAR_MASS_CGS = 1.989e33       # constants.h:84
BOLTZMANN_CGS = 1.38066e-16     # constants.h:87
PROTONMASS_CGS = 1.6726e-24     # constants.h:89

# --- the unit system every sink/few-body test uses: pc - km/s - Msun ----------------------
UNIT_LENGTH_IN_CM = 3.085678e18
UNIT_MASS_IN_G = SOLAR_MASS_CGS
UNIT_VELOCITY_IN_CM_PER_S = 1.0e5


def G_code(unit_length_cm=UNIT_LENGTH_IN_CM,
           unit_mass_g=UNIT_MASS_IN_G,
           unit_velocity_cgs=UNIT_VELOCITY_IN_CM_PER_S):
    """G in code units, exactly as core/begrun.cc:530 computes All.G.

    Defaults give the pc - km/s - Msun value, 4.300710573e-3 -- which is what the tests must
    use, not the 4.300917270e-3 that astropy (or a table) gives.
    """
    return (GRAVITY_G_CGS * unit_mass_g
            / (unit_length_cm * unit_velocity_cgs * unit_velocity_cgs))


# Convenience: the value the pc - km/s - Msun tests want, so they can just import it.
G_CODE = G_code()

# Pure geometry, no dynamical role -- safe to take from a table (this is astropy's value).
AU_PER_PC = 206264.80624709636
