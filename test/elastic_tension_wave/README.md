# elastic_tension_wave

A standing longitudinal wave in a cold elastic solid (compact Tillotson basalt), 1D periodic, 50 cells per wavelength,
followed for 8 periods. The pressure swings into tension every half period, so the run exercises the elastic solver's
negative-pressure (tensile-instability) correction in the simplest geometry. The reference is linear theory: in 1D the
stress deviator vanishes and the wave speed is the adiabatic sound speed of the Tillotson EOS at the initial state.

Checks: mass and momentum conserved; phase speed within 5% of linear theory (measured 0.967 of it: the tensile
correction slows the wave by about 3% at this amplitude); amplitude loss below 15% per period (measured 10%).

    pytest test/elastic_tension_wave          # ~2 minutes; make_ics.py writes the IC
