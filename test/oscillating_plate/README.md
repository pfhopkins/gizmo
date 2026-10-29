# oscillating_plate

The clamped oscillating plate of Gray, Monaghan & Swift (2001) as set up by Zhang et al. (2023, sec. 6.1): a 2D
(plane-strain) elastic plate, H/dp = 10, clamped at one end by fixed boundary cells (`BOX_BND_PARTICLES`, ParticleID 0)
and started in its first bending mode, run for just over one period. Reference: thin-plate theory, period 0.2541
(published SPH results at H/dp = 30: 0.262-0.273).

Checks: clamp unmoved; period within 10% of thin-plate theory (measured +7.5%; +1.7% at H/dp = 20); first tip excursion
within 10% of v_tip / w (measured +2%); amplitude loss below 20% per period (measured 10%).

    pytest test/oscillating_plate             # ~8 minutes (marked slow); make_ics.py writes the IC
