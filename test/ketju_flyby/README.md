# ketju_flyby

A 1 Msun star passes a bound 1 + 1 Msun, 200 AU pair at 30 km/s, at 1500 AU (outside the pair's KETJU region) and at
600 AU (captured into it); the end state is compared with a DOP853 three-body integration of the same ICs.

- `test_ketju_flyby_motion_caps`: with a field star around the region, its step is capped by 0.1 r_region / v_CoM and
  r_region / (6 sigma) (Mannerkoski+ 2023, eqs. 4 and 6; v_CoM and sigma relative to the surrounding stars).
