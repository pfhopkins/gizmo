# isrf_recent_sfr_tallbox

`ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR` for the `RT_ISRF_BACKGROUND` field without SGChem, in the tall-box configuration (`BOX_PERIODIC`, `BOX_LONG_Z=4`, `BOX_OUTFLOW_Z`,
`GRAVITY_PERIODIC_XY_IMAGES`; self-gravity off to keep the run short). The IC is lifted into the middle of the box.

| test | checks |
|---|---|
| `test_isrf_recent_sfr_tallbox` | 2 ranks. 4096 cold gas cells forming stars without feedback, 20 pre-existing stars aging through the 30 Myr window. At each of 65 star-formation calls F equals [M(stars < 30 Myr)/30 Myr]/[Mgas0/4 Gyr] rebuilt from the final snapshot (1e-12) and the code's full scan; the background field is 1.7 max(F, 0.324e-2/1.7) Habing; F rises with star formation and falls as stars leave the window. |

Limits: the running sum does not see a young star that the open z boundary removes (`BOX_OUTFLOW_Z` sets its mass to zero in
the drift). In a first version of this test the gas sat on the z = 0 face, 303 particles left, and the running sum was
0.4 % above the full scan. The test keeps the cloud away from the faces; a hook in the drift that runs on the device is
not part of this change.
