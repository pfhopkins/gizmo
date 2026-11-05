# isrf_recent_sfr_tallbox

`ISRF_AND_CR_RATE_SCALE_WITH_RECENT_SFR` for the `RT_ISRF_BACKGROUND` field without SGChem (stella_dev `GHOSDT_G0_CR` with
`RT_ISRF_BACKGROUND`), in the tall-box configuration (`BOX_PERIODIC`, `BOX_LONG_Z=4`, `BOX_OUTFLOW_Z`,
`GRAVITY_PERIODIC_XY_IMAGES`; self-gravity off to keep the run short). Same IC and parameters as stella_dev
`test/ghosdt_g0_cr`, lifted into the middle of the box.

| test | checks |
|---|---|
| `test_isrf_recent_sfr_tallbox` | 2 ranks. 4096 cold gas cells forming stars without feedback, 20 pre-existing stars aging through the 30 Myr window. At each of 65 star-formation calls F equals [M(stars < 30 Myr)/30 Myr]/[Mgas0/4 Gyr] rebuilt from the final snapshot (1e-12) and the code's full scan; the background field is 1.7 max(F, 0.324e-2/1.7) Habing; F rises with star formation and falls as stars leave the window. |

Against stella_dev (same IC, own run, `GIZMO` at 22017b407 built with the same flags): F(0) = 115.2 and the maximum
116.99 are identical; the minimum is 99.5 here and 97.4 there and F at the end 107.2 and 106.0, because star formation is
stochastic with different random numbers; stella_dev's F equals its own snapshot rebuild to 1.3e-14.

Limits: the running sum does not see a young star that the open z boundary removes (`BOX_OUTFLOW_Z` sets its mass to zero in
the drift). In a first version of this test the gas sat on the z = 0 face, 303 particles left, and the running sum was
0.4 % above the full scan. The test keeps the cloud away from the faces; a hook in the drift that runs on the device is
not part of this change.
