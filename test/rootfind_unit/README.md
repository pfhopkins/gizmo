# rootfind_unit

Unit test for `brent_rootfind()` in `src/system/bracketed_rootfind_functions.h`, the Brent root finder that
`DoCooling` (gas temperature), the H2-fraction solve and `rt_ir_lambdadust` (cooling-side dust temperature) call.

`test_rootfind_unit.cc` stubs the few GIZMO symbols the header uses and runs it on 17 bracketed
problems shaped like the residuals those callers produce: a residual spanning nine decades across the
bracket, |f| ≈ 0 at one end for reasons unrelated to the root, a cubic, a bracket on a negative variable
(x = Tdust − T), jumps (opacity-table zone edges), a flat cubic root, a root at an end point, a very steep
root, and two non-bracketing intervals that exercise the expansion fallback for positive and negative
variables, a 40-decade bracket, a tight tolerance, and residual products that overflow or underflow. Each case
must return a point the solver actually evaluated, within 3× the requested tolerance
of the known root, under the iteration cap, without `endrun()`.

`test_rootfind_unit.py` compiles and runs it twice, with `-O2` and with the production flags
(`-O3 -ffast-math -funroll-loops`). No build of GIZMO is required. `ROOTFIND_HEADER` points it at another
copy of the header, e.g. the gpu_kokkos_unified one, which still has the defects below.

Written 2026-10-13 (for the then macro-based `system/bracketed_rootfind.h`) after the dust-temperature stress
test found that the block returned an unevaluated
interpolation step when the proposed step was shorter than the tolerance (returning an end point for a
root 6× away) and that its expansion fallback walks away from the root for negative variables.
