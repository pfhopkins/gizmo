/* radfb_local_gpu.cc — TRANSITION SHIM (Phase 4 / Wave 3 / radfb_local port).
 *
 * The legacy `radiation_pressure_winds_gpu` body + per-source local_fill
 * + `GPU_ALL_SYNC_FUNC(radfbrp)` previously lived here. All three have been
 * superseded by the runner-template port:
 *   - radiation_pressure_winds_consolidated  → galaxy_sf/radfb_local.cc
 *     (toplevel caller dispatches run_neighbor_loop_iterative<RadFBRPSpec>).
 *   - RadFBRPLocalIn / RadFBRPSpec / inline pair body / radfb_rp_local_fill /
 *     radfb_rp_age_threshold_value  → galaxy_sf/radfb_rp_loop.{h,cc}.
 *   - GPU_ALL_SYNC_FUNC_STUB(radfbrp)  → galaxy_sf/radfb_rp_loop.cc.
 *
 * This file is RETAINED in the port commit only to keep its build-rule and
 * GPU_OBJS entry stable across the transition; the file is fully empty of
 * symbols here. Deleted entirely by the radfb_local cleanup commit (next),
 * along with the orphaned `radiation_pressure_winds_gpu` declaration in
 * galaxy_sf/galsf_gpu_decls.h and the legacy ghost_writeback_radfbrp +
 * snapshot / delta structs in mesh/ghost_writeback.{h,cc}.
 *
 * See OPEN_3d_radfb_local_design.md for the three-commit shape.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */
