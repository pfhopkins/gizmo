/* galsf_gpu_decls.h — consolidated GPU dispatch declarations for galaxy_sf
 * kernels: thermal_fb, mechanical_fb, radfb_local (radiation_pressure_winds).
 * dm_dispersion is now runner-ported (dm_dispersion_loop.h) and has no GPU
 * dispatch header.  Step 5 Phase E1a (2026-04-30) — originally merged four
 * single-line headers (radfb_local_gpu.h / thermal_fb_gpu.h /
 * mechanical_fb_gpu.h / dm_dispersion_gpu.h); dm_dispersion_gpu.h retired 3d.D.
 *
 * Each section preserves its original feature gate.  Includers should pull
 * this single header instead of the per-kernel headers.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GALSF_GPU_DECLS_H
#define GALSF_GPU_DECLS_H

#include "mechanical_fb_types.h"  /* full struct MechFBGasDelta */

/* thermal_fb (B6) GPU decls retired in 3e.2: thermal_fb_evaluate_gpu
 * removed entirely; the runner-template caller in thermal_fb.cc replaces it.
 * Declaration intentionally NOT redeclared here. */

/* ---- mechanical_fb (B8) — runner-template ported (Phase 4 / Wave 3 / 3e.1).
 * Runs all 6 modes (-2, -1, 0, 1, 2, 3) of the default-scheme addFB_evaluate
 * via MechFBSpec on the neighbor-loop runner; toplevel
 * mechanical_fb_calc_toplevel calls mechfb_run_iterative below.  All Kokkos /
 * SharedSpace / runner contact is encapsulated inside galaxy_sf/mechfb_loop.cc
 * so this header (and mechanical_fb.cc, compiled non-GPU) does not need Kokkos.
 * The legacy mechanical_fb_evaluate_gpu evaluator was retired together with
 * galaxy_sf/mechanical_fb_gpu.cc. ---- */
struct MechFBGasDelta *mechfb_alloc_local_gas_delta(int n_gas);
void                   mechfb_free_local_gas_delta(struct MechFBGasDelta *p);
struct MechFBCallScalars;
/* Host-side builder for MechFBCallScalars (codex r6 fix).  Used by
 * MechFBSpec::populate_call_scalars so cosmology / unit-factor / CR-rigidity
 * values are read through the same scalars struct instead of bare All.*.
 * Defined in galaxy_sf/mechfb_loop.cc. */
void mechfb_fill_call_scalars(struct MechFBCallScalars *scalars);
/* Zero the gas-only portion of a SharedSpace LocalGasMechFBInfoTemp buffer.
 * Implemented in the GPU TU (mechfb_loop.cc) via Kokkos parallel_for so the
 * non-GPU mechanical_fb.cc TU doesn't take a backend dependency on whether
 * SharedSpace is host-readable bytewise (codex r4-followup 2026-05-14). */
void mechfb_zero_local_gas_delta(struct MechFBGasDelta *p, int n_gas);
void mechfb_run_iterative(int *active_list, int num_active,
                          struct MechFBGasDelta *LocalGasMechFBInfoTemp,
                          int n_gas);


/* ---- radfb_local: radiation_pressure_winds — runner-template ported in
 * Phase 4 / Wave 3 / radfb_local. Toplevel
 * radiation_pressure_winds_consolidated lives in galaxy_sf/radfb_rp_loop.cc;
 * core/proto.h carries its forward decl. Legacy radiation_pressure_winds_gpu
 * thin-wrapper retired in the radfb_local cleanup commit. ---- */


/* ---- dm_dispersion (GALSF_SUBGRID_WINDS, scaling==2) ----
 * Legacy dispdens_gpu_out / disp_density_evaluate_gpu retired by
 * DMDispersionSpec (3d.D port). galaxy_sf/dm_dispersion_loop.{h,cc}
 * is the replacement; dm_dispersion_gpu.cc has been deleted.
 * -------------------------------------------------------------------- */

#endif /* GALSF_GPU_DECLS_H */
