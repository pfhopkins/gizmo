/* thermal_fb_gpu.cc — GPU neighbor-loop port for addthermalFB_evaluate (B6).
 *
 * Active Type-4 (stellar) particles with SNe_ThisTimeStep > 0 scatter
 * thermal energy / ejecta mass into surrounding gas (Type 0) neighbors
 * via a Kokkos parallel_for kernel.  This replaces the CPU tree-walk
 * inside thermal_fb_calc().
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../declarations/macros.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"

#if defined(GALSF_FB_THERMAL)

#include "thermal_fb_functions.h"

/* Fill ThermalFBLocalIn for source particle i from CPU global arrays. */
static void thermal_fb_local_fill(int i,
                                   struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   struct ThermalFBLocalIn *loc)
{
    loc->Pos            = P_host[i].Pos;
    loc->KernelRadius   = P_host[i].KernelRadius;
    loc->wt_sum         = P_host[i].DensityAroundParticle;

    /* Get ejecta properties via CPU-only helper (mirrors particle2in_addthermalFB) */
    struct addFB_evaluate_data_in_ fb;
    particle2in_addFB_fromstars(&fb, i, 0);
    loc->Msne = fb.Msne;
    loc->Esne = 0.5f * fb.Msne * fb.SNe_v_ejecta * fb.SNe_v_ejecta;

    double kz = 0, dwk_dummy = 0;
    kernel_main(0.0, 1.0, 1.0, &kz, &dwk_dummy, -1);
    loc->kernel_zero = (MyFloat)kz;

#ifdef METALS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) { loc->yields[k] = (MyFloat)fb.yields[k]; }
#endif
}


void thermal_fb_evaluate_gpu(struct particle_data *P_host,
                               struct gas_cell_data *CellP_host,
                               int num_total,
                               int *i_active_host, int num_active,
                               const double *src_radii_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(thermalfb);

    /* Caller supplies LOCAL active-source indices (built from ActiveParticleList +
       addthermalFB_evaluate_active_check). Iterating num_total here would include
       ghost-imported sources and double-deposit on multi-rank runs.
       Note: do NOT early-return when num_active==0; ghost prep and ghost_writeback
       are MPI collectives — every rank must participate even with no local sources. */
    int num_src = num_active;

    /* Prep ghosts (check guard from TRANSPORT_SUBCYCLE pitfall) */
    int imported_ghosts = 0;
    if(ghost_get_num_ghosts() <= 0) {
        gizmo_density_prep_ghosts(gizmo_ghost_safety_factor());
        imported_ghosts = 1;
    }

    /* Fill per-source input structs (size guarded so std::vector(0) is well-defined) */
    std::vector<struct ThermalFBLocalIn> src_local(num_src > 0 ? num_src : 1);
    for(int a = 0; a < num_src; a++) {
        thermal_fb_local_fill(i_active_host[a], P_host, CellP_host, &src_local[a]);
    }

    /* Copy P and CellP to SharedSpace */
    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_acquire(num_all, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    /* Snapshot ghost fields before kernel for delta writeback */
    ghost_write_detector_begin("thermal_fb");
    ghost_writeback_zero_thermalfb();

    /* Copy per-source input to SharedSpace (1-element backstop when num_src==0) */
    int alloc_n = (num_src > 0) ? num_src : 1;
    struct ThermalFBLocalIn *d_local = (struct ThermalFBLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct ThermalFBLocalIn));
    if(num_src > 0) memcpy(d_local, src_local.data(), num_src * sizeof(struct ThermalFBLocalIn));

    /* Output: M_coupled per source */
    struct ThermalFBOut *d_out = (struct ThermalFBOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct ThermalFBOut));
    memset(d_out, 0, alloc_n * sizeof(struct ThermalFBOut));  /* zero M_coupled */

    /* Build cross-type neighbor list: Type 4 sources → gas (j_type_bitmask = 1) */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       i_active_host, num_src,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */,
                       &gnl, gpu_step_sidx_ptr(), 1.0, src_radii_host, NULL, "therm_fb");

    PRINT_STATUS("  GPU thermal_fb: %d sources, %d pairs", num_src, gnl.total_pairs);

    /* Launch kernel */
    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        struct ThermalFBLocalIn *local_arr = d_local;
        struct ThermalFBOut     *out_arr   = d_out;
        struct particle_data    *kp = P_gpu;
        struct gas_cell_data    *kc = CellP_gpu;

        Kokkos::parallel_for("thermal_fb", num_src, KOKKOS_LAMBDA(int aa) {
            const struct ThermalFBLocalIn& loc = local_arr[aa];
            if(loc.KernelRadius <= 0 || loc.wt_sum <= 0 || loc.Msne <= 0) return;
            double h2 = (double)loc.KernelRadius * (double)loc.KernelRadius;

            struct ThermalFBOut myout; myout.M_coupled = 0;

            int start = offsets[aa], end = offsets[aa+1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Type != 0) continue;
                if(kp[j].Mass <= 0) continue;
                Vec3<double> dp = loc.Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(r2 <= 0 || r2 >= h2) continue;
                thermal_fb_pair_kernel(loc, j, kp, kc, r2, dp, myout);
            }
            out_arr[aa].M_coupled = myout.M_coupled;
        });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("thermal_fb", num_src);

    /* Scatter modified arrays back to host */
    memcpy(P_host,     P_gpu,     num_all * sizeof(struct particle_data));
    memcpy(CellP_host, CellP_gpu, num_all * sizeof(struct gas_cell_data));

    /* Ghost writeback: propagate j-particle deltas to home ranks */
    ghost_writeback_thermalfb();
    ghost_write_detector_end();

    /* Copy output back and apply to source particles */
    std::vector<struct ThermalFBOut> h_out(num_src);
    memcpy(h_out.data(), d_out, num_src * sizeof(struct ThermalFBOut));

    for(int a = 0; a < num_src; a++) {
        int i = i_active_host[a];
        double M_coupled = h_out[a].M_coupled;
        if(M_coupled > 0 && P_host[i].Mass > 0) {
            for(int k = 0; k < 3; k++) {
                P_host[i].dp[k] -= M_coupled * P_host[i].Vel[k];
            }
            P_host[i].Mass -= M_coupled;
            if(P_host[i].Mass < 0 || P_host[i].Mass != P_host[i].Mass) P_host[i].Mass = 0;
        }
    }

    /* Full P+CellP scatter syncs arena→host; per-source mass-loss loop above
     * mutates P_host directly, so invalidate to cover that. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_ptr());
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    gpu_particles_arena_invalidate();

    if(imported_ghosts) { ghost_exchange_cleanup(); }
}

GPU_ALL_SYNC_FUNC(thermalfb)

#else

void thermal_fb_evaluate_gpu(struct particle_data *p,
                               struct gas_cell_data *cp,
                               int num_total,
                               int *i_active_host, int num_active,
                               const double *src_radii_host)
{
    (void)p; (void)cp; (void)num_total; (void)i_active_host; (void)num_active; (void)src_radii_host;
}
GPU_ALL_SYNC_FUNC_STUB(thermalfb)

#endif /* GALSF_FB_THERMAL */
