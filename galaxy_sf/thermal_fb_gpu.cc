/* thermal_fb_gpu.cc — GPU neighbor-loop port for addthermalFB_evaluate (B6).
 *
 * Active Type-4 (stellar) particles with SNe_ThisTimeStep > 0 scatter
 * thermal energy / ejecta mass into surrounding gas (Type 0) neighbors
 * via a Kokkos parallel_for kernel.  This replaces the CPU tree-walk
 * inside thermal_fb_calc() when GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY and
 * OPENMP_GPU_OFFLOAD are both defined.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef OPENMP_GPU_OFFLOAD
#include <Kokkos_Core.hpp>
#endif

#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../declarations/macros.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"

#if defined(GALSF_FB_THERMAL) && defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

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
                               int num_total)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(thermalfb);

    /* Collect active sources */
    std::vector<int>    active_src;
    std::vector<double> src_radii;
    for(int i = 0; i < num_total; i++) {
        if(P_host[i].Type != 4) continue;
        if(P_host[i].Mass <= 0) continue;
        if(P_host[i].KernelRadius <= 0) continue;
        if(P_host[i].NumNgb <= 0) continue;
        if(P_host[i].SNe_ThisTimeStep <= 0) continue;
        if(P_host[i].DensityAroundParticle <= 0) continue;
        active_src.push_back(i);
        src_radii.push_back((double)P_host[i].KernelRadius);
    }
    int num_src = (int)active_src.size();
    if(num_src == 0) return;

    /* Prep ghosts (check guard from TRANSPORT_SUBCYCLE pitfall) */
    int imported_ghosts = 0;
    if(ghost_get_num_ghosts() <= 0) {
        gizmo_density_prep_ghosts(P_host, CellP_host, num_total);
        imported_ghosts = 1;
    }

    /* Fill per-source input structs */
    std::vector<struct ThermalFBLocalIn> src_local(num_src);
    for(int a = 0; a < num_src; a++) {
        thermal_fb_local_fill(active_src[a], P_host, CellP_host, &src_local[a]);
    }

    /* Copy P and CellP to SharedSpace */
    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    struct particle_data  *P_gpu = (struct particle_data *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_all * sizeof(struct particle_data));
    struct gas_cell_data  *CellP_gpu = (struct gas_cell_data *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_all * sizeof(struct gas_cell_data));
    memcpy(P_gpu,     P_host,     num_all * sizeof(struct particle_data));
    memcpy(CellP_gpu, CellP_host, num_all * sizeof(struct gas_cell_data));

    /* Snapshot ghost fields before kernel for delta writeback */
    ghost_writeback_zero_thermalfb();

    /* Copy per-source input to SharedSpace */
    struct ThermalFBLocalIn *d_local = (struct ThermalFBLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_src * sizeof(struct ThermalFBLocalIn));
    memcpy(d_local, src_local.data(), num_src * sizeof(struct ThermalFBLocalIn));

    /* Output: M_coupled per source */
    struct ThermalFBOut *d_out = (struct ThermalFBOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_src * sizeof(struct ThermalFBOut));
    memset(d_out, 0, num_src * sizeof(struct ThermalFBOut));  /* zero M_coupled */

    /* Build cross-type neighbor list: Type 4 sources → gas (j_type_bitmask = 1) */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       active_src.data(), num_src,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */,
                       &gnl, NULL, 1.0, src_radii.data());

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

    /* Copy output back and apply to source particles */
    std::vector<struct ThermalFBOut> h_out(num_src);
    memcpy(h_out.data(), d_out, num_src * sizeof(struct ThermalFBOut));

    for(int a = 0; a < num_src; a++) {
        int i = active_src[a];
        double M_coupled = h_out[a].M_coupled;
        if(M_coupled > 0 && P_host[i].Mass > 0) {
            for(int k = 0; k < 3; k++) {
                P_host[i].dp[k] -= M_coupled * P_host[i].Vel[k];
            }
            P_host[i].Mass -= M_coupled;
            if(P_host[i].Mass < 0 || P_host[i].Mass != P_host[i].Mass) P_host[i].Mass = 0;
        }
    }

    gpu_ngb_list_free(&gnl, NULL);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);

    if(imported_ghosts) { ghost_exchange_cleanup(); }
}

GPU_ALL_SYNC_FUNC(thermalfb)

#else

void thermal_fb_evaluate_gpu(struct particle_data *p,
                               struct gas_cell_data *cp,
                               int num_total)
{
    (void)p; (void)cp; (void)num_total;
}
void gizmo_gpu_sync_all_thermalfb(struct global_data_all_processes *p) { (void)p; }

#endif /* GALSF_FB_THERMAL && OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
