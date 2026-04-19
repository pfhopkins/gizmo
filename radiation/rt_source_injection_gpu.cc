/* rt_source_injection_gpu.cc — GPU neighbor-loop port for rt_source_injection (B5).
 *
 * Source particles (Type != 0 with luminosity) scatter radiation into
 * surrounding gas (Type == 0) neighbors via a Kokkos parallel_for kernel.
 * This replaces the CPU tree-walk inside rt_source_injection() when both
 * GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY and OPENMP_GPU_OFFLOAD are defined.
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

#if defined(RT_SOURCE_INJECTION) && defined(OPENMP_GPU_OFFLOAD) && defined(GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY)

#include "rt_source_injection_functions.h"

/* Pre-fill a RtSrcLocalIn struct for source particle i on CPU.
   Mirrors INPUTFUNCTION_NAME in rt_source_injection.cc. */
static void rt_src_local_fill(int i,
                               struct particle_data *P_host,
                               struct gas_cell_data *CellP_host,
                               struct RtSrcLocalIn *loc)
{
    loc->Pos = P_host[i].Pos;
    loc->KernelRadius = P_host[i].KernelRadius;
    loc->KernelSum_Around_RT_Source = P_host[i].KernelSum_Around_RT_Source;
    double lum[N_RT_FREQ_BINS];
    int active_check = rt_get_source_luminosity(i, 0, lum, P_host, CellP_host);
    double dt = 1.;
#if defined(RT_INJECT_PHOTONS_DISCRETELY)
    dt = get_particle_feedback_timestep_in_physical(i, P_host);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(P_host[i].Type == 5) { dt = P_host[i].dt_since_last_gas_search; }
#endif
#if defined(RT_EVOLVE_FLUX)
    for(int k=0; k<3; k++) {
        if(P_host[i].Type==0) { loc->Vel[k] = CellP_host[i].VelPred[k]; }
        else                  { loc->Vel[k] = P_host[i].Vel[k]; }
    }
#endif
#endif
    for(int k=0; k<N_RT_FREQ_BINS; k++) {
        if(P_host[i].Type==0 || active_check==0) { loc->Luminosity[k]=0; }
        else { loc->Luminosity[k] = lum[k] * dt; }
    }
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    if(P_host[i].Type==5 && active_check) {
        loc->Luminosity[N_RT_FREQ_BINS-1] += P_host[i].Sink_accreted_photon_energy;
        P_host[i].Sink_accreted_photon_energy = 0;
    }
#endif
#if defined(RT_REPROCESS_INJECTED_PHOTONS) && defined(RT_CHEM_PHOTOION)
    loc->Dt = dt;
    if(P_host[i].Type>0) { loc->Density = P_host[i].DensityAroundParticle; }
    else                 { loc->Density = CellP_host[i].Density; }
#endif
}


void rt_source_injection_evaluate_gpu(struct particle_data *P_host,
                                       struct gas_cell_data *CellP_host,
                                       int num_total)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(rtsrcinjection);

    /* Collect active source particles (non-gas with nonzero kernel/luminosity) */
    std::vector<int> active_src;
    std::vector<double> src_radii;
    active_src.reserve(64);
    src_radii.reserve(64);
    for(int i=0; i<num_total; i++) {
        if(P_host[i].Type == 0) continue;
        if(P_host[i].Mass <= 0) continue;
        if(P_host[i].KernelRadius <= 0) continue;
        if(P_host[i].KernelSum_Around_RT_Source <= 0) continue;
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
        if(P_host[i].Type == 5 && !P_host[i].do_gas_search_this_timestep) continue;
#endif
        double lum[N_RT_FREQ_BINS];
        if(rt_get_source_luminosity(i, -1, lum, P_host, CellP_host) == 0) continue;
        active_src.push_back(i);
        double sr = P_host[i].KernelRadius;
#ifdef RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
        sr *= 3.0; /* conservative: RT_SINK_ANGLEWEIGHT checks P[j].KernelRadius too */
#endif
        src_radii.push_back(sr);
    }
    int num_src = (int)active_src.size();
    if(num_src == 0) return;

    /* Build per-source input structs on CPU */
    std::vector<struct RtSrcLocalIn> src_local(num_src);
    for(int a=0; a<num_src; a++) {
        rt_src_local_fill(active_src[a], P_host, CellP_host, &src_local[a]);
    }

    /* Copy P and CellP to SharedSpace */
    struct particle_data *P_gpu = (struct particle_data *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct particle_data));
    struct gas_cell_data *CellP_gpu = (struct gas_cell_data *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_total * sizeof(struct gas_cell_data));
    memcpy(P_gpu,     P_host,     num_total * sizeof(struct particle_data));
    memcpy(CellP_gpu, CellP_host, num_total * sizeof(struct gas_cell_data));

    /* Copy per-source local input to SharedSpace */
    struct RtSrcLocalIn *d_local = (struct RtSrcLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_src * sizeof(struct RtSrcLocalIn));
    memcpy(d_local, src_local.data(), num_src * sizeof(struct RtSrcLocalIn));

    /* Build cross-type neighbor list: sources → gas (j_type_bitmask=1).
       Uses gpu_ngb_list_build directly (same pattern as ags_force_gpu.cc). */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total,
                       active_src.data(), num_src,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */,
                       &gnl, NULL, 1.0, src_radii.data());

    PRINT_STATUS("  GPU rt_source_injection: %d sources, %d pairs", num_src, gnl.total_pairs);

    /* Launch kernel */
    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        struct RtSrcLocalIn *local_arr = d_local;
        struct particle_data  *kp = P_gpu;
        struct gas_cell_data  *kc = CellP_gpu;

        Kokkos::parallel_for("rt_src_injection", num_src, KOKKOS_LAMBDA(int aa) {
            const struct RtSrcLocalIn& loc = local_arr[aa];
            if(loc.KernelRadius <= 0 || loc.KernelSum_Around_RT_Source <= 0) return;
            double h2 = loc.KernelRadius * loc.KernelRadius;

            int start = offsets[aa], end = offsets[aa+1];
            for(int nn=start; nn<end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Type != 0) continue;
                if(kp[j].Mass <= 0) continue;
                Vec3<double> dp = loc.Pos - kp[j].Pos;
                nearest_xyz(dp);
                double r2 = dp.norm_sq();
                if(r2 <= 0) continue;
#ifdef RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
                if((All.TimeStep > 0) && (r2 >= h2) && (r2 >= kp[j].KernelRadius*kp[j].KernelRadius)) continue;
#else
                if(r2 >= h2) continue;
#endif
#ifdef SINK_WIND_SPAWN
                if(kp[j].StellarAge == All.Time) continue;
#endif
                rt_source_injection_pair_kernel(loc, j, kp, kc, r2, dp);
            }
        });
    }
    Kokkos::fence();
    gizmo_gpu_check_last_error("rt_src_injection", num_src);

    /* Scatter: copy modified P and CellP back to host */
    memcpy(P_host,    P_gpu,     num_total * sizeof(struct particle_data));
    memcpy(CellP_host, CellP_gpu, num_total * sizeof(struct gas_cell_data));

    gpu_ngb_list_free(&gnl, NULL);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(CellP_gpu);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(P_gpu);
}

GPU_ALL_SYNC_FUNC(rtsrcinjection)

#else

void rt_source_injection_evaluate_gpu(struct particle_data *p,
                                       struct gas_cell_data *cp,
                                       int num_total)
{
    (void)p; (void)cp; (void)num_total;
}
void gizmo_gpu_sync_all_rtsrcinjection(struct global_data_all_processes *p) { (void)p; }

#endif /* RT_SOURCE_INJECTION && OPENMP_GPU_OFFLOAD && GIZMO_USE_NEIGHBOR_LIST_FOR_DENSITY */
