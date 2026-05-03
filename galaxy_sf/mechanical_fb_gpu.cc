/* mechanical_fb_gpu.cc — GPU neighbor-loop port for addFB_evaluate (B8).
 *
 * Dispatches all 6 modes (-2, -1, 0, 1, 2, 3) of the default-scheme
 * mechanical_fb on the GPU, sharing a single neighbor list across modes.
 * Per-gas deltas atomically accumulated into a MechFBGasDelta buffer; after
 * all modes complete, ghost deltas are reduced to home ranks via
 * ghost_writeback_mechfb; home-cell deltas are scattered by the existing
 * host-side verify_and_assign_local_mechfb_integrals (conservation-enforced).
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
#include "galsf_gpu_decls.h"

#if defined(GALSF_FB_MECHANICAL)

#include "mechanical_fb_functions.h"


/* Fill MechFBLocalIn for source particle i in a given mode — mirrors
 * particle2in_addFB in mechanical_fb.cc but writes into our GPU-friendly
 * struct. particle2in_addFB_fromstars has a public declaration in proto.h. */
static void mech_fb_local_fill(int i, int loop_iteration,
                                struct MechFBLocalIn *loc)
{
    struct addFB_evaluate_data_in_ fb;
    fb.Pos = P[i].Pos;
    fb.Vel = P[i].Vel;
    double heff = P[i].KernelRadius / (double)P[i].NumNgb;
    fb.V_i = heff * heff * heff;
    fb.KernelRadius = P[i].KernelRadius;
#ifdef METALS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) fb.yields[k] = 0.0;
#endif
    for(int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; k++) fb.Area_weighted_sum[k] = P[i].Area_weighted_sum[k];
    fb.Msne = 0; fb.unit_mom_SNe = 0; fb.SNe_v_ejecta = 0;

    if((P[i].DensityAroundParticle > 0) && (P[i].Mass > 0)) {
        if(loop_iteration < 0) {
            /* weighting loops — treat every source as a uniform Msne=mass probe */
            fb.Msne = P[i].Mass;
            fb.unit_mom_SNe = 1.0e-4;
            fb.SNe_v_ejecta = 1.0e-4;
        } else {
            particle2in_addFB_fromstars(&fb, i, loop_iteration);
            fb.unit_mom_SNe = fb.Msne * fb.SNe_v_ejecta;
        }
    }

    loc->Pos = fb.Pos;
    loc->Vel = fb.Vel;
    loc->Msne = fb.Msne;
    loc->KernelRadius = fb.KernelRadius;
    loc->V_i = fb.V_i;
    loc->SNe_v_ejecta = fb.SNe_v_ejecta;
    for(int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; k++) loc->Area_weighted_sum[k] = fb.Area_weighted_sum[k];
#ifdef METALS
    for(int k = 0; k < NUM_METAL_SPECIES; k++) loc->yields[k] = fb.yields[k];
#endif
}


/* Host-side out2particle for Area_weighted_sum writeback between weighting modes. */
static void mech_fb_apply_aws_out(const struct MechFBOut *out, int i, int loop_iteration)
{
    if(P[i].Mass <= 0) return;
    int kmin = 0, kmax = 7;
    if(loop_iteration == -1) { kmin = 7; kmax = AREA_WEIGHTED_SUM_ELEMENTS; }
    for(int k = kmin; k < kmax; k++) P[i].Area_weighted_sum[k] = out->Area_weighted_sum[k];
}

/* Host/device mirror of out2particle_addFB() for coupling modes. Apply the
 * source mass loss immediately after each mode so later modes pack from the
 * updated source state, matching the CPU ordering semantics. */
static void mech_fb_apply_source_mass_out(struct particle_data *P_arr,
                                          struct gas_cell_data *CellP_arr,
                                          int i,
                                          MyFloat M_coupled)
{
    if(P_arr[i].Mass <= 0) return;
    for(int k = 0; k < 3; k++) P_arr[i].dp[k] -= M_coupled * P_arr[i].Vel[k];
    P_arr[i].Mass -= M_coupled;
    if(P_arr[i].Mass < 0 || P_arr[i].Mass != P_arr[i].Mass) P_arr[i].Mass = 0;
    if(P_arr[i].Type == 0) CellP_arr[i].Mass = P_arr[i].Mass;
#ifdef SINGLE_STAR_FB_WINDS
    P_arr[i].Sink_Mass -= M_coupled;
    if(P_arr[i].Sink_Mass < 0 || P_arr[i].Sink_Mass != P_arr[i].Sink_Mass) P_arr[i].Sink_Mass = 0;
#endif
}


void mechanical_fb_evaluate_gpu(struct particle_data *P_host,
                                 struct gas_cell_data *CellP_host,
                                 int num_total,
                                 struct MechFBGasDelta *gas_delta_host,
                                 int n_gas,
                                 int *i_active_host, int num_active,
                                 const double *src_radii_host,
                                 int *n_couplings_out)
{
    ghost_write_detector_begin("mechanical_fb");
    GIZMO_GPU_ENSURE_ALL_FRESH(mechfb);

    /* Caller-supplies-active-list rule: i_active_host[] holds LOCAL indices
       built from ActiveParticleList + per-mode active-check (superset across
       modes). Never iterate num_total here — ghost imports would double-deposit.
       Do NOT early-return when num_active==0: gizmo_density_prep_ghosts and
       ghost_writeback_mechfb are MPI collectives; every rank must participate. */
    int num_src = num_active;

    int imported_ghosts = 0;
    if(ghost_get_num_ghosts() <= 0) {
        gizmo_density_prep_ghosts(gizmo_ghost_safety_factor());
        imported_ghosts = 1;
    }

    int num_all = ghost_get_num_local() + ghost_get_num_ghosts();
    if(num_all <= 0) num_all = num_total;

    /* Step 13 Phase 1 arena. */
    gpu_particles_arena_acquire(num_all, P_host, CellP_host);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = gpu_particles_arena_CellP();

    /* Per-source input / output. 1-element backstop when num_src==0. */
    int alloc_n = (num_src > 0) ? num_src : 1;
    struct MechFBLocalIn *d_local = (struct MechFBLocalIn *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct MechFBLocalIn));
    struct MechFBOut *d_out = (struct MechFBOut *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct MechFBOut));
    int *d_active = (int *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(int));
    if(num_src > 0) memcpy(d_active, i_active_host, num_src * sizeof(int));

    /* Per-gas delta accumulator, sized for home + ghost cells. Zero before kernels. */
    struct MechFBGasDelta *d_gas = (struct MechFBGasDelta *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(num_all * sizeof(struct MechFBGasDelta));
    memset(d_gas, 0, num_all * sizeof(struct MechFBGasDelta));

    /* Build ONE neighbor list; reused across all 6 modes. */
    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_all,
                       i_active_host, num_src,
                       NGB_SEARCH_ONEWAY, 1 /* gas only */,
                       &gnl, gpu_step_sidx_ptr(), 1.0, src_radii_host, NULL, "mech_fb");

    PRINT_STATUS("  GPU mech_fb: %d sources, %d pairs", num_src, gnl.total_pairs);

    /* Per-mode dispatch. Modes beyond -2,-1,0 are only meaningful under
       GALSF_FB_FIRE_STELLAREVOLUTION (mass return, r-process, age tracers). */
    int modes[6] = {-2, -1, 0, 1, 2, 3};
    int num_modes = 3;
#if defined(GALSF_FB_FIRE_STELLAREVOLUTION)
    num_modes = 4;  /* adds mode 1 */
#if defined(GALSF_FB_FIRE_RPROCESS)
    num_modes = 5;  /* adds mode 2 */
#endif
#if defined(GALSF_FB_FIRE_AGE_TRACERS)
    num_modes = 6;  /* adds mode 3 */
#endif
#endif

    for(int mi = 0; mi < num_modes; mi++) {
        int loop_iteration = modes[mi];

        /* Host: pack per-source input for this mode (reads P[i].Area_weighted_sum
           which was updated by the previous weighting pass via mech_fb_apply_aws_out). */
        for(int aa = 0; aa < num_src; aa++) {
            mech_fb_local_fill(i_active_host[aa], loop_iteration, &d_local[aa]);
        }

        /* Zero per-source output. */
        memset(d_out, 0, alloc_n * sizeof(struct MechFBOut));

        /* Launch kernel. */
        {
            int  *offsets   = gnl.offsets;
            int  *neighbors = gnl.neighbors;
            struct MechFBLocalIn *local_arr = d_local;
            struct MechFBOut     *out_arr   = d_out;
            int                  *active_arr = d_active;
            struct particle_data *kp = P_gpu;
            struct gas_cell_data *kc = CellP_gpu;
            struct MechFBGasDelta *kg = d_gas;

            Kokkos::parallel_for("mech_fb", num_src, KOKKOS_LAMBDA(int aa) {
                int i = active_arr[aa];
                /* per-mode active mask — superset list contains stars active in
                   ANY mode; skip those not active in this specific mode. */
                if(!mechanical_fb_star_active_check(i, loop_iteration, kp)) return;

                const struct MechFBLocalIn& loc = local_arr[aa];
                if(loc.Msne <= 0) return;
                if(loc.KernelRadius <= 0) return;

                struct MechFBSourceMode mstate;
                mechanical_fb_per_source_setup(loc, loop_iteration, mstate);

                struct MechFBOut myout;
                myout.M_coupled = 0;
                for(int k = 0; k < AREA_WEIGHTED_SUM_ELEMENTS; k++) myout.Area_weighted_sum[k] = 0;

                int start = offsets[aa], end = offsets[aa+1];
                for(int nn = start; nn < end; nn++) {
                    int j = neighbors[nn];
                    if(kp[j].Type != 0) continue;
                    if(kp[j].Mass <= 0) continue;
                    Vec3<double> dp = loc.Pos - kp[j].Pos;
                    nearest_xyz(dp);
                    double r2 = dp.norm_sq();
                    if(r2 <= 0) continue;
                    double h2j = (double)kp[j].KernelRadius * (double)kp[j].KernelRadius;
                    if((r2 > mstate.h2) && (r2 > h2j)) continue;
                    if(r2 > mstate.r2max_phys) continue;
                    mechanical_fb_pair_kernel(loc, mstate, loop_iteration, j,
                                              kp, kc, kg, dp, r2, myout);
                }
                out_arr[aa] = myout;
            });
        }
        Kokkos::fence();
        gizmo_gpu_check_last_error("mech_fb", num_src);

        /* Host: consume per-source output. For weighting modes (-2, -1), write
           Area_weighted_sum back into P[i] so the NEXT mode's local_fill sees
           the updated value. For coupling modes (>= 0), immediately apply
           source mass loss so later modes pack from the updated source state. */
        if(loop_iteration < 0) {
            for(int aa = 0; aa < num_src; aa++) {
                mech_fb_apply_aws_out(&d_out[aa], i_active_host[aa], loop_iteration);
            }
        } else {
            for(int aa = 0; aa < num_src; aa++) {
                int i = i_active_host[aa];
                MyFloat M_coupled = d_out[aa].M_coupled;
                if(M_coupled <= 0) continue;
                mech_fb_apply_source_mass_out(P_host, CellP_host, i, M_coupled);
                if(P_gpu != P_host || CellP_gpu != CellP_host) {
                    mech_fb_apply_source_mass_out(P_gpu, CellP_gpu, i, M_coupled);
                }
            }
        }
    } /* per-mode loop */

    /* Scatter P/CellP back to host (the kernel itself doesn't modify these,
       but keep this for parity with B6 in case the kernel gains direct writes). */
    memcpy(P_host,     P_gpu,     num_all * sizeof(struct particle_data));
    memcpy(CellP_host, CellP_gpu, num_all * sizeof(struct gas_cell_data));

    /* Copy device delta buffer: home cells -> caller's gas_delta_host; ghost
       cells stay in d_gas for the writeback step to ship home. */
    for(int j = 0; j < n_gas; j++) gas_delta_host[j] = d_gas[j];

    /* Ghost writeback: reduce ghost-side deltas into home-rank home_buf entries.
       d_gas is in SharedSpace (host-accessible), passed as the source; caller's
       gas_delta_host is the home destination. */
    ghost_writeback_mechfb(d_gas, gas_delta_host, n_gas);
    ghost_write_detector_end();

    /* Count gas cells that received coupling, matching N_Gas_Couplings_ThisTask. */
    int n_coup = 0;
    for(int j = 0; j < n_gas; j++) if(gas_delta_host[j].N_injected > 0) n_coup++;
    if(n_couplings_out) *n_couplings_out = n_coup;

    /* Full P+CellP scatter syncs arena→host; ghost_writeback_mechfb above
     * already invalidates for any host changes from ghost-side reduction. */
    gpu_ngb_list_free(&gnl, gpu_step_sidx_ptr());
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_gas);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_active);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_local);
    /* Per-mode source updates above mutate host particle state directly, so
     * invalidate the arena before any later GPU consumer can assume freshness. */
    gpu_particles_arena_invalidate();

    if(imported_ghosts) ghost_exchange_cleanup();
}

GPU_ALL_SYNC_FUNC(mechfb)

#else

void mechanical_fb_evaluate_gpu(struct particle_data *p,
                                 struct gas_cell_data *cp,
                                 int num_total,
                                 struct MechFBGasDelta *gas_delta_host,
                                 int n_gas,
                                 int *i_active_host, int num_active,
                                 const double *src_radii_host,
                                 int *n_couplings_out)
{
    (void)p; (void)cp; (void)num_total; (void)gas_delta_host; (void)n_gas;
    (void)i_active_host; (void)num_active; (void)src_radii_host;
    if(n_couplings_out) *n_couplings_out = 0;
}
GPU_ALL_SYNC_FUNC_STUB(mechfb)

#endif /* GALSF_FB_MECHANICAL */
