/* sink_environment_gpu.cc — GPU-accelerated sink environment loop (B4).
 *
 * Ports sink_environment_evaluate (sinks/sink_environment.cc:163-293) to a
 * Kokkos parallel_for kernel.  i-particles are active sinks (Type=5);
 * j-particles are all types (SINK_NEIGHBOR_BITFLAG).  The neighbor list is
 * built with gpu_build_cross_type_neighbor_list (Infra-2).
 *
 * i-side accumulators are written to per-active-sink sink_env_gpu_out structs
 * and scattered into SinkTempInfo on the host after the kernel.
 *
 * Under SINGLE_STAR_SINK_DYNAMICS + SINK_GRAVCAPTURE_GAS, the kernel does
 * Kokkos::atomic_fetch_min on P[j].SwallowTime for bound particles.  The host
 * driver wraps this call with ghost_writeback_{zero_,}swallowtime() so
 * ghost-side minima are reverse-communicated to their home ranks.
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
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/neighbor_list.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"

#include "sink_environment_gpu.h"

#if defined(SINK_PARTICLES) && defined(OPENMP_GPU_OFFLOAD)

#include "sink_functions.h"

void sink_environment_evaluate_gpu(struct particle_data *P_host,
                                   struct gas_cell_data *CellP_host,
                                   int num_total,
                                   int *i_active_host, int num_active,
                                   const double *i_radii_host,
                                   int j_type_bitmask,
                                   struct sink_env_gpu_out *out_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkenv);

    /* Step 13 Phase 1 arena: pass CellP_host=NULL when no gas; arena handles. */
    gpu_particles_arena_acquire(num_total, P_host, (All.TotN_gas > 0) ? CellP_host : NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    struct gas_cell_data *CellP_gpu = (All.TotN_gas > 0) ? gpu_particles_arena_CellP() : NULL;

    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_ONEWAY, j_type_bitmask, &gnl, NULL,
                       1.0, i_radii_host);

    struct sink_env_gpu_out *d_out = (struct sink_env_gpu_out *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            ((num_active > 0) ? num_active : 1) * sizeof(struct sink_env_gpu_out));

    double *d_radii = (double *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(
            ((num_active > 0) ? num_active : 1) * sizeof(double));
    memcpy(d_radii, i_radii_host, num_active * sizeof(double));

    /* Capture cosmological and physical scalars */
    double cf_atime    = All.cf_atime;
    double cf_a2inv    = All.cf_a2inv;
    double cf_a3inv    = All.cf_a3inv;
    double G_val       = All.G;
    double sink_radius_grav = SinkParticle_GravityKernelRadius;

    PRINT_STATUS("  GPU sink_environment: %d active, j_bitmask=%d, %d pairs",
                 num_active, j_type_bitmask, gnl.total_pairs);

    {
        int *offsets   = gnl.offsets;
        int *neighbors = gnl.neighbors;
        int *active    = gnl.d_active;
        struct particle_data   *kp  = P_gpu;
        struct gas_cell_data   *kc  = CellP_gpu;
        double *radii  = d_radii;
        struct sink_env_gpu_out *kout = d_out;

        gizmo_gpu_kernel_launch("sink_env_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            double h_i = radii[aa];
            memset(&kout[aa], 0, sizeof(struct sink_env_gpu_out));
            if(!(kp[ii].Mass > 0) || !(h_i > 0)) { return; }

            double hinv = 1. / h_i, hinv3 = hinv*hinv*hinv;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
            double ags_h_i = kp[ii].AGS_KernelRadius;
#else
            double ags_h_i = sink_radius_grav;
#endif

            Vec3<double> pos_i = kp[ii].Pos;
            Vec3<double> vel_i = kp[ii].Vel;
            MyIDType id_i      = kp[ii].ID;
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
            double mass_i      = kp[ii].Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
            double sink_r_i    = kp[ii].SinkRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
            Vec3<double> Sink_AngMom_i;
            for(int kv = 0; kv < 3; kv++) Sink_AngMom_i[kv] = kp[ii].Sink_Specific_AngMom[kv];
#endif

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0 || kp[j].Type == 5 || kp[j].ID == id_i) { continue; }

                double wt = kp[j].Mass;
                Vec3<double> dP = kp[j].Pos - pos_i;
                Vec3<double> dv = kp[j].Vel - vel_i;
                nearest_xyz(dP, -1);
                NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i, kp[j].Pos, dv, -1);

#ifdef SINK_REPOSITION_ON_POTMIN
                if(kp[j].Type != 0 && kp[j].Type != 5) {
                    double rfac = dP.norm_sq() * (10. / (h_i*h_i) + 0.1 / (sink_radius_grav*sink_radius_grav));
                    double wtfac = wt / (1. + rfac);
                    if(kp[j].Mass > kout[aa].DF_mmax_particles) kout[aa].DF_mmax_particles = kp[j].Mass;
                    for(int kv = 0; kv < 3; kv++) kout[aa].DF_mean_vel[kv] += wtfac * dv[kv];
                    kout[aa].DF_rms_vel += wtfac;
                    kout[aa].DF_rms_vel += wtfac;
                    kout[aa].DF_rms_vel += wtfac;
                }
#endif

                if(kp[j].Type == 0) {
                    kout[aa].Mgas_in_Kernel += wt;
                    if(kc) { kout[aa].Sink_SurroudingGasInternalEnergy += wt * kc[j].InternalEnergy; }
                    Vec3<double> J_gas = cross(dP, dv);
                    for(int kv = 0; kv < 3; kv++) kout[aa].Jgas_in_Kernel[kv] += wt * J_gas[kv];
#if defined(SINK_OUTPUT_MOREINFO)
                    if(kc) { kout[aa].Sfr_in_Kernel += kc[j].Sfr; }
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
                    for(int kv = 0; kv < 3; kv++) kout[aa].Sink_SurroundingGasVel[kv] += wt * dv[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
                    for(int kv = 0; kv < 3; kv++) kout[aa].Sink_SurroundingGasCOM[kv] += wt * dP[kv];
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
                    {
                        double u_wb = dP.norm() / DMAX(h_i, kp[j].KernelRadius);
                        double wk_wb = 0, dwk_wb = 0;
                        if(u_wb < 1) { kernel_main(u_wb, 1., 1., &wk_wb, &dwk_wb, -1); }
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
                        double r2j = dP.norm_sq();
                        double Lrj = dot(Sink_AngMom_i, dP);
                        Vec3<double> Ang_pass = Sink_AngMom_i * r2j - dP * Lrj;
                        for(int kv = 0; kv < 3; kv++) kout[aa].angmom_prepass_sum_for_passback[kv] += wk_wb * wt * Ang_pass[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
                        kout[aa].kernel_norm_topass_in_swallowloop += wk_wb;
#endif
                    }
#endif
#if (SINK_GRAVACCRETION == 8)
                    if(kc) {
                        double u_h = dP.norm() / h_i;
                        double wk_h = 0, dwk_h = 0;
                        if(u_h < 1) { kernel_main(u_h, hinv3, hinv3*hinv, &wk_h, &dwk_h, -1); }
                        double rj = u_h * h_i * cf_atime;
                        double csj = kc[j].effective_soundspeed();
                        double vdotrj = -dot(dP, dv);
                        double vr_mdot = 4*M_PI * wt*(wk_h*cf_a3inv) * rj*vdotrj;
                        if(rj < sink_radius_grav * cf_atime) {
                            double bondi_mdot = 4*M_PI*G_val*G_val * mass_i*mass_i
                                / pow(csj*csj + dv.norm_sq()*cf_a2inv, 1.5) * wt * (wk_h*cf_a3inv);
                            vr_mdot = DMAX(vr_mdot, bondi_mdot);
                            kout[aa].hubber_mdot_bondi_limiter += bondi_mdot;
                        }
                        kout[aa].hubber_mdot_vr_estimator  += vr_mdot;
                        kout[aa].hubber_mdot_disk_estimator += wt * wk_h * sqrt(rj) / (kc[j].Density * csj*csj);
                    }
#endif
                } else if(kp[j].Type == 4 ||
                          ((kp[j].Type == 2 || kp[j].Type == 3) && !All.ComovingIntegrationOn)) {
                    kout[aa].Mstar_in_Kernel += wt;
                    Vec3<double> J_star = cross(dP, dv);
                    for(int kv = 0; kv < 3; kv++) kout[aa].Jstar_in_Kernel[kv] += wt * J_star[kv];
                } else {
                    kout[aa].Malt_in_Kernel += wt;
                    Vec3<double> J_alt = cross(dP, dv);
                    for(int kv = 0; kv < 3; kv++) kout[aa].Jalt_in_Kernel[kv] += wt * J_alt[kv];
                }

#ifdef SINK_GRAVCAPTURE_GAS
#ifdef GRAIN_FLUID
                if(kp[j].Mass > 0 && (kp[j].Type == 0 || ((1<<kp[j].Type) & GRAIN_PTYPES)))
#else
                if(kp[j].Mass > 0 && kp[j].Type == 0)
#endif
                {
                    double dr_code = dP.norm();
                    double vrel = dv.norm() / cf_atime;
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE)
                    if((1<<kp[j].Type) & GRAIN_PTYPES) {
                        Vec3<double> B_vec; for(int kv = 0; kv < 3; kv++) B_vec[kv] = kp[j].Gas_B[kv];
                        double vrel_dot = dot(dv, B_vec), bmag2 = B_vec.norm_sq();
                        vrel = (fabs(vrel_dot) / sqrt(bmag2)) / cf_atime;
                    }
#endif
                    struct gas_cell_data kc_j_local;
                    if(kp[j].Type == 0 && kc) { kc_j_local = kc[j]; }
                    else { memset(&kc_j_local, 0, sizeof(kc_j_local)); }
                    double vbound = sink_vesc_gpu(kp[j], kc_j_local, mass_i, dr_code, ags_h_i);
                    if(vrel < vbound) {
                        double local_sink_radius = sink_radius_grav;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
                        local_sink_radius = sink_r_i;
                        double spec_mom = dot(dv, dP);
                        double r2 = dP.norm_sq();
                        spec_mom = r2*vrel*vrel - spec_mom*spec_mom*cf_a2inv;
                        if(spec_mom >= G_val * (mass_i + kp[j].Mass) * local_sink_radius) { continue; }
#endif
                        if(sink_check_boundedness_gpu(kp[j], kc_j_local, vrel, vbound, dr_code, local_sink_radius) == 1) {
#ifdef SINGLE_STAR_SINK_DYNAMICS
                            double eps = DMAX(dr_code, DMAX(kp[j].KernelRadius, ags_h_i) * KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER);
                            double tff = eps*eps*eps / (mass_i + kp[j].Mass);
                            Kokkos::atomic_fetch_min(&kp[j].SwallowTime, (MyFloat)tff);
#endif
                            if(kp[j].SwallowID < id_i) { kout[aa].mass_to_swallow_edd += kp[j].Mass; }
                        }
                    }
                }
#endif  /* SINK_GRAVCAPTURE_GAS */
            } /* end neighbor loop */
        });
    }

    /* Copy outputs + any j-side SwallowTime atomics back to host.
     * Full P scatter syncs arena→host for P; CellP unscattered, so host
     * sink_environment.cc post-scatter of out_host into CellP fields makes
     * arena stale — invalidate before next kernel acquires. */
    memcpy(out_host, d_out, num_active * sizeof(struct sink_env_gpu_out));
    memcpy(P_host, P_gpu, num_total * sizeof(struct particle_data));

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_radii);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    gpu_ngb_list_free(&gnl, NULL);
    gpu_particles_arena_invalidate();
}

/* ========================================================================
 * Stage E2 — sink_environment_second: Bulge-Disk aggregator (pure, no j-writes)
 * ======================================================================== */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)

void sink_environment_second_evaluate_gpu(struct particle_data *P_host,
                                           struct gas_cell_data *CellP_host,
                                           int num_total,
                                           int *i_active_host, int num_active,
                                           const double *i_radii_host,
                                           const MyFloat (*i_Jgas_host)[3],
                                           const MyFloat (*i_Jstar_host)[3],
                                           int j_type_bitmask,
                                           struct sink_env_second_gpu_out *out_host)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(sinkenv);

    /* Step 13 Phase 1 arena. CellP unused by this kernel; pass NULL. */
    gpu_particles_arena_acquire(num_total, P_host, NULL);
    struct particle_data *P_gpu = gpu_particles_arena_P();
    (void)CellP_host;

    gpu_neighbor_list_t gnl;
    gpu_ngb_list_build(P_gpu, num_total, i_active_host, num_active,
                       NGB_SEARCH_ONEWAY, j_type_bitmask, &gnl, NULL,
                       1.0, i_radii_host);

    int alloc_n = (num_active > 0) ? num_active : 1;
    struct sink_env_second_gpu_out *d_out = (struct sink_env_second_gpu_out *)
        Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(struct sink_env_second_gpu_out));

    /* Stage per-source J vectors into SharedSpace so the kernel can read them. */
    double (*d_Jgas)[3]  = (double (*)[3]) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(double[3]));
    double (*d_Jstar)[3] = (double (*)[3]) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(alloc_n * sizeof(double[3]));
    for(int a = 0; a < num_active; a++) {
        for(int k = 0; k < 3; k++) {
            d_Jgas[a][k]  = (double)i_Jgas_host[a][k];
            d_Jstar[a][k] = (double)i_Jstar_host[a][k];
        }
    }

    PRINT_STATUS("  GPU sink_environment_second: %d active, %d pairs", num_active, gnl.total_pairs);

    int comoving_on = All.ComovingIntegrationOn;

    {
        int  *offsets   = gnl.offsets;
        int  *neighbors = gnl.neighbors;
        int  *active    = gnl.d_active;
        struct particle_data *kp   = P_gpu;
        double (*dJg)[3]  = d_Jgas;
        double (*dJs)[3]  = d_Jstar;
        struct sink_env_second_gpu_out *kout = d_out;

        gizmo_gpu_kernel_launch("sink_env_second_kernel", num_active, KOKKOS_LAMBDA(int aa) {
            int ii = active[aa];
            memset(&kout[aa], 0, sizeof(struct sink_env_second_gpu_out));
            if(!(kp[ii].Mass > 0)) return;

            Vec3<double> pos_i = kp[ii].Pos;
            Vec3<double> vel_i = kp[ii].Vel;
            Vec3<double> Jgas_i{dJg[aa][0], dJg[aa][1], dJg[aa][2]};
            Vec3<double> Jstar_i{dJs[aa][0], dJs[aa][1], dJs[aa][2]};

            int start = offsets[aa], end = offsets[aa + 1];
            for(int nn = start; nn < end; nn++) {
                int j = neighbors[nn];
                if(kp[j].Mass <= 0 || kp[j].KernelRadius <= 0 || kp[j].Type == 5) continue;
                Vec3<double> dP = kp[j].Pos - pos_i;
                Vec3<double> dv = kp[j].Vel - vel_i;
                nearest_xyz(dP, -1);
                NGB_SHEARBOX_BOUNDARY_VELCORR_(pos_i, kp[j].Pos, dv, -1);
                Vec3<double> J_tmp = cross(dP, dv);
                if(kp[j].Type == 0) {
                    if(dot(J_tmp, Jgas_i) < 0) { kout[aa].MgasBulge_in_Kernel += 2 * kp[j].Mass; }
                }
                if(kp[j].Type == 4 ||
                   ((kp[j].Type == 2 || kp[j].Type == 3) && !comoving_on)) {
                    if(dot(J_tmp, Jstar_i) < 0) { kout[aa].MstarBulge_in_Kernel += 2 * kp[j].Mass; }
                }
            }
        });
    }

    /* Read-only kernel on P; out_host is post-scattered by host sink_environment.cc
     * into P/CellP, so invalidate so the next acquire re-syncs from host. */
    memcpy(out_host, d_out, num_active * sizeof(struct sink_env_second_gpu_out));

    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_Jstar);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_Jgas);
    Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(d_out);
    gpu_ngb_list_free(&gnl, NULL);
    gpu_particles_arena_invalidate();
}

#endif /* SINK_GRAVACCRETION == 0 */


GPU_ALL_SYNC_FUNC(sinkenv)

#else /* stubs when disabled */

void sink_environment_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                   int, int *, int, const double *, int,
                                   struct sink_env_gpu_out *) {}
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
void sink_environment_second_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                           int, int *, int, const double *,
                                           const MyFloat (*)[3], const MyFloat (*)[3],
                                           int, struct sink_env_second_gpu_out *) {}
#endif
void gizmo_gpu_sync_all_sinkenv(struct global_data_all_processes *p) { (void)p; }

#endif /* SINK_PARTICLES && OPENMP_GPU_OFFLOAD */
