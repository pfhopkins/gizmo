/* sinks/sink_env1_pair_kernel.h — SSOT pair kernel for sink_env1.
 *
 * Pre-runner-extraction step: consolidates the sink_env1 per-pair body into
 * a single KOKKOS_INLINE_FUNCTION called by BOTH the Mode A GPU lambda
 * (sinks/sink_environment_gpu.cc) AND the Mode B host walker
 * (sinks/sink_environment_mode_b.cc). Kills the SPIKE-duplicate trap
 * required by ~/.claude/memory/reference_neighbor_loop_contract.md before
 * extending Mode B to a second caller.
 *
 * The function takes references to the j-side particle/cell data (so both
 * GPU lambda — passing &kp[j], &kc[j] — and host helper — passing
 * P[j], &CellP[j] — can share the same body). The signature is host/device-
 * neutral; KOKKOS_INLINE_FUNCTION makes the function callable from both
 * the GPU kernel and from host code.
 *
 * The full NeighborLoopSpec runner (mesh/neighbor_loop_runner.h) abstracts
 * j-side construction into NeighborData; this header is the consolidated
 * physics body, callable from there once the SinkEnv1Spec lands.
 */
#ifndef SINK_ENV1_PAIR_KERNEL_H
#define SINK_ENV1_PAIR_KERNEL_H

#ifdef SINK_PARTICLES

#include "../declarations/allvars.h"
/* NOTE: ../mesh/kernel.h has no include guards; caller TUs must include it
 * BEFORE this header (both sink_environment_gpu.cc and sink_environment_mode_b.cc
 * already do). Functions used from kernel.h: kernel_main, NEAREST_XYZ macros,
 * nearest_xyz, NGB_SHEARBOX_BOUNDARY_VELCORR_, etc. */
#include "sinks_gpu_decls.h"           /* struct sink_env_gpu_out */
#include "sink_environment_mode_b.h"   /* struct sink_env1_query_t */

/* Make sure KOKKOS_INLINE_FUNCTION is defined; if Kokkos is unavailable
 * (host-only TUs), fall back to plain inline. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Cosmology + physical scalars the pair body needs. Captured once per
 * evaluator call from globals; passed by value into the kernel so no
 * implicit globals reach inside. */
struct sink_env1_scalars_t {
    double cf_atime;
    double cf_a2inv;
    double cf_a3inv;
    double G_val;
    double sink_radius_grav;
};

/* SSOT pair body for sink_env1.
 *
 * Inputs:
 *   q       — packed active sink data (pos, vel, id, h_search, ags_h, ...)
 *   kp_j    — j-side particle_data (passed by reference; works whether the
 *             caller has &P[j] (host) or &kp[j] inside the GPU kernel).
 *   kc_j    — j-side gas_cell_data pointer (nullptr for non-gas; otherwise
 *             pointer into CellP/UVM array).
 *   sc      — per-call scalars snapshot.
 *
 * Output:
 *   out     — per-active accumulator. Caller zero-initializes; this body
 *             ASSIGN_ADDs contributions from this pair.
 *
 * Caller is responsible for the per-active early-return (Mass<=0 || h<=0).
 *
 * J-side writes:
 *   - SINGLE_STAR_SINK_DYNAMICS active: kp_j.SwallowTime is atomically min'd.
 *     Caller MUST be on the rank that owns j (or using the writeback
 *     channel) to apply this safely. Inactive in fire_m11i / non-SSD builds.
 *   - SINK_GRAVCAPTURE_GAS: out.mass_to_swallow_edd accumulates; no j-write.
 */
KOKKOS_INLINE_FUNCTION
static void sink_env1_pair_kernel(const sink_env1_query_t& q,
                                  const struct particle_data& kp_j,
                                  const struct gas_cell_data* kc_j,
                                  const sink_env1_scalars_t& sc,
                                  struct sink_env_gpu_out& out)
{
    /* Self-skip / mass / type-5 filter. */
    if(kp_j.Mass <= 0 || kp_j.Type == 5 || kp_j.ID == q.id) return;

    const double h_i  = q.h_search;
    const double hinv  = 1.0 / h_i;
    const double hinv3 = hinv * hinv * hinv;
    const double ags_h_i = q.ags_h;

    /* dP, dv with periodic wrap. Sign convention: j - i. */
    Vec3<double> dP;
    dP[0] = (double)kp_j.Pos[0] - q.pos[0];
    dP[1] = (double)kp_j.Pos[1] - q.pos[1];
    dP[2] = (double)kp_j.Pos[2] - q.pos[2];
    Vec3<double> dv;
    dv[0] = (double)kp_j.Vel[0] - q.vel[0];
    dv[1] = (double)kp_j.Vel[1] - q.vel[1];
    dv[2] = (double)kp_j.Vel[2] - q.vel[2];
    nearest_xyz(dP, -1);
    NGB_SHEARBOX_BOUNDARY_VELCORR_(q.pos, kp_j.Pos, dv, -1);

    const double wt = (double)kp_j.Mass;

#ifdef SINK_REPOSITION_ON_POTMIN
    if(kp_j.Type != 0 && kp_j.Type != 5) {
        double rfac  = dP.norm_sq() * (10.0 / (h_i * h_i)
                                     + 0.1 / (sc.sink_radius_grav * sc.sink_radius_grav));
        double wtfac = wt / (1.0 + rfac);
        if((MyFloat)kp_j.Mass > out.DF_mmax_particles) out.DF_mmax_particles = (MyFloat)kp_j.Mass;
        for(int kv = 0; kv < 3; kv++) out.DF_mean_vel[kv] += wtfac * dv[kv];
        out.DF_rms_vel += wtfac;
        out.DF_rms_vel += wtfac;
        out.DF_rms_vel += wtfac;
    }
#endif

    if(kp_j.Type == 0) {
        out.Mgas_in_Kernel += wt;
        if(kc_j) { out.Sink_SurroudingGasInternalEnergy += wt * kc_j->InternalEnergy; }
        Vec3<double> J_gas = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jgas_in_Kernel[kv] += wt * J_gas[kv];
#if defined(SINK_OUTPUT_MOREINFO)
        if(kc_j) { out.Sfr_in_Kernel += kc_j->Sfr; }
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(int kv = 0; kv < 3; kv++) out.Sink_SurroundingGasVel[kv] += wt * dv[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
        for(int kv = 0; kv < 3; kv++) out.Sink_SurroundingGasCOM[kv] += wt * dP[kv];
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
        {
            double u_wb = dP.norm() / DMAX(h_i, (double)kp_j.KernelRadius);
            double wk_wb = 0, dwk_wb = 0;
            if(u_wb < 1) { kernel_main(u_wb, 1.0, 1.0, &wk_wb, &dwk_wb, -1); }
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
            double r2j = dP.norm_sq();
            double Lrj = dot(q.sink_angmom, dP);
            Vec3<double> Ang_pass = q.sink_angmom * r2j - dP * Lrj;
            for(int kv = 0; kv < 3; kv++)
                out.angmom_prepass_sum_for_passback[kv] += wk_wb * wt * Ang_pass[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
            out.kernel_norm_topass_in_swallowloop += wk_wb;
#endif
        }
#endif
#if (SINK_GRAVACCRETION == 8)
        if(kc_j) {
            double u_h = dP.norm() / h_i;
            double wk_h = 0, dwk_h = 0;
            if(u_h < 1) { kernel_main(u_h, hinv3, hinv3 * hinv, &wk_h, &dwk_h, -1); }
            double rj = u_h * h_i * sc.cf_atime;
            double csj = kc_j->effective_soundspeed();
            double vdotrj = -dot(dP, dv);
            double vr_mdot = 4 * M_PI * wt * (wk_h * sc.cf_a3inv) * rj * vdotrj;
            if(rj < sc.sink_radius_grav * sc.cf_atime) {
                double bondi_mdot = 4 * M_PI * sc.G_val * sc.G_val * q.mass * q.mass
                    / pow(csj * csj + dv.norm_sq() * sc.cf_a2inv, 1.5)
                    * wt * (wk_h * sc.cf_a3inv);
                vr_mdot = DMAX(vr_mdot, bondi_mdot);
                out.hubber_mdot_bondi_limiter += bondi_mdot;
            }
            out.hubber_mdot_vr_estimator    += vr_mdot;
            out.hubber_mdot_disk_estimator  += wt * wk_h * sqrt(rj) / (kc_j->Density * csj * csj);
        }
#endif
    } else if(kp_j.Type == 4 ||
              ((kp_j.Type == 2 || kp_j.Type == 3) && !All.ComovingIntegrationOn)) {
        out.Mstar_in_Kernel += wt;
        Vec3<double> J_star = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jstar_in_Kernel[kv] += wt * J_star[kv];
    } else {
        out.Malt_in_Kernel += wt;
        Vec3<double> J_alt = cross(dP, dv);
        for(int kv = 0; kv < 3; kv++) out.Jalt_in_Kernel[kv] += wt * J_alt[kv];
    }

    /* SINK_GRAVCAPTURE_GAS path. Boundedness check + SwallowID-based
     * mass-marked-swallow accumulation. Lifted from the GPU lambda
     * (sinks/sink_environment_gpu.cc:235-274). The j-side atomic write to
     * kp_j.SwallowTime is only present under SINGLE_STAR_SINK_DYNAMICS;
     * sink_environment_mode_b.cc refuses to compile under SSD (#error at
     * its top) so the host TU never instantiates the SSD branch. The GPU
     * lambda calls THIS header AND has a small SSD-only tail in its own
     * .cc for the atomic write. Once the SinkEnv1Spec runner with
     * SymmetricPairScatter writeback lands, even that tail consolidates. */
#ifdef SINK_GRAVCAPTURE_GAS
#ifdef GRAIN_FLUID
    if(kp_j.Mass > 0 && (kp_j.Type == 0 || ((1<<kp_j.Type) & GRAIN_PTYPES)))
#else
    if(kp_j.Mass > 0 && kp_j.Type == 0)
#endif
    {
        double dr_code = dP.norm();
        double vrel = dv.norm() / sc.cf_atime;
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE)
        if((1<<kp_j.Type) & GRAIN_PTYPES) {
            Vec3<double> B_vec; for(int kv = 0; kv < 3; kv++) B_vec[kv] = kp_j.Gas_B[kv];
            double vrel_dot = dot(dv, B_vec), bmag2 = B_vec.norm_sq();
            vrel = (fabs(vrel_dot) / sqrt(bmag2)) / sc.cf_atime;
        }
#endif
        struct gas_cell_data kc_j_local;
        if(kp_j.Type == 0 && kc_j) { kc_j_local = *kc_j; }
        else { /* zero-init */
            for(size_t b = 0; b < sizeof(kc_j_local); b++) {
                ((char*)&kc_j_local)[b] = 0;
            }
        }
        double vbound = sink_vesc_gpu(kp_j, kc_j_local, q.mass, dr_code, ags_h_i);
        if(vrel < vbound) {
            double local_sink_radius = sc.sink_radius_grav;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            local_sink_radius = q.sink_radius;
            double spec_mom = dot(dv, dP);
            double r2 = dP.norm_sq();
            spec_mom = r2*vrel*vrel - spec_mom*spec_mom*sc.cf_a2inv;
            if(spec_mom >= sc.G_val * (q.mass + (double)kp_j.Mass) * local_sink_radius) { return; }
#endif
            if(sink_check_boundedness_gpu(kp_j, kc_j_local, vrel, vbound, dr_code, local_sink_radius) == 1) {
                /* SINGLE_STAR_SINK_DYNAMICS atomic write to kp_j.SwallowTime
                 * lives in the GPU lambda's tail, NOT here — see the
                 * sink_environment_gpu.cc body. (Host helper TU is not built
                 * under SSD per the #error at top of sink_environment_mode_b.cc.) */
                if(kp_j.SwallowID < q.id) { out.mass_to_swallow_edd += (MyFloat)kp_j.Mass; }
            }
        }
    }
#endif  /* SINK_GRAVCAPTURE_GAS */
}

#endif /* SINK_PARTICLES */

#endif /* SINK_ENV1_PAIR_KERNEL_H */
