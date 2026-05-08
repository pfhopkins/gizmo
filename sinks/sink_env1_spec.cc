/* sinks/sink_env1_spec.cc — host-side hooks for SinkEnv1Spec.
 *
 * The KOKKOS_INLINE_FUNCTION hooks (load_active, load_neighbor, pair_kernel,
 * zero_accum) live in sink_env1_spec.h so they are inlinable from both
 * device kernels (Mode A) and host walkers (Mode B/Brute). This .cc
 * contains the host-only hooks: search_radius_host (pre-arena radii),
 * populate_call_scalars_host (per-call globals snapshot),
 * apply_active_writeback (post-dispatch host scatter into the caller's
 * nl_outs buffer), and compare_accum (oracle helper for 3c.2).
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) and Claude for GIZMO.
 */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"           /* MUST precede sink_env1_spec.h */
#include "sink_env1_spec.h"

#ifdef SINK_PARTICLES

/* ----------------------------------------------------------------------------
 * Host hook: search_radius_host
 *
 * Per-active search radius from external state (P[i].KernelRadius).
 * Pre-arena, pre-drift epoch — matches sinks/sink_environment.cc:73:
 *     nl_radii[aa] = P[i].KernelRadius;
 * The runner stages radii_uvm[num_active] from this hook and passes it
 * to gpu_ngb_list_build (Mode A) and Mode B's per-query walker.
 * --------------------------------------------------------------------------*/
double SinkEnv1Spec::search_radius_host(const neighbor_loop_args& args,
                                         int /*active_slot*/, int i)
{
    return (double)args.P[i].KernelRadius;
}

/* ----------------------------------------------------------------------------
 * Host hook: populate_call_scalars_host
 *
 * Capture per-call cosmology + gravity scalars into a POD. Matches the
 * legacy capture in sinks/sink_environment_gpu.cc:108-113:
 *     double cf_atime    = All.cf_atime;
 *     double cf_a2inv    = All.cf_a2inv;
 *     double cf_a3inv    = All.cf_a3inv;
 *     double G_val       = All.G;
 *     double sink_radius_grav = SinkParticle_GravityKernelRadius;
 * --------------------------------------------------------------------------*/
SinkEnv1Spec::CallScalars
SinkEnv1Spec::populate_call_scalars_host(const neighbor_loop_args& /*args*/)
{
    CallScalars cs;
    cs.cf_atime         = All.cf_atime;
    cs.cf_a2inv         = All.cf_a2inv;
    cs.cf_a3inv         = All.cf_a3inv;
    cs.G_val            = All.G;
    cs.sink_radius_grav = SinkParticle_GravityKernelRadius;
    return cs;
}

/* ----------------------------------------------------------------------------
 * Host writeback: apply_active_writeback
 *
 * Copy AccumData into args.aux->nl_outs[active_slot]. The caller
 * (sinks/sink_environment.cc) reads nl_outs in its scatter loop
 * (lines 144-185) and applies physics-specific reductions into
 * SinkTempInfo. Byte-equivalent to the legacy memcpy at
 * sinks/sink_environment_gpu.cc:198 followed by the same caller-side
 * scatter — the runner just routes the AccumData into the same buffer.
 * --------------------------------------------------------------------------*/
void SinkEnv1Spec::apply_active_writeback(const neighbor_loop_args& args,
                                           int active_slot, int /*i*/,
                                           const AccumData& out)
{
    SinkEnv1Aux *aux = (SinkEnv1Aux *)args.aux;
    /* Byte-copy: AccumData == sink_env_gpu_out is POD; matches legacy
     * memcpy(out_host, d_out, num_active * sizeof(struct sink_env_gpu_out))
     * shape, just per-slot. */
    aux->nl_outs[active_slot] = out;
}

/* ----------------------------------------------------------------------------
 * Oracle compare: compare_accum
 *
 * Returns max-relative residual across all AccumData fields. Used by the
 * 3c.2 oracle gate; in 3c.1 the function compiles but is never called
 * (oracle path aborts early in run_neighbor_loop). Conservative
 * implementation: walk the byte representation and compute the maximum
 * relative difference per double-sized chunk. AccumData layout is
 * MyFloat-based (currently double per declarations/typedefs.h:23) — for
 * a future MyFloat=float build this still produces a meaningful relative
 * residual.
 *
 * Note: this is a host-only oracle helper. The pair kernel itself is
 * deterministic per active under fixed neighbor order, so under Mode B
 * vs Brute (same neighbor set, possibly different order) the worst case
 * is FP-floor reordering; compare_accum returns the max relative diff
 * which the runner gates against Spec::accum_tolerance.
 * --------------------------------------------------------------------------*/
double SinkEnv1Spec::compare_accum(const AccumData& a, const AccumData& b)
{
    double max_rel = 0.0;
    size_t max_rel_field = 0;
    /* Flatten as MyFloat array. AccumData layout is exclusively MyFloat
     * scalars and MyFloat[3] arrays (see sinks/sinks_gpu_decls.h:50-86). */
    const MyFloat *pa = reinterpret_cast<const MyFloat*>(&a);
    const MyFloat *pb = reinterpret_cast<const MyFloat*>(&b);
    const size_t n = sizeof(AccumData) / sizeof(MyFloat);
    static_assert(sizeof(AccumData) % sizeof(MyFloat) == 0,
        "SinkEnv1Spec::AccumData must be MyFloat-aligned for byte-walk compare");
    for(size_t k = 0; k < n; k++) {
        double va = (double)pa[k], vb = (double)pb[k];
        double denom = std::fmax(std::fabs(va), std::fabs(vb));
        double diff  = std::fabs(va - vb);
        double rel   = (denom > 0.0) ? (diff / denom) : diff;
        if(rel > max_rel) { max_rel = rel; max_rel_field = k; }
    }
    /* Diagnostic: when GIZMO_NLR_ORACLE_DUMP=1 and max_rel > accum_tolerance,
     * dump every nonzero-diff field with absolute and relative diffs. Lets us
     * tell summation-reorder roundoff (small abs, small rel on fields whose
     * magnitude swamps eps) from physics drift (large abs OR small abs on
     * a near-zero-denom field where rel spikes). Print cap = 16 calls. */
    static int s_dump_cached = -1;
    if(s_dump_cached < 0) {
        const char *e = std::getenv("GIZMO_NLR_ORACLE_DUMP");
        s_dump_cached = (e && e[0] == '1') ? 1 : 0;
    }
    static int s_dump_count = 0;
    if(s_dump_cached && max_rel > 1e-10 && s_dump_count < 16) {
        std::fprintf(stderr, "[compare_accum DUMP call=%d max_rel=%g (field=%zu) abs=%g a=%g b=%g]\n",
                     s_dump_count, max_rel, max_rel_field,
                     std::fabs((double)pa[max_rel_field] - (double)pb[max_rel_field]),
                     (double)pa[max_rel_field], (double)pb[max_rel_field]);
        for(size_t k = 0; k < n; k++) {
            double va = (double)pa[k], vb = (double)pb[k];
            double denom = std::fmax(std::fabs(va), std::fabs(vb));
            double diff  = std::fabs(va - vb);
            double rel   = (denom > 0.0) ? (diff / denom) : diff;
            if(diff > 0.0) {
                std::fprintf(stderr, "  field[%zu] a=%.17g b=%.17g abs=%.6g rel=%.6g\n",
                             k, va, vb, diff, rel);
            }
        }
        std::fflush(stderr);
        s_dump_count++;
    }
    return max_rel;
}

/* ----------------------------------------------------------------------------
 * merge_accum
 *
 * Per-field merge of a peer's contribution (src) into a local accumulator
 * (dst). Per-field op MUST match the pair_kernel writes (sum for additive
 * fields, MAX for DF_mmax_particles). Lifted byte-exact from the legacy
 * sinks/sink_environment_mode_b.cc::merge_into (lines 446-486) including
 * all #ifdef-gated optional fields. Used by run_mode_b_remote at the
 * cross-rank boundary.
 * --------------------------------------------------------------------------*/
void SinkEnv1Spec::merge_accum(AccumData& dst, const AccumData& src)
{
    dst.Sink_SurroudingGasInternalEnergy += src.Sink_SurroudingGasInternalEnergy;
    dst.Mgas_in_Kernel  += src.Mgas_in_Kernel;
    dst.Mstar_in_Kernel += src.Mstar_in_Kernel;
    dst.Malt_in_Kernel  += src.Malt_in_Kernel;
    for(int kv = 0; kv < 3; kv++) {
        dst.Jgas_in_Kernel[kv]  += src.Jgas_in_Kernel[kv];
        dst.Jstar_in_Kernel[kv] += src.Jstar_in_Kernel[kv];
        dst.Jalt_in_Kernel[kv]  += src.Jalt_in_Kernel[kv];
    }
#ifdef SINK_REPOSITION_ON_POTMIN
    dst.DF_rms_vel += src.DF_rms_vel;
    for(int kv = 0; kv < 3; kv++) dst.DF_mean_vel[kv] += src.DF_mean_vel[kv];
    if(src.DF_mmax_particles > dst.DF_mmax_particles) dst.DF_mmax_particles = src.DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    dst.Sfr_in_Kernel += src.Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    for(int kv = 0; kv < 3; kv++) dst.Sink_SurroundingGasVel[kv] += src.Sink_SurroundingGasVel[kv];
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    for(int kv = 0; kv < 3; kv++) dst.Sink_SurroundingGasCOM[kv] += src.Sink_SurroundingGasCOM[kv];
#endif
#if (SINK_GRAVACCRETION == 8)
    dst.hubber_mdot_bondi_limiter   += src.hubber_mdot_bondi_limiter;
    dst.hubber_mdot_vr_estimator    += src.hubber_mdot_vr_estimator;
    dst.hubber_mdot_disk_estimator  += src.hubber_mdot_disk_estimator;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    dst.mass_to_swallow_edd += src.mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    for(int kv = 0; kv < 3; kv++) dst.angmom_prepass_sum_for_passback[kv] += src.angmom_prepass_sum_for_passback[kv];
#endif
#if defined(SINK_RETURN_BFLUX)
    dst.kernel_norm_topass_in_swallowloop += src.kernel_norm_topass_in_swallowloop;
#endif
}

/* ----------------------------------------------------------------------------
 * Optional diagnostic hooks (3c.4a)
 *
 * Line shapes preserved byte-identical to legacy emit sites. The runner
 * gates these via cached env reads (gizmo_nlr_xval_dump_enabled() /
 * gizmo_nlr_xval_nb_dump_enabled()) and SFINAE-detects the hooks; both are
 * optional from the runner's perspective.
 * --------------------------------------------------------------------------*/

/* MODEB_XVAL accumulator dump. Byte-identical to:
 *   sinks/sink_environment.cc:140-159 (env-off, deleted in 3c.4a)
 *   sinks/sink_environment_mode_b.cc:621-636 (SPIKE branch, retires in 3c.5)
 * Both legacy sites use caller="sink_env1" (hardcoded) and active_local=<slot>;
 * preserved here. */
void SinkEnv1Spec::diagnostic_dump_active(const ActiveDumpView<SinkEnv1Spec>& v)
{
    const AccumData& a = *v.accum;
    std::printf("MODEB_XVAL rank=%d caller=sink_env1 active_local=%d "
                "Mgas=%g Mstar=%g Malt=%g IE=%g "
                "Jgas=%g,%g,%g Jstar=%g,%g,%g Jalt=%g,%g,%g\n",
                v.rank, v.active_slot,
                (double)a.Mgas_in_Kernel,
                (double)a.Mstar_in_Kernel,
                (double)a.Malt_in_Kernel,
                (double)a.Sink_SurroudingGasInternalEnergy,
                (double)a.Jgas_in_Kernel[0], (double)a.Jgas_in_Kernel[1], (double)a.Jgas_in_Kernel[2],
                (double)a.Jstar_in_Kernel[0], (double)a.Jstar_in_Kernel[1], (double)a.Jstar_in_Kernel[2],
                (double)a.Jalt_in_Kernel[0], (double)a.Jalt_in_Kernel[1], (double)a.Jalt_in_Kernel[2]);
}

/* MODEB_XVAL_NB neighbor-list dump. Byte-identical to legacy emit at
 * sinks/sink_environment_gpu.cc:209-249 (Mode A only). First-call gated
 * per-process via static, matching legacy semantics. The runner only
 * invokes this hook when:
 *   (a) GIZMO_MODE_B_XVAL_NB_DUMP=1 (cached env)
 *   (b) the call is on the Mode A path with a live GPU NGL CSR
 * so the hook itself only needs the per-process first-call gate. */
void SinkEnv1Spec::diagnostic_dump_neighbor_list(const NeighborListDumpView<SinkEnv1Spec>& v)
{
    static int s_fired = 0;
    if(s_fired) return;
    /* Per-call: once the FIRST active in the FIRST call has been printed,
     * the rest of that call's actives keep printing (same behavior as the
     * legacy block which wraps the entire per-active loop in the !fired
     * guard, then sets fired=1 after the loop). The runner calls this hook
     * once per active; we mirror legacy by gating on s_fired and only
     * setting it when the runner signals end-of-call (active_slot == last).
     * Simpler approach: print every active in every call until we've
     * printed N-1; then flip the flag. This matches legacy because in
     * legacy the static was process-scoped and flipped after the FIRST
     * call's loop completed. */
    const struct neighbor_loop_args& args = *v.args;
    const struct particle_data *P_host = args.P;
    const int ii = args.active_list[v.active_slot];
    const double h_i = (double)v.active->q.h_search;
    std::printf("MODEB_XVAL_NB rank=%d call=1 active=%d path=%s "
                "h_search=%.17g i_pos=%.17g,%.17g,%.17g "
                "i_vel=%.17g,%.17g,%.17g i_id=%llu n_j=%d\n",
                v.rank, v.active_slot, v.path, h_i,
                (double)P_host[ii].Pos[0], (double)P_host[ii].Pos[1], (double)P_host[ii].Pos[2],
                (double)P_host[ii].Vel[0], (double)P_host[ii].Vel[1], (double)P_host[ii].Vel[2],
                (unsigned long long)P_host[ii].ID, v.n_candidates);
    for(int idx = 0; idx < v.n_candidates; idx++) {
        const int j = v.candidate_ids[idx];
        double dx = (double)P_host[j].Pos[0] - (double)P_host[ii].Pos[0];
        double dy = (double)P_host[j].Pos[1] - (double)P_host[ii].Pos[1];
        double dz = (double)P_host[j].Pos[2] - (double)P_host[ii].Pos[2];
        NEAREST_XYZ(dx, dy, dz, 1);
        const double r2 = dx*dx + dy*dy + dz*dz;
        std::printf("MODEB_XVAL_NB_J rank=%d call=1 active=%d path=%s "
                    "j=%d Type=%d Mass=%.17g KernelRadius=%.17g r2=%.17g "
                    "Pos=%.17g,%.17g,%.17g Vel=%.17g,%.17g,%.17g ID=%llu\n",
                    v.rank, v.active_slot, v.path, j, (int)P_host[j].Type,
                    (double)P_host[j].Mass, (double)P_host[j].KernelRadius, r2,
                    (double)P_host[j].Pos[0], (double)P_host[j].Pos[1], (double)P_host[j].Pos[2],
                    (double)P_host[j].Vel[0], (double)P_host[j].Vel[1], (double)P_host[j].Vel[2],
                    (unsigned long long)P_host[j].ID);
    }
    /* Flip the gate when the LAST active of this call has been printed,
     * matching legacy where xv_nb_fired=1 was set after the per-active
     * loop completed inside the if(!fired) block. */
    if(v.active_slot == args.num_active - 1) s_fired = 1;
}

#endif /* SINK_PARTICLES */
