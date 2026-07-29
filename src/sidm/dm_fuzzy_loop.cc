/* sidm/dm_fuzzy_loop.cc — DMGradSpec host hooks + the toplevel
 * DMGrad_gradient_calc, for the runner-template port of the DM_FUZZY
 * higher-order density-gradient estimator (legacy DMGrad_evaluate /
 * dmgrad_evaluate_gpu).
 *
 * Device-callable hooks (pair_kernel, zero_accum, load_active, load_neighbor)
 * live in dm_fuzzy_loop.h so the runner instantiates them from GPU TUs. This
 * file owns: is_active, search_radius, populate_call_scalars,
 * apply_active_writeback, merge_accum, compare_accum, set_oracle_brute_pass,
 * the construct_dmgrad_gradient post-process helper, and DMGrad_gradient_calc.
 *
 * Replaces sidm/dm_fuzzy_gpu.cc (retired in this commit). The unrelated
 * host-only routines (do_dm_fuzzy_drift_kick, do_dm_fuzzy_initialization)
 * stay in sidm/dm_fuzzy.cc — runner-port checklist §5.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"            /* MUST precede dm_fuzzy_loop.h (no include guards) */
#include "dm_fuzzy_loop.h"

#ifdef DM_FUZZY

/* ============================================================================
 * Finite-aware per-field relative-difference accumulator for compare_accum.
 * A non-finite value on either side forces a huge residual so the oracle flags
 * it loudly. The finite test is inline — (x==x) rejects NaN, (x-x==0) rejects
 * Inf — to avoid the isfinite-macro-vs-std::isfinite ambiguity in GPU TUs
 * (runner checklist §2). Same helper as difffilter_loop.cc.
 * ========================================================================== */
namespace {
inline bool nlr_is_finite(double x) { return (x == x) && (x - x == 0.0); }

inline double nlr_rel_update(double max_rel, double a, double b) {
    if (!nlr_is_finite(a) || !nlr_is_finite(b)) return 1e30;
    const double denom = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    const double rel   = std::fabs(a - b) / denom;
    return (rel > max_rel) ? rel : max_rel;
}

/* construct_dmgrad_gradient — apply P[i].NV_T to a gradient vector in place.
 *
 * This is the legacy construct_gradient_DMGrad (gizmo-cpp/sidm/dm_fuzzy.cc:370):
 * the NV_T matrix-based gradient estimator, applied UNCONDITIONALLY. It must
 * NOT be replaced by the hydro construct_gradient (hydro/gradients.cc) — that
 * reads CellP[i].ConditionNumber/Density/NV_T (gas-cell state), which is the
 * documented garbage trap for non-gas particles (reference_legacy_definition_
 * cellp_mass.md). DMGrad runs on Type-1 DM; P[i].NV_T is the correct tensor.
 *
 * Templated on the row type so it accepts both Vec3<MyFloat>& (AGS_Gradients_*)
 * and a Mat3 row proxy (AGS_Gradients2_*[k]).
 * ========================================================================== */
template <typename RowT>
inline void construct_dmgrad_gradient(RowT&& grad, int i) {
    const double v0 = (double)grad[0], v1 = (double)grad[1], v2 = (double)grad[2];
    const Mat3<MyDouble>& M = P[i].NV_T;
    for (int k = 0; k < 3; k++) {
        grad[k] = (MyFloat)(M[k][0] * v0 + M[k][1] * v1 + M[k][2] * v2);
    }
}
} /* anonymous namespace */

/* ============================================================================
 * DMGradSpec host hooks.
 * ========================================================================== */

/* is_active — legacy CONDITIONFUNCTION_FOR_EVALUATION (DMGrad_evaluate):
 * ags_density_isactive(i). NOT a hand-rolled Type==1 check — under combined
 * options (ADAPTIVE_GRAVSOFT_FORALL, DM_SIDM) the predicate legitimately spans
 * more types. The bm!=0 filter is applied in the toplevel partition. */
bool DMGradSpec::is_active(int i)
{
    return ags_density_isactive(i) ? true : false;
}

/* search_radius — fixed per-active radius: P[i].AGS_KernelRadius (no per-iter
 * mutation; DMGrad is non-iterative). */
double DMGradSpec::search_radius(const neighbor_loop_args& args,
                                 int /*active_slot*/, int i)
{
    return (double)args.P[i].AGS_KernelRadius;
}

DMGradSpec::CallScalars
DMGradSpec::populate_call_scalars(const neighbor_loop_args& args)
{
    CallScalars s;
    s.common         = nlr_common_scalars_from_all();
    s.loop_iteration = static_cast<const Aux*>(args.aux)->loop_iteration;
    return s;
}

/* apply_active_writeback — scatter the per-active accumulator into P[i],
 * gated on the external gradient-pass index. The toplevel pre-zeroes the
 * target fields per pass, so += is correct (matches legacy out2particle_DMGrad
 * ASSIGN_ADD_PRESET semantics after the host preloop zero). */
void DMGradSpec::apply_active_writeback(const neighbor_loop_args& args,
                                        int /*active_slot*/, int i,
                                        const AccumData& accum)
{
    const int pass = static_cast<const Aux*>(args.aux)->loop_iteration;
    struct particle_data& Pi = args.P[i];

    if (pass <= 0) {
        for (int k = 0; k < 3; k++)
            Pi.AGS_Gradients_Density[k] += (MyFloat)accum.grad_rho[k];
#if (DM_FUZZY > 0)
        for (int k = 0; k < 3; k++) {
            Pi.AGS_Gradients_Psi_Re[k] += (MyFloat)accum.grad_psi_re[k];
            Pi.AGS_Gradients_Psi_Im[k] += (MyFloat)accum.grad_psi_im[k];
        }
#endif
    } else {
        for (int k = 0; k < 3; k++)
            for (int k2 = 0; k2 < 3; k2++)
                Pi.AGS_Gradients2_Density[k2][k] += (MyFloat)accum.grad2_rho[k2][k];
#if (DM_FUZZY > 0)
        for (int k = 0; k < 3; k++)
            for (int k2 = 0; k2 < 3; k2++) {
                Pi.AGS_Gradients2_Psi_Re[k2][k] += (MyFloat)accum.grad2_psi_re[k2][k];
                Pi.AGS_Gradients2_Psi_Im[k2][k] += (MyFloat)accum.grad2_psi_im[k2][k];
            }
#endif
    }
}

void DMGradSpec::merge_accum(AccumData& local, const AccumData& peer)
{
    for (int k = 0; k < 3; k++) {
        local.grad_rho[k] += peer.grad_rho[k];
        for (int k2 = 0; k2 < 3; k2++)
            local.grad2_rho[k2][k] += peer.grad2_rho[k2][k];
#if (DM_FUZZY > 0)
        local.grad_psi_re[k] += peer.grad_psi_re[k];
        local.grad_psi_im[k] += peer.grad_psi_im[k];
        for (int k2 = 0; k2 < 3; k2++) {
            local.grad2_psi_re[k2][k] += peer.grad2_psi_re[k2][k];
            local.grad2_psi_im[k2][k] += peer.grad2_psi_im[k2][k];
        }
#endif
    }
}

double DMGradSpec::compare_accum(const AccumData& local, const AccumData& oracle)
{
    double m = 0.0;
    for (int k = 0; k < 3; k++) {
        m = nlr_rel_update(m, local.grad_rho[k], oracle.grad_rho[k]);
        for (int k2 = 0; k2 < 3; k2++)
            m = nlr_rel_update(m, local.grad2_rho[k2][k], oracle.grad2_rho[k2][k]);
#if (DM_FUZZY > 0)
        m = nlr_rel_update(m, local.grad_psi_re[k], oracle.grad_psi_re[k]);
        m = nlr_rel_update(m, local.grad_psi_im[k], oracle.grad_psi_im[k]);
        for (int k2 = 0; k2 < 3; k2++) {
            m = nlr_rel_update(m, local.grad2_psi_re[k2][k], oracle.grad2_psi_re[k2][k]);
            m = nlr_rel_update(m, local.grad2_psi_im[k2][k], oracle.grad2_psi_im[k2][k]);
        }
#endif
    }
    return m;
}

/* No j-side writes → oracle brute pass needs no suppression. */
void DMGradSpec::set_oracle_brute_pass(DeviceContext& /*ctx*/, bool /*on*/) {}

/* ============================================================================
 * DMGrad_gradient_calc — toplevel. Two external gradient passes; within each,
 * one runner call per neighbor-type-bitmask group. Replaces the legacy
 * DMGrad_gradient_calc in sidm/dm_fuzzy.cc.
 * ========================================================================== */
void DMGrad_gradient_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    const double t00 = my_second();  const double child0_span = CPU_ChildCharged;
    PRINT_STATUS(" ..calculating higher-order gradients for DM density field\n");

    /* Initialize the numerical quantum potential at the very first step
     * (legacy sidm/dm_fuzzy.cc:145). */
    if (All.Time == All.TimeBegin) {
        for (int i : ActiveParticleList) { P[i].AGS_Numerical_QuantumPotential = 0; }
    }

    /* Partition local actives by shared AGS neighbor-type bitmask. bm==0
     * particles (not AGS-kernel-sharing) are dropped. */
    std::map<int, std::vector<int>> bm_groups;
    for (int i : ActiveParticleList) {
        if (ags_density_isactive(i)) {
            const int bm = ags_gravity_kernel_shared_BITFLAG(P[i].Type);
            if (bm > 0 && bm < 64) { bm_groups[bm].push_back(i); }
        }
    }

    /* Global bm-presence union (collective). Every rank must iterate the same
     * bm set in the same order so the per-bm run_neighbor_loop calls — which
     * are collective — stay synchronized. Mirrors ags_density_loop.cc. */
    uint64_t local_bm_presence = 0;
    for (const auto& kv : bm_groups) { local_bm_presence |= (1ULL << kv.first); }
    uint64_t global_bm_presence = local_bm_presence;
    if (NTask > 1) {
        MPI_Allreduce(&local_bm_presence, &global_bm_presence, 1,
                      MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
    }
    if (global_bm_presence == 0) {
        { const double t_end = my_second();
          CPU_Step[CPU_AGSDENSMISC] += cpu_minus_children(timediff(t00, t_end), child0_span);
          cpu_chain_sync(t_end); }
        return;   /* no DMGrad-active particles anywhere — collective short-circuit */
    }

    /* Local union active list — used for the per-pass pre-zero and the
     * post-process (once over the union, NOT per bm group). */
    std::vector<int> union_actives;
    for (const auto& kv : bm_groups) {
        union_actives.insert(union_actives.end(), kv.second.begin(), kv.second.end());
    }

    /* Two passes: 0 = gradient of AGS_Density (+psi); 1 = gradient-of-gradient
     * tensor. Each pass is a fresh set of runner calls → fresh ghost import,
     * so pass 1 sees pass-0's post-processed P[j].AGS_Gradients_* on ghosts
     * (fixes the legacy import-once staleness bug — design §2(b)). */
    for (int pass = 0; pass < 2; pass++) {
        /* Pre-zero the pass's target fields once over the local union. */
        for (int i : union_actives) {
            if (pass == 0) {
                for (int k = 0; k < 3; k++) P[i].AGS_Gradients_Density[k] = 0;
#if (DM_FUZZY > 0)
                for (int k = 0; k < 3; k++) {
                    P[i].AGS_Gradients_Psi_Re[k] = 0;
                    P[i].AGS_Gradients_Psi_Im[k] = 0;
                }
#endif
            } else {
                for (int k = 0; k < 3; k++)
                    for (int k2 = 0; k2 < 3; k2++) P[i].AGS_Gradients2_Density[k2][k] = 0;
#if (DM_FUZZY > 0)
                for (int k = 0; k < 3; k++)
                    for (int k2 = 0; k2 < 3; k2++) {
                        P[i].AGS_Gradients2_Psi_Re[k2][k] = 0;
                        P[i].AGS_Gradients2_Psi_Im[k2][k] = 0;
                    }
#endif
            }
        }

        /* One runner call per bm in the GLOBAL union, ascending order. A rank
         * with no local actives for a bm still enters with num_active=0 —
         * run_neighbor_loop is collective (unconditional MPI_Allreduce at
         * dispatch entry) and handles an empty local active set. */
        for (int bm = 1; bm < 64; bm++) {
            if (!(global_bm_presence & (1ULL << bm))) { continue; }
            auto it = bm_groups.find(bm);
            std::vector<int>* lst = (it != bm_groups.end()) ? &it->second : nullptr;

            DMGradSpec::Aux aux;
            aux.loop_iteration = pass;

            neighbor_loop_args args = nlr_default_args();
            args.active_list = (lst && !lst->empty()) ? lst->data() : nullptr;
            args.num_active  = lst ? (int)lst->size() : 0;
            args.aux         = &aux;
            args.neighbor_type_mask_override = (unsigned int)bm;
            run_neighbor_loop<DMGradSpec>(args);
        }

        /* Post-process once over the local union: NV_T matrix-gradient
         * estimator on the raw accumulated sums. */
        for (int i : union_actives) {
            if (pass == 0) {
                construct_dmgrad_gradient(P[i].AGS_Gradients_Density, i);
#if (DM_FUZZY > 0)
                construct_dmgrad_gradient(P[i].AGS_Gradients_Psi_Re, i);
                construct_dmgrad_gradient(P[i].AGS_Gradients_Psi_Im, i);
#endif
            } else {
                for (int k = 0; k < 3; k++) {
                    construct_dmgrad_gradient(P[i].AGS_Gradients2_Density[k], i);
#if (DM_FUZZY > 0)
                    construct_dmgrad_gradient(P[i].AGS_Gradients2_Psi_Re[k], i);
                    construct_dmgrad_gradient(P[i].AGS_Gradients2_Psi_Im[k], i);
#endif
                }
                /* Symmetrize the gradient-of-gradient tensors (legacy
                 * sidm/dm_fuzzy.cc:266-277, verbatim). */
                const int k0[3] = {0, 0, 1}, k1[3] = {1, 2, 2};
                for (int k = 0; k < 3; k++) {
                    double tmp = 0.5 * ((double)P[i].AGS_Gradients2_Density[k0[k]][k1[k]]
                                      + (double)P[i].AGS_Gradients2_Density[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Density[k0[k]][k1[k]] = (MyFloat)tmp;
                    P[i].AGS_Gradients2_Density[k1[k]][k0[k]] = (MyFloat)tmp;
#if (DM_FUZZY > 0)
                    tmp = 0.5 * ((double)P[i].AGS_Gradients2_Psi_Re[k0[k]][k1[k]]
                               + (double)P[i].AGS_Gradients2_Psi_Re[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Psi_Re[k0[k]][k1[k]] = (MyFloat)tmp;
                    P[i].AGS_Gradients2_Psi_Re[k1[k]][k0[k]] = (MyFloat)tmp;
                    tmp = 0.5 * ((double)P[i].AGS_Gradients2_Psi_Im[k0[k]][k1[k]]
                               + (double)P[i].AGS_Gradients2_Psi_Im[k1[k]][k0[k]]);
                    P[i].AGS_Gradients2_Psi_Im[k0[k]][k1[k]] = (MyFloat)tmp;
                    P[i].AGS_Gradients2_Psi_Im[k1[k]][k0[k]] = (MyFloat)tmp;
#endif
                }
            }
        }
    } /* end of pass loop */

    { const double t_end = my_second();
          CPU_Step[CPU_AGSDENSMISC] += cpu_minus_children(timediff(t00, t_end), child0_span);
          cpu_chain_sync(t_end); }
}

#endif /* DM_FUZZY */
