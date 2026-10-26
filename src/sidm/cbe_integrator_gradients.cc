/*
 * sidm/cbe_integrator_gradients.cc
 *
 * CBE pre-force gradient module. Out-of-line host hooks for
 * CBEGradSpec plus the toplevel driver CBEGrad_gradient_calc(). Mirrors
 * sidm/dm_fuzzy_loop.cc (DMGradSpec / DMGrad_gradient_calc) — same shape,
 * two passes orchestrated at the toplevel via Aux::loop_iteration.
 *
 * Persistent gradient storage lives on P[i].Gradients_CBE_basis_moments
 * (declarations/particle_data.h, gated on CBE_INTEGRATOR_WITHGRADIENTS).
 * Standard P[]-driven ghost import (gizmo_request_filtered_ghost_import_fresh)
 * carries the field naturally; there is no scratch UVM array and no custom
 * Alltoallv. The earlier scratch architecture (CbeGradScratch,
 * CbeGradScratchOwner, cbe_grad_import_ghosts, CbeGradientsSpec,
 * CbeBjLimiterSpec) is gone.
 *
 * Pass-0 writeback solves M^{-1} . B with the same Tikhonov-style
 * ill-conditioning guard used by hydro NV_T inversion
 * (hydro/density_loop.cc:514-525). Pass-1 writeback rescales the
 * already-persistent field in place by phi[m][k] — it MUST NOT re-zero or
 * re-compute the field.
 *
 * Primitive-gradient swap: the stored field now holds primitive gradients
 * (∂ρ, ∂v, ∂S) rather than moment gradients (∂m, ∂p, ∂T), packed in the
 * same slot layout. The writeback math is unchanged (M^{-1} · B, then
 * per-(m,k) rescale by phi) — only the upstream LSQ accumulator B is now
 * built from primitive deltas (see cbe_integrator_gradients.h pair
 * bodies). The field name `Gradients_CBE_basis_moments` is now STALE;
 * rename out of scope.
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

#include "../declarations/gpu_all_mirror.h"   /* MUST precede allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "cbe_integrator_gradients.h"
#include "../mesh/ghost_symlist_lifecycle.h"   /* gizmo_ghost_safety_factor */
#include "../system/gpu_particles_arena.h"
#include "../gravity/ags_force_loop.h"         /* AgsForceSpec (the corridor builds AGSForce's search), AgsForcePartition */

#if defined(CBE_INTEGRATOR_WITHGRADIENTS)

/* ============================================================================
 * Out-of-line CBEGradSpec hooks.
 * ========================================================================== */

/* search_radius — mirrors AgsForceSpec so the symmetric face set the pass-1
 * limiter walks is a SUPERSET of the AgsForce flux pair set. The pass-0 LSQ
 * pair body internally re-narrows acceptance to r < h_i. */
double CBEGradSpec::search_radius(const neighbor_loop_args& /*args*/,
                                   int /*active_slot*/, int i)
{
    return (double)P[i].AGS_KernelRadius;
}

CBEGradSpec::CallScalars
CBEGradSpec::populate_call_scalars(const neighbor_loop_args& args)
{
    CallScalars s;
    s.common         = nlr_common_scalars_from_all();
    s.loop_iteration = static_cast<const Aux*>(args.aux)->loop_iteration;
    return s;
}

/* apply_active_writeback — pass-gated:
 *   pass 0 (LSQ):       solve M^{-1} . B with Tikhonov-style diagonal-loading
 *                       guard; write P[i].Gradients_CBE_basis_moments.
 *   pass 1 (BJ-style):  rescale the already-persistent gradient by
 *                       phi[m][k] IN PLACE (no re-zero, no recompute). */
void CBEGradSpec::apply_active_writeback(const neighbor_loop_args& args,
                                          int active_slot, int i,
                                          const AccumData& accum)
{
    /* These should be impossible under the runner contract — args.aux is
     * set by CBEGrad_gradient_calc before every per-bm dispatch, and the
     * runner only invokes apply_active_writeback for indices it built into
     * the active list (which is bounds-clamped in toplevel). A failure
     * here is a runner / toplevel bug, not a recoverable physics state;
     * loudly abort rather than silently skipping a gradient. */
    const Aux *aux = static_cast<const Aux*>(args.aux);
    if(aux == nullptr) {
        fprintf(stderr,
                "[CBEGradSpec::apply_active_writeback] args.aux == nullptr "
                "(active_slot=%d, i=%d, NumPart=%d, rank=%d) — runner contract violation\n",
                active_slot, i, NumPart, ThisTask);
        endrun(91420);
        return;   /* soft stop: aux is NULL; returning avoids the aux->loop_iteration deref below */
    }
    if(i < 0 || i >= NumPart) {
        fprintf(stderr,
                "[CBEGradSpec::apply_active_writeback] i=%d out of [0,%d) "
                "(active_slot=%d, rank=%d) — runner handed out-of-range active index\n",
                i, NumPart, active_slot, ThisTask);
        endrun(91421);
        return;   /* soft stop: i is out of range; returning avoids the args.P[i] OOB reference below */
    }

    struct particle_data& Pi = args.P[i];

    if(aux->loop_iteration <= 0) {
        /* ------------------------------------------------------------------
         * Pass 0 — LSQ inversion with Tikhonov-style diagonal-loading guard.
         *
         * Same pattern as hydro NV_T inversion (hydro/density_loop.cc:514-525):
         *   threshold = 10 * CONDITION_NUMBER_DANGER
         *   while cond > threshold:
         *     M += (1.05 * trace/NUMDIMS / threshold) * I on diag
         *     cond_term *= 1.2
         * 50-iter hard cap; if it falls through, zero this row so commit #5's
         * flux body cannot ingest an arbitrarily-large gradient.
         * ------------------------------------------------------------------ */
        double M[3][3];
        double M_inv[3][3];
        for(int a = 0; a < 3; a++)
            for(int b = 0; b < 3; b++)
                M[a][b] = accum.M[a][b];

        const double cond_threshold = 10.0 * CONDITION_NUMBER_DANGER;
        const double trace_initial  = M[0][0] + M[1][1] + M[2][2];
        double       cond_term      = 1.05 * (trace_initial / 3.0) / cond_threshold;
        bool         singular       = false;

        for(int it = 0; it < 50; it++) {
            const double cond = matrix_invert_ndims(M, M_inv);
            if(isfinite(cond) && cond < cond_threshold) {
                bool any_bad = false;
                for(int a = 0; a < 3 && !any_bad; a++)
                    for(int b = 0; b < 3 && !any_bad; b++)
                        if(!isfinite(M_inv[a][b])) any_bad = true;
                if(!any_bad) break;
            }
            if(!isfinite(cond_term) || cond_term <= 0) cond_term = MIN_REAL_NUMBER;
            for(int a = 0; a < 3; a++) M[a][a] += cond_term;
            cond_term *= 1.2;
            if(it == 49) singular = true;
        }

        if(singular) {
            for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++)
                for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++)
                    for(int d = 0; d < 3; d++)
                        Pi.Gradients_CBE_basis_moments[m][k][d] = 0.0;
            return;
        }

        /* grad[m][k][d] = sum_e M_inv[d][e] * B[m][k][e] */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                for(int d = 0; d < 3; d++) {
                    double g = 0.0;
                    for(int e = 0; e < 3; e++) {
                        g += M_inv[d][e] * accum.B[m][k][e];
                    }
                    Pi.Gradients_CBE_basis_moments[m][k][d] = g;
                }
            }
        }
    } else {
        /* ------------------------------------------------------------------
         * Pass 1 — pairwise BJ-style limiter rescale, IN PLACE.
         *
         * Pi.Gradients_CBE_basis_moments was written by pass 0 (and
         * already imported on ghosts via standard P[] ghost transport when
         * the pass-1 runner call ran). We multiply each (m,k,d) component
         * by phi[m][k] clipped to [0, 1] with non-finite -> 0. Also sanitize
         * the persistent value (non-finite -> 0) so a bad pass-0 entry does
         * not propagate into commit #5's flux body.
         * ------------------------------------------------------------------ */
        for(int m = 0; m < CBE_INTEGRATOR_NBASIS; m++) {
            for(int k = 0; k < CBE_INTEGRATOR_NMOMENTS; k++) {
                double phi = accum.phi[m][k];
                if(!isfinite(phi)) phi = 0.0;
                else if(phi < 0.0) phi = 0.0;
                else if(phi > 1.0) phi = 1.0;
                for(int d = 0; d < 3; d++) {
                    const double g_old = Pi.Gradients_CBE_basis_moments[m][k][d];
                    double g_new = (isfinite(g_old) && isfinite(phi))
                                   ? (g_old * phi) : 0.0;
                    if(!isfinite(g_new)) g_new = 0.0;
                    Pi.Gradients_CBE_basis_moments[m][k][d] = g_new;
                }
            }
        }
    }
}


/* ============================================================================
 * THE CBE NEIGHBOUR CORRIDOR.
 *
 * CBEGrad's pass 0, its pass 1 and AGSForce run over the same actives (one
 * partition, ags_force_partition_actives), search the same neighbours (the same
 * type mask per subgroup, symmetric, at each active's AGS_KernelRadius) and are
 * separated only by owner-side writes: the pass-0 gradients, AGSForce's zeroed
 * outputs.  Alone, each would import its own ghosts and build its own list.  The
 * corridor makes ONE A/B decision for all three, and in Mode A ONE ghost import
 * and ONE list per subgroup; between the consumers a value refresh carries the
 * owner writes to the ghosts along the import's own provenance.  Mode B needs
 * neither: the consumers only share the decision.
 *
 * The lists answer AGSForce's search (AgsForceSpec: symmetric, its radius
 * policy, scale 1, AGS_KernelRadius); CBEGrad's is the same search over a mask
 * without gas, and the runner checks every consumer against what the list was
 * built for (struct nlr_external_csr).  A subgroup mask that holds gas does not
 * occur for CBE, whose neighbours are never gas: if one appears, the corridor
 * stands aside for the step and the three loops search as they do alone.
 *
 * Span: cbe_corridor_begin (core/accel.cc, before CBEGrad) partitions and
 * decides; the first cbe_corridor_refresh, made by CBEGrad AFTER its pass-0
 * snapshot of the previous gradients (pass 0 reads that snapshot on its
 * neighbours), imports and builds; later refreshes carry values, or rebuild
 * everything if the pool is no longer the one the lists index; AGSForce_calc
 * ends the span where it tore its own pool down before.
 * ========================================================================== */
#ifdef CBE_NEIGHBOUR_CORRIDOR
namespace {
struct CbeCorridor {
    AgsForcePartition                part;
    bool                             begun  = false;    /* part is this span's (cbe_corridor_begin, or the first consumer) */
    bool                             serves = false;    /* the corridor owns this step's CBE calls */
    NlrForceMode                     mode   = NlrForceMode::None;
    std::vector<std::vector<double>> radii;             /* per subgroup: each active's search radius */
    std::vector<neighbor_list_t>     lists;             /* per subgroup, host CSR in the arena */
    std::vector<nlr_external_csr>    views;
    bool                             built  = false;    /* lists (and at NTask>1 the import) in place */
    unsigned long long               pool_epoch = 0;
};
CbeCorridor  g_cbe_corridor;
int          g_cbe_corridor_no_rows  = 0;   /* row storage of a zero-row list */
const double g_cbe_corridor_no_radii = 0;   /* its row radii (set, never read) */

/* Free the lists, newest first (arena order). */
void cbe_corridor_release_lists(CbeCorridor &c)
{
    for(int s = (int)c.lists.size() - 1; s >= 0; s--) {free_neighbor_list(&c.lists[s]);}
    c.lists.clear();
    c.views.clear();
    c.built = false;
}

/* Import (NTask>1) with AGSForce's recipe, then one list per subgroup. */
void cbe_corridor_build(CbeCorridor &c)
{
    const int nsub = (int)c.part.masks.size();
    if(NTask > 1) {
        std::vector<int>    union_actives;
        std::vector<double> union_radii;
        unsigned int        mask_union = 0;
        for(int s = 0; s < nsub; s++) {
            mask_union |= c.part.masks[s];
            union_actives.insert(union_actives.end(), c.part.actives[s].begin(), c.part.actives[s].end());
            union_radii.insert(union_radii.end(), c.radii[s].begin(), c.radii[s].end());
        }
        nlr_import_union_ghosts<AgsForceSpec>(mask_union, union_actives, union_radii, gizmo_ghost_safety_factor());
    }
    gpu_particles_arena_set_site("cbe_corridor");
    gpu_particles_arena_acquire(NumPart, P, (All.TotN_gas > 0) ? CellP : nullptr);
    struct particle_data *P_shared = gpu_particles_arena_P();

    c.lists.assign(nsub, neighbor_list_t{NULL, NULL, 0, 0});
    c.views.assign(nsub, nlr_external_csr{});
    for(int s = 0; s < nsub; s++) {
        std::vector<int> &rows = c.part.actives[s];
        const int n = (int)rows.size();
        neighbor_list_t &list = c.lists[s];
        list.num_active = n;
        list.on_heap = 0;
        list.offsets = (int64_t *) mymalloc("cbe_corridor_offsets", (size_t)(n + 1) * sizeof(int64_t));
        list.offsets[0] = 0;
        gpu_neighbor_list_t gnl{};
        if(n > 0) {
            gpu_ngb_list_build(P_shared, NumPart, rows.data(), n, AgsForceSpec::search_mode,
                               (int)c.part.masks[s], &gnl, NULL, 1.0, c.radii[s].data(), NULL,
                               "cbe_corridor", nlr_spec_symmetric_j_radius_scale<AgsForceSpec>(),
                               AgsForceSpec::radius_policy);
            /* A build that could not get its memory has asked for the stop and
             * hands back no rows: leave this list empty. */
            if(gnl.d_active != NULL) {
                memcpy(list.offsets, gnl.offsets, (size_t)(n + 1) * sizeof(int64_t));
                list.total_pairs = gnl.total_pairs;
            } else {
                for(int k = 1; k <= n; k++) {list.offsets[k] = 0;}
            }
        }
        list.neighbors = (int *) mymalloc("cbe_corridor_neighbors",
                                          (size_t)(list.total_pairs > 0 ? list.total_pairs : 1) * sizeof(int));
        if(list.total_pairs > 0) {gpu_ngb_copy_neighbors_to_host(&gnl, list.neighbors);}
        gpu_ngb_list_free(&gnl);

        nlr_external_csr &v = c.views[s];
        v.active_indices        = (n > 0) ? rows.data() : &g_cbe_corridor_no_rows;
        v.num_active            = n;
        v.offsets               = list.offsets;
        v.neighbors             = list.neighbors;
        v.total_pairs           = list.total_pairs;
        v.owner_name            = "corridor:cbe";
        v.type_mask             = c.part.masks[s];
        v.search_mode           = AgsForceSpec::search_mode;
        v.radius_policy         = AgsForceSpec::radius_policy;
        v.neighbor_radius_scale = nlr_spec_symmetric_j_radius_scale<AgsForceSpec>();
        v.query_radii           = (n > 0) ? c.radii[s].data() : &g_cbe_corridor_no_radii;
        v.pool_epoch            = ghost_provenance_epoch();
    }
    c.pool_epoch = ghost_provenance_epoch();
    c.built      = true;
}
} /* namespace */

void cbe_corridor_begin(void)
{
    CbeCorridor &c = g_cbe_corridor;
    cbe_corridor_release_lists(c);   /* nothing to release unless a span was never ended */
    c.serves = false;
    c.mode   = NlrForceMode::None;
    ags_force_partition_actives(c.part);
    c.begun  = true;
    if(c.part.masks.empty()) {return;}
    for(unsigned int bm : c.part.masks) {if(bm & 1u) {return;}}   /* gas in a mask: stand aside */

    /* One A/B decision for all three consumers: AGSForce's own rule on the
     * partition's active count, which is what AGSForce would decide alone. */
    int local_n = 0;
    for(const std::vector<int> &rows : c.part.actives) {local_n += (int)rows.size();}
    const long long local_n_64 = local_n;
    long long sum_n = local_n; int max_n = local_n;   /* the sum over ranks can exceed 2^31 */
    if(NTask > 1) {
        MPI_Allreduce(&local_n_64, &sum_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
    /* If this selector ever returns Mode D, that answer needs no list and no
     * import, so the corridor stands aside and each loop decides alone. */
    const NlrMode mode = nlr_select_mode<AgsForceSpec>(sum_n, max_n);
    if(mode == NlrMode::D) {return;}
    c.mode   = (mode == NlrMode::B) ? NlrForceMode::B : NlrForceMode::A;
    c.serves = true;
    if(c.mode != NlrForceMode::A) {return;}

    /* Each row's search radius: the consumers' own (AgsForceSpec::search_radius,
     * which CBEGradSpec::search_radius equals), checked against them per row. */
    const neighbor_loop_args no_args = nlr_default_args();
    c.radii.assign(c.part.masks.size(), std::vector<double>());
    for(size_t s = 0; s < c.part.masks.size(); s++) {
        const std::vector<int> &rows = c.part.actives[s];
        c.radii[s].resize(rows.size());
        for(size_t k = 0; k < rows.size(); k++) {c.radii[s][k] = AgsForceSpec::search_radius(no_args, (int)k, rows[k]);}
    }
}

void cbe_corridor_refresh(void)
{
    CbeCorridor &c = g_cbe_corridor;
    if(!c.serves || c.mode != NlrForceMode::A) {return;}
    if(c.built) {
        if(NTask <= 1) {return;}   /* no ghosts: the consumers read the owners */
        /* Fast path: the live pool is the very import the lists index, so a
         * value-only refresh along its provenance keeps every slot (and list)
         * valid.  Anything else -- a pool torn down or replaced in between, a
         * refused refresh -- rebuilds everything (fail closed). */
        int same_pool = (ghost_pool_is_live() && ghost_provenance_epoch() == c.pool_epoch) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &same_pool, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);   /* the paths below are collective */
        if(same_pool && ghost_refresh_values() == GHOST_REFRESH_OK) {return;}
        cbe_corridor_release_lists(c);
        ghost_exchange_cleanup();
    }
    cbe_corridor_build(c);
}

void cbe_corridor_end(void)
{
    CbeCorridor &c = g_cbe_corridor;
    cbe_corridor_release_lists(c);
    /* The pool is torn down where AGSForce_calc tore down its own: whenever the
     * step had AGSForce actives, whether or not the corridor served. */
    if(NTask > 1 && !c.part.masks.empty()) {ghost_exchange_cleanup();}
    c.part.masks.clear();
    c.part.actives.clear();
    c.radii.clear();
    c.begun  = false;
    c.serves = false;
    c.mode   = NlrForceMode::None;
}

/* A consumer reached without cbe_corridor_begin builds the partition itself, and
 * the corridor stands aside: the loops then search as they do alone. */
AgsForcePartition &cbe_corridor_partition(void)
{
    CbeCorridor &c = g_cbe_corridor;
    if(!c.begun) {
        c.serves = false;
        c.mode   = NlrForceMode::None;
        ags_force_partition_actives(c.part);
        c.begun  = true;
    }
    return c.part;
}
bool               cbe_corridor_serves(void)    {return g_cbe_corridor.serves;}
NlrForceMode       cbe_corridor_mode(void)      {return g_cbe_corridor.mode;}
const nlr_external_csr *cbe_corridor_list(int subgroup)
{
    const CbeCorridor &c = g_cbe_corridor;
    if(!c.built || subgroup < 0 || subgroup >= (int)c.views.size()) {return nullptr;}
    return &c.views[subgroup];
}
#endif /* CBE_NEIGHBOUR_CORRIDOR */


/* ============================================================================
 * CBEGrad_gradient_calc — toplevel. Two passes through the runner, separated
 * so pass 1 sees on its neighbours the persistent
 * P[].Gradients_CBE_basis_moments written by pass 0.
 *
 * Active set: AGSForce_isactive (same predicate as the force consumer).
 * Subgroup partitioning: ags_gravity_kernel_shared_BITFLAG
 *   (ags_force_partition_actives; under the CBE corridor, its partition).
 *
 * Pre-zero: pass 0 only (LSQ accumulator scatter); pass 1 rescales in place.
 * Called from core/accel.cc in the pre-force phase, paralleling
 * DMGrad_gradient_calc.
 * ========================================================================== */
void CBEGrad_gradient_calc(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    const double t00 = my_second();  const double child0_span = CPU_ChildCharged;
    PRINT_STATUS(" ..calculating CBE basis-moment gradients\n");

    /* Local actives by shared AGS neighbour-type mask, in the same global mask
     * order on every rank (the per-mask runner calls are collective). Particles
     * with no shared kernel are dropped, matching DMGrad. */
#ifdef CBE_NEIGHBOUR_CORRIDOR
    AgsForcePartition &part = cbe_corridor_partition();
#else
    AgsForcePartition part;
    ags_force_partition_actives(part);
#endif
    if(part.masks.empty()) {
        { const double t_end = my_second();
          CPU_Step[CPU_AGSDENSMISC] += cpu_minus_children(timediff(t00, t_end), child0_span);
          cpu_chain_sync(t_end); }
        return;   /* no AGSForce-active particles anywhere */
    }
    /* (CPU accounting bucket: parallel DMGrad_gradient_calc — both paths
     * land in CPU_AGSDENSMISC.) */

    /* Two passes: 0 = raw LSQ -> writes P[].Gradients_CBE_basis_moments;
     *             1 = pairwise BJ-style limiter -> rescales the same field
     *                 in place. Each pass reads on its neighbours what the
     *                 owners hold when it starts: a fresh import per runner
     *                 call, or the corridor's refresh. */
    for(int pass = 0; pass < 2; pass++) {
        /* Pre-zero only on pass 0; pass 1 rescales the persistent field. */
        if(pass == 0) {
            for(const std::vector<int> &rows : part.actives) {
                for(int i : rows) {
                    /* Snapshot last step's final gradient as the 'prev' (stale) row
                     * before the pre-zero wipes it — the density-continuity matching
                     * cost reconstructs face densities from this prior-step gradient. */
                    std::memcpy(P[i].Gradients_CBE_basis_moments_prev,
                                P[i].Gradients_CBE_basis_moments,
                                sizeof(P[i].Gradients_CBE_basis_moments));
                    std::memset(P[i].Gradients_CBE_basis_moments, 0,
                                sizeof(P[i].Gradients_CBE_basis_moments));
                }
            }
        }
#ifdef CBE_NEIGHBOUR_CORRIDOR
        cbe_corridor_refresh();   /* pass 0: import + lists; pass 1: the pass-0 gradients */
#endif

        /* One runner call per mask in the global partition, ascending. A rank
         * with no local actives for a mask still enters with num_active=0 —
         * run_neighbor_loop is collective. */
        for(size_t s = 0; s < part.masks.size(); s++) {
            std::vector<int> &rows = part.actives[s];

            CBEGradSpec::Aux aux;
            aux.loop_iteration = pass;

            neighbor_loop_args args = nlr_default_args();
            args.active_list = rows.empty() ? nullptr : rows.data();
            args.num_active  = (int)rows.size();
            args.aux         = &aux;
            args.neighbor_type_mask_override = part.masks[s];
#ifdef CBE_NEIGHBOUR_CORRIDOR
            if(cbe_corridor_serves()) {
                args.dispatch_override = cbe_corridor_mode();
                args.external_csr      = cbe_corridor_list((int)s);
            }
#endif
            run_neighbor_loop<CBEGradSpec>(args);
        }
    }

    { const double t_end = my_second();
          CPU_Step[CPU_AGSDENSMISC] += cpu_minus_children(timediff(t00, t_end), child0_span);
          cpu_chain_sync(t_end); }
}

#endif /* CBE_INTEGRATOR_WITHGRADIENTS */
