/* ghost_symlist_lifecycle.h — ghost-exchange + symmetric-CSR lifecycle helpers.
 *
 * These helpers absorb the ghost_exchange / symlist setup-and-teardown that
 * used to be copy-pasted at every call site of density(), hydro_gradient_calc(),
 * and hydro_force() (in accel.cc, init.cc, run.cc).  The three top-level
 * functions now include the lifecycle inline, so any caller gets the correct
 * sequencing automatically — you cannot forget to prep or clean up.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#ifndef GHOST_SYMLIST_LIFECYCLE_H
#define GHOST_SYMLIST_LIFECYCLE_H

#include "neighbor_list.h"
#include "sfc_tiles.h"
#include "ghost_exchange_spec.h"
#include "ghost_writeback.h"

extern void gpu_build_symmetric_neighbor_list(struct particle_data *P, int num_total,
    int *active_indices, int num_active, neighbor_list_t *out,
    double search_radius_factor);

/* Shared search/ghost-inflation factor (only grows when TURB_DIFF_DYNAMIC
   widens the dynamic-diffusion kernel). */
static inline double gizmo_ghost_safety_factor(void)
{
    double f = 1.0;
#ifdef TURB_DIFF_DYNAMIC
    f = DMAX(f, All.TurbDynamicDiffFac);
#endif
    return f;
}

/* Prologue for density(): drift all particles to current time and import
   ghost particles from neighbouring ranks.  Unconditional — both underlying
   routines early-out for NTask==1.

   NB: this is the ALL-TYPES variant. Many callers (sinks, AGS gravity,
   stellar feedback, DM dispersion, fuzzy DM, grain physics, etc.) need
   cross-type ghosts and route here. Pure-hydro callers (density iteration,
   gradient prep/refresh) should call gizmo_hydro_prep_ghosts / use
   ghost_exchange_hydro directly to avoid DM/star kernel pollution of tile
   hmax that inflates hydro search radii by 10-100x. */
static inline void gizmo_density_prep_ghosts(double safety)
{
    double t0 = my_second();
    move_particles(All.Ti_Current);
    double t_drift = timediff(t0, my_second());
    double t1 = my_second();
    ghost_exchange(safety);
    double t_ghost = timediff(t1, my_second());
    CPU_Step[CPU_DENSMISC] += t_drift;
    CPU_Step[CPU_DENSCOMM] += t_ghost;
}

/* Explicit-query ghost prep for caller-owned active lists.
 *
 * This preserves a single ghost_exchange implementation while letting physics
 * loops provide their exact source list and search semantics. Most loops are
 * symmetric; only use NGB_SEARCH_ONEWAY when the actual kernel predicate is
 * strictly r_ij < h_i (density-like calls). */
static inline void gizmo_explicit_query_prep_ghosts(const char *caller_name,
                                                    int search_mode,
                                                    unsigned int supply_type_mask,
                                                    const int *active_indices,
                                                    int num_active,
                                                    const double *active_radii,
                                                    double safety)
{
    double t0 = my_second();
    move_particles(All.Ti_Current);
    double t_drift = timediff(t0, my_second());

    int alloc_n = (num_active > 0 ? num_active : 1);
    double (*qpos)[3] = (double (*)[3]) malloc((size_t)alloc_n * sizeof(double[3]));
    double *qh = (double *) malloc((size_t)alloc_n * sizeof(double));
    for(int a = 0; a < num_active; a++) {
        int i = active_indices[a];
        qpos[a][0] = P[i].Pos[0];
        qpos[a][1] = P[i].Pos[1];
        qpos[a][2] = P[i].Pos[2];
        qh[a] = active_radii[a];
    }

    struct ghost_exchange_spec_t sp = {
        0u,
        supply_type_mask,
        search_mode,
        safety,
        caller_name,
        num_active,
        (const double (*)[3]) qpos,
        (const double *) qh
    };

    double t1 = my_second();
    ghost_exchange_run(&sp);
    double t_ghost = timediff(t1, my_second());
    free(qh);
    free(qpos);
    CPU_Step[CPU_DENSMISC] += t_drift;
    CPU_Step[CPU_DENSCOMM] += t_ghost;
}

/* Collective-safe explicit-query prep for phases that must always build a
 * caller-specific ghost pool. A local ghost-count guard is illegal here:
 * exporter-only ranks can have zero received ghosts while importer ranks have
 * nonzero ghosts, causing only some ranks to enter the collective exchange. */
static inline int gizmo_explicit_query_prep_ghosts_fresh(const char *caller_name,
                                                        int search_mode,
                                                        unsigned int supply_type_mask,
                                                        const int *active_indices,
                                                        int num_active,
                                                        const double *active_radii,
                                                        double safety)
{
    ghost_exchange_cleanup();
    gizmo_explicit_query_prep_ghosts(caller_name, search_mode, supply_type_mask,
                                     active_indices, num_active, active_radii, safety);
    return 1;
}

/* Hydro-density variant of gizmo_density_prep_ghosts: gas-only pool, gas-only
   active gate, ONE-WAY criterion (r_ij < h_i — correct for density; symmetric
   max(h_i, h_j) is wrong here and over-imports by orders of magnitude per
   Phase-0 waste-ratio diagnostic). Use from hydro density iteration only. */
static inline void gizmo_hydro_density_prep_ghosts(double safety)
{
    double t0 = my_second();
    move_particles(All.Ti_Current);
    double t_drift = timediff(t0, my_second());
    double t1 = my_second();
    ghost_exchange_hydro_oneway(safety);
    double t_ghost = timediff(t1, my_second());
    CPU_Step[CPU_DENSMISC] += t_drift;
    CPU_Step[CPU_DENSCOMM] += t_ghost;
}

/* Epilogue for density(): if h grew beyond the ghost pool during density
   iteration, re-exchange with the converged hmax so subsequent neighbour
   ops have a complete ghost set.  Internally guarded (NTask==1 returns 0).
   ALL-TYPES — must match the prep. AGS / DM-dispersion / fuzzy-DM density
   iterations call this and need cross-type ghosts. */
static inline void gizmo_density_redo_ghosts_if_needed(double safety)
{
    if(ghost_exchange_needs_redo()) {
        double t0 = my_second();
        ghost_exchange_cleanup();
        ghost_exchange(safety);
        CPU_Step[CPU_DENSCOMM] += timediff(t0, my_second());
    }
}

/* Hydro-typed companion to gizmo_density_redo_ghosts_if_needed: gas-only
   pool, one-way criterion. Use from hydro density() iteration so the redo
   matches the gas-only one-way prep (gizmo_hydro_density_prep_ghosts). */
static inline void gizmo_hydro_density_redo_ghosts_if_needed(double safety)
{
    if(ghost_exchange_needs_redo()) {
        double t0 = my_second();
        ghost_exchange_cleanup();
        ghost_exchange_hydro_oneway(safety);
        CPU_Step[CPU_DENSCOMM] += timediff(t0, my_second());
    }
}

/* Prologue for hydro_gradient_calc(): allocate the per-step active-index
   array, refresh ghost CellP with converged density/h, and build the
   symmetric CSR neighbor list used by gradients and hydro_force.

   Multi-rank tiny-N gate: when no rank has any active gas particle, the
   ghost refresh + symlist build are pure overhead (~1-2s/step on Vista
   2-rank). MPI_Allreduce the local active-gas count once; if global == 0,
   skip the collective ghost_exchange and the kernel-side symlist build,
   leaving an empty symlist that the gradient/hydro kernels fast-path on. */
static inline void gizmo_gradients_prep_symlist(double safety, double search_fac)
{
    /* active-index allocation (persists across gradients + hydro) */
    gizmo_sym_num_active = 0;
    for(int ii : ActiveParticleList) {if(P[ii].Type == 0 && P[ii].Mass > 0) gizmo_sym_num_active++;}
    gizmo_sym_active_indices = (int *) mymalloc("sym_active",
        (gizmo_sym_num_active > 0 ? gizmo_sym_num_active : 1) * sizeof(int));
    {int aa = 0; for(int ii : ActiveParticleList) {
        if(P[ii].Type == 0 && P[ii].Mass > 0) gizmo_sym_active_indices[aa++] = ii;}}

    /* Global active-gas count: cached for refresh's matching skip decision.
       Single int Allreduce is microseconds; not measurable next to the
       O(0.1-1s) ghost_exchange we may skip. */
    if(NTask > 1) {
        MPI_Allreduce(&gizmo_sym_num_active, &gizmo_sym_num_active_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    } else {
        gizmo_sym_num_active_global = gizmo_sym_num_active;
    }

    /* When global active-gas count is 0, skip the collective ghost refresh
       (otherwise pure overhead — ~1-2s/step on 2-rank tiny-N). The symlist
       build still runs with num_active=0; it allocates the size-1 backstop
       offsets/neighbors that downstream gradient_evaluate_gpu requires (a
       NULL backing buffer there causes a segfault even though pairs=0). */
    if(gizmo_sym_num_active_global > 0 && NTask > 1) {
        double t_refresh = my_second();
        ghost_exchange_cleanup();
        ghost_exchange_hydro(safety);
        if(ThisTask == 0) {PRINT_STATUS("Ghost refresh before gradients (%.4f s)", timediff(t_refresh, my_second()));}
    }

    /* build the symmetric CSR list (max(h_i,h_j) search radius) */
    double t_sym = my_second();
    gpu_build_symmetric_neighbor_list(P, NumPart, gizmo_sym_active_indices, gizmo_sym_num_active, &gizmo_sym_neighbor_list, search_fac);
    if(ThisTask == 0) {PRINT_STATUS("Symmetric neighbor list: %d active, %d pairs (%.4f s)",
                                    gizmo_sym_num_active, gizmo_sym_neighbor_list.total_pairs, timediff(t_sym, my_second()));}
}

/* Epilogue for hydro_gradient_calc(): refresh ghosts again so hydro_force
   sees updated CellP.Gradients on both sides of each pair, and rebuild the
   CSR (ghost indices may shift after re-exchange). Skipped when global
   active-gas was 0 in prep — no work in either direction so no need to
   rebuild ghosts or the symlist. The size-1 backstop symlist from prep
   stays valid for the (no-op) hydro_force kernel. */
static inline void gizmo_gradients_refresh_symlist(double safety, double search_fac)
{
    if(gizmo_sym_num_active_global == 0) return;
    if(NTask > 1) {
        double t_refresh = my_second();
        free_neighbor_list(&gizmo_sym_neighbor_list);
        ghost_exchange_cleanup();
        ghost_exchange_hydro(safety);
        gpu_build_symmetric_neighbor_list(P, NumPart, gizmo_sym_active_indices, gizmo_sym_num_active, &gizmo_sym_neighbor_list, search_fac);
        if(ThisTask == 0) {PRINT_STATUS("Ghost refresh + CSR rebuild after gradients: %d pairs (%.4f s)",
                                        gizmo_sym_neighbor_list.total_pairs, timediff(t_refresh, my_second()));}
    }
}

/* Epilogue for hydro_force(): free the symlist and remove ghost particles.
   Skipped when TRANSPORT_SUBCYCLE is active — in that case the symlist and
   ghosts stay alive for RT subcycle steps and are freed at the end of the
   subcycle loop in run.cc. */
static inline void gizmo_hydro_cleanup_symlist_and_ghosts(void)
{
#ifndef TRANSPORT_SUBCYCLE
    gizmo_sym_neighbor_list_free();
    ghost_exchange_cleanup();
#endif
}

/* init.cc only: density() was called for initial h-convergence, no subsequent
   gradients/hydro in this context, so just tear down ghosts. */
static inline void gizmo_density_init_cleanup(void)
{
    if(NTask > 1) {ghost_exchange_cleanup();}
}

#endif /* GHOST_SYMLIST_LIFECYCLE_H */
