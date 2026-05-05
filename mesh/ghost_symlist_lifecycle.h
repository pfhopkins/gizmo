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

/* Pure-hydro variant of gizmo_density_prep_ghosts: gas-only pool, gas-only
   active gate, gas-only effective h. Use from density iteration and any
   other context that only needs gas neighbors of gas particles. */
static inline void gizmo_hydro_prep_ghosts(double safety)
{
    double t0 = my_second();
    move_particles(All.Ti_Current);
    double t_drift = timediff(t0, my_second());
    double t1 = my_second();
    ghost_exchange_hydro(safety);
    double t_ghost = timediff(t1, my_second());
    CPU_Step[CPU_DENSMISC] += t_drift;
    CPU_Step[CPU_DENSCOMM] += t_ghost;
}

/* Epilogue for density(): if h grew beyond the ghost pool during density
   iteration, re-exchange with the converged hmax so subsequent neighbour
   ops have a complete ghost set.  Internally guarded (NTask==1 returns 0).
   ALL-TYPES — must match the prep. AGS / DM-dispersion / fuzzy-DM density
   iterations call this and need cross-type ghosts. A hydro-typed redo
   companion can be added when a hydro-typed prep replaces the all-types
   prep at the matching call sites. */
static inline void gizmo_density_redo_ghosts_if_needed(double safety)
{
    if(ghost_exchange_needs_redo()) {
        double t0 = my_second();
        ghost_exchange_cleanup();
        ghost_exchange(safety);
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
    if(gizmo_sym_num_active_global == 0) {
        /* Leave the symlist in zero-init state so callers fast-path on
           total_pairs==0; gradient/hydro kernels skip cleanly when num_active==0. */
        gizmo_sym_neighbor_list.offsets = NULL;
        gizmo_sym_neighbor_list.neighbors = NULL;
        gizmo_sym_neighbor_list.total_pairs = 0;
        return;
    }

    /* refresh ghosts so they carry converged Density/KernelRadius from their
       home rank (the density pass modified local values only). */
    if(NTask > 1) {
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
   active-gas was 0 in prep (matching skip — no ghosts were imported, no
   symlist was built, nothing to refresh or rebuild). */
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
