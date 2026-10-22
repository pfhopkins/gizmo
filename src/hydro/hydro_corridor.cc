/* hydro/hydro_corridor.cc — hydro corridor mode-decision + lifecycle.
 *
 * See hydro/hydro_corridor.h for the API contract and the corridor
 * sequencing rationale.
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include <climits>
#include "../mesh/neighbor_list.h"            /* gizmo_sym_* globals */
#include "../mesh/neighbor_loop_runner.h"     /* nlr_external_csr */
#include "../mesh/ghost_symlist_lifecycle.h"  /* gizmo_gradients_prep_symlist / refresh_symlist */
#include "../mesh/gpu_neighbor_list.h"        /* gizmo_mark_kernel_radius_dirty_range */
#include "../system/gpu_particles_arena.h"
#include "hydro_corridor.h"

/* The corridor takes the runner's Mode-B rule and the built-in thresholds of a
 * symmetric loop (nlr_default_modeb_thresholds), on the summed and per-rank-max
 * count of active GAS.  Density is one-way and decides for itself. */

/* Step-scoped mode state. Reset to UNSET by gizmo_hydro_corridor_end().
 * File-static so consumers must go through the accessor; no extern reads.
 *
 * Lifetime invariant: UNSET -> {MODE_A, MODE_B} via decide_mode at the top
 * of compute_hydro_densities_and_forces, then back to UNSET via end at the
 * bottom of the same function. Reading UNSET from a consumer is a bug
 * (consumer placed outside the corridor span). */
static GizmoHydroCorridorMode g_corridor_mode = GizmoHydroCorridorMode::UNSET;
/* Whether this step's shared list is built to outlive the corridor for the transport sub-steps. */
static int g_corridor_keep_for_substeps = 0;

/* Step-scoped MG decision (see gizmo_hydro_corridor_mg_step in hydro_corridor.h).
 * Set by decide_mode, cleared by end. */
static bool g_mg_step = false;
static bool g_mg_needs_all_gas = false;

/* Global active-gas counts — collective Allreduce so every rank computes
 * the same mode (corridor coherence requires this). Fills the summed and the
 * per-rank-max counts of active gas with mass (matching the runner's sum/max
 * dispatch inputs), and the summed count of all active gas cells, which is the
 * population All.TotN_gas counts. One pass over the active list, two Allreduces.
 * Caller must be inside MPI_COMM_WORLD scope. */
static void
global_active_gas_counts(long long *out_sum, int *out_max, long long *out_sum_all_gas)
{
    long long local_sums[2] = {0, 0};   /* active gas with mass, all active gas */
    for(int ii : ActiveParticleList) {
        if(P[ii].Type != 0) continue;
        local_sums[1]++;
        if(P[ii].Mass > 0) local_sums[0]++;
    }
    int local_max = (int) local_sums[0];
    long long sums[2] = {local_sums[0], local_sums[1]};
    if(NTask > 1) {
        MPI_Allreduce(local_sums, sums, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, out_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    } else {
        *out_max = local_max;
    }
    *out_sum = sums[0]; *out_sum_all_gas = sums[1];
}

/* The global MG solve runs when the active gas fraction reaches
 * ActiveFractionForMGSweep and some gas is active; otherwise the local CG
 * correction is used. Deciding it here, before density, is what lets the gradients
 * pass (which skips the CG correction on MG steps) and the hydro pass of the same
 * step agree on the choice. */
static void decide_mg_step(long long sum_active_gas, long long sum_active_all_gas)
{
    g_mg_step = false;
    g_mg_needs_all_gas = false;
#ifdef MHD_MODIFIED_GRADIENT
    double active_gas_fraction = (All.TotN_gas > 0) ? ((double)sum_active_all_gas / (double)All.TotN_gas) : 0.0;
    g_mg_step = (sum_active_gas > 0) && (active_gas_fraction >= All.ActiveFractionForMGSweep);
    g_mg_needs_all_gas = g_mg_step && (sum_active_all_gas < (long long)All.TotN_gas);
    All.Flag_SkipMGSolve = g_mg_step ? 0 : 1;
    if(!g_mg_step && ThisTask == 0) {PRINT_STATUS("Skipping MG global solve (active gas fraction %g < %g), using CG fallback for active cells", active_gas_fraction, All.ActiveFractionForMGSweep);}
#else
    (void)sum_active_gas; (void)sum_active_all_gas;
#endif
}

bool gizmo_hydro_corridor_mg_step(void) {return g_mg_step;}
bool gizmo_hydro_corridor_mg_needs_all_gas(void) {return g_mg_needs_all_gas;}

/* Bring every local gas cell to the current time: the MG graph spans all gas, and
 * inactive cells are otherwise drifted only lazily. Same bookkeeping as the full
 * drift (kernel-radius dirty marks, arena coherence). Must run with no ghosts
 * materialized, so that [0, N_gas) is exactly the owned gas block. */
static void drift_local_gas_to_current_time(void)
{
    std::vector<int> drifted((size_t) N_gas); int n_drifted = 0;
    drift_particles_batch(NULL, N_gas, All.Ti_Current, drifted.data(), &n_drifted);
    if(n_drifted > 0) {   /* a repeat import in the same step finds nothing behind */
        gizmo_mark_kernel_radius_dirty_indices(drifted.data(), n_drifted);
        gpu_particles_arena_refresh_from_host(NumPart, P, CellP, "mg_all_gas_drift");
    }
}

void gizmo_hydro_corridor_import_gas_ghosts(double safety)
{
    if(!g_mg_needs_all_gas) {ghost_exchange_hydro(safety); return;}
    drift_local_gas_to_current_time();
    std::vector<double> query_pos, query_h;
    query_pos.reserve(3 * (size_t)N_gas); query_h.reserve((size_t)N_gas);
    for(int i = 0; i < N_gas; i++) {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        query_pos.push_back(P[i].Pos[0]); query_pos.push_back(P[i].Pos[1]); query_pos.push_back(P[i].Pos[2]);
        query_h.push_back(P[i].KernelRadius);
    }
    /* Same supply and reach as ghost_exchange_hydro; only the query set widens. */
    struct ghost_exchange_spec_t sp = {GHOST_TYPE_0, GHOST_TYPE_0, NGB_SEARCH_SYMMETRIC, safety, "hydro_symmetric_all_gas",
                                       (int)query_h.size(), (const double (*)[3]) query_pos.data(), query_h.data(),
                                       MODE_B_RADIUS_LEGACY_KERNEL_ALLTYPES, 1.0};
    ghost_exchange_run(&sp);
}

void gizmo_hydro_corridor_decide_mode(void)
{
#ifdef TRANSPORT_SUBCYCLE
#ifdef MHD_MODIFIED_GRADIENT
    {long long sum_active_gas = 0, sum_active_all_gas = 0; int max_active_gas = 0;
     global_active_gas_counts(&sum_active_gas, &max_active_gas, &sum_active_all_gas); decide_mg_step(sum_active_gas, sum_active_all_gas);}
#endif
    /* Transport subcycling reuses the shared symmetric gas CSR
     * (gizmo_sym_neighbor_list) across every subcycle iteration
     * (transport_subcycle_exchange_fluxes consumes it), so the corridor MUST
     * build that CSR — i.e. Mode A — on every hydro step. Mode B is
     * request-driven with no CSR and is therefore incompatible with
     * TRANSPORT_SUBCYCLE; it is not offered in this build, and the mode-force
     * env vars are intentionally not consulted. Forcing Mode A here (rather
     * than at each consumer) is the single policy point; it also lets prep/
     * refresh_symlist skip their Mode-B dead-work path without a subcycle
     * special-case. */
    g_corridor_mode = GizmoHydroCorridorMode::MODE_A;
    return;
#endif
    /* Adaptive dispatch: the runner's Mode-B rule with a symmetric loop's
       defaults standing in where NeighborLoopModeBThreshold{Sum,Max} are unset.
       TRANSPORT_SUBCYCLE has already forced Mode A above. An MG step needs the
       complete gas graph, i.e. the Mode-A ghost pool and CSR. The summed count is
       64-bit for the MG fraction; any count past INT_MAX is far above every
       Mode-B threshold, so it enters the rule clamped. */
    long long sum_active_gas = 0, sum_active_all_gas = 0; int max_active_gas = 0;
    global_active_gas_counts(&sum_active_gas, &max_active_gas, &sum_active_all_gas);
    decide_mg_step(sum_active_gas, sum_active_all_gas);
    int default_sum, default_max;
    nlr_default_modeb_thresholds(MODE_B_SEARCH_SYMMETRIC, &default_sum, &default_max);
    const int sum_for_rule = (sum_active_gas > (long long)INT_MAX) ? INT_MAX : (int)sum_active_gas;
    bool select_mode_b = nlr_mode_b_rule(sum_for_rule, max_active_gas, default_sum, default_max) && !g_mg_step;
    GizmoHydroCorridorMode mode = select_mode_b ? GizmoHydroCorridorMode::MODE_B
                                                : GizmoHydroCorridorMode::MODE_A;
    g_corridor_mode = mode;

}

GizmoHydroCorridorMode gizmo_hydro_corridor_get_mode(void)
{
    return g_corridor_mode;
}

/* Corridor-owned view of the gizmo_sym_* shared CSR. Populated by
 * gizmo_hydro_corridor_begin() in Mode A at ANY rank count (at NTask>1
 * publication additionally requires a live ghost pool — the runner's
 * caller-owned-pool contract); republished by
 * gizmo_hydro_corridor_refresh_ghost_values() after each rebuild; cleared
 * by gizmo_hydro_corridor_end() at corridor exit. Does NOT own the
 * underlying buffer — gizmo_sym_neighbor_list is freed by
 * gizmo_hydro_cleanup_symlist_and_ghosts() after hydro_force(). */
static nlr_external_csr g_corridor_csr;
static bool             g_corridor_csr_valid = false;
/* Each row's search radius in the published list (the contract on struct
 * nlr_external_csr): the ghost safety factor times the row's KernelRadius,
 * frozen for the corridor span. */
static std::vector<double> g_corridor_query_radii;
/* Import epoch of the ghost pool the published CSR was built from (stamped at
 * begin and at every fallback republish). The value-refresh fast path in
 * gizmo_hydro_corridor_refresh_ghost_values runs ONLY when the CURRENT live
 * pool carries this epoch: a live pool from some other import could pass the
 * refresh count checks by coincidence while the CSR still indexes the old
 * slot layout — that case must take the full rebuild path. */
static unsigned long long g_corridor_pool_epoch = 0;

void gizmo_hydro_corridor_begin(void)
{
    if(g_corridor_mode == GizmoHydroCorridorMode::UNSET) return; /* defensive: decide_mode not run */

    /* Mode B: build ONLY the shared active-index list (the request-driven
     * walkers' row source for gradients/hydro_force) and dematerialize any
     * leftover ghosts — no import, no CSR, no arena work (the runner's Mode-B
     * hard-corridor counters enforce this downstream). Consumers no longer
     * build this list themselves. */
    const double gsl_safety = gizmo_ghost_safety_factor();
    if(g_corridor_mode == GizmoHydroCorridorMode::MODE_B) {
        gizmo_gradients_prep_symlist(gsl_safety, gsl_safety, true, 0, NULL);
        return;
    }

    /* Mode A: import the shared gas ghost pool (NTask>1) and build the
     * active list + gizmo_sym_* CSR once for the whole corridor span. With one
     * task there is no import to carry the all-gas drift, so it is done here. */
    if(g_mg_needs_all_gas && NTask == 1) {drift_local_gas_to_current_time();}
#ifdef TRANSPORT_SUBCYCLE
    g_corridor_keep_for_substeps = transport_subcycle_keep_wanted();
#endif
    gizmo_gradients_prep_symlist(gsl_safety, gsl_safety, false, g_corridor_keep_for_substeps, &gizmo_sym_neighbor_list_device);

    /* Publish the corridor's view ONLY when the runner's caller-owned-pool
     * contract can be satisfied: at NTask>1 that requires a LIVE ghost pool
     * (prep skips the import when no rank has active gas — then nothing is
     * published and consumers take their fallback, which handles empty).
     * The gizmo_sym_* globals live until gizmo_hydro_cleanup_symlist_and_
     * ghosts() after hydro_force(), spanning all three consumers. */
    if(NTask > 1 && !ghost_pool_is_live()) return;

    /* The search gizmo_gradients_prep_symlist built (gpu_build_symmetric_neighbor_list:
     * symmetric, gas only, the default radius policy, both radii scaled by the
     * ghost safety factor); each consumer is checked against it. */
    g_corridor_query_radii.resize((size_t)gizmo_sym_num_active);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(gizmo_sym_num_active >= 16)
#endif
    for(int aa = 0; aa < gizmo_sym_num_active; aa++) {
        g_corridor_query_radii[aa] = gsl_safety * (double)P[gizmo_sym_active_indices[aa]].KernelRadius;
    }
    g_corridor_csr.active_indices        = gizmo_sym_active_indices;
    g_corridor_csr.num_active            = gizmo_sym_num_active;
    g_corridor_csr.offsets               = gizmo_sym_neighbor_list.offsets;
    g_corridor_csr.neighbors             = gizmo_sym_neighbor_list.neighbors;
    g_corridor_csr.total_pairs           = gizmo_sym_neighbor_list.total_pairs;
    g_corridor_csr.owner_name            = "corridor:hydro";
    g_corridor_csr.type_mask             = 1u;
    g_corridor_csr.search_mode           = MODE_B_SEARCH_SYMMETRIC;
    g_corridor_csr.radius_policy         = MODE_B_RADIUS_DEFAULT;
    g_corridor_csr.neighbor_radius_scale = gsl_safety;
    g_corridor_csr.query_radii           = g_corridor_query_radii.data();
    g_corridor_csr.device                = &gizmo_sym_neighbor_list_device;
    g_corridor_csr_valid                 = true;
    g_corridor_pool_epoch                = ghost_provenance_epoch();
    g_corridor_csr.pool_epoch            = g_corridor_pool_epoch;

}

const nlr_external_csr * gizmo_hydro_corridor_external_csr(void)
{
    return g_corridor_csr_valid ? &g_corridor_csr : nullptr;
}

bool gizmo_hydro_corridor_csr_arrays(const int **rows, int *n_rows, const int64_t **offsets, const int **neighbors,
                                     const struct gpu_neighbor_list_t **device)
{
    if(!g_corridor_csr_valid) {return false;}
    *rows = g_corridor_csr.active_indices; *n_rows = g_corridor_csr.num_active;
    *offsets = g_corridor_csr.offsets; *neighbors = g_corridor_csr.neighbors;
    *device = g_corridor_csr.device;
    return true;
}

void gizmo_hydro_corridor_refresh_ghost_values(const char *stage)
{
    /* No-op unless the corridor owns a published Mode-A view this step —
     * Mode B / fallback paths keep their own refresh semantics. At NTask==1
     * there are no ghosts and owner values are read directly: nothing to
     * refresh. */
    if(!g_corridor_csr_valid) return;
    if(NTask <= 1) return;

    /* FAST PATH: value-only in-place refresh of the existing ghost pool along
     * the provenance recorded at import (owners re-pack current P/CellP;
     * receivers overwrite the existing slots). Slot identity is preserved by
     * construction, so the pool, the provenance maps, and the published CSR
     * all stay valid — no discovery, no rebuild, no republish. Permitted ONLY
     * when the current live pool is the very import the CSR was built from
     * (epoch match; see g_corridor_pool_epoch). Any other state — pool torn
     * down by an intervening loop's own ghost lifecycle, epoch mismatch, or
     * a refresh-guard failure — falls CLOSED to the full path below. */
    int same_pool = (ghost_pool_is_live() && ghost_provenance_epoch() == g_corridor_pool_epoch) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &same_pool, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);   /* both paths are collective */
    if(same_pool) {
        int refresh_rc = ghost_refresh_values();
        if(refresh_rc == GHOST_REFRESH_OK) {
            return;
        }
        /* fall through: fail-closed */
    }

    /* FULL PATH (also the fail-closed fallback): ghost teardown + re-import +
     * CSR rebuild, then REPUBLISH (the rebuild reallocates offsets/neighbors —
     * consumers re-fetch the view after every refresh; see hydro_corridor.h).
     * Import side-effects (the new ghost provenance the neighbour index keys
     * on) happen inside the re-import itself, so no side-effect list is
     * maintained here. */
    const double gsl_safety = gizmo_ghost_safety_factor();
    /* Only a list that was admitted to the heap at prep tries again (the same answer, once per step). */
    g_corridor_keep_for_substeps = gizmo_sym_neighbor_list.on_heap;
    gizmo_gradients_refresh_symlist(gsl_safety, gsl_safety, false, g_corridor_keep_for_substeps, &gizmo_sym_neighbor_list_device);

    /* Republish. The active set is frozen across the corridor span, so
     * active_indices/num_active are pointer-stable; only the CSR arrays
     * moved. Re-stamp the pool epoch: subsequent refreshes may fast-path
     * off this fresh import. */
    g_corridor_csr.offsets     = gizmo_sym_neighbor_list.offsets;
    g_corridor_csr.neighbors   = gizmo_sym_neighbor_list.neighbors;
    g_corridor_csr.total_pairs = gizmo_sym_neighbor_list.total_pairs;
    g_corridor_csr.device      = &gizmo_sym_neighbor_list_device;
    g_corridor_pool_epoch      = ghost_provenance_epoch();
    g_corridor_csr.pool_epoch  = g_corridor_pool_epoch;

}

void gizmo_hydro_corridor_end(void)
{
    g_corridor_mode      = GizmoHydroCorridorMode::UNSET;
    g_corridor_csr_valid = false;
    g_mg_step            = false;
    g_mg_needs_all_gas   = false;
    /* Note: do NOT free gizmo_sym_* here. The legacy
     * gizmo_hydro_cleanup_symlist_and_ghosts() (called after
     * hydro_force) still owns the free. Double-free would crash. */
}

