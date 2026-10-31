/* hydro/gradients_loop.cc — host hooks + toplevel for GradientsSpec.
 *
 * See hydro/gradients_loop.h for the Spec contract. This file owns:
 *   - populate_device_context (grad_iter ferry to device)
 *   - apply_active_writeback  (replays out2particle_GasGrad{,_iter})
 *   - merge_accum             (peer-rank reduction for Mode B remote)
 *   - GasGrad_isactive        (host-side narrow predicate, migrated from
 *                              the retired hydro/gradients.cc)
 *   - construct_gradient, local_slopelimiter
 *                             (host helpers, migrated from the retired
 *                              hydro/gradients.cc; used by the between-iter
 *                              MHD-CG block + the post-iter finalization
 *                              loop in hydro_gradient_calc below)
 *   - hydro_gradient_calc     (toplevel — replaces the legacy walker in
 *                              the retired hydro/gradients.cc)
 *
 * MHD-CG outer loop ordering: the ghost refresh runs AFTER the host
 * between-iter work and BEFORE the next iteration's kernel, so both sides
 * of every cross-rank pair see the same slope-limited fields. The
 * post-iter finalization loop is SERIAL (OMP parallelization requires a
 * per-i purity audit of the calculate_and_assign_* helpers first).
 *
 * Written by Philip F. Hopkins (phopkins@caltech.edu) for GIZMO. */

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "../declarations/gpu_all_mirror.h"  /* MUST precede allvars.h: installs device-pass `#define All AllDeviceMirror` redirect before cell_data.h is parsed */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../declarations/gpu_dispatch_templates.h"   /* gizmo_gpu_kernel_launch */
#include "../mesh/kernel.h"                   /* MUST precede gradients_loop.h
                                                * — kernel.h has no include guards */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/neighbor_list.h"            /* gizmo_sym_* — still maintained for hydro_force */
#include "hydro_corridor.h"                   /* mode + external_csr accessors */
#include "compute_finitevol_faces_functions.h"
#include "gradients_loop.h"
#ifdef TURB_DIFF_DYNAMIC
#include "../mesh/ghost_symlist_lifecycle.h"  /* gizmo_ghost_safety_factor */
#endif

#include "gradients_tail.h"                   /* the per-particle writeback and finalization bodies (host and device),
                                                * ASSIGN_ADD_PRESET / MAX_ADD / MIN_ADD */

/* ============================================================================
 * Host-side narrow predicate. Used by GasGrad_isactive_gpu() (via this same
 * source via gradient_functions.h's device wrapper) and by other host
 * callers; preserved from the retired hydro/gradients.cc:61.
 * ========================================================================== */
int GasGrad_isactive(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(pp[i].Type != 0) return 0;
    if(pp[i].Mass <= 0) return 0;
    if(cell[i].Density <= 0 || pp[i].KernelRadius <= 0) return 0;
#if defined(GALSF_SUBGRID_WINDS) && !defined(TURB_DIFF_DYNAMIC)
    if(cell[i].DelayTime > 0) return 0;
#endif
    return 1;
}

/* ============================================================================
 * Host helpers migrated from the retired hydro/gradients.cc. Used by the
 * between-iter MHD-CG block and the post-iter finalization loop below.
 * ========================================================================== */
void local_slopelimiter(double *grad, double valmax, double valmin, double alim, double h,
                        double shoot_tol, int pos_preserve, double d_max, double val_cen)
{
    gradients_slopelimiter(grad, valmax, valmin, alim, h, shoot_tol, pos_preserve, d_max, val_cen);
}

void construct_gradient(Vec3<MyDouble>& grad, int i)
{
    gradients_construct(grad, i, P, CellP);
}

#ifdef TURB_DIFF_DYNAMIC
/* ============================================================================
 * Search reach under TURB_DIFF_DYNAMIC. The pair body accumulates Velocity_hat
 * over every pair inside either particle's widened radius
 * (TurbDynamicDiffFac * h, gradient_functions.h), before its ordinary
 * max(h_i, h_j) test, so the search must reach that far on both sides. The
 * factor is never below 1: the ordinary gradients still need support to h. It
 * is the same factor the shared gas neighbour list is built with.
 * ========================================================================== */
double GradientsSpec::search_radius(const neighbor_loop_args& /*args*/, int /*active_slot*/, int i)
{
    return gizmo_ghost_safety_factor() * (double)P[i].KernelRadius;
}

double GradientsSpec::symmetric_neighbor_radius_scale()
{
    return gizmo_ghost_safety_factor();
}
#endif

/* ============================================================================
 * DeviceContext extension: grad_iter ferry. Reads aux->grad_iter set by
 * the toplevel before each runner call.
 * ========================================================================== */
void GradientsSpec::populate_device_context(const neighbor_loop_args& args,
                                             DeviceContext& ctx)
{
    Aux *aux = nlr_aux<GradientsSpec>(args);
    ctx.grad_iter = (aux != nullptr) ? aux->grad_iter : 0;
}

/* ============================================================================
 * apply_active_writeback — replay out2particle_GasGrad (grad_iter==0) or
 * out2particle_GasGrad_iter (grad_iter>0). Iter-0 path is line-for-line
 * out2particle_GasGrad (gradients.cc:267-456). Iter>0 path mirrors
 * out2particle_GasGrad_iter (gradients.cc:253-263) — pulls FaceDotB out of
 * accum, plus PhiGrad (MIDPOINT) reconstructed from accum.Gradients[k].Phi.
 *
 * `mode` parameter from the legacy functions is hard-coded to 0 here
 * because the runner already aggregates all neighbor contributions into
 * `accum` before calling us — ASSIGN_ADD_PRESET / MAX_ADD / MIN_ADD all
 * ignore mode (symmetric re-formulation), so behaviour is unchanged.
 * ========================================================================== */
void GradientsSpec::apply_active_writeback(const neighbor_loop_args& args,
                                            int /*active_slot*/, int i,
                                            const AccumData& accum_in)
{
    Aux *aux = nlr_aux<GradientsSpec>(args);
    const int grad_iter = (aux != nullptr) ? aux->grad_iter : 0;
    struct temporary_data_topass *passer = (aux != nullptr) ? aux->passer : nullptr;
    /* The legacy out2particle_GasGrad functions take a non-const out pointer
     * but only read it. const_cast to reuse legacy assignment macros. */
    struct GasGraddata_out_ *out = const_cast<struct GasGraddata_out_*>(&accum_in);

    /* ---------- iter>0 path: out2particle_GasGrad_iter ---------- */
#ifdef MHD_CONSTRAINED_GRADIENT
    if(grad_iter > 0) {
        ASSIGN_ADD_PRESET(passer[i].FaceDotB, out->FaceDotB, 0);
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
        /* Legacy adapter (gradients.cc:670-682) extracted PhiGrad components
         * from out->Gradients[k].Phi (k=0..2). out2particle_GasGrad_iter
         * (line 259) then did ASSIGN_ADD_PRESET on PhiGrad. Combine here. */
        for(int k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(passer[i].PhiGrad[k], out->Gradients[k].Phi, 0);
        }
#endif
        return;
    }
#else
    (void)grad_iter;
#endif

    /* ---------- iter==0 path: out2particle_GasGrad (line-for-line, hydro/gradients_tail.h) ---------- */
    gradients_writeback_cell(P, CellP, passer[i], i, out);
}


#ifdef MHD_CONSTRAINED_GRADIENT
/* ============================================================================
 * GradientsIterSpec host hooks (grad_iter>0, slim GasGraddata_out_iter_).
 * Each mirrors the corresponding GradientsSpec hook restricted to the fields
 * the iter>0 writeback consumes (FaceDotB + MIDPOINT PhiGrad), so behaviour is
 * bitwise-identical to the full path on the constrained-gradient passes.
 * ========================================================================== */
void GradientsIterSpec::populate_device_context(const neighbor_loop_args& args,
                                                 DeviceContext& ctx)
{
    Aux *aux = nlr_aux<GradientsIterSpec>(args);
    ctx.grad_iter = (aux != nullptr) ? aux->grad_iter : 0;
}

/* Slim replay of out2particle_GasGrad_iter: FaceDotB (+ MIDPOINT PhiGrad) from
 * the slim accum into passer[i]. Mirrors GradientsSpec::apply_active_writeback's
 * grad_iter>0 branch, reading the dedicated slim PhiGrad field in place of
 * out->Gradients[k].Phi. */
void GradientsIterSpec::apply_active_writeback(const neighbor_loop_args& args,
                                               int /*active_slot*/, int i,
                                               const AccumData& accum_in)
{
    Aux *aux = nlr_aux<GradientsIterSpec>(args);
    struct temporary_data_topass *passer = (aux != nullptr) ? aux->passer : nullptr;
    const struct GasGraddata_out_iter_ *out = &accum_in;   /* read-only: slim writeback never mutates accum */

    ASSIGN_ADD_PRESET(passer[i].FaceDotB, out->FaceDotB, 0);
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
    for(int k = 0; k < 3; k++) {
        ASSIGN_ADD_PRESET(passer[i].PhiGrad[k], out->PhiGrad[k], 0);
    }
#endif
}


#endif /* MHD_CONSTRAINED_GRADIENT */

/* ============================================================================
 * Toplevel: hydro_gradient_calc — replaces the legacy walker in the retired
 * hydro/gradients.cc. Structure:
 *
 *   0. TURB_DIFF_DYNAMIC: smooth Velocity_bar on its owners, so the refresh
 *      below ships the value the pair term reads on neighbours.
 *   1. Corridor ghost-value refresh (Mode A; topology was built once at
 *      gizmo_hydro_corridor_begin — this function builds nothing).
 *   2. Allocate GasGradDataPasser off the mymalloc LIFO stack (local
 *      std::vector — symlist stays at LIFO top, refresh is unobstructed).
 *   3. Host-side zero of CellP[i].Gradients.* / passer[i] / etc.
 *   4. Invalidate arena (host mutated state the next kernel reads).
 *   5. MHD-CG outer loop:
 *        a. Zero per-iter passer fields (FaceDotB / PhiGrad).
 *        b. Set aux.grad_iter; build args (corridor CSR in Mode A;
 *           corridor-built active list in Mode B).
 *        c. run_neighbor_loop<GradientsSpec>(args).
 *        d. Host between-iter MHD-CG block (slope-limit B / CG correction
 *           sweeps / MIDPOINT Phi-gradient build).
 *        e. corridor ghost-refresh BEFORE next iter (NTask>1 MHD_CG only;
 *           skipped after final iter).
 *   6. SERIAL post-iter finalization loop (OMP parallelization needs a
 *      per-i purity audit of calculate_and_assign_* first).
 *   7. No final refresh here — hydro_force refreshes at its own top.
 * ========================================================================== */
/* The per-active part of the gradients zero loop below (step 3) that writes the particle storage: the
 * same assignments on the host or, while the corridor keeps the particles on the device, in a kernel. */
KOKKOS_INLINE_FUNCTION
static void gradients_zero_active_fields(struct particle_data *Pp, struct gas_cell_data *Cp, int i, double smoothInv)
{
    int k2;
    (void)smoothInv; (void)Pp;
#ifdef HYDRO_SPH
#ifdef MAGNETIC
    Cp[i].DtB = {};
#endif
#ifdef DIVBCLEANING_DEDNER
    Cp[i].divB = 0;
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
    Cp[i].alpha_limiter = 0;
#endif
#endif
#ifdef TURB_DIFF_DYNAMIC
    Cp[i].Velocity_hat = Cp[i].Velocity_bar * smoothInv;   /* Velocity_bar already smoothed in (0) */
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    Pp[i].AGS_zeta = 0;
#endif
    Cp[i].Gradients.Density  = {};
    Cp[i].Gradients.Pressure = {};
    Cp[i].Gradients.Velocity = {};
#ifdef DOGRAD_INTERNAL_ENERGY
    Cp[i].Gradients.InternalEnergy = {};
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
    Cp[i].Gradients.ElectronNumberDensity = {};
    Cp[i].Gradients.ElectronTemperature   = {};
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
    Cp[i].Gradients.E_battery_T2 = {};
#endif
#ifdef COSMIC_RAY_FLUID
    for(k2 = 0; k2 < N_CR_PARTICLE_BINS; k2++) { Cp[i].Gradients.CosmicRayPressure[k2] = {}; }
#endif
#ifdef DOGRAD_SOUNDSPEED
    Cp[i].Gradients.SoundSpeed = {};
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
    Cp[i].Gradients.B = {};
#else
    Cp[i].Face_Area = {};
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
    Cp[i].Gradients.Phi = {};
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    for(k2 = 0; k2 < NUM_METAL_SPECIES; k2++) { Cp[i].Gradients.Metallicity[k2] = {}; }
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
    for(k2 = 0; k2 < N_RT_FREQ_BINS; k2++) { Cp[i].Gradients.Rad_E_gamma_ET[k2] = {}; }
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
    for(k2 = 0; k2 < N_RT_FREQ_BINS; k2++) {
        for(int k3 = 0; k3 < 3; k3++) { Cp[i].Gradients.Rad_E_gamma_Grad[k2][k3] = 0; }
        for(int k4 = 0; k4 < 3; k4++) for(int k5 = 0; k5 < 3; k5++) {
            Cp[i].Gradients.Rad_Flux_Grad[k2][k4][k5] = 0;
        }
    }
#endif
    (void)k2;
}

/* Step 3 with the particles on the device: the field zeroing as a kernel over the active list, and the
 * host-side passer zeroing driven by the kernel's gas test, so the host reads no particle. */
static void gradients_zero_actives_on_device(struct temporary_data_topass *passer_base, int n, double smoothInv)
{
    static int *d_list = NULL;
    static unsigned char *d_gas = NULL;
    static size_t cap = 0;
    if(n <= 0) {return;}
    if((size_t) n > cap) {
        if(d_list) {gizmo_gpu_free_device(d_list);}
        if(d_gas) {gizmo_gpu_free_device(d_gas);}
        cap = (size_t) n + (size_t) n / 4;
        d_list = (int *) gizmo_gpu_alloc_device(cap * sizeof(int), "gradients_zero_list");
        d_gas = (unsigned char *) gizmo_gpu_alloc_device(cap, "gradients_zero_gas");
        if(!d_list || !d_gas) {cap = 0;}
    }
    std::vector<unsigned char> gas((size_t) n, 0);
    if(cap == 0) {   /* no scratch: the host loop (correct, only slower) */
        for(int idx = 0; idx < n; idx++) {const int i = ActiveParticleList[idx]; gas[idx] = (P[i].Type == 0); if(gas[idx]) {gradients_zero_active_fields(P, CellP, i, smoothInv);}}
    } else {
        Kokkos::View<const int *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_list(ActiveParticleList.data(), (size_t) n);
        Kokkos::View<int *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_list(d_list, (size_t) n);
        Kokkos::deep_copy(dv_list, h_list);
        struct particle_data *Pp = P;
        struct gas_cell_data *Cp = CellP;
        int *dl = d_list;
        unsigned char *dg = d_gas;
        gizmo_gpu_kernel_launch("gradients_zero_actives", n, KOKKOS_LAMBDA(int idx) {
            const int i = dl[idx];
            const unsigned char g = (Pp[i].Type == 0) ? 1 : 0;
            dg[idx] = g;
            if(g) {gradients_zero_active_fields(Pp, Cp, i, smoothInv);}
        });
        Kokkos::View<unsigned char *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_gas(gas.data(), (size_t) n);
        Kokkos::View<const unsigned char *, GIZMO_KOKKOS_DEVICE_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dv_gas(d_gas, (size_t) n);
        Kokkos::deep_copy(h_gas, dv_gas);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n >= 16)
#endif
    for(int idx = 0; idx < n; idx++) {
        if(gas[idx]) {memset(&passer_base[ActiveParticleList[idx]], 0, sizeof(struct temporary_data_topass));}
    }
}

void hydro_gradient_calc(void)
{
    CPU_Step[CPU_DENSMISC] += measure_time();
    double t0 = my_second();
    /* The per-active host loops below each write only their own particle, so they thread over the
     * active list on any step with enough actives to pay for it. */
    const int n_active_list = (int) ActiveParticleList.size();

#ifdef TURB_DIFF_DYNAMIC
    /* (0) Velocity_bar takes its smoothing once, on its owner, before anything
     * reads it on a neighbour: the Velocity_hat pair term differences the
     * neighbour's Velocity_bar against the particle's own, so the ghost refresh
     * below must ship the scaled value. Done in either corridor mode, never
     * inside the refresh (whose fallback can rebuild and repeat). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_active_list >= 16)
#endif
    for(int idx = 0; idx < n_active_list; idx++) {
        const int i = ActiveParticleList[idx];
        if(P[i].Type == 0) {CellP[i].Velocity_bar *= All.TurbDynamicDiffSmoothing;}
    }
#endif

    /* (1) Corridor topology. The shared active list (and, in Mode A, the
     * ghost pool + CSR) was built ONCE at gizmo_hydro_corridor_begin(); this
     * function builds nothing. Mode A with a published view: refresh ghost
     * field values (owner-side hydro fields changed since the corridor
     * import: cellcorrections' Volume_1) — the refresh may
     * rebuild+republish the CSR view, so the view is re-fetched per
     * iteration below, never cached across a refresh. Mode B: nothing to do
     * here (request-driven walkers use the corridor-built active list; no
     * ghosts, no CSR). Mode A WITHOUT a published view and with active gas
     * is a corridor sequencing bug — fail loudly, never quietly rebuild. */
    const GizmoHydroCorridorMode   corridor_mode = gizmo_hydro_corridor_get_mode();
    const bool corridor_built_csr = (gizmo_hydro_corridor_external_csr() != nullptr);
    if(corridor_built_csr) {
        gizmo_hydro_corridor_refresh_ghost_values_unless_owner_clean("pre_gradients");
    } else if(corridor_mode == GizmoHydroCorridorMode::MODE_A
              && gizmo_sym_any_active_global) {
        printf("FATAL: hydro_gradient_calc in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
        fflush(stdout);
        endrun(7315);
    }

    /* (2) Per-active gradient scratch — persistent capacity-managed buffer
     * (grow-only). Transient scratch: every entry that is read is first zeroed
     * by the per-active memset below, and non-active entries are never accessed
     * (apply_active_writeback writes only passer[i] for the active home i; the
     * finalize loops read only over ActiveParticleList). Reusing one grown-
     * never-shrunk buffer is therefore bitwise-identical to a fresh per-call
     * N_gas-sized vector, and drops the per-call O(N_gas) zero-init. */
    static std::vector<struct temporary_data_topass> gas_grad_passer_storage;
    if((size_t)N_gas > gas_grad_passer_storage.size()) {
        gas_grad_passer_storage.resize(N_gas > 0 ? (size_t)N_gas : 1);
    }
    struct temporary_data_topass *passer_base = gas_grad_passer_storage.data();

    /* (3) Zero CellP[i].Gradients.*, P[i].AGS_zeta, passer[i], etc.
     * for every active gas particle. Mirrors hydro/gradients.cc:552-620
     * verbatim (gradients_zero_active_fields). */
#ifdef TURB_DIFF_DYNAMIC
    double smoothInv = 1.0 / All.TurbDynamicDiffSmoothing;
#else
    double smoothInv = 0;
#endif
    if(gizmo_particles_owned_kept_on_device()) {
        gradients_zero_actives_on_device(passer_base, n_active_list, smoothInv);
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_active_list >= 16)
#endif
    for(int idx = 0; idx < n_active_list; idx++) {
        const int i = ActiveParticleList[idx];
        if(P[i].Type != 0) continue;
        memset(&passer_base[i], 0, sizeof(struct temporary_data_topass));
        gradients_zero_active_fields(P, CellP, i, smoothInv);
    }
    }

    /* (4) Arena invalidation: host zeroed CellP[i].Gradients.* / AGS_zeta;
     * arena (if previously populated by density_gpu) is stale. */
    gpu_particles_arena_invalidate();

    /* (5) MHD-CG outer loop. */
    GradientsAux aux;
    aux.passer    = passer_base;
    aux.grad_iter = 0;

    /* Active-list source: both modes use the corridor-built broad list —
     * Mode A via the published CSR view, Mode B via gizmo_sym_active_indices
     * directly. Using gizmo_sym_* (rather than a fresh nlr_build_active_list)
     * matches the legacy GPU walker exactly AND keeps the LIFO clean: an
     * in-iter nlr_build_active_list would mymalloc on top of the sym_* arena,
     * blocking the between-iter corridor refresh from freeing
     * sym_neighbor_list — abort 814. */
    const bool nothing_to_do =
        (!corridor_built_csr) && (!gizmo_sym_any_active_global);

    for(int grad_iter = 0; (!nothing_to_do) && grad_iter < NUMBER_OF_GRADIENT_ITERATIONS; grad_iter++) {
        aux.grad_iter = grad_iter;

        /* (5a) Zero per-iter passer fields (mirrors gradients.cc:637-646). */
        if(grad_iter > 0) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n_active_list >= 16)
#endif
            for(int idx = 0; idx < n_active_list; idx++) {
                const int i = ActiveParticleList[idx];
                if(P[i].Type != 0) continue;
#ifdef MHD_CONSTRAINED_GRADIENT
                passer_base[i].FaceDotB = 0;
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
                passer_base[i].PhiGrad = {};
#endif
#endif
            }
        }

        /* (5b) Build args. Mode A consumes the corridor's external CSR with
         * its broad row list; the view is RE-FETCHED here every iteration
         * because a corridor ghost refresh (pre-gradients or between MHD-CG
         * iters) may rebuild+republish it — a cached pointer would go stale.
         * Mode B uses the corridor-built gizmo_sym_active_indices directly. */
        neighbor_loop_args args = nlr_default_args();
        args.aux = &aux;

        const nlr_external_csr *corridor_csr = gizmo_hydro_corridor_external_csr();
        if(corridor_csr != nullptr) {
            args.active_list       = corridor_csr->active_indices;
            args.num_active        = corridor_csr->num_active;
            args.external_csr      = corridor_csr;
            args.dispatch_override = NlrForceMode::A;
        } else {
            args.active_list = gizmo_sym_active_indices;
            args.num_active  = gizmo_sym_num_active;
            if(corridor_mode == GizmoHydroCorridorMode::MODE_A) {
                args.dispatch_override = NlrForceMode::A;
            } else if(corridor_mode == GizmoHydroCorridorMode::MODE_B) {
                args.dispatch_override = NlrForceMode::B;
            }
        }

        /* (5c) Kernel dispatch. grad_iter==0 runs the full GradientsSpec;
         * grad_iter>0 (constrained-gradient iterations) runs the slim
         * GradientsIterSpec, which carries only the FaceDotB(+PhiGrad) the
         * iter>0 writeback reads — dropping the dead full-struct payload. The
         * runner is pass-agnostic; the Spec type is the only thing that changes. */
#ifdef MHD_CONSTRAINED_GRADIENT
        if(grad_iter > 0) {
            run_neighbor_loop<GradientsIterSpec>(args);
        } else
#endif
        {
            run_neighbor_loop<GradientsSpec>(args);
        }

        /* (5d) Host between-iter MHD-CG block — verbatim from
         * hydro/gradients.cc:689-829. */
#ifdef MHD_CONSTRAINED_GRADIENT
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) if(n_active_list >= 16)
#endif
        for(int idx = 0; idx < n_active_list; idx++) {
            const int i = ActiveParticleList[idx];
            if(P[i].Type != 0) continue;
            int k, k1;
            CellP[i].FlagForConstrainedGradients = 1;
            for(k = 0; k < 3; k++) {
                for(k1 = 0; k1 < 3; k1++) {
                    CellP[i].Gradients.B[k][k1] = passer_base[i].BGrad[k][k1];
                }
            }
            for(k = 0; k < 3; k++) { construct_gradient(CellP[i].Gradients.B[k], i); }
            double v_tmp = P[i].Mass / CellP[i].Density;
            double tmp_d = sqrt(1.0e-37 + (2. * All.cf_atime / CellP[i].Pressure * v_tmp * v_tmp)
                                + CellP[i].BPred.norm_sq());
            double tmp   = 3.0e3 * fabs(CellP[i].divB) * P[i].KernelRadius / tmp_d;
            double alim  = 1. + DMIN(1., tmp * tmp);
#if (MHD_CONSTRAINED_GRADIENT <= 1)
            {
                double dbmax = 0, dbgrad = 0;
                double dh    = 0.25 * P[i].KernelRadius;
                for(k = 0; k < 3; k++) {
                    double b0 = CellP[i].Bfield_component(k);
                    double dd = 2. * fabs(b0)
                                * DMIN(fabs(passer_base[i].Minima.B[k]),
                                       fabs(passer_base[i].Maxima.B[k]));
                    dbmax = DMIN(fabs(dbmax + dd), fabs(dbmax - dd));
                    for(k1 = 0; k1 < 3; k1++) {
                        dbgrad += 2. * dh * fabs(b0 * CellP[i].Gradients.B[k][k1]);
                    }
                }
                dbmax = (dbgrad > 0) ? dbmax / dbgrad : MAX_REAL_NUMBER; /* no field-gradient term (e.g. B=0): only the ordinary slope limit below applies */
                for(k1 = 0; k1 < 3; k1++) {
                    double d_abs = CellP[i].Gradients.B[k1].norm_sq();
                    if(d_abs > 0) {
                        double cfac = 1 / (0.25 * P[i].KernelRadius * sqrt(d_abs));
                        cfac *= DMIN(fabs(passer_base[i].Maxima.B[k1]),
                                     fabs(passer_base[i].Minima.B[k1]));
                        double c_eff = DMIN(cfac, DMAX(cfac / alim, dbmax));
                        if(c_eff < 1) { CellP[i].Gradients.B[k1] *= c_eff; }
                    } else {
                        CellP[i].Gradients.B[k1] = {};
                    }
                }
            }
#endif
            /* Particle area closure check */
            double area = fabs(CellP[i].Face_Area[0])
                         + fabs(CellP[i].Face_Area[1])
                         + fabs(CellP[i].Face_Area[2]);
            area /= Get_Particle_Expected_Area(P[i].KernelRadius);
            if(area > 0.5)                              { CellP[i].FlagForConstrainedGradients = 0; }
            if(CellP[i].ConditionNumber > 1000.)        { CellP[i].FlagForConstrainedGradients = 0; }
            if(SHOULD_I_USE_SPH_GRADIENTS(CellP[i].ConditionNumber))
                                                        { CellP[i].FlagForConstrainedGradients = 0; }

            if(CellP[i].FlagForConstrainedGradients == 1) {
                int do_cg_correction = 1;
#ifdef MHD_MODIFIED_GRADIENT
                do_cg_correction = All.Flag_SkipMGSolve;
#endif
                if(do_cg_correction) {
                    double GB0[3][3];
                    double fsum = 0.0, dmag = 0.0;
                    double h_eff = P[i].Get_Particle_Size();
                    for(k = 0; k < 3; k++) {
                        double grad_limiter_mag = CellP[i].Bfield_component(k) / h_eff;
                        dmag += grad_limiter_mag * grad_limiter_mag;
                        for(k1 = 0; k1 < 3; k1++) {
                            GB0[k][k1] = CellP[i].Gradients.B[k][k1];
                            dmag += GB0[k][k1] * GB0[k][k1];
                            fsum += passer_base[i].FaceCrossX[k][k1] * passer_base[i].FaceCrossX[k][k1];
                        }
                    }
                    if((fsum <= 0) || (dmag <= 0)) {
                        CellP[i].FlagForConstrainedGradients = 0;
                    } else {
                        dmag = 2.0 * sqrt(dmag);
                        fsum = -1 / fsum;
                        for(int j_gloop = 0; j_gloop < 5; j_gloop++) {
                            double asum = passer_base[i].FaceDotB;
                            for(k = 0; k < 3; k++) {
                                for(k1 = 0; k1 < 3; k1++) {
                                    asum += CellP[i].Gradients.B[k][k1] * passer_base[i].FaceCrossX[k][k1];
                                }
                            }
                            double prefac = 1.0 * asum * fsum;
                            double ecorr[3][3];
                            double cmag = 0;
                            for(k = 0; k < 3; k++) {
                                for(k1 = 0; k1 < 3; k1++) {
                                    ecorr[k][k1] = prefac * passer_base[i].FaceCrossX[k][k1];
                                    double grad_limiter_mag = (CellP[i].Gradients.B[k][k1] + ecorr[k][k1]) - GB0[k][k1];
                                    cmag += grad_limiter_mag * grad_limiter_mag;
                                }
                            }
                            cmag = sqrt(cmag);
                            double nnorm = 1.0;
                            if(cmag > dmag) nnorm *= dmag / cmag;
                            for(k = 0; k < 3; k++) {
                                for(k1 = 0; k1 < 3; k1++) {
                                    CellP[i].Gradients.B[k][k1] = GB0[k][k1]
                                        + nnorm * (CellP[i].Gradients.B[k][k1] + ecorr[k][k1] - GB0[k][k1]);
                                }
#if (MHD_CONSTRAINED_GRADIENT <= 1)
                                local_slopelimiter(CellP[i].Gradients.B[k],
                                                   passer_base[i].Maxima.B[k],
                                                   passer_base[i].Minima.B[k],
                                                   0.25, P[i].KernelRadius, 0.25, 0, 0, 0);
#endif
                            }
                        }
                    }
                }
            }
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
            {
                double a_limiter = 0.25;
                if(CellP[i].ConditionNumber > 100)
                    a_limiter = DMIN(0.5, 0.25 + 0.25 * (CellP[i].ConditionNumber - 100) / 100);
                CellP[i].Gradients.Phi = passer_base[i].PhiGrad;
                construct_gradient(CellP[i].Gradients.Phi, i);
                local_slopelimiter(CellP[i].Gradients.Phi,
                                   passer_base[i].Maxima.Phi, passer_base[i].Minima.Phi,
                                   a_limiter, P[i].KernelRadius, 0.0, 0, 0, 0);
            }
#endif
        }
#endif /* MHD_CONSTRAINED_GRADIENT */

        /* (5e) Refresh ghosts between MHD-CG iters AFTER the host
         * between-iter work, BEFORE the next iter's kernel, so cross-rank
         * pairs see the slope-limited fields. NTask>1 only; skipped after
         * the final iter. The corridor's refresh republishes the CSR view
         * (re-fetched at the top of the next iteration). Mode B has no
         * ghosts to refresh (request/reply reads live peer state). */
#if defined(MHD_CONSTRAINED_GRADIENT)
        if(grad_iter + 1 < NUMBER_OF_GRADIENT_ITERATIONS && NTask > 1) {
            if(corridor_built_csr) {
                gizmo_hydro_corridor_refresh_ghost_values("mhd_cg_iter");
            }
        }
#endif
    } /* end grad_iter loop */

    /* (6) Post-iter finalization loop — OMP-parallel.
     *
     * Per-i pure body: every write targets CellP[i] / P[i] only (no
     * CellP[j] / P[j] cross-particle writes — those happened during the
     * pair-kernel and were reduced into accum before apply_active_writeback
     * ran). The three calculate_and_assign_* helpers (eos/eos.cc:372,
     * :492, :567) take (i, P, CellP) and were audited per-i pure: zero
     * cell[non-i] / pp[non-i] indexing.
     *
     * schedule(dynamic) absorbs the substantial #ifdef-gated per-i cost
     * imbalance (RT block + KERNEL_CRK_FACES tensor inversion + MHD-CG
     * slope-limit work). Range-for over ActiveParticleList rewritten to
     * indexed for so OMP can partition. Body verbatim from
     * hydro/gradients.cc:851-1223 (with GasGradDataPasser[i] mechanically
     * rewritten as passer_base[i]), now in hydro/gradients_tail.h. When the
     * writeback ran on the device the same body runs there, on the scratch
     * the device writeback filled. */
    if(!nothing_to_do && !gizmo_gradients_finalize_on_device()) {
#if GIZMO_HYDRO_TAILS_DEVICE
    gradients_finalize_host_loop(passer_base);   /* the same loop, built like the device one */
#else
    const int n_active_total = (int)ActiveParticleList.size();
    const int *active_list_data = ActiveParticleList.data();
#pragma omp parallel for schedule(dynamic, gizmo_omp_particle_chunk())
    for(int aa = 0; aa < n_active_total; aa++) {
        int i = active_list_data[aa];
        if(P[i].Type != 0) continue;
        gradients_finalize_cell(P, CellP, passer_base[i], i);
    } /* end finalization for-each-active */
#endif
    } /* end if !nothing_to_do */

    /* (7) NO ghost refresh here: hydro_force runs the corridor refresh at
     * its own top instead. Refreshing here would be both insufficient (owner
     * values still change between here and hydro_force: MG gradient
     * correction, dynamic-diffusion calc, and any intervening loop whose own
     * runner import+cleanup tears the corridor pool down) and redundant with
     * the hydro_force-top refresh that must happen anyway. Mode B has no
     * ghosts to refresh. */
    double t1 = my_second(); cpu_chain_sync(t1);

    (void)t0; (void)t1;
}
