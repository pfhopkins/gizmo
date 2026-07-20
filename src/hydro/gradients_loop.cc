/* hydro/gradients_loop.cc — host hooks + toplevel for GradientsSpec.
 *
 * See hydro/gradients_loop.h for the Spec contract. This file owns:
 *   - populate_device_context (grad_iter ferry to device)
 *   - apply_active_writeback  (replays out2particle_GasGrad{,_iter})
 *   - merge_accum             (peer-rank reduction for Mode B remote)
 *   - compare_accum           (oracle gate)
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
#include "../core/step_phases.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/kernel.h"                   /* MUST precede gradients_loop.h
                                                * — kernel.h has no include guards */
#include "../mesh/neighbor_loop_runner.h"
#include "../mesh/neighbor_list.h"            /* gizmo_sym_* — still maintained for hydro_force */
#include "hydro_corridor.h"                   /* mode + external_csr accessors */
#include "compute_finitevol_faces_functions.h"
#include "gradients_loop.h"

/* MAX/MIN macros mirroring the legacy preset versions from
 * hydro/gradients.cc (mode parameter ignored — these are the symmetric
 * re-formulation versions). Local to this file. */
#define ASSIGN_ADD_PRESET(x, y, mode) (x += y)
#define MAX_ADD(x, y, mode)           ((y > x) ? (x = y) : (1))
#define MIN_ADD(x, y, mode)           ((y < x) ? (x = y) : (1))

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
    Vec3<double>& g = *reinterpret_cast<Vec3<double>*>(grad);
    double d_abs = g.norm_sq();
    if(d_abs > 0) {
        d_abs = sqrt(d_abs); double cfac = 1 / (alim * h * d_abs);
        double fabs_max = fabs(valmax), fabs_min = fabs(valmin), abs_max = fabs_max, abs_min = fabs_min, f_corr_overshoot;
        if(abs_max < abs_min) { abs_max = fabs_min; abs_min = fabs_max; }
        f_corr_overshoot = DMIN(abs_min + shoot_tol * abs_max, abs_max);
        cfac *= f_corr_overshoot;
        if(pos_preserve == 1) {
            double fmin = DMIN(val_cen, DMAX(0, DMAX(MIN_REAL_NUMBER * val_cen,
                                                    DMIN(0.5 * (val_cen + valmin),
                                                         val_cen - f_corr_overshoot))));
            cfac = DMIN(((val_cen - fmin) / d_max) / d_abs, cfac);
        }
        if(cfac < 1) { g *= cfac; }
    }
}

void construct_gradient(Vec3<MyDouble>& grad, int i)
{
    if(SHOULD_I_USE_SPH_GRADIENTS(CellP[i].ConditionNumber)) {
        if(CellP[i].Density > 0) { grad *= P[i].DrkernNgbFactor / CellP[i].Density; }
    } else {
        grad = CellP[i].NV_T.matvec(grad);
    }
}

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

    /* ---------- iter==0 path: out2particle_GasGrad (line-for-line) ---------- */

#ifdef MHD_CONSTRAINED_GRADIENT
    {
        ASSIGN_ADD_PRESET(passer[i].FaceDotB, out->FaceDotB, 0);
#ifdef MHD_CONSTRAINED_GRADIENT_MIDPOINT
        for(int k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(passer[i].PhiGrad[k], out->Gradients[k].Phi, 0);
        }
#endif
    }
#endif

    {
        int j, k;
        MAX_ADD(passer[i].MaxDistance, out->MaxDistance, 0);
#ifdef TURB_DIFF_DYNAMIC
        for(j = 0; j < 3; j++) {
            MAX_ADD(passer[i].Maxima.Velocity_bar[j], out->Maxima.Velocity_bar[j], 0);
            MIN_ADD(passer[i].Minima.Velocity_bar[j], out->Minima.Velocity_bar[j], 0);
            ASSIGN_ADD_PRESET(CellP[i].Velocity_hat[j], out->Velocity_hat[j], 0);
            for(k = 0; k < 3; k++) {
                ASSIGN_ADD_PRESET(passer[i].GradVelocity_bar[j][k], out->Gradients[k].Velocity_bar[j], 0);
            }
        }
#endif

#if defined(KERNEL_CRK_FACES)
        ASSIGN_ADD_PRESET(passer[i].m0, out->m0, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(passer[i].dm0[k], out->dm0[k], 0); }
        for(j = 0; j < 3; j++) {
            ASSIGN_ADD_PRESET(passer[i].m1[j], out->m1[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(passer[i].dm1[j][k], out->dm1[j][k], 0); }
        }
        for(j = 0; j < 6; j++) {
            ASSIGN_ADD_PRESET(passer[i].m2[j], out->m2[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(passer[i].dm2[j][k], out->dm2[j][k], 0); }
        }
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
        ASSIGN_ADD_PRESET(passer[i].GlassAcc, out->GlassAcc, 0);
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
        ASSIGN_ADD_PRESET(CellP[i].alpha_limiter, out->alpha_limiter, 0);
#endif

        MAX_ADD(passer[i].Maxima.Density,  out->Maxima.Density,  0);
        MIN_ADD(passer[i].Minima.Density,  out->Minima.Density,  0);
        MAX_ADD(passer[i].Maxima.Pressure, out->Maxima.Pressure, 0);
        MIN_ADD(passer[i].Minima.Pressure, out->Minima.Pressure, 0);
        for(k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(CellP[i].Gradients.Density[k],  out->Gradients[k].Density,  0);
            ASSIGN_ADD_PRESET(CellP[i].Gradients.Pressure[k], out->Gradients[k].Pressure, 0);
        }
#ifdef DOGRAD_INTERNAL_ENERGY
        MAX_ADD(passer[i].Maxima.InternalEnergy, out->Maxima.InternalEnergy, 0);
        MIN_ADD(passer[i].Minima.InternalEnergy, out->Minima.InternalEnergy, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.InternalEnergy[k], out->Gradients[k].InternalEnergy, 0); }
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        MAX_ADD(passer[i].Maxima.ElectronNumberDensity, out->Maxima.ElectronNumberDensity, 0);
        MIN_ADD(passer[i].Minima.ElectronNumberDensity, out->Minima.ElectronNumberDensity, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.ElectronNumberDensity[k], out->Gradients[k].ElectronNumberDensity, 0); }
        MAX_ADD(passer[i].Maxima.ElectronTemperature, out->Maxima.ElectronTemperature, 0);
        MIN_ADD(passer[i].Minima.ElectronTemperature, out->Minima.ElectronTemperature, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.ElectronTemperature[k], out->Gradients[k].ElectronTemperature, 0); }
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(j = 0; j < 3; j++) {
            MAX_ADD(passer[i].Maxima.E_battery_T2[j], out->Maxima.E_battery_T2[j], 0);
            MIN_ADD(passer[i].Minima.E_battery_T2[j], out->Minima.E_battery_T2[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.E_battery_T2[j][k], out->Gradients[k].E_battery_T2[j], 0); }
        }
#endif
#ifdef COSMIC_RAY_FLUID
        for(j = 0; j < N_CR_PARTICLE_BINS; j++) {
            MAX_ADD(passer[i].Maxima.CosmicRayPressure[j], out->Maxima.CosmicRayPressure[j], 0);
            MIN_ADD(passer[i].Minima.CosmicRayPressure[j], out->Minima.CosmicRayPressure[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.CosmicRayPressure[j][k], out->Gradients[k].CosmicRayPressure[j], 0); }
        }
#endif
#ifdef DOGRAD_SOUNDSPEED
        MAX_ADD(passer[i].Maxima.SoundSpeed, out->Maxima.SoundSpeed, 0);
        MIN_ADD(passer[i].Minima.SoundSpeed, out->Minima.SoundSpeed, 0);
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.SoundSpeed[k], out->Gradients[k].SoundSpeed, 0); }
#endif

        for(j = 0; j < 3; j++) {
            MAX_ADD(passer[i].Maxima.Velocity[j], out->Maxima.Velocity[j], 0);
            MIN_ADD(passer[i].Minima.Velocity[j], out->Minima.Velocity[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.Velocity[j][k], out->Gradients[k].Velocity[j], 0); }
        }

#ifdef MAGNETIC
#ifdef HYDRO_SPH
#ifdef DIVBCLEANING_DEDNER
        ASSIGN_ADD_PRESET(CellP[i].divB, out->divB, 0);
#endif
        ASSIGN_ADD_PRESET(CellP[i].DtB, out->DtB, 0);
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
        for(j = 0; j < 3; j++) {
            ASSIGN_ADD_PRESET(CellP[i].Face_Area[j], out->Face_Area[j], 0);
            for(k = 0; k < 3; k++) {
                ASSIGN_ADD_PRESET(passer[i].BGrad[j][k],       out->Gradients[k].B[j],     0);
                ASSIGN_ADD_PRESET(passer[i].FaceCrossX[j][k],  out->FaceCrossX[j][k],      0);
            }
        }
#endif
        for(j = 0; j < 3; j++) {
            MAX_ADD(passer[i].Maxima.B[j], out->Maxima.B[j], 0);
            MIN_ADD(passer[i].Minima.B[j], out->Minima.B[j], 0);
            for(k = 0; k < 3; k++) {
#ifndef MHD_CONSTRAINED_GRADIENT
                ASSIGN_ADD_PRESET(CellP[i].Gradients.B[j][k], out->Gradients[k].B[j], 0);
#endif
            }
        }
#ifdef DIVBCLEANING_DEDNER
        MAX_ADD(passer[i].Maxima.Phi, out->Maxima.Phi, 0);
        MIN_ADD(passer[i].Minima.Phi, out->Minima.Phi, 0);
#ifndef MHD_CONSTRAINED_GRADIENT_MIDPOINT
        for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.Phi[k], out->Gradients[k].Phi, 0); }
#endif
#endif
#endif /* MAGNETIC */

#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(j = 0; j < NUM_METAL_SPECIES; j++) {
            MAX_ADD(passer[i].Maxima.Metallicity[j], out->Maxima.Metallicity[j], 0);
            MIN_ADD(passer[i].Minima.Metallicity[j], out->Minima.Metallicity[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(CellP[i].Gradients.Metallicity[j][k], out->Gradients[k].Metallicity[j], 0); }
        }
#endif

#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        for(j = 0; j < N_RT_FREQ_BINS; j++) {
            MAX_ADD(passer[i].Maxima.Rad_E_gamma[j], out->Maxima.Rad_E_gamma[j], 0);
            MIN_ADD(passer[i].Minima.Rad_E_gamma[j], out->Minima.Rad_E_gamma[j], 0);
            for(k = 0; k < 3; k++) { ASSIGN_ADD_PRESET(passer[i].Gradients_Rad_E_gamma[j][k], out->Gradients[k].Rad_E_gamma[j], 0); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int k_d = 0; k_d < 3; k_d++) {
                MAX_ADD(passer[i].Maxima.Rad_Flux[j][k_d], out->Maxima.Rad_Flux[j][k_d], 0);
                MIN_ADD(passer[i].Minima.Rad_Flux[j][k_d], out->Minima.Rad_Flux[j][k_d], 0);
                for(k = 0; k < 3; k++) {
                    ASSIGN_ADD_PRESET(CellP[i].Gradients.Rad_Flux_Grad[j][k_d][k], out->Gradients[k].Rad_Flux[j][k_d], 0);
                }
            }
#endif
        }
        /* The gradient dotted into the Eddington tensor (NV_T matvec replay) —
         * gradients.cc:432-448 verbatim. */
        for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
            for(int k_xyz = 0; k_xyz < 3; k_xyz++) {
                for(int j_xyz = 0; j_xyz < 3; j_xyz++) {
                    for(int i_xyz = 0; i_xyz < 3; i_xyz++) {
                        CellP[i].Gradients.Rad_E_gamma_ET[k_freq][k_xyz] +=
                            CellP[i].NV_T[j_xyz][i_xyz] *
                            out->Gradients[i_xyz].Rad_E_gamma_ET[k_freq][k_xyz][j_xyz];
                    }
                }
            }
        }
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
        ASSIGN_ADD_PRESET(P[i].AGS_zeta, out->AGS_zeta, 0);
#endif
    }
}

/* ============================================================================
 * merge_accum — peer-rank accum reduction (Mode B remote). Field-by-field
 * additive for Gradients[k].*, MAX/MIN for Maxima/Minima, MAX for
 * MaxDistance, additive for all CRK/face/SPH/AGS accumulators. Must match
 * the merge semantics that the pair body implicitly performs across passes.
 * ========================================================================== */
void GradientsSpec::merge_accum(AccumData& dst, const AccumData& src)
{
#define MERGE_ADD(field)  do { dst.field += src.field; } while(0)
#define MERGE_MAX(field)  do { if(src.field > dst.field) dst.field = src.field; } while(0)
#define MERGE_MIN(field)  do { if(src.field < dst.field) dst.field = src.field; } while(0)

    /* Gradients[k].* — additive for every quantity in Quantities_for_Gradients. */
    for(int k = 0; k < 3; k++) {
        MERGE_ADD(Gradients[k].Density);
        MERGE_ADD(Gradients[k].Pressure);
        for(int j = 0; j < 3; j++) { MERGE_ADD(Gradients[k].Velocity[j]); }
#ifdef MAGNETIC
        for(int j = 0; j < 3; j++) { MERGE_ADD(Gradients[k].B[j]); }
#ifdef DIVBCLEANING_DEDNER
        MERGE_ADD(Gradients[k].Phi);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(int j = 0; j < NUM_METAL_SPECIES; j++) { MERGE_ADD(Gradients[k].Metallicity[j]); }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
        for(int j = 0; j < N_RT_FREQ_BINS; j++) {
            MERGE_ADD(Gradients[k].Rad_E_gamma[j]);
            /* SymmetricTensor2 stores 6 unique elements behind [i][j]==[j][i]
             * aliasing — iterate raw storage to avoid double-adding the
             * off-diagonals (xy/yz/xz appear via both index orderings). */
            for(int kd = 0; kd < 6; kd++) { MERGE_ADD(Gradients[k].Rad_E_gamma_ET[j].data[kd]); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int kd = 0; kd < 3; kd++) { MERGE_ADD(Gradients[k].Rad_Flux[j][kd]); }
#endif
        }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        MERGE_ADD(Gradients[k].InternalEnergy);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        MERGE_ADD(Gradients[k].ElectronNumberDensity);
        MERGE_ADD(Gradients[k].ElectronTemperature);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(int j = 0; j < 3; j++) { MERGE_ADD(Gradients[k].E_battery_T2[j]); }
#endif
#ifdef COSMIC_RAY_FLUID
        for(int j = 0; j < N_CR_PARTICLE_BINS; j++) { MERGE_ADD(Gradients[k].CosmicRayPressure[j]); }
#endif
#ifdef DOGRAD_SOUNDSPEED
        MERGE_ADD(Gradients[k].SoundSpeed);
#endif
#ifdef TURB_DIFF_DYNAMIC
        for(int j = 0; j < 3; j++) { MERGE_ADD(Gradients[k].Velocity_bar[j]); }
#endif
    }

    /* Maxima/Minima — element-wise MAX/MIN. */
    MERGE_MAX(Maxima.Density);  MERGE_MIN(Minima.Density);
    MERGE_MAX(Maxima.Pressure); MERGE_MIN(Minima.Pressure);
    for(int j = 0; j < 3; j++) { MERGE_MAX(Maxima.Velocity[j]); MERGE_MIN(Minima.Velocity[j]); }
#ifdef MAGNETIC
    for(int j = 0; j < 3; j++) { MERGE_MAX(Maxima.B[j]); MERGE_MIN(Minima.B[j]); }
#ifdef DIVBCLEANING_DEDNER
    MERGE_MAX(Maxima.Phi); MERGE_MIN(Minima.Phi);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    for(int j = 0; j < NUM_METAL_SPECIES; j++) { MERGE_MAX(Maxima.Metallicity[j]); MERGE_MIN(Minima.Metallicity[j]); }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
    for(int j = 0; j < N_RT_FREQ_BINS; j++) {
        MERGE_MAX(Maxima.Rad_E_gamma[j]); MERGE_MIN(Minima.Rad_E_gamma[j]);
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        for(int kd = 0; kd < 3; kd++) { MERGE_MAX(Maxima.Rad_Flux[j][kd]); MERGE_MIN(Minima.Rad_Flux[j][kd]); }
#endif
    }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
    MERGE_MAX(Maxima.InternalEnergy); MERGE_MIN(Minima.InternalEnergy);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
    MERGE_MAX(Maxima.ElectronNumberDensity); MERGE_MIN(Minima.ElectronNumberDensity);
    MERGE_MAX(Maxima.ElectronTemperature);   MERGE_MIN(Minima.ElectronTemperature);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
    for(int j = 0; j < 3; j++) { MERGE_MAX(Maxima.E_battery_T2[j]); MERGE_MIN(Minima.E_battery_T2[j]); }
#endif
#ifdef COSMIC_RAY_FLUID
    for(int j = 0; j < N_CR_PARTICLE_BINS; j++) { MERGE_MAX(Maxima.CosmicRayPressure[j]); MERGE_MIN(Minima.CosmicRayPressure[j]); }
#endif
#ifdef DOGRAD_SOUNDSPEED
    MERGE_MAX(Maxima.SoundSpeed); MERGE_MIN(Minima.SoundSpeed);
#endif
#ifdef TURB_DIFF_DYNAMIC
    for(int j = 0; j < 3; j++) { MERGE_MAX(Maxima.Velocity_bar[j]); MERGE_MIN(Minima.Velocity_bar[j]); }
#endif

    MERGE_MAX(MaxDistance);

#if defined(KERNEL_CRK_FACES)
    MERGE_ADD(m0);
    for(int k = 0; k < 3; k++) {
        MERGE_ADD(m1[k]); MERGE_ADD(dm0[k]);
        for(int kx = 0; kx < 3; kx++) { MERGE_ADD(dm1[k][kx]); }
    }
    for(int k = 0; k < 6; k++) {
        MERGE_ADD(m2[k]);
        for(int kx = 0; kx < 3; kx++) { MERGE_ADD(dm2[k][kx]); }
    }
#endif
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    for(int j = 0; j < 3; j++) { MERGE_ADD(GlassAcc[j]); }
#endif
#ifdef HYDRO_SPH
#ifdef MAGNETIC
    for(int j = 0; j < 3; j++) { MERGE_ADD(DtB[j]); }
#ifdef DIVBCLEANING_DEDNER
    MERGE_ADD(divB);
#endif
#endif
    MERGE_ADD(alpha_limiter);
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
    for(int j = 0; j < 3; j++) {
        MERGE_ADD(Face_Area[j]);
        for(int k = 0; k < 3; k++) { MERGE_ADD(FaceCrossX[j][k]); }
    }
    MERGE_ADD(FaceDotB);
#endif
#ifdef TURB_DIFF_DYNAMIC
    for(int j = 0; j < 3; j++) { MERGE_ADD(Velocity_hat[j]); }
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    MERGE_ADD(AGS_zeta);
#endif

#undef MERGE_ADD
#undef MERGE_MAX
#undef MERGE_MIN
}

/* ============================================================================
 * compare_accum — oracle gate. Field-wise relative residual, taking the MAX
 * across every scalar pair_kernel can write. Per-field
 *   rel(a, b) = |a - b| / max(max(|a|, |b|), floor)
 * with floor = 1e-30, same form as cellcorrections::compare_accum. The MAX
 * (rather than L2 sum) gives a sharper per-field signal and a single
 * scalar return for the runner's accum_tolerance gate.
 *
 * Coverage mirrors merge_accum (same field set, same #ifdef structure) so
 * the oracle and the Mode B remote merge see the same surface. Reading the
 * struct as a flat double[] would mix MyFloat / MyDouble, read padding as
 * payload, and miss tail bytes — explicitly per-field instead.
 * ========================================================================== */
double GradientsSpec::compare_accum(const AccumData& a, const AccumData& b)
{
    double worst = 0.0;
    auto rel = [](double x, double y) {
        /* Absolute-difference floor: many MHD-CG fields (Face_Area,
         * FaceCrossX) involve subtractive sums that legitimately cancel
         * to ~machine-eps for symmetric configurations. A pure relative
         * residual (cellcorrections-style) divides O(1e-22) numerators by
         * O(1e-22) denominators and produces spurious O(1) "mismatches".
         * Treat any |a-b| below abs_tol as below the signal floor; above
         * it, use the relative residual with a relative-magnitude floor.
         * abs_tol is set 10 orders of magnitude below typical realistic
         * field magnitudes (Face_Area ~ h^(NUMDIMS-1) ~ 1e-1..1e-2 for
         * standard test problems), so genuine physics bugs that move
         * Face_Area by O(1e-6) or larger still trigger the gate. */
        constexpr double abs_tol = 1e-12;
        double d = fabs(x - y);
        if(d < abs_tol) return 0.0;
        double m = fmax(fmax(fabs(x), fabs(y)), 1e-30);
        return d / m;
    };
#define CHECK(field) do { double r = rel((double)a.field, (double)b.field); \
                          if(r > worst) worst = r; } while(0)

    for(int k = 0; k < 3; k++) {
        CHECK(Gradients[k].Density);
        CHECK(Gradients[k].Pressure);
        for(int j = 0; j < 3; j++) { CHECK(Gradients[k].Velocity[j]); }
#ifdef MAGNETIC
        for(int j = 0; j < 3; j++) { CHECK(Gradients[k].B[j]); }
#ifdef DIVBCLEANING_DEDNER
        CHECK(Gradients[k].Phi);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(int j = 0; j < NUM_METAL_SPECIES; j++) { CHECK(Gradients[k].Metallicity[j]); }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
        for(int j = 0; j < N_RT_FREQ_BINS; j++) {
            CHECK(Gradients[k].Rad_E_gamma[j]);
            /* SymmetricTensor2 stores 6 unique elements — iterate raw .data
             * to avoid double-counting off-diagonals via [i][j]/[j][i] alias. */
            for(int kd = 0; kd < 6; kd++) { CHECK(Gradients[k].Rad_E_gamma_ET[j].data[kd]); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            for(int kd = 0; kd < 3; kd++) { CHECK(Gradients[k].Rad_Flux[j][kd]); }
#endif
        }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        CHECK(Gradients[k].InternalEnergy);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        CHECK(Gradients[k].ElectronNumberDensity);
        CHECK(Gradients[k].ElectronTemperature);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(int j = 0; j < 3; j++) { CHECK(Gradients[k].E_battery_T2[j]); }
#endif
#ifdef COSMIC_RAY_FLUID
        for(int j = 0; j < N_CR_PARTICLE_BINS; j++) { CHECK(Gradients[k].CosmicRayPressure[j]); }
#endif
#ifdef DOGRAD_SOUNDSPEED
        CHECK(Gradients[k].SoundSpeed);
#endif
#ifdef TURB_DIFF_DYNAMIC
        for(int j = 0; j < 3; j++) { CHECK(Gradients[k].Velocity_bar[j]); }
#endif
    }

    /* Maxima/Minima — relative residual on each scalar (MAX/MIN merge
     * semantics; tolerance gate is on rel diff, not on commutativity). */
    CHECK(Maxima.Density);  CHECK(Minima.Density);
    CHECK(Maxima.Pressure); CHECK(Minima.Pressure);
    for(int j = 0; j < 3; j++) { CHECK(Maxima.Velocity[j]); CHECK(Minima.Velocity[j]); }
#ifdef MAGNETIC
    for(int j = 0; j < 3; j++) { CHECK(Maxima.B[j]); CHECK(Minima.B[j]); }
#ifdef DIVBCLEANING_DEDNER
    CHECK(Maxima.Phi); CHECK(Minima.Phi);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    for(int j = 0; j < NUM_METAL_SPECIES; j++) { CHECK(Maxima.Metallicity[j]); CHECK(Minima.Metallicity[j]); }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
    for(int j = 0; j < N_RT_FREQ_BINS; j++) {
        CHECK(Maxima.Rad_E_gamma[j]); CHECK(Minima.Rad_E_gamma[j]);
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        for(int kd = 0; kd < 3; kd++) { CHECK(Maxima.Rad_Flux[j][kd]); CHECK(Minima.Rad_Flux[j][kd]); }
#endif
    }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
    CHECK(Maxima.InternalEnergy); CHECK(Minima.InternalEnergy);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
    CHECK(Maxima.ElectronNumberDensity); CHECK(Minima.ElectronNumberDensity);
    CHECK(Maxima.ElectronTemperature);   CHECK(Minima.ElectronTemperature);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
    for(int j = 0; j < 3; j++) { CHECK(Maxima.E_battery_T2[j]); CHECK(Minima.E_battery_T2[j]); }
#endif
#ifdef COSMIC_RAY_FLUID
    for(int j = 0; j < N_CR_PARTICLE_BINS; j++) { CHECK(Maxima.CosmicRayPressure[j]); CHECK(Minima.CosmicRayPressure[j]); }
#endif
#ifdef DOGRAD_SOUNDSPEED
    CHECK(Maxima.SoundSpeed); CHECK(Minima.SoundSpeed);
#endif
#ifdef TURB_DIFF_DYNAMIC
    for(int j = 0; j < 3; j++) { CHECK(Maxima.Velocity_bar[j]); CHECK(Minima.Velocity_bar[j]); }
#endif

    CHECK(MaxDistance);

#if defined(KERNEL_CRK_FACES)
    CHECK(m0);
    for(int k = 0; k < 3; k++) {
        CHECK(m1[k]); CHECK(dm0[k]);
        for(int kx = 0; kx < 3; kx++) { CHECK(dm1[k][kx]); }
    }
    for(int k = 0; k < 6; k++) {
        CHECK(m2[k]);
        for(int kx = 0; kx < 3; kx++) { CHECK(dm2[k][kx]); }
    }
#endif
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    for(int j = 0; j < 3; j++) { CHECK(GlassAcc[j]); }
#endif
#ifdef HYDRO_SPH
#ifdef MAGNETIC
    for(int j = 0; j < 3; j++) { CHECK(DtB[j]); }
#ifdef DIVBCLEANING_DEDNER
    CHECK(divB);
#endif
#endif
    CHECK(alpha_limiter);
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
    for(int j = 0; j < 3; j++) {
        CHECK(Face_Area[j]);
        for(int k = 0; k < 3; k++) { CHECK(FaceCrossX[j][k]); }
    }
    CHECK(FaceDotB);
#endif
#ifdef TURB_DIFF_DYNAMIC
    for(int j = 0; j < 3; j++) { CHECK(Velocity_hat[j]); }
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    CHECK(AGS_zeta);
#endif

#undef CHECK
    return worst;
}

/* ============================================================================
 * Toplevel: hydro_gradient_calc — replaces the legacy walker in the retired
 * hydro/gradients.cc. Structure:
 *
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
void hydro_gradient_calc(void)
{
    CPU_Step[CPU_DENSMISC] += measure_time();
    double t0 = my_second();
    double t_grad_outer_start = my_second();

    /* (1) Corridor topology. The shared active list (and, in Mode A, the
     * ghost pool + CSR) was built ONCE at gizmo_hydro_corridor_begin(); this
     * function builds nothing. Mode A with a published view: refresh ghost
     * field values (owner-side hydro fields changed since the corridor
     * import: cellcorrections, stellar feedback) — the refresh may
     * rebuild+republish the CSR view, so the view is re-fetched per
     * iteration below, never cached across a refresh. Mode B: nothing to do
     * here (request-driven walkers use the corridor-built active list; no
     * ghosts, no CSR). Mode A WITHOUT a published view and with active gas
     * is a corridor sequencing bug — fail loudly, never quietly rebuild. */
    const GizmoHydroCorridorMode   corridor_mode = gizmo_hydro_corridor_get_mode();
    const bool corridor_built_csr = (gizmo_hydro_corridor_external_csr() != nullptr);
    double t_diag_symlist_start = my_second();
    if(corridor_built_csr) {
        gizmo_hydro_corridor_refresh_ghost_values("pre_gradients");
    } else if(corridor_mode == GizmoHydroCorridorMode::MODE_A
              && gizmo_sym_num_active_global > 0) {
        printf("FATAL: hydro_gradient_calc in Mode A with active gas but no published corridor CSR on task %d.\n", ThisTask);
        fflush(stdout);
        endrun(7315);
    }
    double t_grad_after_symlist = my_second();
    gizmo_step_phase_record("gradient_prep_symlist", timediff(t_diag_symlist_start, t_grad_after_symlist));
    if(ThisTask == 0 && gizmo_verbose_diag()) {
        printf("[DIAG_SYMNL step=%d N=%d pairs=%lld] symlist_build=%.3f%s\n",
               (int)All.NumCurrentTiStep, gizmo_sym_num_active,
               (long long)(gizmo_sym_neighbor_list.total_pairs),
               timediff(t_diag_symlist_start, my_second()),
               corridor_built_csr ? " (skipped: corridor built CSR)" : "");
        fflush(stdout);
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

    /* (3) Zero CellP[i].Gradients.*, P[i].AGS_zeta, passer[i], etc. on host
     * for every active gas particle. Mirrors hydro/gradients.cc:552-620
     * verbatim. */
#ifdef TURB_DIFF_DYNAMIC
    double smoothInv = 1.0 / All.TurbDynamicDiffSmoothing;
#endif
    for(int i : ActiveParticleList) {
        if(P[i].Type != 0) continue;
        int k2;
        memset(&passer_base[i], 0, sizeof(struct temporary_data_topass));
#ifdef HYDRO_SPH
#ifdef MAGNETIC
        CellP[i].DtB = {};
#endif
#ifdef DIVBCLEANING_DEDNER
        CellP[i].divB = 0;
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
        CellP[i].alpha_limiter = 0;
#endif
#endif
#ifdef TURB_DIFF_DYNAMIC
        CellP[i].Velocity_bar *= All.TurbDynamicDiffSmoothing;
        CellP[i].Velocity_hat = CellP[i].Velocity_bar * smoothInv;
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
        P[i].AGS_zeta = 0;
#endif
        CellP[i].Gradients.Density  = {};
        CellP[i].Gradients.Pressure = {};
        CellP[i].Gradients.Velocity = {};
#ifdef DOGRAD_INTERNAL_ENERGY
        CellP[i].Gradients.InternalEnergy = {};
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        CellP[i].Gradients.ElectronNumberDensity = {};
        CellP[i].Gradients.ElectronTemperature   = {};
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        CellP[i].Gradients.E_battery_T2 = {};
#endif
#ifdef COSMIC_RAY_FLUID
        for(k2 = 0; k2 < N_CR_PARTICLE_BINS; k2++) { CellP[i].Gradients.CosmicRayPressure[k2] = {}; }
#endif
#ifdef DOGRAD_SOUNDSPEED
        CellP[i].Gradients.SoundSpeed = {};
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
        CellP[i].Gradients.B = {};
#else
        CellP[i].Face_Area = {};
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
        CellP[i].Gradients.Phi = {};
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(k2 = 0; k2 < NUM_METAL_SPECIES; k2++) { CellP[i].Gradients.Metallicity[k2] = {}; }
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        for(k2 = 0; k2 < N_RT_FREQ_BINS; k2++) { CellP[i].Gradients.Rad_E_gamma_ET[k2] = {}; }
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        for(k2 = 0; k2 < N_RT_FREQ_BINS; k2++) {
            for(int k3 = 0; k3 < 3; k3++) { CellP[i].Gradients.Rad_E_gamma_Grad[k2][k3] = 0; }
            for(int k4 = 0; k4 < 3; k4++) for(int k5 = 0; k5 < 3; k5++) {
                CellP[i].Gradients.Rad_Flux_Grad[k2][k4][k5] = 0;
            }
        }
#endif
        (void)k2;
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
        (!corridor_built_csr) && (gizmo_sym_num_active_global <= 0);

    for(int grad_iter = 0; (!nothing_to_do) && grad_iter < NUMBER_OF_GRADIENT_ITERATIONS; grad_iter++) {
        aux.grad_iter = grad_iter;

        /* (5a) Zero per-iter passer fields (mirrors gradients.cc:637-646). */
        if(grad_iter > 0) {
            for(int i : ActiveParticleList) {
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

        /* (5c) Kernel dispatch. */
        run_neighbor_loop<GradientsSpec>(args);

        /* (5d) Host between-iter MHD-CG block — verbatim from
         * hydro/gradients.cc:689-829. */
#ifdef MHD_CONSTRAINED_GRADIENT
        for(int i : ActiveParticleList) {
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
                dbmax /= dbgrad;
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
     * rewritten as passer_base[i]). */
    if(!nothing_to_do) {
    const int n_active_total = (int)ActiveParticleList.size();
    const int *active_list_data = ActiveParticleList.data();
#pragma omp parallel for schedule(dynamic)
    for(int aa = 0; aa < n_active_total; aa++) {
        int i = active_list_data[aa];
        if(P[i].Type != 0) continue;
        int k, k1;
        construct_gradient(CellP[i].Gradients.Density, i);
        construct_gradient(CellP[i].Gradients.Pressure, i);
        for(k = 0; k < 3; k++) { construct_gradient(CellP[i].Gradients.Velocity[k], i); }
#ifdef TURB_DIFF_DYNAMIC
        for(k = 0; k < 3; k++) { construct_gradient(passer_base[i].GradVelocity_bar[k], i); }
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
        construct_gradient(CellP[i].Gradients.InternalEnergy, i);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
        construct_gradient(CellP[i].Gradients.ElectronNumberDensity, i);
        construct_gradient(CellP[i].Gradients.ElectronTemperature, i);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
        for(k = 0; k < 3; k++) { construct_gradient(CellP[i].Gradients.E_battery_T2[k], i); }
#endif
#ifdef COSMIC_RAY_FLUID
        for(k = 0; k < N_CR_PARTICLE_BINS; k++) { construct_gradient(CellP[i].Gradients.CosmicRayPressure[k], i); }
        int is_particle_local_extremum[N_CR_PARTICLE_BINS] = {0};
        is_particle_local_extremum[0] = 0;
#endif
#ifdef DOGRAD_SOUNDSPEED
        construct_gradient(CellP[i].Gradients.SoundSpeed, i);
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
        for(k = 0; k < 3; k++) { construct_gradient(CellP[i].Gradients.B[k], i); }
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
        construct_gradient(CellP[i].Gradients.Phi, i);
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
        for(k = 0; k < NUM_METAL_SPECIES; k++) { construct_gradient(CellP[i].Gradients.Metallicity[k], i); }
#endif
#ifdef RT_COMPGRAD_EDDINGTON_TENSOR
        for(k = 0; k < N_RT_FREQ_BINS; k++) { construct_gradient(passer_base[i].Gradients_Rad_E_gamma[k], i); }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
        for(int k_f = 0; k_f < N_RT_FREQ_BINS; k_f++) {
            for(int k_d = 0; k_d < 3; k_d++) { construct_gradient(CellP[i].Gradients.Rad_Flux_Grad[k_f][k_d], i); }
        }
#endif
#endif

#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        P[i].GradRho = CellP[i].Gradients.Density;
#endif

#if defined(TURB_DRIVING) || defined(OUTPUT_VORTICITY)
        CellP[i].Vorticity = CellP[i].Gradients.Velocity.curl();
#endif

#ifdef SPH_TP12_ARTIFICIAL_RESISTIVITY
        {
            double GradBMag = CellP[i].Gradients.B.frobenius_norm_sq();
            double rho_over_m = CellP[i].Density / P[i].Mass;
            double BMag = CellP[i].BPred.norm_sq() * rho_over_m * rho_over_m;
            CellP[i].Balpha = DMAX(DMIN(P[i].KernelRadius * sqrt(GradBMag / (BMag + 1.0e-33)),
                                        0.1 * All.ArtMagDispConst), 0.005);
        }
#endif

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
        {
            double ngb_eff = pow(P[i].NumNgb, NUMDIMS);
            if((fabs(ngb_eff - All.DesNumNgb) / All.DesNumNgb < 0.05)
               && (P[i].KernelRadius > 1.001 * All.MinKernelRadius)
               && (P[i].KernelRadius < 0.999 * All.MaxKernelRadius)) {
                double ndenNGB = ngb_eff / (VOLUME_NORM_COEFF_FOR_NDIMS * pow(P[i].KernelRadius, NUMDIMS));
                P[i].AGS_zeta *= P[i].Mass * P[i].KernelRadius / (NUMDIMS * ndenNGB) * P[i].DrkernNgbFactor;
            } else {
                P[i].AGS_zeta = 0;
            }
        }
#endif

#ifdef HYDRO_SPH
#ifdef MAGNETIC
        if(CellP[i].Density > 0) {
            CellP[i].DtB *= P[i].DrkernNgbFactor * P[i].Mass / (CellP[i].Density * CellP[i].Density) / All.cf_atime;
#ifdef DIVBCLEANING_DEDNER
            CellP[i].divB *= P[i].DrkernNgbFactor * P[i].Mass / (CellP[i].Density * CellP[i].Density);
            if((!isnan(CellP[i].divB)) && (P[i].KernelRadius > 0) && (CellP[i].divB != 0) && (CellP[i].Density > 0)) {
                double tmp_ded = 0.5 * CellP[i].MaxSignalVel;
                double rho_over_m_divb = CellP[i].Density / P[i].Mass;
                double b2_max = CellP[i].BPred.norm_sq() * rho_over_m_divb * rho_over_m_divb;
                b2_max = 100.0 * fabs(sqrt(b2_max) * All.cf_a2inv * P[i].Mass
                          / (CellP[i].Density * All.cf_a3inv) * 1.0
                          / (P[i].KernelRadius * All.cf_atime));
                if(fabs(CellP[i].divB) > b2_max) { CellP[i].divB *= b2_max / fabs(CellP[i].divB); }
                CellP[i].DtPhi = -tmp_ded * tmp_ded * All.DivBcleanHyperbolicSigma
                                  * CellP[i].divB * CellP[i].Density * All.cf_a3inv;
            } else {
                CellP[i].DtPhi = 0; CellP[i].divB = 0; CellP[i].DtB = {};
            }
            CellP[i].divB = 0.0;
#endif
        } else {
            CellP[i].DtB = {};
#ifdef DIVBCLEANING_DEDNER
            CellP[i].divB = 0; CellP[i].DtPhi = 0;
#endif
        }
#endif

#ifdef SPHAV_CD10_VISCOSITY_SWITCH
        {
            CellP[i].alpha_limiter /= CellP[i].Density;
            double NV_dt = get_particle_timestep_in_physical(i);
            double NV_dummy = fabs(1.0 * pow(1.0 - CellP[i].alpha_limiter, 4.0) * CellP[i].NV_DivVel);
            double NV_limiter = NV_dummy * NV_dummy / (NV_dummy * NV_dummy + CellP[i].NV_trSSt);
            double NV_A = DMAX(-CellP[i].NV_dt_DivVel, 0.0);
            double divVel_physical = CellP[i].NV_DivVel;
            if(All.ComovingIntegrationOn) { divVel_physical += 3 * All.cf_hubble_a; }
            if(divVel_physical >= 0.0) { NV_A = 0.0; }
            double h_eff = P[i].Get_Particle_Size() * All.cf_atime / 0.5;
            double cs_nv = CellP[i].effective_soundspeed();
            double alphaloc = All.ViscosityAMax * h_eff * h_eff * NV_A / (0.36 * cs_nv * cs_nv + h_eff * h_eff * NV_A);
            if(CellP[i].alpha < alphaloc) { CellP[i].alpha = alphaloc; }
            else if(CellP[i].alpha > alphaloc) {
                CellP[i].alpha = alphaloc + (CellP[i].alpha - alphaloc)
                                  * exp(-NV_dt * (0.5 * fabs(CellP[i].MaxSignalVel)) / (0.5 * h_eff) * 0.05);
            }
            if(CellP[i].alpha < All.ViscosityAMin) { CellP[i].alpha = All.ViscosityAMin; }
            CellP[i].alpha_limiter = DMAX(NV_limiter, All.ViscosityAMin / CellP[i].alpha);
        }
#else
        {
            double divVel = All.cf_a2inv * fabs(CellP[i].Gradients.Velocity.trace());
            if(All.ComovingIntegrationOn) { divVel += 3 * All.cf_hubble_a; }
            Vec3<double> CurlVel = CellP[i].Gradients.Velocity.curl();
            double MagCurl = All.cf_a2inv * CurlVel.norm();
            double fac_mu = 1 / All.cf_atime;
            CellP[i].alpha_limiter = divVel / (divVel + MagCurl
                                       + 0.0001 * CellP[i].effective_soundspeed()
                                                 / (P[i].Get_Particle_Size()) / fac_mu);
        }
#endif
#endif

        calculate_and_assign_conduction_and_viscosity_coefficients(i, P, CellP);
        calculate_and_assign_nonideal_mhd_coefficients(i, P, CellP);

#ifdef RADTRANSFER
        for(int k_freq = 0; k_freq < N_RT_FREQ_BINS; k_freq++) {
            CellP[i].Rad_Kappa[k_freq] = rt_kappa(i, k_freq, P, CellP);
#if defined(RT_FLUXLIMITER) && defined(RT_COMPGRAD_EDDINGTON_TENSOR)
            double lambda = 1;
            if(CellP[i].Rad_E_gamma_Pred[k_freq] > 0) {
                double R_ET = CellP[i].Gradients.Rad_E_gamma_ET[k_freq].norm()
                              / (MIN_REAL_NUMBER + CellP[i].Rad_E_gamma_Pred[k_freq]
                                                    * CellP[i].Density / (MIN_REAL_NUMBER + P[i].Mass));
                R_ET = 3. * DMAX(R_ET, 1.e-6 / P[i].Get_Particle_Size())
                        / (1.e-55 + All.cf_atime * CellP[i].Rad_Kappa[k_freq]
                                     * (CellP[i].Density * All.cf_a3inv));
                lambda = DMIN(1., DMAX(3. * (2. + R_ET) / (6. + 3. * R_ET + R_ET * R_ET), MIN_REAL_NUMBER));
#ifdef RT_OTVET
                double chi = DMAX(1./3., DMIN(1., (3. + 4. * lambda * lambda)
                              / (5. + 2. * sqrt(4. - 3. * lambda * lambda))));
                double chifac_iso = 3. * (1 - chi) / 2., chifac_ot = (3. * chi - 1.) / 2.;
                CellP[i].Gradients.Rad_E_gamma_ET[k_freq] =
                    chifac_ot * CellP[i].Gradients.Rad_E_gamma_ET[k_freq]
                  + (chifac_iso / 3.) * passer_base[i].Gradients_Rad_E_gamma[k_freq];
#endif
            }
            CellP[i].Rad_Flux_Limiter[k_freq] = lambda;
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_OTVET)
            {
                Vec3<MyDouble> g{passer_base[i].Gradients_Rad_E_gamma[k_freq][0],
                                 passer_base[i].Gradients_Rad_E_gamma[k_freq][1],
                                 passer_base[i].Gradients_Rad_E_gamma[k_freq][2]};
                CellP[i].Gradients.Rad_E_gamma_ET[k_freq] = CellP[i].ET[k_freq].matvec(g);
            }
#endif
#if defined(GRAIN_RDI_TESTPROBLEM_LIVE_RADIATION_INJECTION)
            if(CellP[i].Interpolated_Opacity[0] < 1.e-3 * All.Dust_to_Gas_Mass_Ratio * 0.75
                     * All.Grain_Q_at_MaxGrainSize
                     / ((All.Grain_Internal_Density / UNIT_DENSITY_IN_CGS)
                        * (All.Grain_Size_Max / UNIT_LENGTH_IN_CGS))) {
                double gmax = -1;
                if(P[i].GravAccel[GRAV_DIRECTION_RDI] < gmax) {
                    P[i].GravAccel[GRAV_DIRECTION_RDI] = gmax;
                }
            }
#endif
        }
#endif /* RADTRANSFER */

#if defined(EOS_ELASTIC)
        elastic_body_update_driftkick(i, 1., 2);
#endif

        /* Slope limiter block */
        {
            double stol = 0.0, stol_tmp, stol_diffusion;
            stol_diffusion = 0.1;
            stol_tmp = stol;
            double h_lim = P[i].KernelRadius;
            double d_max = DMAX(P[i].KernelRadius, passer_base[i].MaxDistance);
            h_lim = d_max;
            double a_limiter = 0.25;
            if(CellP[i].ConditionNumber > 100)
                a_limiter = DMIN(0.5, 0.25 + 0.25 * (CellP[i].ConditionNumber - 100) / 100);
#if defined(SELFGRAVITY_OFF) && (!defined(MAGNETIC) && !defined(GALSF))
            h_lim = P[i].KernelRadius; stol = 0.1;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 2)
            h_lim = P[i].KernelRadius; a_limiter *= 0.5; stol = 0.125;
#endif
#if (SLOPE_LIMITER_TOLERANCE == 0)
            a_limiter *= 2.0; stol = 0.0;
#endif

#if (SINGLE_STAR_SINK_FORMATION & 4)
            CellP[i].Density_Relative_Maximum_in_Kernel = passer_base[i].Maxima.Density;
#endif
            local_slopelimiter(CellP[i].Gradients.Density, passer_base[i].Maxima.Density, passer_base[i].Minima.Density,
                               a_limiter, h_lim, 0, 1, d_max, CellP[i].Density);
            int pressure_is_positive_definite = 1;
#if defined(EOS_TILLOTSON) || defined(EOS_ELASTIC) || defined(EOS_ANEOS)
            pressure_is_positive_definite = 0;
#endif
            local_slopelimiter(CellP[i].Gradients.Pressure, passer_base[i].Maxima.Pressure, passer_base[i].Minima.Pressure,
                               a_limiter, h_lim, stol, pressure_is_positive_definite, d_max, CellP[i].Pressure);
            stol_tmp = stol;
#if defined(VISCOSITY)
            stol_tmp = DMAX(stol, stol_diffusion);
#endif
#ifdef TURB_DIFF_DYNAMIC
            for(k1 = 0; k1 < 3; k1++) {
                local_slopelimiter(passer_base[i].GradVelocity_bar[k1],
                                   passer_base[i].Maxima.Velocity_bar[k1],
                                   passer_base[i].Minima.Velocity_bar[k1],
                                   a_limiter, h_lim, stol, 0, 0, 0);
            }
#endif
            for(k1 = 0; k1 < 3; k1++) {
                local_slopelimiter(CellP[i].Gradients.Velocity[k1],
                                   passer_base[i].Maxima.Velocity[k1],
                                   passer_base[i].Minima.Velocity[k1],
                                   a_limiter, h_lim, stol_tmp, 0, 0, 0);
            }
#ifdef DOGRAD_INTERNAL_ENERGY
            stol_tmp = stol;
#if defined(CONDUCTION)
            stol_tmp = DMAX(stol, stol_diffusion);
#endif
            local_slopelimiter(CellP[i].Gradients.InternalEnergy,
                               passer_base[i].Maxima.InternalEnergy, passer_base[i].Minima.InternalEnergy,
                               a_limiter, h_lim, stol_tmp, 1, d_max, CellP[i].InternalEnergyPred);
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & 1)
            local_slopelimiter(CellP[i].Gradients.ElectronNumberDensity,
                               passer_base[i].Maxima.ElectronNumberDensity, passer_base[i].Minima.ElectronNumberDensity,
                               a_limiter, h_lim, stol, 1, d_max, CellP[i].n_e());
            local_slopelimiter(CellP[i].Gradients.ElectronTemperature,
                               passer_base[i].Maxima.ElectronTemperature, passer_base[i].Minima.ElectronTemperature,
                               a_limiter, h_lim, stol, 1, d_max, CellP[i].T_e());
#endif
#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (2|4|8))
            for(k1 = 0; k1 < 3; k1++) {
                local_slopelimiter(CellP[i].Gradients.E_battery_T2[k1],
                                   passer_base[i].Maxima.E_battery_T2[k1], passer_base[i].Minima.E_battery_T2[k1],
                                   a_limiter, h_lim, stol, 0, 0, 0);
            }
#endif
#ifdef DOGRAD_SOUNDSPEED
            local_slopelimiter(CellP[i].Gradients.SoundSpeed,
                               passer_base[i].Maxima.SoundSpeed, passer_base[i].Minima.SoundSpeed,
                               a_limiter, h_lim, stol, 1, d_max, CellP[i].effective_soundspeed());
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(k1 = 0; k1 < NUM_METAL_SPECIES; k1++) {
                local_slopelimiter(CellP[i].Gradients.Metallicity[k1],
                                   passer_base[i].Maxima.Metallicity[k1], passer_base[i].Minima.Metallicity[k1],
                                   a_limiter, h_lim, DMAX(stol, stol_diffusion), 1, d_max, P[i].Metallicity[k1]);
            }
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && !defined(RT_EVOLVE_FLUX)
            for(k1 = 0; k1 < N_RT_FREQ_BINS; k1++) {
                local_slopelimiter(CellP[i].Gradients.Rad_E_gamma_ET[k1],
                                   passer_base[i].Maxima.Rad_E_gamma[k1], passer_base[i].Minima.Rad_E_gamma[k1],
                                   a_limiter, h_lim, stol, 1, d_max,
                                   CellP[i].Rad_E_gamma_Pred[k1] * CellP[i].Density / P[i].Mass);
                local_slopelimiter(passer_base[i].Gradients_Rad_E_gamma[k1],
                                   passer_base[i].Maxima.Rad_E_gamma[k1], passer_base[i].Minima.Rad_E_gamma[k1],
                                   a_limiter, h_lim, DMAX(stol, stol_diffusion), 1, d_max,
                                   CellP[i].Rad_E_gamma_Pred[k1] * CellP[i].Density / P[i].Mass);
            }
#endif
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
            {
                double V_i_inv_rt = CellP[i].Density / P[i].Mass;
                for(k1 = 0; k1 < N_RT_FREQ_BINS; k1++) {
                    double val_cen_e = CellP[i].Rad_E_gamma_Pred[k1] * V_i_inv_rt;
                    local_slopelimiter(passer_base[i].Gradients_Rad_E_gamma[k1],
                                       passer_base[i].Maxima.Rad_E_gamma[k1], passer_base[i].Minima.Rad_E_gamma[k1],
                                       a_limiter, h_lim, stol, 1, d_max, val_cen_e);
                    for(k = 0; k < 3; k++) {
                        CellP[i].Gradients.Rad_E_gamma_Grad[k1][k] = passer_base[i].Gradients_Rad_E_gamma[k1][k];
                    }
                    for(int k_d = 0; k_d < 3; k_d++) {
                        double val_cen_f = CellP[i].Rad_Flux_Pred[k1][k_d] * V_i_inv_rt;
                        local_slopelimiter(CellP[i].Gradients.Rad_Flux_Grad[k1][k_d],
                                           passer_base[i].Maxima.Rad_Flux[k1][k_d], passer_base[i].Minima.Rad_Flux[k1][k_d],
                                           a_limiter, h_lim, stol, 0, d_max, val_cen_f);
                    }
                }
            }
#endif
#ifdef MAGNETIC
#ifndef MHD_CONSTRAINED_GRADIENT
            {
                double v_tmp = P[i].Mass / CellP[i].Density;
                double tmp_d = sqrt(1.0e-37 + (2. * All.cf_atime / CellP[i].Pressure * v_tmp * v_tmp)
                                    + CellP[i].BPred.norm_sq());
                double q = fabs(CellP[i].divB) * P[i].KernelRadius / tmp_d;
                double alim2 = a_limiter * (1. + q * q);
                if(alim2 > 0.5) alim2 = 0.5;
                stol_tmp = stol;
#ifdef MHD_NON_IDEAL
                stol_tmp = DMAX(stol, stol_diffusion);
#endif
                for(k1 = 0; k1 < 3; k1++) {
                    local_slopelimiter(CellP[i].Gradients.B[k1],
                                       passer_base[i].Maxima.B[k1], passer_base[i].Minima.B[k1],
                                       alim2, h_lim, stol_tmp, 0, 0, 0);
                }
            }
#endif
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
            local_slopelimiter(CellP[i].Gradients.Phi,
                               passer_base[i].Maxima.Phi, passer_base[i].Minima.Phi,
                               a_limiter, h_lim, stol, 0, 0, 0);
#endif
#endif
        }

#ifdef TURB_DIFFUSION
#ifdef TURB_DIFF_DYNAMIC
        for(int k1_ = 0; k1_ < 3; k1_++) for(int k2_ = 0; k2_ < 3; k2_++) {
            CellP[i].VelShear_bar[k1_][k2_] = 0.5 * (passer_base[i].GradVelocity_bar[k1_][k2_]
                                                  + passer_base[i].GradVelocity_bar[k2_][k1_]);
        }
#endif
        calculate_and_assign_turbulent_diffusion_coefficients(i, P, CellP);
#endif

#if defined(COSMIC_RAY_FLUID) && !defined(CRFLUID_EVOLVE_SCATTERINGWAVES)
        for(k = 0; k < N_CR_PARTICLE_BINS; k++) { CellP[i].CosmicRayDiffusionCoeff[k] = 0; }
        if(CellP[i].Density > 0 && P[i].Mass > 0) {
            CalculateAndAssign_CosmicRay_DiffusionAndStreamingCoefficients(i, P, CellP);
        }
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
        if(All.Time > 0) {
            double cs_invelunits = CellP[i].effective_soundspeed() * All.cf_atime;
            double L_i_code = P[i].Get_Particle_Size();
            Vec3<double> dvel = L_i_code * L_i_code * passer_base[i].GlassAcc;
            double velnorm = dvel.norm();
            double dtx = get_particle_timestep_in_physical(i);
            if(velnorm > 0 && dtx > 0) {
                double v00 = 0.5 * DMIN(cs_invelunits * (0.5 * velnorm),
                                       All.CourantFac * (L_i_code / dtx) / All.cf_a2inv);
                CellP[i].ParticleVel += v00 * (dvel / velnorm);
            }
        }
#endif

#if defined(KERNEL_CRK_FACES)
        {
            double m0, dm0[3], m1[3], dm1[3][3], m2[3][3], m2i[3][3], dm2[3][3][3], Cnum_m2;
            m0 = passer_base[i].m0;
            for(k = 0; k < 3; k++) {
                dm0[k] = passer_base[i].dm0[k];
                m1[k]  = passer_base[i].m1[k];
                for(int k_x = 0; k_x < 3; k_x++) {
                    dm1[k][k_x] = passer_base[i].dm1[k][k_x];
                    int k_tmp = 0;
                    if((k == 0) && (k_x == 0)) { k_tmp = 0; }
                    if((k == 1) && (k_x == 1)) { k_tmp = 1; }
                    if((k == 2) && (k_x == 2)) { k_tmp = 2; }
                    if((k == 0) && (k_x == 1)) { k_tmp = 3; }
                    if((k == 1) && (k_x == 0)) { k_tmp = 3; }
                    if((k == 0) && (k_x == 2)) { k_tmp = 4; }
                    if((k == 2) && (k_x == 0)) { k_tmp = 4; }
                    if((k == 1) && (k_x == 2)) { k_tmp = 5; }
                    if((k == 2) && (k_x == 1)) { k_tmp = 5; }
                    m2[k][k_x] = passer_base[i].m2[k_tmp]; m2i[k][k_x] = 0;
                    for(int k_y = 0; k_y < 3; k_y++) { dm2[k][k_x][k_y] = passer_base[i].dm2[k_tmp][k_y]; }
                }
            }
            for(k = 0; k < 3; k++) { dm1[k][k] += m0; }
            for(k = 0; k < 3; k++) for(int k_x = 0; k_x < 3; k_x++) {
                dm2[k][k_x][k_x] += m1[k]; dm2[k_x][k][k_x] += m1[k];
            }
            Cnum_m2 = matrix_invert_ndims(m2, m2i);
            (void)Cnum_m2;
            double A = 0, B[3] = {0}, Bdotm1 = 0, dB[3][3] = {{0}}, dA[3] = {0};
            for(k = 0; k < 3; k++) {
                for(int k_x = 0; k_x < 3; k_x++) { B[k] += -m2i[k][k_x] * m1[k_x]; }
                Bdotm1 += B[k] * m1[k];
            }
            A = 1. / (m0 + Bdotm1);

            double minus_m2i_dm1_dotm1[3] = {0}, contracted_twotensor[3][3] = {{0}};
            double contracted_twotensor_x[3][3] = {{0}}, contracted_twotensor_dotm1[3] = {0};
            for(int k_gamma = 0; k_gamma < 3; k_gamma++) {
                for(int k_alpha = 0; k_alpha < 3; k_alpha++) {
                    for(int k_beta = 0; k_beta < 3; k_beta++) {
                        contracted_twotensor[k_beta][k_gamma] = 0;
                        for(int k_delta = 0; k_delta < 3; k_delta++) {
                            contracted_twotensor[k_beta][k_gamma] += dm2[k_beta][k_delta][k_gamma] * B[k_delta];
                        }
                        contracted_twotensor_x[k_alpha][k_gamma] += dm2[k_alpha][k_beta][k_gamma] * B[k_beta];
                        dB[k_alpha][k_gamma] += -m2i[k_alpha][k_beta]
                            * (dm1[k_beta][k_gamma] + contracted_twotensor[k_beta][k_gamma]);
                    }
                    minus_m2i_dm1_dotm1[k_gamma] += 2. * B[k_alpha] * dm1[k_alpha][k_gamma];
                    contracted_twotensor_dotm1[k_gamma] += B[k_alpha] * contracted_twotensor_x[k_alpha][k_gamma];
                }
                dA[k_gamma] = -A * A * (dm0[k_gamma] + minus_m2i_dm1_dotm1[k_gamma]
                                + contracted_twotensor_dotm1[k_gamma]);
            }

            double vector_corr[3] = {0}, tensor_corr[3][3] = {{0}};
            for(k = 0; k < 3; k++) {
                vector_corr[k] = dA[k] + A * B[k];
                for(int k_x = 0; k_x < 3; k_x++) {
                    tensor_corr[k][k_x] = B[k] * dA[k_x] + A * dB[k][k_x];
                }
            }
            CellP[i].Tensor_CRK_Face_Corrections[0] = A;
            for(k = 0; k < 3; k++) { CellP[i].Tensor_CRK_Face_Corrections[1 + k] = B[k]; }
            for(k = 0; k < 3; k++) { CellP[i].Tensor_CRK_Face_Corrections[1 + 3 + k] = vector_corr[k]; }
            for(k = 0; k < 3; k++) for(int k_x = 0; k_x < 3; k_x++) {
                CellP[i].Tensor_CRK_Face_Corrections[1 + 3 + 3 + 3 * k + k_x] = tensor_corr[k][k_x];
            }
        }
#endif
    } /* end finalization for-each-active */
    } /* end if !nothing_to_do */

    /* (7) NO ghost refresh here: hydro_force runs the corridor refresh at
     * its own top instead. Refreshing here would be both insufficient (owner
     * values still change between here and hydro_force: MG gradient
     * correction, dynamic-diffusion calc, and any intervening loop whose own
     * runner import+cleanup tears the corridor pool down) and redundant with
     * the hydro_force-top refresh that must happen anyway. Mode B has no
     * ghosts to refresh. */
    double t1 = WallclockTime = my_second();
    double t_grad_before_refresh = my_second();
    double t_grad_outer_end = my_second();
    gizmo_step_phase_record("gradient_zero_iter_loops", timediff(t_grad_after_symlist, t_grad_before_refresh));
    gizmo_step_phase_record("gradient_refresh_symlist", timediff(t_grad_before_refresh, t_grad_outer_end));
    gizmo_step_phase_record("gradient_outer_total",     timediff(t_grad_outer_start,    t_grad_outer_end));

    (void)t0; (void)t1;
}
