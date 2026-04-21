/* radfb_local_functions.h — GPU-callable per-source struct and per-pair kernel
 * for radiation_pressure_winds_consolidated GPU port.
 *
 * Handles GALSF_FB_FIRE_RT_LOCALRP: local stellar radiation-pressure momentum
 * kicks and UV/IR opacity coupling to gas neighbors.
 *
 * Two-pass kernel strategy: pass 1 computes wt_sum = Σ h_j² over neighbors
 * (within one Kokkos thread, serial over neighbor list); pass 2 applies kicks.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef GALSF_FB_FIRE_RT_LOCALRP

#include "../declarations/gpu_rng.h"
#include "../radiation/rt_functions.h"

/* Per-source input filled on CPU before GPU kernel launch. */
struct RadFBRPLocalIn {
    Vec3<MyDouble> Pos;
    MyFloat KernelRadius;      /* search radius used for neighbor list */
    MyFloat dE_over_c;         /* total photon momentum budget this step (code units) */
    MyFloat f_lum_ion;         /* ionizing luminosity fraction */
    MyIDType ID;               /* source particle ID, for GPU RNG */
#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    MyFloat delta_v_imparted_rp; /* per-kick target velocity (stochastic threshold) */
#endif
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    MyFloat jet_momentum_tocouple; /* extra jet momentum budget (0 if none) */
#endif
};

/* Per-source output: source-side jet budget consumed (STELLAREVOLUTION > 2 only). */
struct RadFBRPOut {
    MyDouble jet_momentum_used;  /* sum of wk*jet_momentum_tocouple over all neighbors */
};


/* Per-pair kick application.  Called from within the two-pass kernel loop.
 * wt_sum is pre-computed by pass 1.  Writes Vel, VelPred, dp on j-particle.
 * Returns dv total magnitude for diagnostics (0 if no kick applied). */
KOKKOS_INLINE_FUNCTION
static double radfb_rp_pair_kick(
    const struct RadFBRPLocalIn& loc,
    int j,
    struct particle_data *kp,
    struct gas_cell_data *kc,
    double r2,
    const Vec3<double>& dp_ij,   /* loc.Pos - kp[j].Pos, nearest_xyz applied */
    double wt_sum,
    int nn,                       /* neighbor index within this source's list */
    struct RadFBRPOut& out)
{
    if(kp[j].Type != 0 || kp[j].Mass <= 0 || r2 <= 0) return 0.0;
    double h_j = kp[j].Get_Particle_Size();
    double wk  = (h_j * h_j) / wt_sum;
    if(wk <= 0) return 0.0;

    double dE   = (double)loc.dE_over_c;
    double Mass_j = (double)kp[j].Mass;

    /* --- single-scattering kick (UV/optical absorbed fraction) --- */
    double dv_ss = wk * dE / Mass_j;

#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
    /* estimate absorbed fraction per cell */
    double cf_a  = All.cf_atime;
    double h_phys = h_j * cf_a;
    double sigma_cell = (wt_sum / (h_j * h_j)) * (Mass_j / (h_phys * h_phys));
    double tau_uv = rt_kappa(j, RT_FREQ_BIN_FIRE_UV, kp, kc) * sigma_cell;
    double tau_op = rt_kappa(j, RT_FREQ_BIN_FIRE_OPT, kp, kc) * sigma_cell;
    double frac_abs = (double)loc.f_lum_ion
                    + (1.0 - (double)loc.f_lum_ion)
                      * (1.0 - 0.5*(exp(-tau_uv) + exp(-tau_op)));
    dv_ss *= frac_abs;

    /* jet contribution */
    double jet_kick = 0.0;
    if((double)loc.jet_momentum_tocouple > 0) {
        jet_kick = wk * (double)loc.jet_momentum_tocouple / Mass_j;
        Kokkos::atomic_add(&out.jet_momentum_used, (MyDouble)(wk * (double)loc.jet_momentum_tocouple));
    }
    dv_ss += jet_kick;
#endif

    /* --- multiple-scattering (IR re-radiation) kick --- */
    double kappa_ir = rt_kappa(j, RT_FREQ_BIN_FIRE_IR, kp, kc);
    double dv_ms    = All.RP_Local_Momentum_Renormalization
                    * dE / Mass_j * kappa_ir
                    * (Mass_j / (4.0 * M_PI * r2 * All.cf_atime * All.cf_atime));

#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2)
    /* stochastic gate per neighbor */
    double dv_imparted = (double)loc.delta_v_imparted_rp;
    double prob = (dv_ms + dv_ss) / dv_imparted;
    if(prob > 1.0) { dv_imparted *= prob; prob = 1.0; }
    /* GPU RNG: use source ID ^ j ID, counter shifts by neighbor index for independence */
    double p_rand = gizmo_gpu_rand_double((uint64_t)loc.ID ^ (uint64_t)kp[j].ID,
                                          (uint64_t)(All.Ti_Current) + 3 + (uint64_t)nn);
    if(p_rand >= prob) return 0.0;

    /* cap total dv */
    if(dv_imparted > 1.0e4 / UNIT_VEL_IN_KMS) dv_imparted = 1.0e4 / UNIT_VEL_IN_KMS;

    /* direction: follow the stronger kick */
    Vec3<double> dir;
    if(dv_ss > dv_ms) { dir = dp_ij; }   /* UV: radially outward from star */
    else              { dir = -kp[j].GradRho; }  /* IR: along opacity gradient */
    double norm = dir.norm_sq();
    if(norm > 0) { norm = sqrt(norm); dir /= norm; } else { dir = {0,0,1}; }

    Vec3<double> dv_kick = (dv_imparted * All.cf_atime) * dir;

#else /* STELLAREVOLUTION > 2: always apply both components, no stochastic gate */

    /* IR kick: along opacity gradient */
    Vec3<double> dir_ir = -kp[j].GradRho;
    double norm_ir = dir_ir.norm_sq();
    if(norm_ir > 0) { norm_ir = sqrt(norm_ir); dir_ir /= norm_ir; } else { dir_ir = {0,0,1}; }

    /* UV/jet kick: radially outward from star */
    Vec3<double> dir_uv = dp_ij;
    double norm_uv = dir_uv.norm_sq();
    if(norm_uv > 0) { norm_uv = sqrt(norm_uv); dir_uv /= norm_uv; } else { dir_uv = {0,0,1}; }

    /* cap each component */
    if(dv_ms  > 1.0e4 / UNIT_VEL_IN_KMS) dv_ms  = 1.0e4 / UNIT_VEL_IN_KMS;
    if(dv_ss  > 1.0e4 / UNIT_VEL_IN_KMS) dv_ss  = 1.0e4 / UNIT_VEL_IN_KMS;

    /* combined kick (split into two directions) */
    Vec3<double> dv_kick = (dv_ms * All.cf_atime) * dir_ir
                         + (dv_ss * All.cf_atime) * dir_uv;
    double dv_imparted = (dv_ms + dv_ss);
#endif

    /* apply kick to j-particle (atomic since multiple sources may share neighbor j) */
    for(int k = 0; k < 3; k++) {
        Kokkos::atomic_add(&kp[j].Vel[k],        (MyDouble)dv_kick[k]);
        Kokkos::atomic_add(&kc[j].VelPred[k],    (MyDouble)dv_kick[k]);
        Kokkos::atomic_add(&kp[j].dp[k],         (MyDouble)(dv_kick[k] * Mass_j));
    }
    return dv_imparted;
}

#endif /* GALSF_FB_FIRE_RT_LOCALRP */
