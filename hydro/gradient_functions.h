/* gradient_functions.h — GPU-callable gradient kernel functions (AthenaK pattern).
 *
 * Contains: struct Quantities_for_Gradients, struct kernel_GasGrad,
 *   struct GasGraddata_in_, struct GasGraddata_out_,
 *   GasGrad_isactive_gpu(), gradient_accumulate_neighbor().
 *
 * These are the per-neighbor-pair accumulation functions extracted from
 * GasGrad_evaluate() in gradients.cc. The tree walk, MPI dispatch,
 * post-processing (slope limiting, NV_T inversion) remain in gradients.cc.
 *
 * For CPU builds: gradients.cc includes this header with KOKKOS_INLINE_FUNCTION
 * redefined to empty, providing externally-visible symbols.
 *
 * For GPU builds: the GPU TU includes this header with KOKKOS_INLINE_FUNCTION
 * as __device__ __host__ inline, making these functions callable from kernels.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef GRADIENT_FUNCTIONS_H
#define GRADIENT_FUNCTIONS_H

/* Requires: allvars.h, proto.h, kernel.h included before this header. */
#ifdef COSMIC_RAY_FLUID
#include "../eos/cosmic_ray_fluid/cosmic_ray_functions.h"
#endif

#ifndef MINMAX_CHECK
#define MINMAX_CHECK(x,xmin,xmax) ((x<xmin)?(xmin=x):((x>xmax)?(xmax=x):(1)))
#endif
#ifndef SHOULD_I_USE_SPH_GRADIENTS
#if defined(COOLING)
#define SHOULD_I_USE_SPH_GRADIENTS(condition_number) ((condition_number > CONDITION_NUMBER_DANGER) ? (1):(0))
#else
#define SHOULD_I_USE_SPH_GRADIENTS(condition_number) ((condition_number > CONDITION_NUMBER_DANGER) ? (0):(0))
#endif
#endif
#ifndef NV_MYSIGN
#define NV_MYSIGN(x) (( x > 0 ) - ( x < 0 ))
#endif


/* define a common 'gradients' structure to hold everything we take derivatives of.
   Also defined in gradients.cc (for the CPU tree-walk path). The two definitions
   must remain identical. Guard prevents redefinition if both are visible. */
#ifndef QUANTITIES_FOR_GRADIENTS_DEFINED
#define QUANTITIES_FOR_GRADIENTS_DEFINED
struct Quantities_for_Gradients
{
    MyDouble Density;
    MyDouble Pressure;
    Vec3<MyDouble> Velocity;
#ifdef MAGNETIC
    Vec3<MyDouble> B;
#ifdef DIVBCLEANING_DEDNER
    MyDouble Phi;
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    MyDouble Metallicity[NUM_METAL_SPECIES];
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
    MyFloat Rad_E_gamma[N_RT_FREQ_BINS];
    SymmetricTensor2<MyFloat> Rad_E_gamma_ET[N_RT_FREQ_BINS];
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
    MyFloat Rad_Flux[N_RT_FREQ_BINS][3];
#endif
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
    MyDouble InternalEnergy;
#endif
#ifdef COSMIC_RAY_FLUID
    MyDouble CosmicRayPressure[N_CR_PARTICLE_BINS];
#endif
#ifdef DOGRAD_SOUNDSPEED
    MyDouble SoundSpeed;
#endif
#ifdef TURB_DIFF_DYNAMIC
    Vec3<MyDouble> Velocity_bar;
#endif
};
#endif /* QUANTITIES_FOR_GRADIENTS_DEFINED */


/* working struct for kernel quantities computed per neighbor pair */
struct kernel_GasGrad
{
    Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i;
};


/* input struct: data from the 'searching' element */
struct GasGraddata_in_
{
    Vec3<MyDouble> Pos;
    MyFloat Mass;
    MyFloat KernelRadius;
#ifdef MHD_CONSTRAINED_GRADIENT
    MyDouble ConditionNumber;
    SymmetricTensor2<MyDouble> NV_T;
    MyFloat BGrad[3][3];
#ifdef MHD_MODIFIED_GRADIENT
    MyFloat MG_cgcoeff;
#endif
#ifdef MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV
    Vec3<MyFloat> PhiGrad;
#endif
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
    MyFloat NV_DivVel;
#endif
    struct Quantities_for_Gradients GQuant;
#ifdef TURB_DIFF_DYNAMIC
    MyDouble Norm_hat;
#ifdef GALSF_SUBGRID_WINDS
    MyFloat DelayTime;
#endif
#endif
};


/* output struct: data accumulated across neighbor pairs */
struct GasGraddata_out_
{
#if defined(KERNEL_CRK_FACES)
    MyDouble m0;
    MyDouble m1[3];
    MyDouble m2[6];
    MyDouble dm0[3];
    MyDouble dm1[3][3];
    MyDouble dm2[6][3];
#endif
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    Vec3<MyFloat> GlassAcc;
#endif
#ifdef HYDRO_SPH
    MyFloat alpha_limiter;
#ifdef MAGNETIC
#ifdef DIVBCLEANING_DEDNER
    MyFloat divB;
#endif
    Vec3<MyFloat> DtB;
#endif
#endif
#ifdef MHD_CONSTRAINED_GRADIENT
    Vec3<MyFloat> Face_Area;
    MyFloat FaceDotB;
    MyFloat FaceCrossX[3][3];
#endif
    struct Quantities_for_Gradients Gradients[3];
    struct Quantities_for_Gradients Maxima;
    struct Quantities_for_Gradients Minima;
    MyFloat MaxDistance;
#ifdef TURB_DIFF_DYNAMIC
    Vec3<MyDouble> Velocity_hat;
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    MyFloat AGS_zeta;
#endif
};


/* check if particle j is valid for gradient computation */
KOKKOS_INLINE_FUNCTION
int GasGrad_isactive_gpu(int j, struct particle_data *P, struct gas_cell_data *CellP)
{
    if(P[j].Type != 0) return 0;
    if(P[j].Mass <= 0) return 0;
    if(CellP[j].Density <= 0 || P[j].KernelRadius <= 0) return 0;
#if defined(GALSF_SUBGRID_WINDS) && !defined(TURB_DIFF_DYNAMIC)
    if(CellP[j].DelayTime > 0) return 0;
#endif
    return 1;
}


/* Per-neighbor-pair accumulation for gradient_iteration==0 (main gradient pass).
   Processes one neighbor j for the searching particle described by 'local'.
   This is the inner body of the GasGrad_evaluate neighbor loop. */
KOKKOS_INLINE_FUNCTION
void gradient_accumulate_neighbor(struct GasGraddata_in_ *local, struct GasGraddata_out_ *out,
                                  struct kernel_GasGrad *kernel, int j,
                                  int sph_gradients_flag_i,
                                  double V_i, double hinv, double hinv3, double hinv4,
                                  int kernel_mode_i,
                                  struct particle_data *P, struct gas_cell_data *CellP)
{
    if(GasGrad_isactive_gpu(j, P, CellP) == 0) return;

    kernel->dp = local->Pos - P[j].Pos;
    nearest_xyz(kernel->dp);
    double r2 = kernel->dp.norm_sq();
    double h_j = P[j].KernelRadius;

#if !defined(HYDRO_SPH) && !defined(KERNEL_CRK_FACES)
    if(r2 <= 0) return;
#endif

    double h2_i = kernel->h_i * kernel->h_i;

#ifdef TURB_DIFF_DYNAMIC
#ifdef GALSF_SUBGRID_WINDS
    if((CellP[j].DelayTime == 0 && local->DelayTime == 0) || (CellP[j].DelayTime > 0 && local->DelayTime > 0)) {
#else
    {
#endif
        double hhat_i, hhat_j, hhatinv_i, hhatinv3_i, hhatinv4_i, wkhat_i, dwkhat_i;
        hhat_i = All.TurbDynamicDiffFac * kernel->h_i; hhat_j = All.TurbDynamicDiffFac * h_j;
        if((r2 < (hhat_i * hhat_i)) || (r2 < (hhat_j * hhat_j))) {
            double h_avg = 0.5 * (hhat_i + hhat_j), particle_distance = sqrt(r2);
            kernel_hinv(h_avg, &hhatinv_i, &hhatinv3_i, &hhatinv4_i);
            double u = DMIN(particle_distance * hhatinv_i, 1.0);
            if(u<1) {kernel_main(u, hhatinv3_i, hhatinv4_i, &wkhat_i, &dwkhat_i, 0);} else {wkhat_i=dwkhat_i=0;}
            double mean_weight = wkhat_i * 0.5 * (CellP[j].Norm_hat + local->Norm_hat) / (local->Norm_hat * CellP[j].Norm_hat);
            double weight_i = P[j].Mass * mean_weight;
            if(particle_distance < h_avg) {
                Vec3<MyDouble> Velocity_bar_diff = CellP[j].Velocity_bar - local->GQuant.Velocity_bar;
                out->Velocity_hat += Velocity_bar_diff * weight_i;
            }
        }
    }
#endif

    if((r2 >= h2_i) && (r2 >= h_j * h_j)) return;

    kernel->r = sqrt(r2);
    double u;
    if(kernel->r < kernel->h_i)
    {
        u = kernel->r * hinv;
        kernel_main(u, hinv3, hinv4, &kernel->wk_i, &kernel->dwk_i, kernel_mode_i);
    }
    else
    {
        kernel->dwk_i = kernel->wk_i = 0;
    }

    int sph_gradients_flag_j = 0;
#if defined(MHD_CONSTRAINED_GRADIENT) || defined(KERNEL_CRK_FACES)
    if(kernel->r < h_j)
    {
        sph_gradients_flag_j = SHOULD_I_USE_SPH_GRADIENTS(CellP[j].ConditionNumber);
        int kernel_mode_j;
#if defined(HYDRO_SPH) || defined(KERNEL_CRK_FACES)
        kernel_mode_j = 0;
#else
        if(sph_gradients_flag_j) {kernel_mode_j=0;} else {kernel_mode_j=-1;}
#endif
        double hinv_j, hinv3_j, hinv4_j;
        kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
        u = kernel->r * hinv_j;
        kernel_main(u, hinv3_j, hinv4_j, &kernel->wk_j, &kernel->dwk_j, kernel_mode_j);
    }
    else
#endif
    {
        kernel->dwk_j = kernel->wk_j = 0;
    }

    double Particle_Size_j = P[j].Get_Particle_Size();
    double Particle_Size_i = pow(local->Mass / local->GQuant.Density, 1./NUMDIMS);

    /* ---- MHD constrained gradient face computation ---- */
#if defined(MHD_CONSTRAINED_GRADIENT)
    {
        double V_j = P[j].Mass / CellP[j].Density;
        double Face_Area_Norm, cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - local->ConditionNumber*local->ConditionNumber;
        Vec3<double> Face_Area_Vec;
        int k, k2;
        double rinv = 1.0 / (MIN_REAL_NUMBER + kernel->r);

        /* compute_finitevol_faces.h expects 'kernel' and 'local' as value types (not pointers),
           so create local references to match its syntax */
        auto &kernel_ref = *kernel;
        auto &local_ref = *local;
        #define kernel kernel_ref
        #define local local_ref
#include "compute_finitevol_faces.h"
        #undef kernel
        #undef local

        for(k=0;k<3;k++)
        {
            out->Face_Area[k] += Face_Area_Vec[k];
            for(k2=0;k2<3;k2++)
            {
                double q = -0.5 * Face_Area_Vec[k] * kernel->dp[k2];
                out->FaceCrossX[k][k2] += q;
            }

            /* B_L,R state reconstruction + slope limiting */
            double Bjk = CellP[j].Bfield_component(k);
            NGB_SHEARBOX_BOUNDARY_BCORR_(local->Pos, P[j].Pos, Bjk, -1);
            double db_c = 0.5 * dot(CellP[j].Gradients.B[k], kernel->dp);
            double db_cR = -0.5 * (local->BGrad[k][0]*kernel->dp[0] + local->BGrad[k][1]*kernel->dp[1] + local->BGrad[k][2]*kernel->dp[2]);

            double Q_L, Q_R;
            if(Bjk == local->GQuant.B[k])
            {
                Q_L = Q_R = Bjk;
            } else {
                Q_L = Bjk + db_c;
                Q_R = local->GQuant.B[k] + db_cR;
                double Qmax, Qmin, Qmed = 0.5*(local->GQuant.B[k] + Bjk);
                if(local->GQuant.B[k] < Bjk) {Qmax=Bjk; Qmin=local->GQuant.B[k];} else {Qmax=local->GQuant.B[k]; Qmin=Bjk;}
                double fac = MHD_CONSTRAINED_GRADIENT_FAC_MINMAX * (Qmax-Qmin);
                fac += MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM * fabs(Qmed);
                double Qmax_eff = Qmax + fac;
                double Qmin_eff = Qmin - fac;
                fac = MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV * (Qmax-Qmin);
                fac += MHD_CONSTRAINED_GRADIENT_FAC_MED_PM * fabs(Qmed);
                double Qmed_max = Qmed + fac;
                double Qmed_min = Qmed - fac;
                if(Qmed_max>Qmax_eff) Qmed_max=Qmax_eff;
                if(Qmed_min<Qmin_eff) Qmed_min=Qmin_eff;
                if(local->GQuant.B[k] < Bjk)
                {
                    if(Q_L>Qmax_eff) Q_L=Qmax_eff;
                    if(Q_L<Qmed_min) Q_L=Qmed_min;
                    if(Q_R<Qmin_eff) Q_R=Qmin_eff;
                    if(Q_R>Qmed_max) Q_R=Qmed_max;
                } else {
                    if(Q_L<Qmin_eff) Q_L=Qmin_eff;
                    if(Q_L>Qmed_max) Q_L=Qmed_max;
                    if(Q_R>Qmax_eff) Q_R=Qmax_eff;
                    if(Q_R<Qmed_min) Q_R=Qmed_min;
                }
            }

            out->FaceDotB += Face_Area_Vec[k] * (local->GQuant.B[k] + Q_L);
        }

#ifdef MHD_MODIFIED_GRADIENT
        {
            double A_dot_dp = dot(Face_Area_Vec, kernel->dp);
            double mg_delta = 0.25 * (CellP[j].MG_cgcoeff - local->MG_cgcoeff) * A_dot_dp * A_dot_dp;
            out->FaceDotB += mg_delta;
        }
#endif

#if defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT) && !defined(DIVBCLEANING_DEDNER)
        {
            double dphi = CellP[j].PhiPred / P[j].Mass - local->GQuant.Phi;
            MINMAX_CHECK(dphi, out->Minima.Phi, out->Maxima.Phi);
            double dphi_grad_j = 0.5 * dot(kernel->dp, CellP[j].Gradients.Phi);
            double dphi_grad_i = -0.5 * dot(kernel->dp, local->PhiGrad);
            if(dphi > 0) {
                if(dphi_grad_j>0) {dphi_grad_j=0;} else {if(dphi_grad_j<0.5*dphi) dphi_grad_j=0.5*dphi;}
                if(dphi_grad_i<0) {dphi_grad_i=0;} else {if(dphi_grad_i>0.5*dphi) dphi_grad_i=0.5*dphi;}
            } else {
                if(dphi_grad_j<0) {dphi_grad_j=0;} else {if(dphi_grad_j>0.5*dphi) dphi_grad_j=0.5*dphi;}
                if(dphi_grad_i>0) {dphi_grad_i=0;} else {if(dphi_grad_i<0.5*dphi) dphi_grad_i=0.5*dphi;}
            }
            double dphi_j = dphi + dphi_grad_j;
            if(sph_gradients_flag_i) {dphi_j *= -2*kernel->wk_i;} else {dphi_j *= kernel->dwk_i/kernel->r * P[j].Mass;}
            for(int kk=0;kk<3;kk++) {out->Gradients[kk].Phi += dphi_j * kernel->dp[kk];}
        }
#endif
    }
#endif /* MHD_CONSTRAINED_GRADIENT */


    /* ---- Difference & slope limiting: track min/max of quantities in kernel ---- */
    if(kernel->r > out->MaxDistance) {out->MaxDistance = kernel->r;}

    double d_rho = CellP[j].Density - local->GQuant.Density;
    MINMAX_CHECK(d_rho, out->Minima.Density, out->Maxima.Density);

    double dp_pres = CellP[j].Pressure - local->GQuant.Pressure;
    MINMAX_CHECK(dp_pres, out->Minima.Pressure, out->Maxima.Pressure);

    auto dv = CellP[j].VelPred - local->GQuant.Velocity;
    NGB_SHEARBOX_BOUNDARY_VELCORR_(local->Pos, P[j].Pos, dv, -1);
    for(int k=0;k<3;k++) {
        MINMAX_CHECK(dv[k], out->Minima.Velocity[k], out->Maxima.Velocity[k]);
    }

#ifdef TURB_DIFF_DYNAMIC
    {
        Vec3<MyDouble> dv_bar = CellP[j].Velocity_bar - local->GQuant.Velocity_bar;
        NGB_SHEARBOX_BOUNDARY_VELCORR_(local->Pos, P[j].Pos, dv_bar, -1);
        for(int k=0;k<3;k++) {MINMAX_CHECK(dv_bar[k], out->Minima.Velocity_bar[k], out->Maxima.Velocity_bar[k]);}
    }
#endif

#if defined(KERNEL_CRK_FACES)
    {
        double V_j_crk = P[j].Mass/CellP[j].Density;
        double wk_ij = 0.5*(kernel->wk_i + kernel->wk_j), dwk_ij = 0.5*(kernel->dwk_i + kernel->dwk_j), rinv = 1./(MIN_REAL_NUMBER + kernel->r);
        double Vj_wki = V_j_crk*wk_ij, Vj_dwki = V_j_crk*dwk_ij*rinv;
        out->m0 += Vj_wki;
        for(int k=0;k<3;k++) {out->dm0[k] += Vj_dwki*kernel->dp[k];}
        for(int k2=0;k2<3;k2++)
        {
            out->m1[k2] += Vj_wki*kernel->dp[k2];
            for(int k=0;k<3;k++) {out->dm1[k2][k] += Vj_dwki*kernel->dp[k2]*kernel->dp[k];}
        }
        for(int k2=0;k2<6;k2++)
        {
            int kk0[6]={0,1,2,0,0,1};
            int kk1[6]={0,1,2,1,2,2};
            out->m2[k2] += Vj_wki*kernel->dp[kk0[k2]]*kernel->dp[kk1[k2]];
            for(int k=0;k<3;k++) {out->dm2[k2][k] += Vj_dwki*kernel->dp[kk0[k2]]*kernel->dp[kk1[k2]]*kernel->dp[k];}
        }
    }
#endif

#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION==6)
    for(int k=0;k<3;k++) { out->GlassAcc[k] += kernel->dp[k] / (kernel->r*kernel->r*kernel->r); }
#endif

#ifdef DOGRAD_INTERNAL_ENERGY
    double du = CellP[j].InternalEnergyPred - local->GQuant.InternalEnergy;
    MINMAX_CHECK(du, out->Minima.InternalEnergy, out->Maxima.InternalEnergy);
#endif

#ifdef COSMIC_RAY_FLUID
    double dpCR[N_CR_PARTICLE_BINS];
    for(int k=0;k<N_CR_PARTICLE_BINS;k++) {
        dpCR[k] = Get_Gas_CosmicRayPressure(j, k, CellP) - local->GQuant.CosmicRayPressure[k];
        MINMAX_CHECK(dpCR[k], out->Minima.CosmicRayPressure[k], out->Maxima.CosmicRayPressure[k]);
    }
#endif

#ifdef DOGRAD_SOUNDSPEED
    double dc = CellP[j].effective_soundspeed() - local->GQuant.SoundSpeed;
    MINMAX_CHECK(dc, out->Minima.SoundSpeed, out->Maxima.SoundSpeed);
#endif

#ifdef MAGNETIC
    Vec3<double> Bj = CellP[j].BPred * (CellP[j].Density / P[j].Mass);
    NGB_SHEARBOX_BOUNDARY_BCORR_(local->Pos, P[j].Pos, Bj, -1);
    Vec3<double> dB = Bj - local->GQuant.B;
    for(int k=0;k<3;k++) { MINMAX_CHECK(dB[k], out->Minima.B[k], out->Maxima.B[k]); }
#endif

#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
    {
        double dphi = CellP[j].PhiPred / P[j].Mass - local->GQuant.Phi;
        MINMAX_CHECK(dphi, out->Minima.Phi, out->Maxima.Phi);
    }
#endif

#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
    double dmetal[NUM_METAL_SPECIES];
    for(int k=0;k<NUM_METAL_SPECIES;k++) {
        dmetal[k] = P[j].Metallicity[k] - local->GQuant.Metallicity[k];
        MINMAX_CHECK(dmetal[k], out->Minima.Metallicity[k], out->Maxima.Metallicity[k]);
    }
#endif

#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
    SymmetricTensor2<double> dnET[N_RT_FREQ_BINS];
    double dn[N_RT_FREQ_BINS];
    double V_i_inv = 1/V_i, V_j_inv = CellP[j].Density/P[j].Mass;
    for(int k=0;k<N_RT_FREQ_BINS;k++) {
        dnET[k] = (CellP[j].Rad_E_gamma_Pred[k]*V_j_inv) * CellP[j].ET[k] - (local->GQuant.Rad_E_gamma[k]*V_i_inv) * local->GQuant.Rad_E_gamma_ET[k];
        dn[k] = CellP[j].Rad_E_gamma_Pred[k]*V_j_inv - local->GQuant.Rad_E_gamma[k]*V_i_inv;
        MINMAX_CHECK(dn[k], out->Minima.Rad_E_gamma[k], out->Maxima.Rad_E_gamma[k]);
    }
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
    double dflux_rt[N_RT_FREQ_BINS][3];
    {int k_f; for(k_f=0;k_f<N_RT_FREQ_BINS;k_f++) {
        int k_d; for(k_d=0;k_d<3;k_d++) {
            dflux_rt[k_f][k_d] = CellP[j].Rad_Flux_Pred[k_f][k_d]*V_j_inv - local->GQuant.Rad_Flux[k_f][k_d]*V_i_inv;
            MINMAX_CHECK(dflux_rt[k_f][k_d], out->Minima.Rad_Flux[k_f][k_d], out->Maxima.Rad_Flux[k_f][k_d]);
        }
    }}
#endif
#endif


    /* ---- Additional operators fitted into the gradients loop ---- */

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || (ADAPTIVE_GRAVSOFT_FORALL & 1)
    if(kernel->r > 0 && local->Mass > 0 && P[j].Mass > 0 && kernel->h_i > 0 && h_j > 0)
    {
        double prefac_ags_a=0.5, prefac_ags_b=0.5, h_a_inv=hinv, h_b_inv=1./h_j, m_b=P[j].Mass;
#if !defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
        if(h_a_inv < h_b_inv) {prefac_ags_a=1; prefac_ags_b=0; h_b_inv=h_a_inv;} else {prefac_ags_a=0; prefac_ags_b=1; h_a_inv=h_b_inv;}
#endif
        if((kernel->r*h_a_inv < 1) && (prefac_ags_a > 0)) {out->AGS_zeta += prefac_ags_a * m_b * kernel_gravity(kernel->r*h_a_inv, h_a_inv, h_a_inv*h_a_inv*h_a_inv, 0);}
    }
#endif

#ifdef HYDRO_SPH
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
    out->alpha_limiter += NV_MYSIGN(CellP[j].NV_DivVel) * P[j].Mass * kernel->wk_i;
#endif
#ifdef MAGNETIC
    {
        double mji_dwk_r = P[j].Mass * kernel->dwk_i / kernel->r;
        double B_dot_dp_i = dot(local->GQuant.B, kernel->dp) * mji_dwk_r;
        out->DtB += B_dot_dp_i * dv;
#ifdef DIVBCLEANING_DEDNER
        out->divB += dot(dB, kernel->dp) * mji_dwk_r;
#endif
    }
#endif
#endif


    /* ---- GRADIENT ACCUMULATION ---- */
    if(kernel->r < kernel->h_i)
    {
        double wk_for_grad = kernel->wk_i;
        if(sph_gradients_flag_i && kernel->r > 0) {wk_for_grad = -kernel->dwk_i/kernel->r * P[j].Mass;}
        for(int k=0;k<3;k++)
        {
            double wk_xyz_i = -wk_for_grad * kernel->dp[k];
            out->Gradients[k].Density += wk_xyz_i * d_rho;
            out->Gradients[k].Pressure += wk_xyz_i * dp_pres;
            out->Gradients[k].Velocity += wk_xyz_i * dv;
#ifdef TURB_DIFF_DYNAMIC
            {Vec3<MyDouble> dv_bar2 = CellP[j].Velocity_bar - local->GQuant.Velocity_bar;
             NGB_SHEARBOX_BOUNDARY_VELCORR_(local->Pos, P[j].Pos, dv_bar2, -1);
             for(int k2=0;k2<3;k2++) {out->Gradients[k].Velocity_bar[k2] += wk_xyz_i * dv_bar2[k2];}}
#endif
#ifdef DOGRAD_INTERNAL_ENERGY
            out->Gradients[k].InternalEnergy += wk_xyz_i * du;
#endif
#ifdef COSMIC_RAY_FLUID
            for(int k2=0;k2<N_CR_PARTICLE_BINS;k2++) {out->Gradients[k].CosmicRayPressure[k2] += wk_xyz_i * dpCR[k2];}
#endif
#ifdef DOGRAD_SOUNDSPEED
            out->Gradients[k].SoundSpeed += wk_xyz_i * dc;
#endif
#ifdef MAGNETIC
            out->Gradients[k].B += wk_xyz_i * dB;
#if defined(DIVBCLEANING_DEDNER) && !defined(MHD_CONSTRAINED_GRADIENT_MIDPOINT)
            {double dphi_loc = CellP[j].PhiPred / P[j].Mass - local->GQuant.Phi;
             out->Gradients[k].Phi += wk_xyz_i * dphi_loc;}
#endif
#endif
#if defined(TURB_DIFF_METALS) && !defined(TURB_DIFF_METALS_LOWORDER)
            for(int k2=0;k2<NUM_METAL_SPECIES;k2++) {out->Gradients[k].Metallicity[k2] += wk_xyz_i * dmetal[k2];}
#endif
#if defined(RT_COMPGRAD_EDDINGTON_TENSOR) && (N_RT_FREQ_BINS > 0)
            for(int k2=0;k2<N_RT_FREQ_BINS;k2++) {
                out->Gradients[k].Rad_E_gamma[k2] += wk_xyz_i * dn[k2];
                out->Gradients[k].Rad_E_gamma_ET[k2] += wk_xyz_i * dnET[k2];
#if defined(RT_M1_SECONDORDER) && defined(RT_EVOLVE_FLUX)
                int k_d; for(k_d=0;k_d<3;k_d++) {out->Gradients[k].Rad_Flux[k2][k_d] += wk_xyz_i * dflux_rt[k2][k_d];}
#endif
            }
#endif
        }
    }
}


#endif /* GRADIENT_FUNCTIONS_H */
