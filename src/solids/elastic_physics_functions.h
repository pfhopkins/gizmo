/* elastic_physics_functions.h — KOKKOS_INLINE_FUNCTION inline body for
 * get_negative_pressure_tensilecorrfac, the tensile-instability kernel
 * correction factor used by the hydro pair body under solid-EOS Configs
 * (EOS_TILLOTSON / EOS_ELASTIC / EOS_ANEOS).
 *
 * Why this header: hydro_functions.h:hydro_accumulate_neighbor (a
 * __host__ __device__ pair body, in the corridor hot path) calls this
 * function on the device pass. Without an inline body visible at the call
 * site, the nvc++ device pass sees only the host-only forward declaration
 * in core/proto.h and emits warning #20011-D ("calling a __host__ function
 * from a __host__ __device__ function is not allowed"). That is a silent
 * physics error on GPU.
 *
 * elastic_physics.cc continues to provide the non-inline host external
 * symbol via the standard #undef KOKKOS_INLINE_FUNCTION / re-include
 * pattern; other host call sites (proto.h forward-declared) link against
 * that one strong symbol.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Requires the caller to have already included mesh/kernel.h (provides
 * kernel_main) and declarations/allvars.h (provides All.DesNumNgb).
 * NOT included here because hydro_functions.h has already included it via its
 * own include chain, so this header does not need to repeat it. (kernel.h has
 * carried a #pragma once since the drift-helper work; it is safe either way.) */


#ifdef EOS_ELASTIC
/* apply_drucker_prager / damage_update_grady_kipp, used by the stress update below
   under the EOS_DAMAGE_POROSITY bits. Safe to include here: it carries an include
   guard and does not pull in Kokkos. */
#include "damage_porosity_functions.h"

/* routine to update the deviatoric stress tensor */
KOKKOS_INLINE_FUNCTION
void elastic_body_update_driftkick_P(int i, double dt_entr, int mode,
                                     struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k,l,NDim=NUMDIMS;
    double dv0[3][3], R[3][3], S[3][3], S_new[3][3], dS=0, mu, Y0, J2=0, I1=0;
#ifdef EOS_TILLOTSON
    mu = All.Tillotson_EOS_params[cell[i].CompositionType][10]; Y0 = All.Tillotson_EOS_params[cell[i].CompositionType][11]; // set for composition
#else
    mu = All.Tillotson_EOS_params[0][10]; Y0 = All.Tillotson_EOS_params[0][11];  // set to universal constants
#endif

    if(mode < 2) // drift or kick operation
    {
        for(j=0;j<NDim;j++) {
            for(k=0;k<NDim;k++) {
                // determine which variable we are updating (mode=0/1 is kick/drift)
                if(mode==0) {S_new[j][k]=cell[i].Elastic_Stress_Tensor[j][k];} else {S_new[j][k]=cell[i].Elastic_Stress_Tensor_Pred[j][k];}
                S_new[j][k] += dt_entr * cell[i].Dt_Elastic_Stress_Tensor[j][k]; // apply time evolution
                if(k==j) {I1 += S_new[j][k];} // first invariant of the tensor
                J2 += 0.5 * S_new[j][k]*S_new[j][k]; // second invariant of the tensor
            }}
        // now apply the yield criterion (von Mises by default; Drucker-Prager
        // extension Y_eff = Y0 + mu_DP * P_hydro under EOS_DAMAGE_POROSITY bit 1)
        double Y_eff = Y0;
#if defined(EOS_DAMAGE_POROSITY) && DAMAGE_POROSITY_BIT_DRUCKER_PRAGER
        Y_eff = apply_drucker_prager(Y0, cell[i].Pressure, cell[i].CompositionType);
#endif
        if(J2 > 0)
        {
            double f_Y = Y_eff*Y_eff/(NDim*J2);
            if(f_Y < 1) {for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {S_new[j][k] *= f_Y;}}}
        }
#if defined(EOS_DAMAGE_POROSITY) && DAMAGE_POROSITY_BIT_GRADY_KIPP
        /* bit 0: Grady-Kipp damage update at kick (mode==0) only */
        if(mode == 0)
        {
            double D_new, A_new;
            damage_update_grady_kipp(cell[i].Damage, cell[i].ActiveCracks,
                                     J2, mu, cell[i].SoundSpeed, dt_entr,
                                     cell[i].CompositionType,
                                     &D_new, &A_new);
            cell[i].Damage       = (MyFloat)D_new;
            cell[i].ActiveCracks = (MyFloat)A_new;
        }
#endif
        // write out to variable //
        for(j=0;j<NDim;j++) {
            for(k=0;k<NDim;k++) {
                if(mode==0) {cell[i].Elastic_Stress_Tensor[j][k]=S_new[j][k];} else {cell[i].Elastic_Stress_Tensor_Pred[j][k]=S_new[j][k];}
            }}

    } else {

        // ok all below is for mode = 2, which is the actual calculation of the time derivative of the stress tensor
        for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {dv0[j][k] = cell[i].Gradients.Velocity[j][k];}}
        for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {S[j][k] = cell[i].Elastic_Stress_Tensor_Pred[j][k];}}
        for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {R[j][k] = 0.5*(dv0[j][k] - dv0[k][j]);}}
        double trace_vel=0; for(j=0;j<NDim;j++) {trace_vel += dv0[j][j];}
        for(j=0;j<NDim;j++) // velocity index
        {
            for(k=0;k<NDim;k++) // gradient index
            {
                dS = mu * (dv0[j][k] + dv0[k][j]); // symmetric strain component
                if(k==j) {dS -= 2.*mu*trace_vel/NDim;} // trace component
                for(l=0;l<NDim;l++) {dS += S[j][l]*R[l][k] - R[j][l]*S[l][k];} // rotation components
                cell[i].Dt_Elastic_Stress_Tensor[j][k] = dS; // save it to variable
            }
        }

    }
    
    return; // all done here
}
#endif /* EOS_ELASTIC */

#if defined(EOS_ELASTIC) || defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* routine to get and define the correction factor needed to prevent tensile instability for negative pressures, for arbitrary kernels & dimensions */
KOKKOS_INLINE_FUNCTION
double get_negative_pressure_tensilecorrfac(double r, double h_i, double h_j)
{
    double dx_ips=0, wk_0=0, dwk_tmp=0, wk_r=0, r_over_heff=0;
#if (NUMDIMS==1)
    dx_ips = 2. / All.DesNumNgb; // 1D inter-node separation for desired NNgb, relative to radius of compact support
#elif(NUMDIMS==2)
    dx_ips = sqrt(M_PI / All.DesNumNgb); // 2D inter-node separation for desired NNgb, relative to radius of compact support
#else
    dx_ips = pow(4.*M_PI/3. / All.DesNumNgb, 1./3.); // 3D inter-node separation for desired NNgb, relative to radius of compact support
#endif
    kernel_main(dx_ips, 1., 1., &wk_0, &dwk_tmp, -1); // use kernels because of their stability properties: here weight for 'mean separation'
    r_over_heff = r / DMAX(h_i, h_j);
    kernel_main(r_over_heff, 1., 1., &wk_r, &dwk_tmp, -1); // here weight for actual half-separation
    return 0.2 * pow(wk_r / wk_0, 4); // correction factor for n=4 from Monaghan et al. 2000, Gray et al. 2001
}
#endif
