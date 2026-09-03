#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "damage_porosity_functions.h"

/*
 
 This module contains the relevant module physics for various elastic,
   visco-elastic, and plastic body simulations.
 
 This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 
 */



#ifdef EOS_TILLOTSON

/* routine that defines the Tillotson EOS parameters for various solid and liquid materials */
void tillotson_eos_init(void)
{   // order: parameter a,b,u0,rho0,A,B,u_s,u_s_prime,alpha,beta,elastic shear modulus, hugoniot elastic limit [all cgs] //

    // initialize pre-computed material properties //
    // columns 0-11: a, b, u_0, rho_0, A, B, u_s, u_s_prime, alpha, beta, mu (shear modulus), Y_0 (HEL) [CGS]
    // columns 12-15 (damage/porosity, used only under EOS_DAMAGE_POROSITY):
    //   12 = k_Weibull [1/cm^3], 13 = m_Weibull [-], 14 = mu_DP (Drucker-Prager friction) [-], 15 = alpha_0 (initial distention) [-]
    // columns 16-17 (Jutzi 2008 P-alpha crush thresholds, EOS_DAMAGE_POROSITY bit 2):
    //   16 = P_e [dyn/cm^2] (elastic limit), 17 = P_s [dyn/cm^2] (solid/full-compaction pressure)
    double qtmp[6][18]={
        {0.5,1.3,1.60e11,2.700,1.80e11,1.80e11,3.50e10,1.80e11,5.0,5.0,2.17e11,3.8e10, 4.0e29, 9.5, 1.0, 1.0, 3.0e8, 1.5e10},  // granite
        {0.5,1.5,4.87e12,2.700,2.67e11,2.67e11,4.72e10,1.82e11,5.0,5.0,2.27e11,3.5e10, 4.0e29, 9.5, 1.0, 1.0, 1.0e8, 7.0e10},  // basalt
        {0.5,1.5,9.50e10,7.860,1.28e12,1.05e12,1.42e10,8.45e10,5.0,5.0,7.75e11,8.5e10, 1.0e25, 9.0, 1.0, 1.0, 1.0e10,1.0e12},  // iron
        {0.3,0.1,1.00e11,0.917,9.47e10,9.47e10,7.730e9,3.04e10,10.,5.0,2.80e10,1.0e10, 1.4e32, 9.1, 1.0, 1.0, 1.0e7, 1.5e9 },  // ice
        {0.5,1.4,5.50e12,3.500,1.31e12,4.90e11,4.50e10,1.50e11,5.0,5.0,8.13e11,9.0e10, 4.0e29, 9.5, 1.0, 1.0, 2.0e8, 1.0e10},  // olivine/dunite
        {0.5,0.9,2.00e10,1.000,2.00e11,1.00e11,4.000e9,2.00e10,5.0,5.0,1.000e0,1.00e0, 0.0,    1.0, 0.0, 1.0, 0.0,   1.0e10}}; // water
    int j_t,k_t;
    for(j_t=1;j_t<7;j_t++)
    {
        for(k_t=0;k_t<18;k_t++)
        {
            All.Tillotson_EOS_params[j_t][k_t] = qtmp[j_t-1][k_t];
            if((k_t==2)||(k_t==6)||(k_t==7)) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_SPECEGY_IN_CGS;}
            if(k_t==3) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_DENSITY_IN_CGS;}
            if((k_t==4)||(k_t==5)||(k_t==10)||(k_t==11)||(k_t==16)||(k_t==17)) {All.Tillotson_EOS_params[j_t][k_t] /= UNIT_PRESSURE_IN_CGS;}
            // slots 12-15 (damage/porosity dimensionless except k_Weibull) intentionally not converted here; k_Weibull conversion is applied where consumed
        }
    }
    return;
}


#endif



/* elastic_body_update_driftkick: body is elastic_body_update_driftkick_P in
   elastic_physics_functions.h; the host wrapper is below the include block. */

#if defined(EOS_ELASTIC) || defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* get_negative_pressure_tensilecorrfac body moved to elastic_physics_functions.h
 * as KOKKOS_INLINE_FUNCTION so the hydro pair body (a __host__ __device__
 * function in the corridor hot path) can call it from the device pass without
 * the nvc++ #20011-D silent-physics warning. This block re-includes the header
 * with empty KOKKOS_INLINE_FUNCTION to emit the non-inline host external
 * symbol for any caller resolved via the core/proto.h forward declaration. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "elastic_physics_functions.h"
#endif


#ifdef EOS_ELASTIC
/* routine to update the deviatoric stress tensor. Sits after the include block
   above so that block stays the FIRST inclusion of the header in this file and
   still emits the non-inline host symbols; an earlier include would make it a
   #pragma once no-op. */
void elastic_body_update_driftkick(int i, double dt_entr, int mode)
{
    elastic_body_update_driftkick_P(i, dt_entr, mode, P, CellP);
}
#endif
