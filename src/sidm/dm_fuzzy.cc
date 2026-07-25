#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"

/*! \file dm_fuzzy.c
 *  \brief routines needed for fuzzy-DM implementation
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef DM_FUZZY



/* do_dm_fuzzy_flux_computation / _old moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION so both the CPU tree-walk and the AGSForce
   GPU kernel can share a single body. */




/* kicks for fuzzy-dm integration: just put relevant drift-kick operators here to keep the code clean
 mode=0 -> 'kick', mode=1 -> 'drift' */
void do_dm_fuzzy_drift_kick(int i, double dt, int mode)
{
    if(mode==0)
    {
        // calculate various energies: quantum potential QP0, 'stored' numerical pressure NQ0, kinetic energy KE0
        double dNQ=P[i].AGS_Dt_Numerical_QuantumPotential*dt, NQ0=P[i].AGS_Numerical_QuantumPotential, NQ1=NQ0+dNQ, KE0=0.5*P[i].Mass*P[i].Vel.norm_sq()*All.cf_a2inv;
        double f00 = 0.5 * All.ScalarField_hbar_over_mass; // this encodes the coefficient with the mass of the particle: units vel*L = hbar / particle_mass
        double d2rho = P[i].AGS_Gradients2_Density[0][0] + P[i].AGS_Gradients2_Density[1][1] + P[i].AGS_Gradients2_Density[2][2]; // laplacian
        double drho2 = P[i].AGS_Gradients_Density.norm_sq();
        double QP0 = (f00*f00 / P[i].AGS_Density) * (d2rho - 0.5*drho2/P[i].AGS_Density); // quantum 'potential'
        NQ1 = DMAX(0,DMAX(NQ1,0.1*NQ0)); NQ1 = DMIN(NQ1,1.1*DMAX(DMAX(KE0+NQ0,fabs(QP0)),KE0+NQ0+QP0)); // limit kick to not produce unphysical energy over-or-under-shoot
        P[i].AGS_Numerical_QuantumPotential = NQ1;
    }

#if (DM_FUZZY > 0) /* if using direct-wavefunction integration methods */
    double vol_inv = P[i].AGS_Density / P[i].Mass;
    if(mode == 0)
    {
        //double psimag_mass_old = (P[i].AGS_Psi_Re*P[i].AGS_Psi_Re + P[i].AGS_Psi_Im*P[i].AGS_Psi_Im) * vol_inv;
        P[i].AGS_Psi_Re += P[i].AGS_Dt_Psi_Re * dt;
        P[i].AGS_Psi_Im += P[i].AGS_Dt_Psi_Im * dt;
        double mass_old = P[i].Mass, dmass = P[i].AGS_Dt_Psi_Mass * dt, mass_new = mass_old + dmass;
        dmass = DMIN(DMAX(dmass,-0.5*mass_old),0.5*mass_old);
        mass_new = mass_old + dmass;
        double psimag_mass_new = (P[i].AGS_Psi_Re*P[i].AGS_Psi_Re + P[i].AGS_Psi_Im*P[i].AGS_Psi_Im) * vol_inv;
#if (DM_FUZZY == 2)
        mass_new = psimag_mass_new; /* uses direct [NON-MASS-CONSERVING] integration of psi field */
#endif
        double psi_corr_fac = sqrt(mass_new / (MIN_REAL_NUMBER + psimag_mass_new));
        P[i].Mass = mass_new; P[i].AGS_Psi_Re *= psi_corr_fac; P[i].AGS_Psi_Im *= psi_corr_fac;

        P[i].AGS_Density = P[i].Mass * vol_inv;
        P[i].AGS_Psi_Re_Pred = P[i].AGS_Psi_Re;
        P[i].AGS_Psi_Im_Pred = P[i].AGS_Psi_Im;
    } else {
        /* in drift mode, AGS_Density should automatically be drifted already by the predictor step, but not the other quantities here */
        P[i].AGS_Psi_Re_Pred += P[i].AGS_Dt_Psi_Re * dt;
        P[i].AGS_Psi_Im_Pred += P[i].AGS_Dt_Psi_Im * dt;
        P[i].AGS_Density *= 1. + DMIN(DMAX(P[i].AGS_Dt_Psi_Mass*dt/P[i].Mass,-0.5),0.5);
    }
#endif
}


/* initialize wavefunction values in the code ICs */
void do_dm_fuzzy_initialization(void)
{
#if (DM_FUZZY > 0)
    int i;
    for(i = 0; i < NumPart; i++)
    {
        double volume = P[i].AGS_Density / P[i].Mass, psimag = sqrt(P[i].AGS_Density), phase = 0;
        /* approximation for initial phase below is fine for slowly-varying k, otherwise not ideal */
        Vec3<double> pos_d = {(double)P[i].Pos[0], (double)P[i].Pos[1], (double)P[i].Pos[2]};
        Vec3<double> vel_d = {(double)P[i].Vel[0], (double)P[i].Vel[1], (double)P[i].Vel[2]};
        phase = dot(pos_d, vel_d) / All.ScalarField_hbar_over_mass;

        P[i].AGS_Psi_Re = psimag * volume * cos(phase); /* remember, we evolve the volume-integrated value of psi */
        P[i].AGS_Psi_Im = psimag * volume * sin(phase);

        P[i].AGS_Dt_Psi_Mass = 0; P[i].AGS_Dt_Psi_Re = 0; P[i].AGS_Dt_Psi_Im = 0;
        P[i].AGS_Psi_Re_Pred = P[i].AGS_Psi_Re; P[i].AGS_Psi_Im_Pred = P[i].AGS_Psi_Im;
    }
#endif
}



/* dm_fuzzy_reconstruct_and_slopelimit{,_sub} moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION. */

#endif // DM_FUZZY
