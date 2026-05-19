#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/*! \file cbe_integrator.cc
 *  \brief startup initialization for the CBE integrator basis moments.
 *         The per-step drift-kick (do_cbe_drift_kick_kernel) and the per-
 *         active post-gravity finalization (do_cbe_postgravity_kernel) now
 *         live in cbe_integrator_functions.h as KOKKOS_INLINE_FUNCTION
 *         kernels, dispatched via sidm/cbe_integrator_gpu.cc.
 *
 *         Per-pair CBE flux (cbe_integrator_flux_compute_pair) is invoked
 *         from the AGSForce iterative neighbor loop (gravity/ags_force_loop.h).
 *
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef CBE_INTEGRATOR


// moment ordering convention: 0, x, y, z, xx, yy, zz, xy, xz, yz
//                             0, 1, 2, 3,  4,  5,  6,  7,  8,  9


/* variable initialization (called once from core/init.cc) */
void do_cbe_initialization(void)
{
    int i,j,k;
    for(i=0;i<NumPart;i++)
    {
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments_dt[j][k]=0;}} // no time derivatives //
        double v2=0, v0=0;
        v2 = P[i].Vel.norm_sq();
        if(v2>0) {v0=sqrt(v2);} else {v0=1.e-10;}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
        {
            for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
            {
                // zeros will be problematic, instead initialize a random distribution //
if(j > 1)
{

                if(k==0) {P[i].CBE_basis_moments[j][0] = 1.e-5 * P[i].Mass * (0.5 + 0.01 + get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j));}
                if((k>0)&&(k<4)) {P[i].CBE_basis_moments[j][k] = P[i].CBE_basis_moments[j][0] * 1.e-8 * (0*2.*P[i].Vel[k]*(get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j + 2)-0.5) + 1.e-5*v0*(get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j + 2)-0.5));}
                if(k>=4 && k<7) {P[i].CBE_basis_moments[j][k] = P[i].CBE_basis_moments[j][0] * 1.e-15;} //(P[i].Vel[k]*P[i].Vel[k] + 1.e-2*v0*v0 + 1.e-3*1.e-3);}
                if(k>=7) {P[i].CBE_basis_moments[j][k] = 0;}

} else {

 P[i].CBE_basis_moments[0][0] = 0.5*P[i].Mass;
 P[i].CBE_basis_moments[1][0] = 0.5*P[i].Mass;
 P[i].CBE_basis_moments[0][1] =  1.0*P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[1][1] = -1.0*P[i].CBE_basis_moments[1][0];
 P[i].CBE_basis_moments[0][2] = P[i].CBE_basis_moments[0][3] = 0;
 P[i].CBE_basis_moments[1][2] = P[i].CBE_basis_moments[1][3] = 0;
#if (CBE_INTEGRATOR_NMOMENTS > 4)
 P[i].CBE_basis_moments[0][4] = P[i].CBE_basis_moments[1][4] = 0.5 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][5] = P[i].CBE_basis_moments[1][5] = 0.1 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][6] = P[i].CBE_basis_moments[1][6] = 0.1 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][7] = P[i].CBE_basis_moments[1][7] = 0.0 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][8] = P[i].CBE_basis_moments[1][8] = 0.0 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][9] = P[i].CBE_basis_moments[1][9] = 0.0 * P[i].CBE_basis_moments[0][0];
#endif

}

#if (NUMDIMS==1)
                if((k!=0)&&(k!=1)&&(k!=4)) {P[i].CBE_basis_moments[j][k] = 0;}
#endif
#if (NUMDIMS==2)
                if((k==3)||(k==6)||(k==8)||(k==9)) {P[i].CBE_basis_moments[j][k] = 0;}
#endif
            }
        }
        double mom_tot[CBE_INTEGRATOR_NMOMENTS]={0};
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]+=P[i].CBE_basis_moments[j][k];}}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments[j][k] *= P[i].Mass / mom_tot[0];}}
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]=0;}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]+=P[i].CBE_basis_moments[j][k];}}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=1;k<4;k++) {P[i].CBE_basis_moments[j][k] += P[i].CBE_basis_moments[j][0]*(P[i].Vel[k-1]-mom_tot[k]/P[i].Mass);}}
    }
    return;
}


#endif
