#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/*! \file cbe_integrator.c
 *  \brief routines needed for CBE integrator implementation
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef CBE_INTEGRATOR



// moment ordering convention: 0, x, y, z, xx, yy, zz, xy, xz, yz
//                             0, 1, 2, 3,  4,  5,  6,  7,  8,  9


/* variable initialization */
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




/* drift-kick updates to distribution functions */
// we evolve conserved quantities directly (in physical units): minor conversion in fluxes required later //
void do_cbe_drift_kick(int i, double dt)
{
    int j, k;
    double moment[CBE_INTEGRATOR_NMOMENTS]={0}, dmoment[CBE_INTEGRATOR_NMOMENTS]={0}, minv=1./P[i].Mass;
    Vec3<double> v0 = {};
    // evaluate total fluxes //
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        {
            moment[k] += P[i].CBE_basis_moments[j][k];
            dmoment[k] += dt*P[i].CBE_basis_moments_dt[j][k];
        }
    }
    // define the current velocity, force-sync update to match it //
    v0 = P[i].Vel / All.cf_atime; // physical units //
    double biggest_dm = 1.e10;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        double q = (dt*P[i].CBE_basis_moments_dt[j][0] - P[i].CBE_basis_moments[j][0]*minv*dmoment[0]) / (P[i].CBE_basis_moments[j][0] * (1.+minv*dmoment[0]));
        if(!isnan(q)) {if(q < biggest_dm) {biggest_dm=q;}}
    }
    double nfac = 1; // normalization factor for fluxes below //
    double threshold_dm;
    threshold_dm = -0.75; // maximum allowed fractional change in m //
    if(biggest_dm < threshold_dm) {nfac = threshold_dm/biggest_dm;} // re-normalize flux so it doesn't overshoot //
    // ok now do the actual update //
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        P[i].CBE_basis_moments[j][0] += nfac * (dt*P[i].CBE_basis_moments_dt[j][0] - P[i].CBE_basis_moments[j][0]*minv*dmoment[0]); // update mass (strictly ensuring total mass matches updated particle)
        for(k=1;k<4;k++) {P[i].CBE_basis_moments[j][k] += nfac * (dt*P[i].CBE_basis_moments_dt[j][k] - P[i].CBE_basis_moments[j][0]*minv*dmoment[k]);} // update momentum (strictly ensuring total momentum matches updated particle)
#if (CBE_INTEGRATOR_NMOMENTS > 4)
        {
            // second moments need some checking //
            for(k=4;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments[j][k] += nfac * (dt*P[i].CBE_basis_moments_dt[j][k]);} // pure dispersion, no re-normalization here
        }

        // now a series of checks for ensuring the second-moments retain positive-definite higher-order moments (positive-definite determinants, etc)
        double eps_tmp = 1.e-8;
        for(k=4;k<7;k++) {if(P[i].CBE_basis_moments[j][k] < MIN_REAL_NUMBER) {P[i].CBE_basis_moments[j][k]=MIN_REAL_NUMBER;}}
        double xyMax = sqrt(P[i].CBE_basis_moments[j][4]*P[i].CBE_basis_moments[j][5]) * (1.-eps_tmp); // xy < sqrt[xx*yy]
        double xzMax = sqrt(P[i].CBE_basis_moments[j][4]*P[i].CBE_basis_moments[j][6]) * (1.-eps_tmp); // xz < sqrt[xx*zz]
        double yzMax = sqrt(P[i].CBE_basis_moments[j][5]*P[i].CBE_basis_moments[j][6]) * (1.-eps_tmp); // yz < sqrt[yy*zz]
        if(P[i].CBE_basis_moments[j][7] > xyMax) {P[i].CBE_basis_moments[j][7] = xyMax;}
        if(P[i].CBE_basis_moments[j][8] > xzMax) {P[i].CBE_basis_moments[j][8] = xzMax;}
        if(P[i].CBE_basis_moments[j][9] > yzMax) {P[i].CBE_basis_moments[j][9] = yzMax;}
        if(P[i].CBE_basis_moments[j][7] < -xyMax) {P[i].CBE_basis_moments[j][7] = -xyMax;}
        if(P[i].CBE_basis_moments[j][8] < -xzMax) {P[i].CBE_basis_moments[j][8] = -xzMax;}
        if(P[i].CBE_basis_moments[j][9] < -yzMax) {P[i].CBE_basis_moments[j][9] = -yzMax;}
        double crossnorm = 1;
        double detSMatrix_Diag = P[i].CBE_basis_moments[j][4]*P[i].CBE_basis_moments[j][5]*P[i].CBE_basis_moments[j][6];
        double detSMatrix_Cross = 2.*P[i].CBE_basis_moments[j][7]*P[i].CBE_basis_moments[j][8]*P[i].CBE_basis_moments[j][9]
        - (  P[i].CBE_basis_moments[j][4]*P[i].CBE_basis_moments[j][9]*P[i].CBE_basis_moments[j][9]
           + P[i].CBE_basis_moments[j][5]*P[i].CBE_basis_moments[j][8]*P[i].CBE_basis_moments[j][8]
           + P[i].CBE_basis_moments[j][6]*P[i].CBE_basis_moments[j][7]*P[i].CBE_basis_moments[j][7] );
        if(detSMatrix_Diag <= 0)
        {
            crossnorm=0; for(k=4;k<7;k++) {if(P[i].CBE_basis_moments[j][k] < MIN_REAL_NUMBER) {P[i].CBE_basis_moments[j][k]=MIN_REAL_NUMBER;}}
        } else {
            if(detSMatrix_Diag + detSMatrix_Cross <= 0)
            {
                double crossmin = -detSMatrix_Diag * (1.-eps_tmp);
                crossnorm = crossmin / detSMatrix_Cross;
            }
        }
        if(crossnorm < 1) {for(k=7;k<10;k++) {P[i].CBE_basis_moments[j][k] *= crossnorm;}}

// simplify to 1D dispersion along direction of motion
if(2==2)
{
double S0 = P[i].CBE_basis_moments[j][4]+P[i].CBE_basis_moments[j][5]+P[i].CBE_basis_moments[j][6], vhat[3]={0}, vmag=0;
for(k=0;k<3;k++) {vhat[k] = P[i].CBE_basis_moments[j][k+1]; vmag += vhat[k]*vhat[k];}
if(vmag > 0)
{
vmag = 1./sqrt(vmag); for(k=0;k<3;k++) {vhat[k] *= vmag;}
P[i].CBE_basis_moments[j][4] = S0*vhat[0]*vhat[0]; P[i].CBE_basis_moments[j][5] = S0*vhat[1]*vhat[1];
P[i].CBE_basis_moments[j][6] = S0*vhat[2]*vhat[2]; P[i].CBE_basis_moments[j][7] = S0*vhat[0]*vhat[1];
P[i].CBE_basis_moments[j][8] = S0*vhat[0]*vhat[2]; P[i].CBE_basis_moments[j][9] = S0*vhat[1]*vhat[2];
}
}

#endif
    }
    
    /* need to deal with cases where one of the basis functions becomes extremely small --
        below simply takes the biggest and splits it (with small perturbation to velocities
        for degeneracy-breaking purposes) */
    double mmax=-1, mmin=1.e10*P[i].Mass; int jmin=-1,jmax=-1;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        double m=P[i].CBE_basis_moments[j][0];
        if(m<mmin) {mmin=m; jmin=j;}
        if(m>mmax) {mmax=m; jmax=j;}
    }
    if((mmin < 1.e-5 * mmax) && (jmin >= 0) && (jmax >= 0) && (All.Time > All.TimeBegin))
    {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        {
            double dq = 0.5*P[i].CBE_basis_moments[jmax][k];
            if(k>0 && k<4) {dq *= 1. + 0.001*(get_random_number(ThisTask+i+32*jmax+12427*k)-0.5);}
            P[i].CBE_basis_moments[jmax][k] -= dq;
            P[i].CBE_basis_moments[jmin][k] += dq; // since we're just splitting mass, this is ok, since these are all mass-weighted quantities
        }
    }
    
    return;
}






/* do_cbe_flux_computation moved to sidm/cbe_integrator_functions.h as a
   KOKKOS_INLINE_FUNCTION so both the CPU tree-walk and the B2 AGSForce
   GPU kernel can call the same body. */




/* this routine contains operations which are needed after the main forcetree-walk loop (where the CBE integration terms are calculated).
     we shift the net momentum flux into the GravAccel vector so that the tree and everything else behaves correctly, velocities
     drift, etc, all as they should. The 'residual' terms are then saved, which can kicked separately from the main particle kick.
*/
void do_postgravity_cbe_calcs(int i)
{
    int j,k; double dmom_tot[CBE_INTEGRATOR_NMOMENTS]={0}, m_inv = 1./P[i].Mass;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {dmom_tot[k] += P[i].CBE_basis_moments_dt[j][k];}} // total change for each moment
    /* total momentum flux will be transferred */
    Vec3<double> dv0 = {m_inv * dmom_tot[1], m_inv * dmom_tot[2], m_inv * dmom_tot[3]}; // total acceleration
    P[i].GravAccel += dv0 / All.cf_a2inv; // write as gravitational acceleration, convert to cosmological units // currently incompatible with hermite integrator -- need to update to Other_Accel
    // now need to add that shift back into the momentum-change terms //
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        P[i].CBE_basis_moments_dt[j][0] -= P[i].CBE_basis_moments[j][0]*(m_inv * dmom_tot[0]); // re-ensure that this is zero to floating-point precision (should be, we are just eliminating summed FP errors here) //
        for(k=0;k<3;k++) {P[i].CBE_basis_moments_dt[j][k+1] -= P[i].CBE_basis_moments[j][0]*dv0[k];} // shift the momentum flux, so now zero net 'residual' momentum flux (should be, we are just eliminating summed FP errors here) //
#if (CBE_INTEGRATOR_NMOMENTS > 4)
        {
            // first, shift the second-moment derivatives so they are dS, not dT, where T = S + v.v (outer product v.v),
            //   so dS = dT - (dv.v + v.dv) = dT - (dv.v + transpose[dv.v])
            double dS[6]={0};
            dS[0] = P[i].CBE_basis_moments_dt[j][4] - m_inv * (P[i].CBE_basis_moments_dt[j][1]*P[i].CBE_basis_moments[j][1] + P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments_dt[j][1]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments[j][1]; // xx
            dS[1] = P[i].CBE_basis_moments_dt[j][5] - m_inv * (P[i].CBE_basis_moments_dt[j][2]*P[i].CBE_basis_moments[j][2] + P[i].CBE_basis_moments[j][2]*P[i].CBE_basis_moments_dt[j][2]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][2]*P[i].CBE_basis_moments[j][2]; // yy
            dS[2] = P[i].CBE_basis_moments_dt[j][6] - m_inv * (P[i].CBE_basis_moments_dt[j][3]*P[i].CBE_basis_moments[j][3] + P[i].CBE_basis_moments[j][3]*P[i].CBE_basis_moments_dt[j][3]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][3]*P[i].CBE_basis_moments[j][3]; // zz
            dS[3] = P[i].CBE_basis_moments_dt[j][7] - m_inv * (P[i].CBE_basis_moments_dt[j][1]*P[i].CBE_basis_moments[j][2] + P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments_dt[j][2]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments[j][2]; // xy
            dS[4] = P[i].CBE_basis_moments_dt[j][8] - m_inv * (P[i].CBE_basis_moments_dt[j][1]*P[i].CBE_basis_moments[j][3] + P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments_dt[j][3]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][1]*P[i].CBE_basis_moments[j][3]; // xz
            dS[5] = P[i].CBE_basis_moments_dt[j][9] - m_inv * (P[i].CBE_basis_moments_dt[j][2]*P[i].CBE_basis_moments[j][3] + P[i].CBE_basis_moments[j][2]*P[i].CBE_basis_moments_dt[j][3]) + m_inv*m_inv * P[i].CBE_basis_moments_dt[j][0] * P[i].CBE_basis_moments[j][2]*P[i].CBE_basis_moments[j][3]; // yz
            if(P[i].CBE_basis_moments_dt[j][0]>0) {for(k=0;k<3;k++) {dS[k]=DMAX(dS[k],0.);}}
            for(k=4;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments_dt[j][k] = dS[k-4];} // set the flux of the stress terms to the change in the dispersion alone //
        }
#endif
    } // for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    return;
}


#endif
