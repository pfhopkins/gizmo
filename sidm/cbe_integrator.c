#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"


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
        for(k=0;k<3;k++) {v2+=P[i].Vel[k]*P[i].Vel[k];}
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
    double moment[CBE_INTEGRATOR_NMOMENTS]={0}, dmoment[CBE_INTEGRATOR_NMOMENTS]={0}, minv=1./P[i].Mass, v0[3]={0};
    // evaluate total fluxes //
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    {
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
        {
            moment[k] += P[i].CBE_basis_moments[j][k];
            dmoment[k] += dt*P[i].CBE_basis_moments_dt[j][k];
        }
    }
#ifdef CBE_DEBUG
    // total mass flux should vanish identically -- check this //
    if(ThisTask==0) {printf("Total mass flux == %g (Task=%d i=%d)\n",dmoment[0],ThisTask,i);}
    if(ThisTask==0) {printf("Momentum flux == %g/%g/%g (Task=%d i=%d)\n",dmoment[1],dmoment[2],dmoment[3],ThisTask,i);}
    if(fabs(minv*dmoment[0]) > 0.05) {printf("MF=%g/%g/%g.",minv*dt*P[i].CBE_basis_moments_dt[0][0],minv*dt*P[i].CBE_basis_moments_dt[1][0],minv*dmoment[0]); endrun(91918282);}
#endif
    // define the current velocity, force-sync update to match it //
    for(k=0;k<3;k++) {v0[k] = P[i].Vel[k] / All.cf_atime;} // physical units //
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
#ifdef CBE_DEBUG
            // check against negatives //
            if(P[i].CBE_basis_moments[j][0]<0 || P[i].CBE_basis_moments[j][4]<0 || P[i].CBE_basis_moments[j][5]<0 || P[i].CBE_basis_moments[j][6]<0)
            {
                printf("ZZa: Task=%d i=%d j=%d   : m=%g v=%g/%g/%g Sxx,yy,zz=%g/%g/%g Sxy,xz,yz=%g/%g/%g \n",ThisTask,i,j,
                       P[i].CBE_basis_moments[j][0],P[i].CBE_basis_moments[j][1],P[i].CBE_basis_moments[j][2],P[i].CBE_basis_moments[j][3],
                       P[i].CBE_basis_moments[j][4],P[i].CBE_basis_moments[j][5],P[i].CBE_basis_moments[j][6],P[i].CBE_basis_moments[j][7],
                       P[i].CBE_basis_moments[j][8],P[i].CBE_basis_moments[j][9]);
                printf("ZZb: Task=%d i=%d j=%d dt: m=%g v=%g/%g/%g Sxx,yy,zz=%g/%g/%g Sxy,xz,yz=%g/%g/%g \n",ThisTask,i,j,
                       P[i].CBE_basis_moments_dt[j][0],P[i].CBE_basis_moments_dt[j][1],P[i].CBE_basis_moments_dt[j][2],P[i].CBE_basis_moments_dt[j][3],
                       P[i].CBE_basis_moments_dt[j][4],P[i].CBE_basis_moments_dt[j][5],P[i].CBE_basis_moments_dt[j][6],P[i].CBE_basis_moments_dt[j][7],
                       P[i].CBE_basis_moments_dt[j][8],P[i].CBE_basis_moments_dt[j][9]);
                fflush(stdout);
            }
#endif
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






/* this computes the actual single-sided fluxes at the face, integrating over a distribution function to use the moments */
double do_cbe_flux_computation(double moments[CBE_INTEGRATOR_NMOMENTS], double vface_dot_A, double vface[3], double Area[3], double moments_ngb[CBE_INTEGRATOR_NMOMENTS], double fluxes[CBE_INTEGRATOR_NMOMENTS])
{
    // couple dot-products must be pre-computed for fluxes //
    double m_inv = 1. / moments[0]; // need for weighting, below [e.g. v_x = moments[1] / moments[0]]
    double v[3], f00_vsig=1; v[0] = m_inv*moments[1]; v[1] = m_inv*moments[2]; v[2] = m_inv*moments[3]; // get velocities
    double vsig = v[0]*Area[0] + v[1]*Area[1] + v[2]*Area[2] - vface_dot_A; // v_alpha . A_face
    if(fabs(vsig) <= 0) return 0; // trap for invalid velocity
    fluxes[0] = vsig * moments[0]; // calculate and assign mass flux
    int k; for(k=1;k<CBE_INTEGRATOR_NMOMENTS;k++) {fluxes[k] =  (m_inv * moments[k]) * fluxes[0];} // specific flux carried just by mass flux

    // now need to deal with the (more complicated) stress/second-moment terms //
#if (CBE_INTEGRATOR_NMOMENTS > 4)
    {
        double dv2 = (v[0]-vface[0])*(v[0]-vface[0]) + (v[1]-vface[1])*(v[1]-vface[1]) + (v[2]-vface[2])*(v[2]-vface[2]); // magnitude of velocity in face frame
        double c_eff_over_vsig_A = sqrt(m_inv*(moments[4]+moments[5]+moments[6]) / dv2); // ratio of c_eff [effective 'signal velocity' in dotted into face direction from stress tensor] to bulk velocity dotted same
        double SM_vsig = 1 + c_eff_over_vsig_A*c_eff_over_vsig_A/(1 + c_eff_over_vsig_A); // star-state wavespeed [from HLLC-type Reimann solver] over v_effective
        double f00_SdotA = 1 - c_eff_over_vsig_A/(SM_vsig + c_eff_over_vsig_A); // groups numerical diffusivity terms that apply to the stress tensor terms below
        f00_vsig = (SM_vsig * (1 + c_eff_over_vsig_A)) / (SM_vsig + c_eff_over_vsig_A); // groups numerical diffusivity terms that apply to the mass flux
        /*
        double S0 = m_inv * (moments[4]+moments[5]+moments[6]); // trace[S] = total velocity dispersion //
        double ANorm=sqrt(Area[0]*Area[0]+Area[1]*Area[1]+Area[2]*Area[2]), vsig_mag=fabs(vsig)/ANorm, cos_vsig_A=vsig_mag/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); // get dot product of velocity and face
        double c_eff = sqrt(S0) * cos_vsig_A; // effective 'signal velocity' in dotted into face direction from stress tensor, for numerical dissipation
        double S_M = vsig_mag + c_eff*c_eff/(vsig_mag + c_eff); // star-state wavespeed [from HLLC-type Reimann solver]
        double f00_SdotA = 1 - c_eff/(c_eff + S_M); // groups numerical diffusivity terms that apply to the stress tensor terms below
        f00_vsig = (S_M * (v0_mag + c_eff)) / (v0_mag * (S_M + c_eff)); // groups numerical diffusivity terms that apply to the mass flux
        */
        /*
        double S[6]; // dispersion part of stress tensor (need to subtract mean-v parts if not doing so in pre-step)
        // note that we are actually evolving S, although we will compute the -flux- of T, the conserved quantity //
        S[0] = m_inv*moments[4];// - v[0]*v[0]; // xx
        S[1] = m_inv*moments[5];// - v[1]*v[1]; // yy
        S[2] = m_inv*moments[6];// - v[2]*v[2]; // zz
        S[3] = m_inv*moments[7];// - v[0]*v[1]; // xy
        S[4] = m_inv*moments[8];// - v[0]*v[2]; // xz
        S[5] = m_inv*moments[9];// - v[1]*v[2]; // yz
        */
        double S_dot_A[3]; // stress tensor dotted into face. note our moments 4-9 are the -dispersions- not T (otherwise need to convert here)
        S_dot_A[0] = f00_SdotA * (moments[4]*Area[0] + moments[7]*Area[1] + moments[8]*Area[2]); // (S_alpha . A_face)_x * mass
        S_dot_A[1] = f00_SdotA * (moments[7]*Area[0] + moments[5]*Area[1] + moments[9]*Area[2]); // (S_alpha . A_face)_y * mass
        S_dot_A[2] = f00_SdotA * (moments[8]*Area[0] + moments[9]*Area[1] + moments[6]*Area[2]); // (S_alpha . A_face)_z * mass
        //S_dot_A[0]=S_dot_A[1]=S_dot_A[2]=0;//
        fluxes[1] += S_dot_A[0]; // add momentum flux from stress tensor - x
        fluxes[2] += S_dot_A[1]; // add momentum flux from stress tensor - y
        fluxes[3] += S_dot_A[2]; // add momentum flux from stress tensor - z
        fluxes[4] += 2.*v[0]*S_dot_A[0] + fluxes[0]*v[0]*v[0]; // add stress flux from stress tensor -- xx
        fluxes[5] += 2.*v[1]*S_dot_A[1] + fluxes[0]*v[1]*v[1]; // add stress flux from stress tensor -- yy
        fluxes[6] += 2.*v[2]*S_dot_A[2] + fluxes[0]*v[2]*v[2]; // add stress flux from stress tensor -- zz
        fluxes[7] += v[0]*S_dot_A[1] + v[1]*S_dot_A[0] + fluxes[0]*v[0]*v[1]; // add stress flux from stress tensor -- xy
        fluxes[8] += v[0]*S_dot_A[2] + v[2]*S_dot_A[0] + fluxes[0]*v[0]*v[2]; // add stress flux from stress tensor -- xz
        fluxes[9] += v[1]*S_dot_A[2] + v[2]*S_dot_A[1] + fluxes[0]*v[1]*v[2]; // add stress flux from stress tensor -- yz
    }
#endif
    return vsig * f00_vsig;
}




/* this routine contains operations which are needed after the main forcetree-walk loop (where the CBE integration terms are calculated).
     we shift the net momentum flux into the GravAccel vector so that the tree and everything else behaves correctly, velocities
     drift, etc, all as they should. The 'residual' terms are then saved, which can kicked separately from the main particle kick.
*/
void do_postgravity_cbe_calcs(int i)
{
    int j,k; double dmom_tot[CBE_INTEGRATOR_NMOMENTS]={0}, m_inv = 1./P[i].Mass;
    for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {dmom_tot[k] += P[i].CBE_basis_moments_dt[j][k];}} // total change for each moment
#ifdef CBE_DEBUG
#if (CBE_INTEGRATOR_NMOMENTS > 4)
    if(ThisTask==0) {printf("P[i].CBE_basis_moments=%g/%g/%g/%g \n",P[i].CBE_basis_moments[0][3],P[i].CBE_basis_moments[2][7],P[i].CBE_basis_moments[1][5],P[i].CBE_basis_moments[0][8]);}
    if(ThisTask==0) {printf("P[i].CBE_basis_moments_dt=%g/%g/%g/%g \n",P[i].CBE_basis_moments_dt[0][3],P[i].CBE_basis_moments_dt[2][7],P[i].CBE_basis_moments_dt[1][5],P[i].CBE_basis_moments_dt[0][8]);}
#endif
    /* total mass change should be zero to floating-point accuracy, so don't need to worry about it, but check! */
    if(ThisTask==0) {printf("PG: Total mass flux == %g (Task=%d i=%d)\n",dmom_tot[0],ThisTask,i);}
    if(ThisTask==0) {printf("PG: Momentum flux == %g/%g/%g (Task=%d i=%d)\n",dmom_tot[1],dmom_tot[2],dmom_tot[3],ThisTask,i);}
#endif
    /* total momentum flux will be transferred */
    double dv0[3];
    for(k=0;k<3;k++)
    {
        dv0[k] = m_inv * dmom_tot[k+1]; // total acceleration
        P[i].GravAccel[k] += dv0[k] / All.cf_a2inv; // write as gravitational acceleration, convert to cosmological units
    }
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
#ifdef CBE_DEBUG
            if(ThisTask==0)
                if((P[i].CBE_basis_moments[j][0] > 0) && (P[i].CBE_basis_moments[j][4]+P[i].CBE_basis_moments[j][5]+P[i].CBE_basis_moments[j][6] < 0))
                {
                    printf("FLX: i=%d m=%g P[i].CBE_basis_moments_dt=%g %g/%g/%g %g/%g/%g %g/%g/%g \n",i,P[i].Mass,P[i].CBE_basis_moments_dt[j][0],
                           P[i].CBE_basis_moments_dt[j][1],P[i].CBE_basis_moments_dt[j][2],P[i].CBE_basis_moments_dt[j][3],P[i].CBE_basis_moments_dt[j][4],P[i].CBE_basis_moments_dt[j][5],P[i].CBE_basis_moments_dt[j][6],P[i].CBE_basis_moments_dt[j][7],
                           P[i].CBE_basis_moments_dt[j][8],P[i].CBE_basis_moments_dt[j][9]);
                    fflush(stdout);
                }
#endif
        }
#endif
    } // for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
    return;
}


#endif
