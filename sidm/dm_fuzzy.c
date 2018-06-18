#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../allvars.h"
#include "../proto.h"
#include "../kernel.h"
#define LOCK_NEXPORT
#define UNLOCK_NEXPORT


/*! \file dm_fuzzy.c
 *  \brief routines needed for fuzzy-DM implementation
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef DM_FUZZY


/* --------------------------------------------------------------------------
 Actual evaluation of fluxes from the quantum pressure tensor
 -------------------------------------------------------------------------- */
void do_dm_fuzzy_flux_computation(double HLLwt, double dt, double m0, double prev_a, double dp[3], double dv[3],
                                  double GradRho_L[3], double GradRho_R[3],
                                  double GradRho2_L[3][3], double GradRho2_R[3][3],
                                  double rho_L, double rho_R, double dv_Right_minus_Left,
                                  double Area[3], double fluxes[3], double AGS_Numerical_QuantumPotential, double *dt_egy_Numerical_QuantumPotential)
{
    if(dt <= 0) return; // no timestep, no flux
    int m,n;
    double f00 = 0.5 * 591569.0 / (All.FuzzyDM_Mass_in_eV * All.UnitVelocity_in_cm_per_s * All.UnitLength_in_cm/All.HubbleParam); // this encodes the coefficient with the mass of the particle: units vel*L = hbar / particle_mass
    // (0.5=1/2, pre-factor from dimensionless equations; 591569 = hbar/eV in cm^2/s; add mass in eV, and put in code units
    double f2 = f00*f00, rhoL_i=1./rho_L, rhoR_i=1./rho_R, r2=0, rSi=1./(rho_L+rho_R); *dt_egy_Numerical_QuantumPotential=0;
    for(m=0;m<3;m++) {r2+=dp[m]*dp[m]; fluxes[m]=0;} /* zero fluxes and calculate separation */
    if(r2 <= 0) return; // same element
    double r=sqrt(r2), wavespeed=2.*f00*(M_PI/r); // approximate k = 2pi/lambda = 2pi/(2*dr) as the maximum k the code will allow locally */
    /* note that the QPT admits waves parallel to k, with wavespeed omega = pm 2*f00*k, so include these for HLLC solution */
    if(dv_Right_minus_Left > wavespeed) return; // elements are receding super-sonically, no way to communicate pressure //
    
    double fluxmag=0, Face_Area_Norm=0; for(m=0;m<3;m++) {Face_Area_Norm+=Area[m]*Area[m];}
    Face_Area_Norm=sqrt(Face_Area_Norm);
    
    for(m=0;m<3;m++)
    {
        for(n=0;n<3;n++)
        {
            double QPT_L = f2*(rhoL_i*GradRho_L[m]*GradRho_L[n] - GradRho2_L[m][n]); // convert grad^2_rho ~ rho/L^2 from code units to physical [should already all be physical here]
            double QPT_R = f2*(rhoR_i*GradRho_R[m]*GradRho_R[n] - GradRho2_R[m][n]); // convert grad^2_rho ~ rho/L^2 from code units to physical [should already all be physical here]
            /* calculate 'star' solution (interface moving with contact wave, since we have a Lagrangian code)
             for HLLC reimann problem based on these pressure tensors */
            double P_star = (QPT_L*rho_R + QPT_R*rho_L) * rSi; // if there were no waves and all at rest
            fluxes[m] += Area[n] * P_star; /* momentum flux into direction 'm' given by Area.Pressure */
            // sign convention here: -Area[m] * (positive definite) => repulsive force, +Area[m] => attractive
        }
        fluxmag += fluxes[m]*fluxes[m];
    }
    fluxmag = sqrt(fluxmag);
    double fluxmax = 100. * Face_Area_Norm * f2 * 0.5*(rho_L+rho_R) / (r*r); // limiter to prevent crazy values where derivatives are ill-posed (e.g. discontinuities)
    if(fluxmag > fluxmax) {for(m=0;m<3;m++) {fluxes[m] *= fluxmax/fluxmag;}}
    
    for(m=0;m<3;m++)
    {
        double ftmp = (2./3.)*AGS_Numerical_QuantumPotential*Area[m]; // 2/3 b/c the equation-of-state of the 'quantum pressure tensor' is gamma=5/3 under isotropic compression/expansion //
        double fmax = 0.5 * m0 * fabs(dv[m]) / dt; fmax = DMAX(fmax, 100.*fluxmag); fmax = DMAX(DMIN(fmax , 100.*prev_a), fluxmag); if(fabs(ftmp) > fmax) {ftmp *= fmax/fabs(ftmp);} // limit pressure-induced acceleration to prevent unphysical cases
        *dt_egy_Numerical_QuantumPotential -= 0.5*ftmp*dv[m]; // PdV work from this pressure term //
        fluxes[m] += ftmp; // add numerical 'pressure' stored from previous timesteps //
    }
    fluxmag=0; for(m=0;m<3;m++) {fluxmag += fluxes[m]*fluxes[m];} if(fluxmag > 0) {fluxmag = sqrt(fluxmag);} else {fluxmag = 0;}

    /* now we have to introduce the numerical diffusivity (the up-wind mixing part from the Reimann problem);
     this can have one of a couple forms, but the most accurate and stable appears to be the traditional HLLC form which we use by default below */
    if(dv_Right_minus_Left < 0) // converging flow, upwind dissipation terms appear //
    {
        // estimate wavenumber needed to calculate wavespeed below, for dissipation terms //
        double g_kL = sqrt(GradRho_L[0]*GradRho_L[0]+GradRho_L[1]*GradRho_L[1]+GradRho_L[2]*GradRho_L[2]); // gradient magnitude
        double g_kR = sqrt(GradRho_R[0]*GradRho_R[0]+GradRho_R[1]*GradRho_R[1]+GradRho_R[2]*GradRho_R[2]); // gradient magnitude
        double k_g1 = 0.5*(rhoL_i*g_kL + rhoR_i*g_kR); // gradient scale-length as proxy for wavenumber [crude] //
        double k_eff=0, k_g2=0, k_g3=0, k_d1 = fabs(rho_L-rho_R)/(0.5*(rho_L+rho_R)) / r; // even cruder delta-based k-estimate
        if(isnan(k_d1)) {k_d1=0;} // trap for division by zero above (should give zero wavespeed)
        if(isnan(k_g1)) {k_g1=0;} // trap
        double k2L = GradRho2_L[0][0]+GradRho2_L[1][1]+GradRho2_L[2][2]; // laplacian
        double k2R = GradRho2_R[0][0]+GradRho2_R[1][1]+GradRho2_R[2][2]; // laplacian
        k_g2 = sqrt( (fabs(k2L)+fabs(k2R)) * rSi ); // sqrt of second derivative -- again crude, doesn't necessarily recover full k but lower value
        k_g3 = 0.5 * sqrt(fabs(k2L-k2R) / (0.5*(g_kL+g_kR)*r)); // third-derivative to first-derivative ratio: exact for resolved wave
        double k_gtan = (fabs(k2L)+fabs(k2R)) / (MIN_REAL_NUMBER + fabs(g_kL) + fabs(g_kR));
        if(isnan(k_g2)) {k_g2=0;} // trap
        if(isnan(k_g3)) {k_g3=0;} // trap
        if(isnan(k_gtan)) {k_gtan=0;} // trap
        k_eff = DMIN(DMAX(k_gtan , DMAX(DMAX((3.+HLLwt)*DMAX(k_d1,k_g1), (2.+HLLwt)*k_g2)  , (1.5+HLLwt)*k_g3)) , 1./r);
        if(isnan(k_eff)) {k_eff=0;} // trap
        double Pstar = (-dv_Right_minus_Left)*(f00*k_eff + (-dv_Right_minus_Left))*rho_L*rho_R*rSi; // HLLC diffusive term
        fluxmag = 0; for(m=0;m<3;m++) {fluxmag += fluxes[m]*fluxes[m];}
        fluxmag = sqrt(fluxmag);
        for(m=0;m<3;m++)
        {
            double f_dir = Area[m]*Pstar, fmax = 0.5 * m0 * fabs(dv[m]) / dt; // assume the face points along the line between particles (very similar, but slightly more stable/diffusive if faces are highly-irregular)
            fmax = DMAX(fmax, 10.*fluxmag); fmax = DMAX(DMIN(fmax , 40.*prev_a), fluxmag); // limit diffusive flux to multiplier of physical flux
            if(fabs(f_dir) > fmax) {f_dir *= fmax/fabs(f_dir);} // limit diffusive flux to avoid overshoot (numerical stability of the diffusion terms) //
            fluxes[m] += f_dir; /* momentum flux into direction 'm' given by Area.Pressure */
            *dt_egy_Numerical_QuantumPotential -= 0.5 * f_dir * dv[m];
        }
    } // approach velocities lead to up-wind mixing
    return;
}


/* kicks for fuzzy-dm integration: just put relevant drift-kick operators here to keep the code clean */
void do_dm_fuzzy_drift_kick(int pindex, double dt_entr)
{
    // calculate various energies: quantum potential QP0, 'stored' numerical pressure NQ0, kinetic energy KE0
    double dNQ=P[pindex].AGS_Dt_Numerical_QuantumPotential*dt_entr, NQ0=P[pindex].AGS_Numerical_QuantumPotential, NQ1=NQ0+dNQ, KE0=0.5*P[pindex].Mass*(P[pindex].Vel[0]*P[pindex].Vel[0]+P[pindex].Vel[1]*P[pindex].Vel[1]+P[pindex].Vel[2]*P[pindex].Vel[2])*All.cf_a2inv;
    double f00 = 0.5 * 591569.0 / (All.FuzzyDM_Mass_in_eV * All.UnitVelocity_in_cm_per_s * All.UnitLength_in_cm/All.HubbleParam); // this encodes the coefficient with the mass of the particle: units vel*L = hbar / particle_mass
    double d2rho = P[pindex].AGS_Gradients2_Density[0][0] + P[pindex].AGS_Gradients2_Density[1][1] + P[pindex].AGS_Gradients2_Density[2][2]; // laplacian
    double drho2 = P[pindex].AGS_Gradients_Density[0]*P[pindex].AGS_Gradients_Density[0] + P[pindex].AGS_Gradients_Density[1]*P[pindex].AGS_Gradients_Density[1] + P[pindex].AGS_Gradients_Density[2]*P[pindex].AGS_Gradients_Density[2];
    double QP0 = (f00*f00 / P[pindex].AGS_Density) * (d2rho - 0.5*drho2/P[pindex].AGS_Density); // quantum 'potential'
    NQ1 = DMAX(0,DMAX(NQ1,0.1*NQ0)); NQ1 = DMIN(NQ1,1.1*DMAX(DMAX(KE0+NQ0,fabs(QP0)),KE0+NQ0+QP0)); // limit kick to not produce unphysical energy over-or-under-shoot
    P[pindex].AGS_Numerical_QuantumPotential = NQ1;
}



/* --------------------------------------------------------------------------
 Everything below here is a giant block to define the sub-routines needed
 to calculate the higher-order matrix gradient estimators for the density
 field, around each DM element (based on its interacting neighbor set,
 within the AGS_Hsml volume). This will give the density gradients
 AGS_Gradients_Density needed to actually compute the quantum pressure tensor
 -------------------------------------------------------------------------- */


/* define a common 'gradients' structure to hold everything we're going to take derivatives of */
struct Quantities_for_Gradients_DM
{
    MyDouble AGS_Density;
    MyDouble AGS_Gradients_Density[3];
};

struct DMGraddata_in
{
    MyDouble Pos[3];
    MyDouble AGS_Hsml;
    struct Quantities_for_Gradients_DM GQuant;
    int NodeList[NODELISTLENGTH];
    int Type;
}
*DMGradDataIn, *DMGradDataGet;

struct DMGraddata_out
{
    struct Quantities_for_Gradients_DM Gradients[3];
    struct Quantities_for_Gradients_DM Maxima;
    struct Quantities_for_Gradients_DM Minima;
}
*DMGradDataResult, *DMGradDataOut;

/* this is a temporary structure for quantities used ONLY in the loop below, for example for computing the slope-limiters (for the Reimann problem) */
static struct temporary_data_topass
{
    struct Quantities_for_Gradients_DM Maxima;
    struct Quantities_for_Gradients_DM Minima;
}
*DMGradDataPasser;

struct kernel_DMGrad
{
    double dp[3],r,wk_i, wk_j, dwk_i, dwk_j,h_i;
};

static inline void particle2in_DMGrad(struct DMGraddata_in *in, int i);

static inline void particle2in_DMGrad(struct DMGraddata_in *in, int i)
{
    int k; for(k=0;k<3;k++) {in->Pos[k] = P[i].Pos[k];}
    in->AGS_Hsml = PPP[i].AGS_Hsml;
    in->GQuant.AGS_Density = P[i].AGS_Density;
    in->Type = P[i].Type;
    for(k=0;k<3;k++) {in->GQuant.AGS_Gradients_Density[k] = P[i].AGS_Gradients_Density[k];}
}

#define ASSIGN_ADD_PRESET(x,y,mode) (mode == 0 ? (x=y) : (x+=y))
#define MINMAX_CHECK(x,xmin,xmax) ((x<xmin)?(xmin=x):((x>xmax)?(xmax=x):(1)))
#define MAX_ADD(x,y,mode) ((y > x) ? (x = y) : (1)) // simpler definition now used
#define MIN_ADD(x,y,mode) ((y < x) ? (x = y) : (1))

static inline void out2particle_DMGrad(struct DMGraddata_out *out, int i, int mode, int gradient_iteration);

static inline void out2particle_DMGrad(struct DMGraddata_out *out, int i, int mode, int gradient_iteration)
{
    if(gradient_iteration <= 0)
    {
        MAX_ADD(DMGradDataPasser[i].Maxima.AGS_Density,out->Maxima.AGS_Density,mode);
        MIN_ADD(DMGradDataPasser[i].Minima.AGS_Density,out->Minima.AGS_Density,mode);
        int k; for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients_Density[k],out->Gradients[k].AGS_Density,mode);}
    } else {
        int k,k2;
        for(k=0;k<3;k++) 
        {
            MAX_ADD(DMGradDataPasser[i].Maxima.AGS_Gradients_Density[k],out->Maxima.AGS_Gradients_Density[k],mode);
            MIN_ADD(DMGradDataPasser[i].Minima.AGS_Gradients_Density[k],out->Minima.AGS_Gradients_Density[k],mode);
            for(k2=0;k2<3;k2++) {ASSIGN_ADD_PRESET(P[i].AGS_Gradients2_Density[k2][k],out->Gradients[k].AGS_Gradients_Density[k2],mode);}
        }
    }
}


void construct_gradient_DMGrad(double *grad, int i);

void construct_gradient_DMGrad(double *grad, int i)
{
    /* use the NV_T matrix-based gradient estimator */
    int k; double v_tmp[3];
    for(k=0;k<3;k++) {v_tmp[k] = grad[k];}
    for(k=0;k<3;k++) {grad[k] = P[i].NV_T[k][0]*v_tmp[0] + P[i].NV_T[k][1]*v_tmp[1] + P[i].NV_T[k][2]*v_tmp[2];}
}




void DMGrad_gradient_calc(void)
{
    /* define the number of iterations needed to compute gradients and optional gradients-of-gradients */
    int gradient_iteration, number_of_gradient_iterations = 1;
    number_of_gradient_iterations = 2; // need extra iteration to get gradients-of-gradients
    /* loop over the number of iterations needed to actually compute the gradients fully */
    if(All.Time==All.TimeBegin) {int i; for(i=FirstActiveParticle; i>=0; i=NextActiveParticle[i]) {P[i].AGS_Numerical_QuantumPotential=0;}}
    for(gradient_iteration=0; gradient_iteration<number_of_gradient_iterations; gradient_iteration++)
    {
        int i, j, k, ngrp, ndone, ndone_flag, recvTask, place, save_NextParticle;
        double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 = 0, timewait2 = 0;
        double timecomp, timecomm, timewait, tstart, tend, t0, t1;
        long long n_exported = 0, NTaskTimesNumPart;
        /* allocate buffers to arrange communication */
        DMGradDataPasser = (struct temporary_data_topass *) mymalloc("DMGradDataPasser",NumPart * sizeof(struct temporary_data_topass));

        NTaskTimesNumPart = maxThreads * NumPart;
        size_t MyBufferSize = All.BufferSize;
        All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                                               sizeof(struct DMGraddata_in) + sizeof(struct DMGraddata_out) + sizemax(sizeof(struct DMGraddata_in),sizeof(struct DMGraddata_out))));
        CPU_Step[CPU_AGSDENSMISC] += measure_time();
        t0 = my_second();
        Ngblist = (int *) mymalloc("Ngblist", NTaskTimesNumPart * sizeof(int));
        DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
        DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));

        /* before doing any operations, need to zero the appropriate memory so we can correctly do pair-wise operations */
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {memset(&DMGradDataPasser[i], 0, sizeof(struct temporary_data_topass));}
        
        /* begin the main gradient loop */
        NextParticle = FirstActiveParticle;    /* begin with this index */
        do
        {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;
            for(j = 0; j < NTask; j++)
            {
                Send_count[j] = 0;
                Exportflag[j] = -1;
            }
            /* do local particles and prepare export list */
            tstart = my_second();
            
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else
                int mainthreadid = 0;
#endif
                DMGrad_evaluate_primary(&mainthreadid, gradient_iteration);    /* do local particles and prepare export list */
            }
            
            tend = my_second();
            timecomp1 += timediff(tstart, tend);
            
            if(BufferFullFlag)
            {
                int last_nextparticle = NextParticle;
                NextParticle = save_NextParticle;
                while(NextParticle >= 0)
                {
                    if(NextParticle == last_nextparticle) break;
                    if(ProcessedFlag[NextParticle] != 1) break;
                    ProcessedFlag[NextParticle] = 2;
                    NextParticle = NextActiveParticle[NextParticle];
                }
                if(NextParticle == save_NextParticle)
                {
                    endrun(123708); /* in this case, the buffer is too small to process even a single particle */
                }
                int new_export = 0;
                for(j = 0, k = 0; j < Nexport; j++)
                    if(ProcessedFlag[DataIndexTable[j].Index] != 2)
                    {
                        if(k < j + 1) {k = j + 1;}
                        for(; k < Nexport; k++)
                            if(ProcessedFlag[DataIndexTable[k].Index] == 2)
                            {
                                int old_index = DataIndexTable[j].Index;
                                DataIndexTable[j] = DataIndexTable[k];
                                DataNodeList[j] = DataNodeList[k];
                                DataIndexTable[j].IndexGet = j;
                                new_export++;
                                DataIndexTable[k].Index = old_index;
                                k++;
                                break;
                            }
                    }
                    else {new_export++;}
                Nexport = new_export;
            }
            n_exported += Nexport;
            for(j = 0; j < NTask; j++) {Send_count[j] = 0;}
            for(j = 0; j < Nexport; j++) {Send_count[DataIndexTable[j].Task]++;}
            MYSORT_DATAINDEX(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
            tstart = my_second();
            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
            tend = my_second();
            timewait1 += timediff(tstart, tend);
            for(j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
            {
                Nimport += Recv_count[j];
                if(j > 0)
                {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }
            
            /* prepare particle data for export */
            DMGradDataGet = (struct DMGraddata_in *) mymalloc("DMGradDataGet", Nimport * sizeof(struct DMGraddata_in));
            DMGradDataIn = (struct DMGraddata_in *) mymalloc("DMGradDataIn", Nexport * sizeof(struct DMGraddata_in));
            for(j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                particle2in_DMGrad(&DMGradDataIn[j], place);
                memcpy(DMGradDataIn[j].NodeList,DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
            }
            
            /* exchange particle data */
            tstart = my_second();
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;
                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* get the particles */
                        MPI_Sendrecv(&DMGradDataIn[Send_offset[recvTask]],
                                     Send_count[recvTask] * sizeof(struct DMGraddata_in), MPI_BYTE,
                                     recvTask, TAG_GRADLOOP_A,
                                     &DMGradDataGet[Recv_offset[recvTask]],
                                     Recv_count[recvTask] * sizeof(struct DMGraddata_in), MPI_BYTE,
                                     recvTask, TAG_GRADLOOP_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            tend = my_second();
            timecommsumm1 += timediff(tstart, tend);
            myfree(DMGradDataIn);
            DMGradDataResult = (struct DMGraddata_out *) mymalloc("DMGradDataResult", Nimport * sizeof(struct DMGraddata_out));
            DMGradDataOut = (struct DMGraddata_out *) mymalloc("DMGradDataOut", Nexport * sizeof(struct DMGraddata_out));
            
            /* now do the particles that were sent to us */
            tstart = my_second();
            NextJ = 0;
            
#ifdef _OPENMP
#pragma omp parallel
#endif
            {
#ifdef _OPENMP
                int mainthreadid = omp_get_thread_num();
#else
                int mainthreadid = 0;
#endif
                DMGrad_evaluate_secondary(&mainthreadid, gradient_iteration);
            }
            
            tend = my_second();
            timecomp2 += timediff(tstart, tend);
            
            if(NextParticle < 0) {ndone_flag = 1;} else {ndone_flag = 0;}
            tstart = my_second();
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            tend = my_second();
            timewait2 += timediff(tstart, tend);
            
            /* get the result */
            tstart = my_second();
            for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
            {
                recvTask = ThisTask ^ ngrp;
                if(recvTask < NTask)
                {
                    if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                    {
                        /* send the results */
                        MPI_Sendrecv(&DMGradDataResult[Recv_offset[recvTask]],
                                     Recv_count[recvTask] * sizeof(struct DMGraddata_out),
                                     MPI_BYTE, recvTask, TAG_GRADLOOP_B,
                                     &DMGradDataOut[Send_offset[recvTask]],
                                     Send_count[recvTask] * sizeof(struct DMGraddata_out),
                                     MPI_BYTE, recvTask, TAG_GRADLOOP_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
            tend = my_second();
            timecommsumm2 += timediff(tstart, tend);
            
            /* add the result to the local particles */
            tstart = my_second();
            for(j = 0; j < Nexport; j++)
            {
                place = DataIndexTable[j].Index;
                out2particle_DMGrad(&DMGradDataOut[j], place, 1, gradient_iteration);
            }
            tend = my_second();
            timecomp1 += timediff(tstart, tend);
            myfree(DMGradDataOut);
            myfree(DMGradDataResult);
            myfree(DMGradDataGet);
        }
        while(ndone < NTask);
        
        myfree(DataNodeList);
        myfree(DataIndexTable);
        myfree(Ngblist);
        
        /* do final operations on results: these are operations that can be done after the complete set of iterations */
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
        {
            if(gradient_iteration <= 0)
            {
                /* now we can properly calculate (second-order accurate) gradients of hydrodynamic quantities from this loop */
                construct_gradient_DMGrad(P[i].AGS_Gradients_Density,i);
                /* finally, we need to apply a sensible slope limiter to the gradients, to prevent overshooting */
                /* (actually not clear that we need to slope-limit these, because we are not using the gradients for reconstruction.
                    testing this now. if not, we can remove the limiter information entirely and save some time in these computations) */
                //local_slopelimiter(P[i].AGS_Gradients_Density,DMGradDataPasser[i].Maxima.AGS_Density,DMGradDataPasser[i].Minima.AGS_Density,0.5,PPP[i].AGS_Hsml,0);
            } else {
                int k;
                for(k=0;k<3;k++)
                {
                    /* construct the gradient-of-gradient */
                    construct_gradient_DMGrad(P[i].AGS_Gradients2_Density[k],i);
                    //local_slopelimiter(P[i].AGS_Gradients2_Density[k],DMGradDataPasser[i].Maxima.AGS_Gradients_Density[k],DMGradDataPasser[i].Minima.AGS_Gradients_Density[k],0.5,PPP[i].AGS_Hsml,0);
                }
                /* symmetrize the gradients */
                double tmp;
                tmp = 0.5*(P[i].AGS_Gradients2_Density[0][1] + P[i].AGS_Gradients2_Density[1][0]);
                P[i].AGS_Gradients2_Density[0][1] = P[i].AGS_Gradients2_Density[1][0] = tmp;
                tmp = 0.5*(P[i].AGS_Gradients2_Density[0][2] + P[i].AGS_Gradients2_Density[2][0]);
                P[i].AGS_Gradients2_Density[0][2] = P[i].AGS_Gradients2_Density[2][0] = tmp;
                tmp = 0.5*(P[i].AGS_Gradients2_Density[1][2] + P[i].AGS_Gradients2_Density[2][1]);
                P[i].AGS_Gradients2_Density[1][2] = P[i].AGS_Gradients2_Density[2][1] = tmp;
            }
        }
        myfree(DMGradDataPasser); /* free the temporary structure we created for the MinMax and additional data passing */
        MPI_Barrier(MPI_COMM_WORLD); // force barrier so we know the first derivatives are fully-computed //

        /* collect some timing information */
        t1 = WallclockTime = my_second();
        timeall += timediff(t0, t1);
        timecomp = timecomp1 + timecomp2;
        timewait = timewait1 + timewait2;
        timecomm = timecommsumm1 + timecommsumm2;
        CPU_Step[CPU_AGSDENSCOMPUTE] += timecomp;
        CPU_Step[CPU_AGSDENSWAIT] += timewait;
        CPU_Step[CPU_AGSDENSCOMM] += timecomm;
        CPU_Step[CPU_AGSDENSMISC] += timeall - (timecomp + timewait + timecomm);
    } // close loop over gradient_iterations (master loop)
}


int DMGrad_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int gradient_iteration)
{
    /* define variables */
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double hinv, hinv3, hinv4, r2, u;
    struct kernel_DMGrad kernel;
    struct DMGraddata_in local;
    struct DMGraddata_out out;
    /* zero memory and import data for local target */
    memset(&out, 0, sizeof(struct DMGraddata_out));
    memset(&kernel, 0, sizeof(struct kernel_DMGrad));
    if(mode == 0) {particle2in_DMGrad(&local, target);} else {local = DMGradDataGet[target];}
    /* check if we should bother doing a neighbor loop */
    if(local.AGS_Hsml <= 0) return 0;
    if(local.GQuant.AGS_Density <= 0) return 0;
    /* now set particle-i centric quantities so we don't do it inside the loop */
    kernel.h_i = local.AGS_Hsml;
    double h2_i = kernel.h_i*kernel.h_i;
    kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
    int AGS_kernel_shared_BITFLAG = ags_gravity_kernel_shared_BITFLAG(local.Type); // determine allowed particle types for search for adaptive gravitational softening terms
    
    /* Now start the actual neighbor computation for this particle */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DMGradDataGet[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads_targeted(local.Pos, local.AGS_Hsml, target, &startnode, mode, exportflag,
                                                                  exportnodecount, exportindex, ngblist, AGS_kernel_shared_BITFLAG);
            if(numngb_inbox < 0) {return -1;} /* no neighbors! */
            for(n = 0; n < numngb_inbox; n++) /* neighbor loop */
            {
                j = ngblist[n];
                if((P[j].Mass <= 0)||(P[j].AGS_Density <= 0)) {continue;} /* make sure neighbor is valid */
                /* calculate position relative to target */
                kernel.dp[0] = local.Pos[0] - P[j].Pos[0]; kernel.dp[1] = local.Pos[1] - P[j].Pos[1]; kernel.dp[2] = local.Pos[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC            /*  now find the closest image in the given box size  */
                NEAREST_XYZ(kernel.dp[0],kernel.dp[1],kernel.dp[2],1);
#endif
                r2 = kernel.dp[0]*kernel.dp[0] + kernel.dp[1]*kernel.dp[1] + kernel.dp[2]*kernel.dp[2];
                if((r2 <= 0) || (r2 >= h2_i)) continue;
                /* calculate kernel quantities needed below */
                kernel.r = sqrt(r2); u = kernel.r * hinv;
                kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, -1);
                /* DIFFERENCE & SLOPE LIMITING: need to check maxima and minima of particle values in the kernel, to avoid 'overshoot' with our gradient estimators. this check should be among all interacting pairs */
                if(gradient_iteration <= 0)
                {
                    double d_rho = P[j].AGS_Density - local.GQuant.AGS_Density;
                    MINMAX_CHECK(d_rho,out.Minima.AGS_Density,out.Maxima.AGS_Density);
                    for(k=0;k<3;k++) {out.Gradients[k].AGS_Density += -kernel.wk_i * kernel.dp[k] * d_rho;} /* sign is important here! */
                } else {
                    for(k=0;k<3;k++)
                    {
                        double d_grad_rho = P[j].AGS_Gradients_Density[k] - local.GQuant.AGS_Gradients_Density[k];
                        MINMAX_CHECK(d_grad_rho,out.Minima.AGS_Gradients_Density[k],out.Maxima.AGS_Gradients_Density[k]);
                        int k2; for(k2=0;k2<3;k2++) {out.Gradients[k2].AGS_Gradients_Density[k] += -kernel.wk_i * kernel.dp[k2] * d_grad_rho;}
                    }
                } // gradient_iteration
            } // numngb_inbox loop
        } // while(startnode)
        /* continue to open leaves if needed */
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DMGradDataGet[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
            }
        }
    }
    /* Collect the result at the right place */
    if(mode == 0) {out2particle_DMGrad(&out, target, 0, gradient_iteration);} else {DMGradDataResult[target] = out;}
    return 0;
}





void *DMGrad_evaluate_primary(void *p, int gradient_iteration)
{
#define CONDITION_FOR_EVALUATION if(ags_density_isactive(i))
#define EVALUATION_CALL DMGrad_evaluate(i,0,exportflag,exportnodecount,exportindex,ngblist,gradient_iteration)
#include "../system/code_block_primary_loop_evaluation.h"
#undef CONDITION_FOR_EVALUATION
#undef EVALUATION_CALL
}

void *DMGrad_evaluate_secondary(void *p, int gradient_iteration)
{
#define EVALUATION_CALL DMGrad_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, gradient_iteration);
#include "../system/code_block_secondary_loop_evaluation.h"
#undef EVALUATION_CALL
}


#endif // DM_FUZZY

