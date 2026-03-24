/*! \file sink_environment.c
*  \brief routines for evaluating sink particle environment
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/


#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //


#define CORE_FUNCTION_NAME sink_environment_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(sink_isactive(i)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */

/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    int NodeList[NODELISTLENGTH]; Vec3<MyDouble> Pos; Vec3<MyFloat> Vel; MyFloat KernelRadius; MyIDType ID;
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    MyDouble Mass;
#endif
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    MyFloat SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    MyFloat AGS_KernelRadius;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<MyFloat> Sink_Specific_AngMom;
#endif
}
*DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    in->Pos=P[i].Pos; in->Vel=P[i].Vel; /* good example - always needed */
    in->KernelRadius = P[i].KernelRadius; in->ID = P[i].ID;
#if defined(SINK_GRAVCAPTURE_GAS) || (SINK_GRAVACCRETION == 8)
    in->Mass = P[i].Mass;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    in->SinkRadius = P[i].SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    in->AGS_KernelRadius = ForceSoftening_KernelRadius(i);
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    in->Sink_Specific_AngMom = P[i].Sink_Specific_AngMom;
#endif
}


/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{ /* define variables below as e.g. "double X;" */
MyFloat Sink_SurroudingGasInternalEnergy, Mgas_in_Kernel, Mstar_in_Kernel, Malt_in_Kernel;
MyFloat Jgas_in_Kernel[3], Jstar_in_Kernel[3], Jalt_in_Kernel[3]; // mass/angular momentum for GAS/STAR/TOTAL components computed always now
#ifdef SINK_REPOSITION_ON_POTMIN
    MyFloat DF_rms_vel, DF_mean_vel[3], DF_mmax_particles;
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    MyFloat Sfr_in_Kernel;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    MyFloat Sink_SurroundingGasVel[3];
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
    MyFloat Sink_SurroundingGasCOM[3];
#endif    
#if (SINK_GRAVACCRETION == 8)
    MyFloat hubber_mdot_vr_estimator, hubber_mdot_disk_estimator, hubber_mdot_bondi_limiter;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    MyFloat mass_to_swallow_edd;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    MyFloat angmom_prepass_sum_for_passback[3];
#endif
#if defined(SINK_RETURN_BFLUX)
    MyFloat kernel_norm_topass_in_swallowloop;
#endif    
}
*DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* simple routine to add quantities to SinkTempInfo */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int target = P[i].IndexMapToTempStruc, k=0;
    ASSIGN_ADD(SinkTempInfo[target].Sink_SurroudingGasInternalEnergy,out->Sink_SurroudingGasInternalEnergy,mode);
    ASSIGN_ADD(SinkTempInfo[target].Mgas_in_Kernel,out->Mgas_in_Kernel,mode);
    ASSIGN_ADD(SinkTempInfo[target].Mstar_in_Kernel,out->Mstar_in_Kernel,mode);
    ASSIGN_ADD(SinkTempInfo[target].Malt_in_Kernel,out->Malt_in_Kernel,mode);
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].Jgas_in_Kernel[k],out->Jgas_in_Kernel[k],mode);}
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].Jstar_in_Kernel[k],out->Jstar_in_Kernel[k],mode);}
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].Jalt_in_Kernel[k],out->Jalt_in_Kernel[k],mode);}
#ifdef SINK_REPOSITION_ON_POTMIN
    ASSIGN_ADD(SinkTempInfo[target].DF_rms_vel,out->DF_rms_vel,mode);
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].DF_mean_vel[k],out->DF_mean_vel[k],mode);}
    if(mode==0) {SinkTempInfo[target].DF_mmax_particles = out->DF_mmax_particles;}
        else {if(out->DF_mmax_particles > SinkTempInfo[target].DF_mmax_particles) {SinkTempInfo[target].DF_mmax_particles = out->DF_mmax_particles;}}
#endif
#if defined(SINK_OUTPUT_MOREINFO)
    ASSIGN_ADD(SinkTempInfo[target].Sfr_in_Kernel,out->Sfr_in_Kernel,mode);
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].Sink_SurroundingGasVel[k],out->Sink_SurroundingGasVel[k],mode);}
#endif
#if defined(JET_DIRECTION_FROM_KERNEL_AND_SINK)
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].Sink_SurroundingGasCOM[k],out->Sink_SurroundingGasCOM[k],mode);}
#endif    
#if (SINK_GRAVACCRETION == 8)
    ASSIGN_ADD(SinkTempInfo[target].hubber_mdot_bondi_limiter,out->hubber_mdot_bondi_limiter,mode);
    ASSIGN_ADD(SinkTempInfo[target].hubber_mdot_vr_estimator,out->hubber_mdot_vr_estimator,mode);
    ASSIGN_ADD(SinkTempInfo[target].hubber_mdot_disk_estimator,out->hubber_mdot_disk_estimator,mode);
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    ASSIGN_ADD(SinkTempInfo[target].mass_to_swallow_edd, out->mass_to_swallow_edd, mode);
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    for(k=0;k<3;k++) {ASSIGN_ADD(SinkTempInfo[target].angmom_prepass_sum_for_passback[k],out->angmom_prepass_sum_for_passback[k],mode);}
#endif
#if defined(SINK_RETURN_BFLUX)
    ASSIGN_ADD(SinkTempInfo[target].kernel_norm_topass_in_swallowloop,out->kernel_norm_topass_in_swallowloop,mode);
#endif    
}


/* for new quantities calculated in environment loop, divide out weights and convert to physical units */
void sink_normalize_temp_info_struct_after_environment_loop(int i);
void sink_normalize_temp_info_struct_after_environment_loop(int i)
{
    int k; k=0;
    if(SinkTempInfo[i].Mgas_in_Kernel > 0)
    {
        SinkTempInfo[i].Sink_SurroudingGasInternalEnergy /= SinkTempInfo[i].Mgas_in_Kernel;
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
        for(k=0;k<3;k++) {SinkTempInfo[i].Sink_SurroundingGasVel[k] /= SinkTempInfo[i].Mgas_in_Kernel * All.cf_atime;}
#endif
    }
    else {SinkTempInfo[i].Sink_SurroudingGasInternalEnergy = 0;}
    // DAA: add GAS/STAR mass/angular momentum to the TOTAL mass/angular momentum in kernel
    SinkTempInfo[i].Malt_in_Kernel += (SinkTempInfo[i].Mgas_in_Kernel + SinkTempInfo[i].Mstar_in_Kernel);
    for(k=0;k<3;k++) {SinkTempInfo[i].Jalt_in_Kernel[k] += (SinkTempInfo[i].Jgas_in_Kernel[k] + SinkTempInfo[i].Jstar_in_Kernel[k]);}
#ifdef SINK_REPOSITION_ON_POTMIN  // DAA: normalize by the appropriate MASS in kernel depending on selected option
    double Mass_in_Kernel = SinkTempInfo[i].DF_rms_vel; // normalize by mass in kernel - this variable was used here as a placeholder for this
    if(Mass_in_Kernel > 0) {for(k=0;k<3;k++) {SinkTempInfo[i].DF_mean_vel[k] /= Mass_in_Kernel * All.cf_atime;}}
#endif
}


/* routine to return the values we need of the properties of the gas, stars, etc in the vicinity of the BH -- these all factor into the BHAR */
/*!   -- this subroutine writes to shared memory [updating the neighbor values, albeit just for one claude for the LowestSinkTimeBin check]: need to protect these writes for openmp below. none of the modified values are read, so only the write block is protected. */
int sink_environment_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    /* initialize variables before loop is started */
    int startnode, numngb, listindex = 0, j, k, n; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */
    double ags_h_i, h_i, hinv, hinv3, wk, dwk, u; wk=0; dwk=0; u=0; h_i=local.KernelRadius; hinv=1./h_i; hinv3=hinv*hinv*hinv; ags_h_i=SinkParticle_GravityKernelRadius;
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    ags_h_i = local.AGS_KernelRadius;
#endif
    /* Now start the actual neighbor computation for this particle */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0) {
        while(startnode >= 0) {
            numngb = ngb_treefind_pairs_threads_targeted(local.Pos, h_i, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist, SINK_NEIGHBOR_BITFLAG);
            if(numngb < 0) {return -2;}
            for(n = 0; n < numngb; n++)
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if( (P[j].Mass > 0) && (P[j].Type != 5) && (P[j].ID != local.ID) )
                {
                    double wt = P[j].Mass;
                    Vec3<double> dP = P[j].Pos - local.Pos; Vec3<double> dv = P[j].Vel - local.Vel;
                    nearest_xyz(dP,-1); /*  find the closest image in the given box size  */
                    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,dv,-1); /* wrap velocities for shearing boxes if needed */

#ifdef SINK_REPOSITION_ON_POTMIN
                    if( (P[j].Type != 0) && (P[j].Type != 5) )
                    {
                        double wtfac = wt, rfac = (dP[0]*dP[0] + dP[1]*dP[1] + dP[2]*dP[2]) * (10./(h_i*h_i) + 0.1/(SinkParticle_GravityKernelRadius*SinkParticle_GravityKernelRadius));
                        wtfac = wt / (1. + rfac); // simple function scaling ~ 1/r^2 for large r, to weight elements closer to the BH, so doesnt get 'pulled' by far-away elements //
                        if(P[j].Mass>out.DF_mmax_particles) out.DF_mmax_particles=P[j].Mass;
                        for (k=0;k<3;k++)
                        {
                            out.DF_mean_vel[k] += wtfac*dv[k];
                            out.DF_rms_vel += wtfac;
                        }
                    }
#endif
                    
                    /* DAA: compute mass/angular momentum for GAS/STAR/DM components within BH kernel
                            this is done always now (regardless of the specific BH options used) */
                    if(P[j].Type==0)
                    {
                        /* we found gas in BH's kernel */
                        out.Mgas_in_Kernel += wt;
                        out.Sink_SurroudingGasInternalEnergy += wt*CellP[j].InternalEnergy;
                        out.Jgas_in_Kernel[0] += wt*(dP[1]*dv[2] - dP[2]*dv[1]); out.Jgas_in_Kernel[1] += wt*(dP[2]*dv[0] - dP[0]*dv[2]); out.Jgas_in_Kernel[2] += wt*(dP[0]*dv[1] - dP[1]*dv[0]);
#if defined(SINK_OUTPUT_MOREINFO)
                        out.Sfr_in_Kernel += CellP[j].Sfr;
#endif
#if (SINK_GRAVACCRETION >= 5) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINGLE_STAR_TIMESTEPPING)
                        for(k=0;k<3;k++) {out.Sink_SurroundingGasVel[k] += wt*dv[k];}
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
                        for(k=0;k<3;k++) {out.Sink_SurroundingGasCOM[k] += wt*dP[k];}
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
                        u=dP.norm()/DMAX(h_i, P[j].KernelRadius); if(u<1) {kernel_main(u,1., 1.,&wk,&dwk,-1);} else {wk=dwk=0;} // spline weighting function for conserved quantity return
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) /* We need a normalization factor for angular momentum feedback so we will go over all the neighbours */
                        double r2j=dP.norm_sq(), Lrj=dot(local.Sink_Specific_AngMom,dP);
                        for(k=0;k<3;k++) {out.angmom_prepass_sum_for_passback[k] += wk * wt*(local.Sink_Specific_AngMom[k]*r2j - dP[k]*Lrj);} // this is now kernel-weighted so that the kicks fall off smoothly as r approaches H
#endif
#if defined(SINK_RETURN_BFLUX)                        
                        out.kernel_norm_topass_in_swallowloop += wk;
#endif                  
#if (SINK_GRAVACCRETION == 8)
                        u=dP.norm()/h_i; if(u<1) {kernel_main(u,hinv3,hinv3*hinv,&wk,&dwk,-1);} else {wk=dwk=0;}
                        double rj=u*h_i*All.cf_atime; double csj=Get_Gas_effective_soundspeed_i(j);
                        double vdotrj=-dot(dP,dv);
                        double vr_mdot = 4*M_PI * wt*(wk*All.cf_a3inv) * rj*vdotrj;
                        if(rj < SinkParticle_GravityKernelRadius*All.cf_atime)
                        {
                            double bondi_mdot = 4*M_PI*All.G*All.G * local.Mass*local.Mass / pow(csj*csj + (dv[0]*dv[0]+dv[1]*dv[1]+dv[2]*dv[2])*All.cf_a2inv, 1.5) * wt * (wk*All.cf_a3inv);
                            vr_mdot = DMAX(vr_mdot , bondi_mdot); out.hubber_mdot_bondi_limiter += bondi_mdot;
                        }
                        out.hubber_mdot_vr_estimator += vr_mdot; /* physical */
                        out.hubber_mdot_disk_estimator += wt*wk * sqrt(rj) / (CellP[j].Density * csj*csj); /* physical */
#endif
                    }
                    else if( P[j].Type==4 || ((P[j].Type==2||P[j].Type==3) && !(All.ComovingIntegrationOn)) ) /* stars */
                    {
                        out.Mstar_in_Kernel += wt; out.Jstar_in_Kernel[0] += wt*(dP[1]*dv[2] - dP[2]*dv[1]); out.Jstar_in_Kernel[1] += wt*(dP[2]*dv[0] - dP[0]*dv[2]); out.Jstar_in_Kernel[2] += wt*(dP[0]*dv[1] - dP[1]*dv[0]);
                    }
                    else /* dark matter */ // DAA: Jalt_in_Kernel and Malt_in_Kernel are updated in sink_normalize_temp_info_struct() to be TOTAL angular momentum and mass
                    {
                        out.Malt_in_Kernel += wt; out.Jalt_in_Kernel[0] += wt*(dP[1]*dv[2] - dP[2]*dv[1]); out.Jalt_in_Kernel[1] += wt*(dP[2]*dv[0] - dP[0]*dv[2]); out.Jalt_in_Kernel[2] += wt*(dP[0]*dv[1] - dP[1]*dv[0]);
                    }

#if defined(SINK_GRAVCAPTURE_GAS) /* XM: I formally distinguish SINK_GRAVCAPTURE_GAS and SINK_GRAVCAPTURE_NONGAS. The former applies to gas ONLY, as an accretion model. The later can be combined with any accretion model.
                                    Currently, I only allow gas accretion to contribute to Sink_Mdot (consistent with the energy radiating away). For star particles, if there is an alpha-disk, they are captured to the disk. If not, they directly go
                                    to the hole, without any contribution to Sink_Mdot and feedback. This can be modified in the swallow loop for other purposes. The goal of the following part is to estimate Sink_Mdot, which will be used to evaluate feedback strength.
                                    Therefore, we only need it when we enable SINK_GRAVCAPTURE_GAS as gas accretion model. */
#ifdef GRAIN_FLUID                    
                    if( (P[j].Mass > 0) && ((P[j].Type == 0) || ((1<<P[j].Type) & GRAIN_PTYPES)))
#else
                    if( (P[j].Mass > 0) && (P[j].Type == 0))
#endif                        
                      
                    {
                        double r2=dP.norm_sq(), vrel=dv.norm() / All.cf_atime;
                        double dr_code = sqrt(r2);
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE) /* need to project grain velocities, shouldn't include gyro motion */
                        if((1<<P[j].Type) & GRAIN_PTYPES) {double bmag2=0; double vrel_dot=0; for(k=0;k<3;k++) {vrel_dot+=dv[k]*P[j].Gas_B[k]; bmag2+=P[j].Gas_B[k]*P[j].Gas_B[k];}
                            vrel = (fabs(vrel_dot)/sqrt(bmag2)) / All.cf_atime;}
#endif
                        double vbound = sink_vesc(j, local.Mass, dr_code, ags_h_i);
                        if(vrel < vbound) { /* bound */
                            double local_sink_radius = SinkParticle_GravityKernelRadius;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
                            local_sink_radius = local.SinkRadius;
                            double spec_mom=dot(dv,dP); // delta_x.delta_v
                            spec_mom = (r2*vrel*vrel - spec_mom*spec_mom*All.cf_a2inv);  // specific angular momentum^2 = r^2(delta_v)^2 - (delta_v.delta_x)^2;
                            if(spec_mom < All.G * (local.Mass + P[j].Mass) * local_sink_radius) // check Bate 1995 angular momentum criterion (in addition to bounded-ness)
#endif
                            if( sink_check_boundedness(j,vrel,vbound,dr_code,local_sink_radius)==1 )
                            { /* apocenter within epsilon (softening length) */
#ifdef SINGLE_STAR_SINK_DYNAMICS
                                double eps = DMAX( dr_code , DMAX(P[j].KernelRadius , ags_h_i) * KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER ); // plummer-equivalent vs r
                                double tff = eps*eps*eps / (local.Mass + P[j].Mass); if(tff < P[j].SwallowTime) {P[j].SwallowTime = tff;}
#endif
                                if(P[j].SwallowID < local.ID) {out.mass_to_swallow_edd += P[j].Mass;} /* mark as 'will be swallowed' on next loop, to correct accretion rate */
                            } /* if( apocenter in tolerance range ) */
                        } /* if(vrel < vbound) */
                    } /* type check */
#endif // SINK_GRAVCAPTURE_GAS
                } // ( (P[j].Mass > 0) && (P[j].Type != 5) && (P[j].ID != local.ID) ) - condition for entering primary loop
            } // numngb_inbox loop
        } // while(startnode)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    }
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
}


void sink_environment_loop(void)
{
    #include "../system/code_block_xchange_perform_ops_malloc.h" /* this calls the large block of code which contains the memory allocations for the MPI/OPENMP/Pthreads parallelization block which must appear below */
    #include "../system/code_block_xchange_perform_ops.h" /* this calls the large block of code which actually contains all the loops, MPI/OPENMP/Pthreads parallelization */
    #include "../system/code_block_xchange_perform_ops_demalloc.h" /* this de-allocates the memory for the MPI/OPENMP/Pthreads parallelization block which must appear above */
    /* final operations on results */
    {int i; for(i=0; i<N_active_loc_Sink; i++) {sink_normalize_temp_info_struct_after_environment_loop(i);}}
    CPU_Step[CPU_SINKS] += measure_time(); /* collect timings and reset clock for next timing */
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */








/* -----------------------------------------------------------------------------------------------------
 * DAA: modified versions of sink_environment_loop and sink_environment_evaluate for a second
 * environment loop. Here we do a Bulge-Disk kinematic decomposition for gravitational torque accretion
 * ----------------------------------------------------------------------------------------------------- */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)

#define CORE_FUNCTION_NAME sink_environment_second_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(sink_isactive(i)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */

/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    int NodeList[NODELISTLENGTH]; Vec3<MyDouble> Pos; Vec3<MyFloat> Vel; MyFloat KernelRadius, Jgas_in_Kernel[3], Jstar_in_Kernel[3];
}
*DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k, j_tempinfo = P[i].IndexMapToTempStruc; in->KernelRadius = P[i].KernelRadius; /* link to the location in the shared structure where this is stored */
    in->Pos=P[i].Pos; in->Vel=P[i].Vel; /* good example - always needed */
    for(k=0;k<3;k++) {in->Jgas_in_Kernel[k]=SinkTempInfo[j_tempinfo].Jgas_in_Kernel[k]; in->Jstar_in_Kernel[k]=SinkTempInfo[j_tempinfo].Jstar_in_Kernel[k];}
}

/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{ /* define variables below as e.g. "double X;" */
    MyFloat MgasBulge_in_Kernel, MstarBulge_in_Kernel;
}
*DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* simple routine to add quantities to SinkTempInfo */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int target = P[i].IndexMapToTempStruc;
    ASSIGN_ADD(SinkTempInfo[target].MgasBulge_in_Kernel,out->MgasBulge_in_Kernel,mode);
    ASSIGN_ADD(SinkTempInfo[target].MstarBulge_in_Kernel,out->MstarBulge_in_Kernel,mode);
}

/* this subroutine does the actual neighbor-element calculations (this is the 'core' of the loop, essentially) */
/*!   -- this subroutine contains no writes to shared memory -- */
int sink_environment_second_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, n; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0) {
        while(startnode >= 0) {
            numngb_inbox = ngb_treefind_pairs_threads_targeted(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist, SINK_NEIGHBOR_BITFLAG);
            if(numngb_inbox < 0) {return -2;} /* no neighbors! */
            for(n = 0; n < numngb_inbox; n++) /* neighbor loop */
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if((P[j].Mass <= 0)||(P[j].KernelRadius <= 0)||(P[j].Type == 5)) {continue;} /* make sure neighbor is valid */
                int k; Vec3<double> dP = P[j].Pos - local.Pos; Vec3<double> dv = P[j].Vel - local.Vel; /* position offset */
                nearest_xyz(dP,-1);
                NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,dv,-1); /* wrap velocities for shearing boxes if needed */
                double J_tmp[3]; J_tmp[0]=dP[1]*dv[2]-dP[2]*dv[1]; J_tmp[1]=dP[2]*dv[0]-dP[0]*dv[2]; J_tmp[2]=dP[0]*dv[1]-dP[1]*dv[0]; /* just need direction not magnitude */
                if(P[j].Type==0) {if(J_tmp[0]*local.Jgas_in_Kernel[0] + J_tmp[1]*local.Jgas_in_Kernel[1] + J_tmp[2]*local.Jgas_in_Kernel[2] < 0) {out.MgasBulge_in_Kernel += 2*P[j].Mass;}} /* DAA: assume the bulge component contains as many particles with positive azimuthal velocities as with negative azimuthal velocities relative to the angular momentum vector */
                if(P[j].Type==4 || ((P[j].Type==2||P[j].Type==3) && !(All.ComovingIntegrationOn))) {if(J_tmp[0]*local.Jstar_in_Kernel[0] + J_tmp[1]*local.Jstar_in_Kernel[1] + J_tmp[2]*local.Jstar_in_Kernel[2] < 0) {out.MstarBulge_in_Kernel += 2*P[j].Mass;}}
            } // numngb_inbox loop
        } // while(startnode)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    }
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
}

void sink_environment_second_loop(void)
{
#include "../../system/code_block_xchange_perform_ops_malloc.h" /* this calls the large block of code which contains the memory allocations for the MPI/OPENMP/Pthreads parallelization block which must appear below */
#include "../../system/code_block_xchange_perform_ops.h" /* this calls the large block of code which actually contains all the loops, MPI/OPENMP/Pthreads parallelization */
#include "../../system/code_block_xchange_perform_ops_demalloc.h" /* this de-allocates the memory for the MPI/OPENMP/Pthreads parallelization block which must appear above */
CPU_Step[CPU_SINKS] += measure_time(); /* collect timings and reset clock for next timing */
}
#include "../../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */

#endif   //SINK_GRAVACCRETION == 0




#endif // top-level flag
