/*! \file sink_swallow_and_kick.c
*  \brief routines for gas accretion onto sink particles, and sink particle mergers
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/


#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //


static int N_gas_swallowed, N_star_swallowed, N_dm_swallowed, N_sink_swallowed;

#ifdef SINK_ALPHADISK_ACCRETION
#define out_accreted_Sink_Mass_alphaornot out.accreted_Sink_Mass_reservoir
#else
#define out_accreted_Sink_Mass_alphaornot out.accreted_Sink_Mass
#endif


#define CORE_FUNCTION_NAME sink_swallow_and_kick_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(sink_isactive(i) && P[i].SwallowID==0) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */


/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    int NodeList[NODELISTLENGTH]; Vec3<MyDouble> Pos; Vec3<MyFloat> Vel; MyFloat KernelRadius, Mass, Sink_Mass, Dt, Mdot; MyIDType ID, ID_child_number, ID_generation;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
    MyFloat Jgas_in_Kernel[3];
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    MyFloat Sink_angle_weighted_kernel_sum;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    Vec3<MyFloat> Sink_Specific_AngMom; MyFloat angmom_norm_topass_in_swallowloop;
#endif
#if defined(SINK_RETURN_BFLUX)
    MyFloat B[3];
    MyFloat kernel_norm_topass_in_swallowloop;
#endif
#ifdef SINGLE_STAR_FB_LOCAL_RP
    MyFloat Luminosity;
#endif    
}
*DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k, j_tempinfo; j_tempinfo = P[i].IndexMapToTempStruc; /* link to the location in the shared structure where this is stored */
    in->Pos=P[i].Pos; in->Vel=P[i].Vel; /* good example - always needed */
    in->KernelRadius = P[i].KernelRadius; in->Mass = P[i].Mass; in->Sink_Mass = P[i].Sink_Mass; in->ID = P[i].ID; in->ID_child_number=P[i].ID_child_number; in->ID_generation=P[i].ID_generation; in->Mdot = P[i].Sink_Mdot;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    in->Jgas_in_Kernel[0]=P[i].Sink_Specific_AngMom[0]; in->Jgas_in_Kernel[1]=P[i].Sink_Specific_AngMom[1]; in->Jgas_in_Kernel[2]=P[i].Sink_Specific_AngMom[2];
#else
    in->Jgas_in_Kernel[0]=SinkTempInfo[j_tempinfo].Jgas_in_Kernel[0]; in->Jgas_in_Kernel[1]=SinkTempInfo[j_tempinfo].Jgas_in_Kernel[1]; in->Jgas_in_Kernel[2]=SinkTempInfo[j_tempinfo].Jgas_in_Kernel[2];
#endif
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    in->Sink_Mass_Reservoir = P[i].Sink_Mass_Reservoir;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    in->Sink_angle_weighted_kernel_sum = SinkTempInfo[j_tempinfo].Sink_angle_weighted_kernel_sum;
#endif
    in->Dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    in->Dt = P[i].dt_since_last_gas_search;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    in->Sink_Specific_AngMom = P[i].Sink_Specific_AngMom;
    in->angmom_norm_topass_in_swallowloop = SinkTempInfo[j_tempinfo].angmom_norm_topass_in_swallowloop;
#endif
#if defined(SINK_RETURN_BFLUX)
    in->B[0]=P[i].B[0]; in->B[1]=P[i].B[1]; in->B[2]=P[i].B[2];
    in->kernel_norm_topass_in_swallowloop = SinkTempInfo[j_tempinfo].kernel_norm_topass_in_swallowloop;
#endif
#ifdef SINGLE_STAR_FB_LOCAL_RP
    in->Luminosity = sink_lum_bol(in->Mdot, in->Sink_Mass, i);
#endif    
}


/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{ /* define variables below as e.g. "double X;" */
    MyDouble accreted_Mass;
    MyDouble accreted_Sink_Mass;
    MyDouble accreted_Sink_Mass_reservoir;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyDouble Sink_AccretionDeficit;
#endif
#ifdef GRAIN_FLUID
    MyDouble accreted_dust_Mass;
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    MyDouble accreted_photon_energy;
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    MyDouble accreted_momentum[3];
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
    MyDouble accreted_centerofmass[3];
#endif
#if defined(SINK_RETURN_BFLUX)
    MyDouble accreted_B[3];
//    MyDouble accreted_Phi;
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    MyDouble accreted_J[3];
#endif
#ifdef SINK_COUNTPROGS
    int Sink_CountProgs;
#endif
#ifdef GALSF
    MyFloat Accreted_Age;
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

#define ASSIGN_ADD_PRESET(x,y,mode) (mode == 0 ? (x=y) : (x+=y))
/* this subroutine assigns the values to the variables that need to be sent -back to- the 'searching' element */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k, target = P[i].IndexMapToTempStruc; k=0;
    ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_Mass, out->accreted_Mass, mode);
    ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_Sink_Mass, out->accreted_Sink_Mass, mode);
    ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_Sink_Mass_reservoir, out->accreted_Sink_Mass_reservoir, mode);
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    ASSIGN_ADD_PRESET(SinkTempInfo[target].Sink_AccretionDeficit, out->Sink_AccretionDeficit, mode);
#endif
#ifdef GRAIN_FLUID
    ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_dust_Mass, out->accreted_dust_Mass, mode);
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
    ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_photon_energy, out->accreted_photon_energy, mode);
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
    for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_momentum[k], out->accreted_momentum[k], mode);}
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
    for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_centerofmass[k], out->accreted_centerofmass[k], mode);}
#endif
#if defined(SINK_RETURN_BFLUX)
    for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_B[k], out->accreted_B[k], mode);}
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    for(k=0;k<3;k++) {ASSIGN_ADD_PRESET(SinkTempInfo[target].accreted_J[k], out->accreted_J[k], mode);}
#endif
#ifdef SINK_COUNTPROGS
    P[i].Sink_CountProgs += out->Sink_CountProgs;
#endif
#ifdef GALSF
    if(P[i].StellarAge > out->Accreted_Age) {P[i].StellarAge = out->Accreted_Age;}
#endif
}


/* subroutine that handles the 'final' swallowing and kicking operations for coupling feedback or accretion between sinks and neighboring elements */
/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below. there's a lot of writing, so this needs to be done carefully. */
int sink_swallow_and_kick_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration);
int sink_swallow_and_kick_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb, listindex = 0, j, k, n, bin; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */
    double h_i=local.KernelRadius, hinv=1/h_i, hinv3, f_accreted; hinv3=hinv*hinv*hinv; f_accreted=0;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    double kernel_zero,dwk; kernel_main(0.0,1.0,1.0,&kernel_zero,&dwk,-1); dwk=0;
#ifdef SINGLE_STAR_FB_LOCAL_RP
    double mom = local.Luminosity * local.Dt / C_LIGHT_CODE, mom_wt = 0;
#else    
    double mom = sink_lum_bol(local.Mdot, local.Sink_Mass, -1) * local.Dt / C_LIGHT_CODE, mom_wt = 0;
#endif
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) || defined(SINK_WIND_KICK)
    Vec3<double> J_dir{(double)local.Jgas_in_Kernel[0], (double)local.Jgas_in_Kernel[1], (double)local.Jgas_in_Kernel[2]};
    double norm = J_dir.norm_sq();
    if(norm>0) {norm=1/sqrt(norm); J_dir *= norm;} else {J_dir[0]=J_dir[1]=0; J_dir[2]=1;}
#endif
#if defined(SINK_WIND_KICK)
    double sink_mass_withdisk=local.Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
    sink_mass_withdisk += local.Sink_Mass_Reservoir;
#endif
#endif
#ifdef GALSF
    out.Accreted_Age = MAX_REAL_NUMBER;
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
                MyIDType OriginallyMarkedSwallowID; OriginallyMarkedSwallowID = P[j].SwallowID; // record this to help prevent double-counting below
                double Mass_j, Vel_j[3], InternalEnergy_j; // velocity kicks to apply to 'j's, accumulate here to apply at bottom of code
                #pragma omp atomic read
                Mass_j = P[j].Mass; // this can get modified -a lot- below, so we need to read it carefully right now
                if(P[j].Type==0)
                {
                    #pragma omp atomic read
                    InternalEnergy_j = CellP[j].InternalEnergy; // this can get modified -a lot- below, so we need to read it carefully right now
                }
                for(k=0;k<3;k++)
                {
                    #pragma omp atomic read
                    Vel_j[k] = P[j].Vel[k]; // this can get modified -a lot- below, so we need to read it carefully right now
                }
                double Mass_j_0 = Mass_j, InternalEnergy_j_0 = InternalEnergy_j, Vel_j_0[3]; for(k=0;k<3;k++) {Vel_j_0[k]=Vel_j[k];} // save initial values to know if we need to update neighbor values below
                
                Vec3<double> dpos = P[j].Pos - local.Pos;
                Vec3<double> dvel{Vel_j[0]-local.Vel[0], Vel_j[1]-local.Vel[1], Vel_j[2]-local.Vel[2]};
                nearest_xyz(dpos,-1); /*  find the closest image in the given box size  */
                NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,dvel,-1); /* wrap velocities for shearing boxes if needed */
                double r2 = dpos.norm_sq();

#if defined(SINK_RETURN_ANGMOM_TO_GAS) || defined(SINK_RETURN_BFLUX)
                double wk, dwk, u=0;
                if(P[j].Type == 0){
                    u=sqrt(r2)/DMAX(h_i, P[j].KernelRadius); if(u<1) { kernel_main(u,1., 1.,&wk,&dwk,-1); } else {wk=dwk=0;}
                }
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS) /* this should go here [right before the loop that accretes it back onto the BH] */
                if(P[j].Type == 0)
                {
                    double dlv[3]; dlv[0]=local.Sink_Specific_AngMom[1]*dpos[2]-local.Sink_Specific_AngMom[2]*dpos[1]; dlv[1]=local.Sink_Specific_AngMom[2]*dpos[0]-local.Sink_Specific_AngMom[0]*dpos[2]; dlv[2]=local.Sink_Specific_AngMom[0]*dpos[1]-local.Sink_Specific_AngMom[1]*dpos[0];
                    for(k=0;k<3;k++) {
                        dlv[k] *= wk * local.angmom_norm_topass_in_swallowloop;
                        Vel_j[k] += dlv[k];
                        out.accreted_momentum[k]-=Mass_j*dlv[k];
                    }
                    out.accreted_J[0]-=Mass_j*(dpos[1]*dlv[2] - dpos[2]*dlv[1]); out.accreted_J[1]-=Mass_j*(dpos[2]*dlv[0] - dpos[0]*dlv[2]); out.accreted_J[2]-=Mass_j*(dpos[0]*dlv[1] - dpos[1]*dlv[0]);
                }
#endif
#if defined(SINK_RETURN_BFLUX) // do a kernel-weighted redistribution of the magnetic flux in the sink into surrounding particles
                if((P[j].Type == 0) && (local.kernel_norm_topass_in_swallowloop > 0)){
                    double dB, b_fraction_toreturn = DMIN(0.1, local.Dt / (local.Sink_Mass_Reservoir / local.Mdot)) * wk / local.kernel_norm_topass_in_swallowloop; // return a fraction dt/t_accretion of the total flux, with simple kernel weighting for each particle
                    for(k=0; k<3;k++) {
                        dB = b_fraction_toreturn * local.B[k];
                        #pragma omp atomic
                        CellP[j].B[k] += dB;
                        #pragma omp atomic
                        CellP[j].BPred[k] +=  dB;
                        out.accreted_B[k] -= dB;
                    }
                }
#endif
                
                
                /* we've found a particle to be swallowed.  This could be a sink merger, DM particle, or baryon w/ feedback */
                if(P[j].SwallowID == local.ID && Mass_j > 0 && r2 > 0)
                {   /* accreted quantities to be added [regardless of particle type] */
                    f_accreted = 1; /* default to accreting entire particle */
#ifdef SINK_WIND_KICK
                    if(P[j].Type == 0)
                    {
                        f_accreted = All.Sink_accreted_fraction; /* if particle is gas, only a fraction gets accreted in these particular modules */
#ifndef SINK_GRAVCAPTURE_GAS
                        if((All.SinkFeedbackFactor > 0) && (All.SinkFeedbackFactor != 1.)) {f_accreted /= All.SinkFeedbackFactor;} else {if(All.Sink_outflow_velocity > 0) f_accreted = 1./(1. + fabs(1.*SINK_WIND_KICK)*All.SinkRadiativeEfficiency*C_LIGHT_CODE/(All.Sink_outflow_velocity));}
                        if((sink_mass_withdisk - local.Mass) <= 0) {f_accreted=0;} // DAA: no need to accrete gas particle to enforce mass conservation (we will simply kick),  note that here the particle mass P.Mass is larger than the physical sink mass P.Sink_Mass
#endif
                    }
#endif

                    
                    /* handle accretion/conservation of certain conserved quantities, depending on whether we are intending our sub-grid model to follow them */
                    double mcount_for_conserve; mcount_for_conserve = f_accreted * Mass_j;
#if (SINK_FOLLOW_ACCRETED_ANGMOM == 1) /* in this case we are only counting this if its coming from sink particles */
                    if(P[j].Type != 5) {mcount_for_conserve=0;} else {mcount_for_conserve=P[j].Sink_Mass;}
#ifdef SINK_ALPHADISK_ACCRETION
                    if(P[j].Type == 5) {mcount_for_conserve += P[j].Sink_Mass_Reservoir;}
#endif
#endif
#ifdef GRAIN_FLUID
                    if((1<<P[j].Type) & GRAIN_PTYPES) {out.accreted_dust_Mass += (Mass_j);}
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS 
		            if(P[j].Type == 0) { // we have to keep track of how much radiation energy is lost when we accrete this gas cell, and reinject it later
			        double photon_energy = 0; int kfreq;
			        for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++) {photon_energy += CellP[j].Rad_E_gamma[kfreq];}
                    out.accreted_photon_energy += photon_energy;
		            }
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM)
                    for(k=0;k<3;k++) {out.accreted_momentum[k] += ( mcount_for_conserve * dvel[k]);}
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM)
                    for(k=0;k<3;k++) {out.accreted_centerofmass[k] += (mcount_for_conserve * dpos[k]);}
#endif
#ifdef SINK_RETURN_BFLUX
                    for(k=0;k<3;k++) {out.accreted_B[k] += (CellP[j].BPred[k]);}
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
                    out.accreted_J[0] += (mcount_for_conserve * ( dpos[1]*dvel[2] - dpos[2]*dvel[1] ));
                    out.accreted_J[1] += (mcount_for_conserve * ( dpos[2]*dvel[0] - dpos[0]*dvel[2] ));
                    out.accreted_J[2] += (mcount_for_conserve * ( dpos[0]*dvel[1] - dpos[1]*dvel[0] ));
                    if(P[j].Type == 5) {for(k=0;k<3;k++) {out.accreted_J[k] += (mcount_for_conserve * P[j].Sink_Specific_AngMom[k]);}}
#endif

                    
                    
                    if(P[j].Type == 5)  /* this is a sink-sink merger */
                    {
#ifdef SINK_OUTPUT_MOREINFO
                        fprintf(FdSinkMergerDetails,"%.16g  %llu %llu %llu %g %2.16g %2.16g %2.16g  %llu %llu %llu %g %2.16g %2.16g %2.16g\n",All.Time,(unsigned long long)local.ID,(unsigned long long)local.ID_child_number,(unsigned long long)local.ID_generation,local.Sink_Mass,local.Pos[0],local.Pos[1],local.Pos[2],  (unsigned long long)P[j].ID,(unsigned long long)P[j].ID_child_number,(unsigned long long)P[j].ID_generation,P[j].Sink_Mass,P[j].Pos[0],P[j].Pos[1],P[j].Pos[2]); fflush(FdSinkMergerDetails);
#elif defined(OUTPUT_ADDITIONAL_RUNINFO)
                        fprintf(FdSinksDetails,"Sink-Sink Merger Occuring: ThisTask=%d, time=%.16g: id=%llu swallows %llu (%g %g)\n", ThisTask, All.Time, (unsigned long long)local.ID, (unsigned long long)P[j].ID, local.Sink_Mass, P[j].Sink_Mass); fflush(FdSinksDetails);
#endif
#ifdef SINK_INCREASE_DYNAMIC_MASS
                        /* the true dynamical mass of the merging sink is Mass_j/SINK_INCREASE_DYNAMIC_MASS unless exceeded by physical growth
                         - in the limit P[j].Sink_Mass > SINK_INCREASE_DYNAMIC_MASS x m_b, then sink_mass=Mass_j on average and we are good as well  */
                        out.accreted_Mass    += ( DMAX(P[j].Sink_Mass, Mass_j/SINK_INCREASE_DYNAMIC_MASS) );
#else
                        out.accreted_Mass    += (Mass_j);
#endif
                        out.accreted_Sink_Mass += (P[j].Sink_Mass);
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
                        out.Sink_AccretionDeficit += (P[j].Sink_AccretionDeficit);
#endif
#ifdef SINK_ALPHADISK_ACCRETION
                        out.accreted_Sink_Mass_reservoir += (P[j].Sink_Mass_Reservoir);
#endif
#ifdef SINK_WIND_SPAWN
                        out_accreted_Sink_Mass_alphaornot += (P[j].unspawned_wind_mass);
#endif
#ifdef SINK_COUNTPROGS
                        out.Sink_CountProgs += P[j].Sink_CountProgs;
#endif
                        bin = P[j].TimeBin;
                        #pragma omp atomic
                        TimeBin_Sink_mass[bin] -= P[j].Sink_Mass;
                        #pragma omp atomic
                        TimeBin_Sink_dynamicalmass[bin] -= Mass_j;
                        #pragma omp atomic
                        TimeBin_Sink_Mdot[bin] -= P[j].Sink_Mdot;
                        if(P[j].Sink_Mass > 0) {
                            #pragma omp atomic
                            TimeBin_Sink_Medd[bin] -= P[j].Sink_Mdot / P[j].Sink_Mass;
                        }
                        Mass_j = 0;
#ifdef SINK_ALPHADISK_ACCRETION
                        #pragma omp atomic write
                        P[j].Sink_Mass_Reservoir = 0; // make sure the mass is -actually- zero'd here
#endif
                        #pragma omp atomic write
                        P[j].Sink_Mdot = 0; // make sure the mass is -actually- zero'd here
                        #pragma omp atomic write
                        P[j].Sink_Mass = 0; // make sure the mass is -actually- zero'd here
#ifdef GALSF
                        out.Accreted_Age = P[j].StellarAge;
#endif
                        #pragma omp atomic
                        N_sink_swallowed++;
                    } // if(P[j].Type == 5) -- sink+sink merger


                    
                    
#if defined(SINK_GRAVCAPTURE_NONGAS) /* DM and star particles can only be accreted ifdef SINK_GRAVCAPTURE_NONGAS */
                    if((P[j].Type > 0) && (P[j].Type < 5))
                    {
                        out.accreted_Mass += (Mass_j); /* account for the swallowed mass in the total mass budget */
                        if((P[j].Type == 1) || (All.ComovingIntegrationOn && (P[j].Type==2||P[j].Type==3)) ) { /* this is a DM particle: In this case, no kick, so just zero out the mass and 'get rid of' the particle (preferably by putting it somewhere irrelevant) */
                            out.accreted_Sink_Mass += (Mass_j); /* if using simple excision, adds to the particle mass, but not assumed to actually be accreted */
                            #pragma omp atomic
                            N_dm_swallowed++;
                        } else { /* this is a star particle: If there is an alpha-disk, we let them go to the disk. If there is no alpha-disk, stars go to the sink directly and won't affect feedback. (Can be simply modified if we need something different.) */
                            out_accreted_Sink_Mass_alphaornot += (Mass_j); /* if using simple excision, adds to the particle mass, but not assumed to actually be accreted */
                            #pragma omp atomic
                            N_star_swallowed++;
                        }
                        Mass_j = 0; /* zero the mass because its been accreted now */
                    }
#endif // close for -- sink + DM or Star merger


                    
                    
                    /* this is a gas particle: DAA: we need to see if the gas particle has to be accreted in full or not, depending on SINK_WIND_KICK
                     the only difference with SINK_ALPHADISK_ACCRETION should be that the mass goes first to the reservoir */
                    if(P[j].Type == 0)
                    {
                        out.accreted_Mass += (f_accreted*Mass_j);
#ifdef SINK_GRAVCAPTURE_GAS
                        out_accreted_Sink_Mass_alphaornot += (f_accreted*Mass_j);
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
                        out.Sink_AccretionDeficit -= (f_accreted*Mass_j); /* account for this in the 'continuous accretion' budget, since it is part of the continuous Mdot onto the sink */
#endif
                        double Mass_initial = Mass_j; // save this for possible IO below
                        Mass_j *= (1-f_accreted);
#ifdef SINK_WIND_KICK     /* BAL kicking operations. NOTE: we have two separate BAL wind models, particle kicking and smooth wind model. This is where we do the particle kicking BAL model. This should also work when there is alpha-disk. */
                        double v_kick=All.Sink_outflow_velocity, dir[3]={dpos[0],dpos[1],dpos[2]}; // DAA: default direction is radially outwards
#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS) /* inject cosmic rays alongside wind injection */
                        double dEcr = All.Sink_CosmicRay_Injection_Efficiency * Mass_j * (All.Sink_accreted_fraction/(1.-All.Sink_accreted_fraction)) * C_LIGHT_CODE*C_LIGHT_CODE;
                        inject_cosmic_rays(dEcr,All.Sink_outflow_velocity,5,j,dir);
#endif
#if (SINK_WIND_KICK < 0)  /* DAA: along polar axis defined by angular momentum within Kernel (we could add finite opening angle) work out the geometry w/r to the plane of the disk */
                        if(dot(Vec3<double>{dir[0],dir[1],dir[2]},J_dir) > 0){dir[0]=J_dir[0];dir[1]=J_dir[1];dir[2]=J_dir[2];} else {dir[0]=-J_dir[0];dir[1]=-J_dir[1];dir[2]=-J_dir[2];}
#endif
                        for(k=0,norm=0;k<3;k++) {norm+=dir[k]*dir[k];} if(norm<=0) {dir[0]=0;dir[1]=0;dir[2]=1;norm=1;} else {norm=sqrt(norm); dir[0]/=norm;dir[1]/=norm;dir[2]/=norm;}
                        for(k=0;k<3;k++) {Vel_j[k]+=v_kick*All.cf_atime*dir[k];}
#ifdef GALSF_SUBGRID_WINDS // if sub-grid galactic winds are decoupled from the hydro, we decouple the sink kick winds as well
                        #pragma omp atomic write
                        CellP[j].DelayTime = All.WindFreeTravelMaxTimeFactor / All.cf_hubble_a;
#endif
#ifdef SINK_OUTPUT_MOREINFO
                        printf(" ..BAL kick: P[j].ID %llu ID %llu Type(j) %d f_acc %g M(j) %g V(j).xyz %g/%g/%g P(j).xyz %g/%g/%g p(i).xyz %g/%g/%g v_out %g \n",(unsigned long long) P[j].ID, (unsigned long long) P[j].SwallowID,P[j].Type, All.Sink_accreted_fraction,Mass_j,Vel_j[0],Vel_j[1],Vel_j[2],P[j].Pos[0],P[j].Pos[1],P[j].Pos[2],local.Pos[0],local.Pos[1],local.Pos[2],v_kick);
                        fprintf(FdSinkWindDetails,"%.16g  %llu %g  %2.16g %2.16g %2.16g  %2.16g %2.16g %2.16g %g %g %g %llu  %2.16g %2.16g %2.16g\n",All.Time, (unsigned long long)P[j].ID, Mass_j, P[j].Pos[0],P[j].Pos[1],P[j].Pos[2],  Vel_j[0],Vel_j[1],Vel_j[2],dir[0]/norm,dir[1]/norm,dir[2]/norm, (unsigned long long)local.ID, local.Pos[0],local.Pos[1],local.Pos[2]); fflush(FdSinkWindDetails);
#endif
#endif // #ifdef SINK_WIND_KICK
                        #pragma omp atomic
                        N_gas_swallowed++;
#ifdef OUTPUT_SINK_ACCRETION_HIST
                        MyDouble tempB[3]={0,0,0};
#ifdef MAGNETIC
                        for(k=0;k<3;k++) {tempB[k]=Get_Gas_BField(j,k);} //use particle magnetic field
#endif
                        fprintf(FdSinkSwallowDetails,"%.16g %llu %llu %llu %g %2.16g %2.16g %2.16g %llu %llu %llu %g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g %2.16g\n", All.Time, (unsigned long long)local.ID,(unsigned long long)local.ID_child_number,(unsigned long long)local.ID_generation,local.Mass,local.Pos[0],local.Pos[1],local.Pos[2],  (unsigned long long)P[j].ID, (unsigned long long)P[j].ID_child_number, (unsigned long long)P[j].ID_generation, Mass_initial, (P[j].Pos[0]-local.Pos[0]),(P[j].Pos[1]-local.Pos[1]),(P[j].Pos[2]-local.Pos[2]), (Vel_j[0]-local.Vel[0]),(Vel_j[1]-local.Vel[1]),(Vel_j[2]-local.Vel[2]), CellP[j].InternalEnergy, tempB[0], tempB[1], tempB[2], CellP[j].Density); fflush(FdSinkSwallowDetails);
#endif
                    }  // if(P[j].Type == 0)
                    //P[j].SwallowID = 0; /* DAA: make sure it is not accreted (or ejected) by the same sink again if inactive in the next timestep [PFH: no longer necessary with the new way we re-initialize the SwallowIDs] */
                } // if(P[j].SwallowID == id)  -- particles being entirely or partially swallowed

                
                
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
                /* now, do any other feedback "kick" operations (which used the previous loops to calculate weights) */
                if(mom>0 && local.Dt>0 && OriginallyMarkedSwallowID==0 && P[j].SwallowID==0 && Mass_j>0 && P[j].Type==0) // particles NOT being swallowed!
                {
                    double r=dpos.norm_sq(); double dir[3]={dpos[0],dpos[1],dpos[2]}; // should be away from BH
                    if(r>0)
                    {
                        r=sqrt(r); dir[0]/=r; dir[1]/=r; dir[2]/=r; /* cos_theta with respect to disk of sink is given by dot product of r and Jgas */
                        norm=dot(Vec3<double>{dir[0],dir[1],dir[2]},J_dir);
                        mom_wt = sink_fb_angleweight_localcoupling(j,norm,r,h_i) / local.Sink_angle_weighted_kernel_sum;
                        if(local.Sink_angle_weighted_kernel_sum<=0) {mom_wt=0;}
#ifdef SINK_PHOTONMOMENTUM /* inject radiation pressure: add initial L/c optical/UV coupling to the gas at the dust sublimation radius */
                        double v_kick = All.Sink_Rad_MomentumFactor * mom_wt * mom / Mass_j;
                        Vel_j[0]+=v_kick*All.cf_atime*dir[0]; Vel_j[1]+=v_kick*All.cf_atime*dir[1]; Vel_j[2]+=v_kick*All.cf_atime*dir[2];
#endif
                    } // r > 0
                } // (check if valid gas neighbor of interest)
#endif // defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
                
                /* ok, now deal with coupling back terms to the actual neighbor 'j'. to do this we need to be careful to ensure thread-safety */
                if(Mass_j != Mass_j_0 || Vel_j[0]-Vel_j_0[0] != 0 || Vel_j[1]-Vel_j_0[1] != 0 || Vel_j[2]-Vel_j_0[2] != 0 || InternalEnergy_j != InternalEnergy_j_0)
                {
                    /* we updated variables that need to get assigned to element 'j' -- let's do it */
                    if(Mass_j > 0)
                    {
                        // do one more check here to make sure we don't accidentally 'resurrect' an element with zero mass set by a different process. technically it is conceivable something could come in the meantime but this makes it vanishingly unlikely [we hope]
                        double Mass_j_0_n = 0;
                        #pragma omp atomic read
                        Mass_j_0_n = P[j].Mass;
                        if(Mass_j_0_n > 0)
                        {
                            #pragma omp atomic
                            P[j].Mass += Mass_j - Mass_j_0; // finite mass update [delta difference added here, allowing for another element to update in the meantime]
                        }
                    } else {
                        #pragma omp atomic write
                        P[j].Mass = 0; // make sure the mass is -actually- zero'd here
                    }
                    
                    for(k=0;k<3;k++) {
                        #pragma omp atomic
                        P[j].Vel[k] += Vel_j[k] - Vel_j_0[k]; // discrete momentum change
                        #pragma omp atomic
                        P[j].dp[k] += Mass_j * Vel_j[k] - Mass_j_0*Vel_j_0[k]; // discrete momentum change
                    }

                    if(P[j].Type==0) // gas variable, need to update the 'pred' and truemass and related variables as well
                    {
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                        if(Mass_j > 0)
                        {
                            #pragma omp atomic
                            CellP[j].MassTrue += Mass_j - Mass_j_0; // finite mass update
                        } else {
                            #pragma omp atomic write
                            CellP[j].MassTrue = 0; // make sure the mass is -actually- zero'd here
                        }
#endif
                        for(k=0;k<3;k++) {
                            #pragma omp atomic
                            CellP[j].VelPred[k] += Vel_j[k] - Vel_j_0[k]; // delta-update
                        }
                        #pragma omp atomic
                        CellP[j].InternalEnergy += InternalEnergy_j - InternalEnergy_j_0; // delta-update
                        #pragma omp atomic
                        CellP[j].InternalEnergyPred += InternalEnergy_j - InternalEnergy_j_0; // delta-update
                    }
                }
                
            } // for(n = 0; n < numngb; n++)
        } // while(startnode >= 0)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    } // while(startnode >= 0) (outer of the double-loop)
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
} /* closes evaluate_swallow routine */



void sink_swallow_and_kick_loop(void)
{
    N_gas_swallowed = N_star_swallowed = N_dm_swallowed = N_sink_swallowed = 0;
    #include "../system/code_block_xchange_perform_ops_malloc.h" /* this calls the large block of code which contains the memory allocations for the MPI/OPENMP/Pthreads parallelization block which must appear below */
    #include "../system/code_block_xchange_perform_ops.h" /* this calls the large block of code which actually contains all the loops, MPI/OPENMP/Pthreads parallelization */
    #include "../system/code_block_xchange_perform_ops_demalloc.h" /* this de-allocates the memory for the MPI/OPENMP/Pthreads parallelization block which must appear above */
    /* collect and print results on any swallow operations in this pass */
    int Ntot_gas_swallowed=0, Ntot_star_swallowed=0, Ntot_dm_swallowed=0, Ntot_sink_swallowed=0;
    MPI_Reduce(&N_gas_swallowed, &Ntot_gas_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_sink_swallowed, &Ntot_sink_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_star_swallowed, &Ntot_star_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_dm_swallowed, &Ntot_dm_swallowed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if((ThisTask == 0)&&(Ntot_gas_swallowed+Ntot_star_swallowed+Ntot_dm_swallowed+Ntot_sink_swallowed>0))
    {
        printf("Accretion done: swallowed %d gas, %d star, %d dm, and %d sink particles\n",
               Ntot_gas_swallowed, Ntot_star_swallowed, Ntot_dm_swallowed, Ntot_sink_swallowed);
    }
    CPU_Step[CPU_SINKS] += measure_time(); /* collect timings and reset clock for next timing */
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */





#ifdef SINK_WIND_SPAWN
void spawn_sink_wind_feedback(void)
{
    int i, n_particles_split = 0, MPI_n_particles_split, dummy_gas_tag=0;
    for(i = 0; i < NumPart; i++)
        if(P[i].Type==0)
        {
            dummy_gas_tag=i;
            break;
        }

    /* don't loop or go forward if there are no gas particles in the domain, or the code will crash */
    for (int i : ActiveParticleList)
    {
        long nmax = (int)(0.99*All.MaxPart); if(All.MaxPart-20 < nmax) nmax=All.MaxPart-20; int ptype_can_spawn = 0; if(P[i].Type == 5) {ptype_can_spawn = 1;}
#ifdef SNE_NONSINK_SPAWN
        if(P[i].Type == 4) {ptype_can_spawn = 1;}
#endif
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4)
        if(is_particle_a_special_zoom_target(i)) {ptype_can_spawn = 1;}
#endif
        if((NumPart+n_particles_split+(int)(2.*(SINK_WIND_SPAWN+0.1)) < nmax) && (ptype_can_spawn==1)) // basic condition: particle is a 'spawner' (sink), and code can handle the event safely without crashing.
        {
            int sink_eligible_to_spawn = 0; // flag to check eligibility for spawning
            if(P[i].unspawned_wind_mass >= (SINK_WIND_SPAWN)*target_mass_for_wind_spawning(i)) {sink_eligible_to_spawn=1;} // have 'enough' mass to spawn
#if defined(SINGLE_STAR_SINK_DYNAMICS)
            if(P[i].Type==5) {if((P[i].Mass <= 3.5*P[i].Sink_Formation_Mass) || (P[i].Sink_Mass*UNIT_MASS_IN_SOLAR < 0.01)) {sink_eligible_to_spawn=0;}}  // spawning causes problems in these modules for low-mass sinks, so arbitrarily restrict to this, since it's roughly a criterion on the minimum particle mass. and for <0.01 Msun, in pre-collapse phase, no jets
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
            if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {sink_eligible_to_spawn=1;}} // spawn the SNe ejecta no matter what the sink or 'unspawned' mass flag actually is
#endif
#endif
            if(sink_eligible_to_spawn)
            {
                int j; dummy_gas_tag=-1; double r2=MAX_REAL_NUMBER;
                for(j=0; j<N_gas; j++) /* find the closest gas particle on the domain to act as the dummy */
                {
                    if(P[j].Type==0)
                    {
                        if((P[j].Mass>0) && (CellP[j].Density>0) && (CellP[j].recent_refinement_flag==0))
                        {
                            double dx2=(P[j].Pos[0]-P[i].Pos[0])*(P[j].Pos[0]-P[i].Pos[0]) + (P[j].Pos[1]-P[i].Pos[1])*(P[j].Pos[1]-P[i].Pos[1]) + (P[j].Pos[2]-P[i].Pos[2])*(P[j].Pos[2]-P[i].Pos[2]);
                            if(dx2 < r2) {r2=dx2; dummy_gas_tag=j;}
                        }
                    }
                }
                if(dummy_gas_tag >= 0)
                {
                    n_particles_split += sink_spawn_particle_wind_shell( i , dummy_gas_tag, n_particles_split);
                }
            }
        }
    }
    MPI_Allreduce(&n_particles_split, &MPI_n_particles_split, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(MPI_n_particles_split>0) {
#ifdef MAINTAIN_TREE_IN_REARRANGE
        All.NumForcesSinceLastDomainDecomp +=  0.0001 * All.TreeDomainUpdateFrequency * All.TotNumPart; // we can insert spawned particles in the tree, but still a good idea to rebuild the tree every now and then, so we make the next domain+treebuild come a bit sooner; additional cost should be small
#else
        TreeReconstructFlag = 1; // otherwise just wipe and rebuild the tree next chance you get - more expensive but more accurate
#endif
        if(ThisTask == 0) {printf(" ..Sink-Spawn Event: %d particles spawned \n", MPI_n_particles_split);}
    }

    /* rearrange_particle_sequence -must- be called immediately after this routine! */
    All.TotNumPart += (long long)MPI_n_particles_split;
    All.TotN_gas   += (long long)MPI_n_particles_split;
    Gas_split       = n_particles_split;                    // specific to the local processor //
}


void get_random_orthonormal_basis(int seed, Vec3<double>& nx, Vec3<double>& ny, Vec3<double>& nz)
{
    double phi, cos_theta, sin_theta, sin_phi, cos_phi;
    phi=2.*M_PI*get_random_number(seed+1+ThisTask), cos_theta=2.*(get_random_number(seed+3+2*ThisTask)-0.5); sin_theta=sqrt(1-cos_theta*cos_theta), sin_phi=sin(phi), cos_phi=cos(phi);
    /* velocities (determined by wind velocity direction) */
    nz = {sin_theta*cos_phi, sin_theta*sin_phi, cos_theta}; // random z axis

    double norm=0;
    while(norm==0){ // necessary in case ny is parallel to nz - believe it or not this happened once!
        phi=2.*M_PI*get_random_number(seed+4+ThisTask), cos_theta=2.*(get_random_number(seed+5+2*ThisTask)-0.5); sin_theta=sqrt(1-cos_theta*cos_theta), sin_phi=sin(phi), cos_phi=cos(phi);
        ny = {sin_theta*cos_phi, sin_theta*sin_phi, cos_theta}; // random y axis, needs to have its z component deprojected
        // do Gram-Schmidt to get an orthonormal basis
        ny -= nz * dot(ny, nz); // deproject component along z
        norm = ny.norm_sq();
        if(norm==0) continue;
        ny *= 1./sqrt(norm);
    }
    nx = cross(ny, nz);
    return;
}

/* Convenience function to compute the direction to launch a wind particle                                      */
/*                                                                                                              */
/* i - index of particle doing the spawning                                                                     */
/* num_spawned_this_call - how many we have already spawned in this call of sink_spawn_particle_wind_shell */
/* mode - 0 for random, 1 for collimated, 2 for isotropized random, 3 for angular grid                          */
/* ny, nz - shape (3,) arrays containing 2 vectors in the fixed orthonormal basis - for collimated winds, nz    */
/*          is the axis                                                                                         */
/* dir - shape (3,) array containing the direction - pass as an input to remember the previous direction        */

void get_wind_spawn_direction(int i, int num_spawned_this_call, int mode, Vec3<double>& ny, Vec3<double>& nz, Vec3<double>& veldir, Vec3<double>& dpdir)
{
    int k;
    if((mode != 3) && (num_spawned_this_call % 2)) { // every second particle is spawned in the opposite direction to the last, conserving momentum and COM
        veldir = -veldir; dpdir = -dpdir;
        return; // we're done
    }
    Vec3<double> nx = cross(ny, nz);
    // now do the actual direction based on the mode we're in
    double phi, cos_theta, sin_theta, sin_phi, cos_phi;
    if(mode==0){ // fully random
        phi=2.*M_PI*get_random_number(num_spawned_this_call+1+ThisTask), cos_theta=2.*(get_random_number(num_spawned_this_call+3+2*ThisTask)-0.5); sin_theta=sqrt(1-cos_theta*cos_theta), sin_phi=sin(phi), cos_phi=cos(phi);
        veldir = {sin_theta*cos_phi, sin_theta*sin_phi, cos_theta}; dpdir = veldir;
    } else if (mode==1){ // collimated according to a conical velocity field
        double theta0=0.01, thetamax=30.*(M_PI/180.); // "flattening parameter" and max opening angle of jet velocity distribution from Matzner & McKee 1999, sets the collimation of the jets
#if !defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
        theta0=1.e-4; thetamax=10.*(M_PI/180.); // narrower opening angle distribution for agn jets
#endif
        double theta=atan(theta0*tan(get_random_number(num_spawned_this_call+7+5*ThisTask)*atan(sqrt(1+theta0*theta0)*tan(thetamax)/theta0))/sqrt(1+theta0*theta0)); // biased sampling to get collimation
        phi=2.*M_PI*get_random_number(num_spawned_this_call+1+ThisTask);
        cos_theta = cos(theta), sin_theta=sin(theta), sin_phi=sin(phi), cos_phi=cos(phi);
        veldir = nx*(sin_theta*cos_phi) + ny*(sin_theta*sin_phi) + nz*cos_theta; dpdir = veldir; //converted from angular momentum relative to into standard coordinates
    }
#if defined(SINGLE_STAR_FB_WINDS) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    else if (mode==2){ //random 3-axis isotropized - spawn along z axis, then y, then x
        if(((P[i].ID_generation-1) % 6) == 0) { // need to generate a brand new coordinate frame
            get_random_orthonormal_basis(P[i].ID_generation, nx, ny, nz);
            veldir = nz; for(k=0; k<3; k++) {P[i].Wind_direction[k]=nx[k]; P[i].Wind_direction[k+3]=ny[k];} dpdir = veldir;
        }
        else if(((P[i].ID_generation-1) % 6) == 2) {for(k=0; k<3; k++) {veldir[k] = P[i].Wind_direction[k];} dpdir = veldir;}
        else {for(k=0; k<3; k++) {veldir[k] = P[i].Wind_direction[k+3];} dpdir = veldir;}
    }
#endif
#if (defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)) || defined(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)
    else if (mode==3) { // spawn on a specific angular grid
        int dir_ind = num_spawned_this_call % SINGLE_STAR_FB_SNE_N_EJECTA;
        veldir = nx*All.SN_Ejecta_Direction[dir_ind][0] + ny*All.SN_Ejecta_Direction[dir_ind][1] + nz*All.SN_Ejecta_Direction[dir_ind][2]; //use directions pre-computed to isotropically cover a sphere with SINGLE_STAR_FB_SNE_N_EJECTA particles
        dpdir = veldir;
    }
#endif
    return;
}


/* return desired cell launch speed for spawned cells, in physical (not comoving) units */
double get_spawned_cell_launch_speed(int i)
{
    double v_magnitude = All.Sink_outflow_velocity; // velocity of the jet: default mode is to set this manually to a specific value in physical units

#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES == 3)
    if(is_particle_a_special_zoom_target(i)) {return 1.e5/UNIT_VEL_IN_KMS;} // need an initial velocity for launch here //
#endif
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4)
    if(is_particle_a_special_zoom_target(i)) {return 3.e4/UNIT_VEL_IN_KMS;} // need an initial velocity for launch here //
#endif
    
#ifdef SNE_NONSINK_SPAWN
    if(P[i].Type == 4) {
        double t_gyr = evaluate_stellar_age_Gyr(i); int SNeIaFlag=0; if(t_gyr > 0.03753) {SNeIaFlag=1;}; /* assume SNe before critical time are core-collapse, later are Ia */
        double Msne=10.5/UNIT_MASS_IN_SOLAR; if(SNeIaFlag) {Msne=1.4/UNIT_MASS_IN_SOLAR;} // average ejecta mass for single event (normalized to give total mass loss correctly)
        double SNeEgy = (1.0e51/UNIT_ENERGY_IN_CGS);
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        if(SNeIaFlag==0) {double z_eff = P[i].Metallicity[10]/All.SolarAbundances[10]; if(z_eff < 1) {SNeEgy *= pow(z_eff + 1.e-5 , -0.12);}} // updated to use same metallicity used for stellar evolution, rather than total metallicity, if this derives from pre-explosion winds, etc, for consistency
#if (FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT > 1)
        if(i>0) {double z0 = P[i].Metallicity[0]/All.SolarAbundances[0];
#if (FIRE_SNE_ENERGY_METAL_DEPENDENCE_EXPERIMENT > 2)
            SNeEgy *= pow(z0/0.1 + 1.e-3 , -0.2);
#else
            SNeEgy *= pow(z0/0.1 + 1.e-3 , -0.1);
#endif
        }
#endif
#endif
        return sqrt(2.0*SNeEgy/Msne); // v_ej in code units: assume all SNe = 1e51 erg //
    }
#endif

#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
    double Mdot_wind = P[i].Sink_Mdot_ROI - P[i].Sink_Mdot;
    if(Mdot_wind < 0) {return MIN_REAL_NUMBER;} // should be invalid
    double mdot = P[i].Sink_Mdot / (P[i].Sink_Mass / (4.e7 / UNIT_TIME_IN_YR));
    double L_over_c = evaluate_sink_radiative_efficiency(P[i].Sink_Mdot,P[i].Sink_Mass,i) * P[i].Sink_Mdot * C_LIGHT_CODE;
    double Pdot_rad = 0.;
    if(mdot > 0.01) {Pdot_rad = L_over_c * DMIN(DMAX(mdot,1.),10.);}
    double sigma_ROI = sqrt(All.G * P[i].Sink_Mass / P[i].Sink_ROI);
    double Pdot_turb = 3. * Mdot_wind * sigma_ROI;
    double Pdot_wind = Pdot_rad + Pdot_turb;
    v_magnitude = Pdot_wind / Mdot_wind; 
    /* // (older deprecated model here)
    double MSINK_4 = P[i].Sink_Mass * UNIT_MASS_IN_SOLAR / 1.e4; // sink mass in 1e4 Msun to scale
    double lambda_edd_eff = DMAX( P[i].Sink_Mdot / sink_eddington_mdot(P[i].Sink_Mass) , 1.e-10 ); // eddington ratio, with floor just to prevent unphysical behaviors
    if(lambda_edd_eff > (SINK_RIAF_SUBEDDINGTON_MODEL))
    {
        double v_eff_esc_BLR = 270. * sqrt(sqrt(MSINK_4 / lambda_edd_eff)) / UNIT_VEL_IN_KMS; // escape velocity from BLR in km/s, using canonical RBLR ~ 20 light-days * (L_bol/1e45)^(1/2)-ish scaling
        v_magnitude = DMIN(v_magnitude , v_eff_esc_BLR); // the input Sink_outflow_velocity parameter now sets the maximum efficiency/velocity this is allowed to reach, but it can be arbitrarily lower
    } else {
        v_magnitude = DMAX(v_magnitude , 1.e5 / UNIT_VEL_IN_KMS); // fast jet speed
    }
    */
#endif
    
#ifdef SINGLE_STAR_FB_JETS
    v_magnitude = single_star_jet_velocity(i); // get velocity from our more detailed function
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(SINGLE_STAR_FB_WINDS)
    if((P[i].ProtoStellarStage == 5) && (P[i].wind_mode==1)) {v_magnitude = single_star_wind_velocity(i);} // only MS stars launch winds: get velocity from fancy model
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(SINGLE_STAR_FB_SNE)
    if(P[i].ProtoStellarStage == 6) {v_magnitude = single_star_SN_velocity(i);} // this star is about to go SNe: get velocity from fancy model
#endif
    return v_magnitude;
}


#ifdef MAGNETIC
void get_wind_spawn_magnetic_field(int j, int mode, Vec3<double>& ny, Vec3<double>& nz, Vec3<double>& dpdir, double d_r)
{
    int k; CellP[j].divB = 0; CellP[j].DtB = {};
#ifdef DIVBCLEANING_DEDNER
    CellP[j].DtPhi = CellP[j].PhiPred = CellP[j].Phi = 0; CellP[j].DtB_PhiCorr = {};
#endif

    double volume_for_BtoVB = P[j].Mass / CellP[j].Density;
#ifdef SINK_WIND_SPAWN_SET_BFIELD_POLTOR /* user manually sets the poloidal and toroidal components here */
    double inj_scale=All.Sink_spawn_injectionradius/All.cf_atime; Vec3<double> Bfield={}, nx=cross(ny,nz);
    double cos_theta=dot(nz,dpdir), sin_theta=sqrt(1-cos_theta*cos_theta), cos_phi=dot(nx,dpdir)/sin_theta, sin_phi=dot(ny,dpdir)/sin_theta;
    /* initialize poloidal component, in the nx,ny,nz coordinate frame */
    Bfield[0]+= All.B_spawn_pol*d_r*cos_theta*d_r*sin_theta/inj_scale/inj_scale*cos_phi*exp(-1.0*d_r*d_r/inj_scale/inj_scale)/exp(-1.0);
    Bfield[1]+= All.B_spawn_pol*d_r*cos_theta*d_r*sin_theta/inj_scale/inj_scale*sin_phi*exp(-1.0*d_r*d_r/inj_scale/inj_scale)/exp(-1.0);
    Bfield[2]+= All.B_spawn_pol*(1-d_r*cos_theta*d_r*sin_theta/inj_scale/inj_scale)    *exp(-1.0*d_r*d_r/inj_scale/inj_scale)/exp(-1.0);
    /* initialize toroidal component, in the nx,ny,nz coordinate frame */
    Bfield[0]+= -1*All.B_spawn_tor*(d_r/inj_scale)*sin_theta*sin_phi*exp(-1.0*d_r*d_r/inj_scale/inj_scale)/exp(-1.0);
    Bfield[1]+=    All.B_spawn_tor*(d_r/inj_scale)*sin_theta*cos_phi*exp(-1.0*d_r*d_r/inj_scale/inj_scale)/exp(-1.0);
    /* assign it back to the actual evolved B in the lab/simulation coordinate frame */
    CellP[j].IniB = nx*Bfield[0] + ny*Bfield[1] + nz*Bfield[2]; CellP[j].DtB = {};
    CellP[j].BPred = CellP[j].B = CellP[j].IniB * ((All.UnitMagneticField_in_gauss/UNIT_B_IN_GAUSS)*(volume_for_BtoVB/All.cf_a2inv));
    
#else /* set B-fields to be weak relative to local ISM values */

    double Bmag=0, Bmag_0=0;
    for(k=0;k<3;k++) {double B=CellP[j].B[k]*All.cf_a2inv/volume_for_BtoVB; Bmag+=B*B; Bmag_0+=CellP[j].B[k]*CellP[j].B[k];} // get actual Bfield
    double Bmag_low_rel_to_progenitor = 1.e-10 * sqrt(Bmag); // set to some extremely low value relative to cloned element
    double u_internal_new_cell = All.Sink_outflow_temperature / (  0.59 * (5./3.-1.) * U_TO_TEMP_UNITS ); // internal energy of new wind cell
    double Bmag_low_rel_to_pressure = 1.e-3 * sqrt(2.*CellP[j].Density*All.cf_a3inv * u_internal_new_cell); // set to beta = 1e6
    Bmag = DMAX(Bmag_low_rel_to_progenitor , Bmag_low_rel_to_pressure); // pick the larger of these (still small) B-field values
#ifdef MHD_B_SET_IN_PARAMS
    double Bmag_IC = sqrt(All.BiniX*All.BiniX + All.BiniY*All.BiniY + All.BiniZ*All.BiniZ) * All.UnitMagneticField_in_gauss / UNIT_B_IN_GAUSS; // IC B-field sets floor as well
    Bmag = DMAX(Bmag , 0.1 * Bmag_IC);
#endif
#if defined(SINGLE_STAR_FB_SNE)
    if(P[j].Type==5) {if(P[j].ProtoStellarStage == 6) {Bmag *= 1.e-3;}} // No need to have flux in SN ejecta - note that this assumes we inherited this attribute from the spawning sink before calling this routine
#endif
    Bmag = DMAX(Bmag, MIN_REAL_NUMBER); // floor to prevent underflow errors
    /* add magnetic flux here to 'Bmag' if desired */
    Bmag *= volume_for_BtoVB / All.cf_a2inv; // convert back to code units
    for(k=0;k<3;k++) {if(Bmag_0>0) {CellP[j].B[k]*=Bmag/sqrt(Bmag_0);} else {CellP[j].B[k]=Bmag;}} // assign if valid values
    CellP[j].BPred=CellP[j].B; CellP[j].DtB={}; // set predicted = actual, derivative to null
#endif
    CellP[j].BField_prerefinement = CellP[j].B * (1.0 / volume_for_BtoVB); /* record the real value of B pre-split to know what we need to correctly re-initialize to once the volume partition can be recomputed */
    CellP[j].BPred = CellP[j].B; /* set predicted/drifted equal to the value above */
    return;
}
#endif


/*! this code copies what was used in merge_split.c for the gas particle split case */
int sink_spawn_particle_wind_shell( int i, int dummy_cell_i_to_clone, int num_already_spawned )
{
    double total_mass_in_winds = P[i].unspawned_wind_mass;

    int n_particles_split   = (int) floor( total_mass_in_winds / target_mass_for_wind_spawning(i) ); /* if we set SINK_WIND_SPAWN we presumably wanted to do this in an exactly-conservative manner, which means we want to have an even number here. */
    int k=0; long j;

#if defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P[i].Type==5) {
        if(P[i].ProtoStellarStage == 6) {
            n_particles_split = (int) floor( total_mass_in_winds / (P[i].Sink_Formation_Mass) );
            double m_relic = single_star_relic_SN_mass(i); // get the intended relic mass //
            if(P[i].Sink_Mass <= m_relic) { // last batch to be spawned
                n_particles_split = SINGLE_STAR_FB_SNE_N_EJECTA; // we are going to spawn a bunch of low mass particles to take the last bit of mass away
                printf("Spawning last SN ejecta of star %llu with %g mass and %d particles \n",(unsigned long long) P[i].ID,total_mass_in_winds,n_particles_split);
                P[i].Mass = DMAX(0, m_relic); // set mass to zero so that this sink will get cleaned up (TreeReconstructFlag = 1 should be already set in sink.c)
#ifdef SINK_ALPHADISK_ACCRETION
                P[i].Sink_Mass_Reservoir = 0; // just to be safe
#endif
                if(P[i].Sink_Mass > 0 && P[i].Sink_Mass > P[i].Sink_Formation_Mass) {P[i].ProtoStellarStage == 7;} // this is a relic now, move it to the next stage
	    }
	}
    }
#if (defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)) || defined(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)
    if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6)
    {
        if (n_particles_split<SINGLE_STAR_FB_SNE_N_EJECTA) {return 0;} // we have to wait until we get a full shell
        else {n_particles_split = n_particles_split - (n_particles_split % SINGLE_STAR_FB_SNE_N_EJECTA);} // we only eject full shells, in practice this will be one shell at a time
    }}
#endif
#endif
    if((((int)SINK_WIND_SPAWN) % 2) == 0) {if(( n_particles_split % 2 ) != 0) {n_particles_split -= 1;}} /* n_particles_split was not even. we'll wait to spawn this last particle, to keep an even number, rather than do it right now and break momentum conservation */	
    if( (n_particles_split == 0) || (n_particles_split < 1) ) {return 0;}
    int n0max = DMAX(20 , (int)(3.*(SINK_WIND_SPAWN)+0.1));
#if defined(SNE_NONSINK_SPAWN)
    n0max = DMAX(6 , (int)(3.*(SINK_WIND_SPAWN)+0.1)); // more conservative to spread over more timesteps to avoid nasty overlaps //
#if (defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)) || defined(SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT)
    n0max = DMAX(SINGLE_STAR_FB_SNE_N_EJECTA , (int)(3.*(SINK_WIND_SPAWN)+0.1)); if((n0max % 2) != 0) {n0max += 1;} // should ensure n0max is always an even number //
#endif
#endif
    if((n0max % 2) != 0) {n0max += 1;} // should ensure n0max is always an even number //
#if defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {n0max = DMAX(n0max, SINGLE_STAR_FB_SNE_N_EJECTA);}} // so that we can spawn the number of wind particles we want, by setting SINK_WIND_SPAWN high it ispossible to spawn multitudes of SINGLE_STAR_FB_SNE_N_EJECTA, but in practice we usually spawn just one
#endif
    if(n_particles_split > n0max) {n_particles_split = n0max;}


    /* here is where the details of the split are coded, the rest is bookkeeping */
    //double mass_of_new_particle = total_mass_in_winds / n_particles_split; /* don't do this, as can produce particles with extremely large masses; instead wait to spawn */
    double mass_of_new_particle = target_mass_for_wind_spawning(i); double mass_of_new_particle_default,mass_of_new_particle_prev; mass_of_new_particle_default=mass_of_new_particle; mass_of_new_particle_prev=mass_of_new_particle;
#if defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {mass_of_new_particle = total_mass_in_winds/(double) n_particles_split;}} // ejecta will have the gas mass resolution except the last batch which will lower masses
#endif
    printf("Task %d wants to create %g mass in wind with %d new particles each of mass %g \n .. splitting sink %d using hydro element %d\n", ThisTask,total_mass_in_winds, n_particles_split, mass_of_new_particle, i, dummy_cell_i_to_clone);

    if(NumPart + num_already_spawned + n_particles_split >= All.MaxPart)
    {
        printf("On Task=%d with NumPart=%d (+N_spawned=%d) we tried to split a particle, but there is no space left...(All.MaxPart=%d). Try using more nodes, or raising PartAllocFac, or changing the split conditions to avoid this.\n", ThisTask, NumPart, num_already_spawned, All.MaxPart);
        fflush(stdout); endrun(8888);
    }
    double d_r = 0.25 * KERNEL_CORE_SIZE*P[i].KernelRadius; // needs to be epsilon*KernelRadius where epsilon<<1, to maintain stability //
    double r2 = (P[dummy_cell_i_to_clone].Pos - P[i].Pos).norm_sq();
    d_r = DMIN(d_r, 0.5*sqrt(r2));
#ifndef SELFGRAVITY_OFF
    d_r = DMAX(d_r , 2.0*EPSILON_FOR_TREERND_SUBNODE_SPLITTING * All.ForceSoftening[0]);
#endif
#ifdef SINK_WIND_SPAWN_SET_BFIELD_POLTOR
    d_r = DMIN(d_r , All.Sink_spawn_injectionradius/All.cf_atime); /* KYSu: sets spawn scale manually */
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    d_r = DMIN(P[i].SinkRadius, d_r); //launch close to the sink
#endif
#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM) && defined(PARTICLE_EXCISION)
    if(is_particle_a_special_zoom_target(i)) {double rmin=All.ForceSoftening[3], r=sqrt(r2), r0=0.5*(rmin+r)*(1.+0.1*get_random_number(i+j)); d_r=r0;} // make sure to spawn OUTSIDE of the excision radius!
#endif
#if defined(SNE_NONSINK_SPAWN)
    if(P[i].Type == 4) {double rmin=All.ForceSoftening[4], r=sqrt(r2), r0=0.5*(rmin+r)*(0.5+1.5*get_random_number(i+j)); d_r=r0;} // need a generous padding to ensure no overlaps
#endif
    long bin, bin_0; for(bin = 0; bin < TIMEBINS; bin++) {if(TimeBinCount[bin] > 0) break;} /* gives minimum active timebin of any particle */
    bin_0 = bin; int i0 = i; /* save minimum timebin, also save ID of sink particle for use below */
    bin = P[i0].TimeBin; /* make this particle active on the BH/star timestep */
    Vec3<double> veldir, dpdir; // velocity direction to spawn in - declare outside the loop so we remember it from the last iteration
    int mode = 0; // 0 if doing totally random directions, 1 if collimated, 2 for 3-axis isotropized, and 3 if using an angular grid,  4 old collimatation script, position isotropic velicity coliminated within certain open angle (might be useful to still keep this option owing to the free open angle choice and better sampling the magnetic field geometry)
#if defined(SINGLE_STAR_FB_JETS) || defined(JET_DIRECTION_FROM_KERNEL_AND_SINK) || defined(SINK_FB_COLLIMATED)
    mode = 1; // collimated mode
#endif
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
    mode=0; // broad-angle default, but will modify below
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    mode = 0;
#if defined(SINGLE_STAR_FB_JETS)
    mode = 1; // launch polar jets
#endif
#if defined(SINGLE_STAR_FB_WINDS)
    if(P[i].Type==5) {if(P[i].ProtoStellarStage == 5) {
        mode = 2; // winds use 3-axis isotropized directions
#if defined(SINGLE_STAR_FB_JETS)
        if(P[i].wind_mode == 2) {mode = 1;} // we inject winds with the FIRE module, we only spawn polar jets here
#endif
    }}
#endif
#if defined(SINGLE_STAR_FB_SNE)
    if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {mode = 3;}} // SNe use an angular grid
#endif
#endif // single-star evolution clause
    if(P[i].Type==3) {mode = 1;} // special particle spawn is collimated
    if(P[i].Type==4) {mode = 0;} // star particle spawn is isotropic
#ifdef SINGLE_STAR_FB_SNE_N_EJECTA_QUADRANT
    if(P[i].Type==4) {mode = 3;} // star particle spawn is isotropic but regularized
#endif

    // based on the mode we're in, let's pick a fixed orthonormal basis that all spawned elements are aware of
    Vec3<double> jz={0,0,1},jy={0,1,0},jx={1,0,0};  /* set up a coordinate system [xyz if we don't have any other information */
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM  /* use local angular momentum to estimate preferred directions/coordinates for spawning */
    if(mode==1){ // set up so that the z axis is the angular momentum vector
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK // Jgas stores total angmom in COM frame of sink-gas system; use this for direction
        double Jtot=P[i].Jgas_in_Kernel.norm_sq();
        if(Jtot>0) {Jtot=1/sqrt(Jtot); jz = P[i].Jgas_in_Kernel * Jtot;}
#else
        double Jtot=P[i].Sink_Specific_AngMom.norm_sq();
        if(Jtot>0) {Jtot=1/sqrt(Jtot); jz = P[i].Sink_Specific_AngMom * Jtot;}
#endif
        Jtot=jz[1]*jz[1]+jz[2]*jz[2]; if(Jtot>0) {Jtot=1/sqrt(Jtot); jy={0, jz[2]*Jtot, -jz[1]*Jtot};} else {jy={0, 1, 0};}
        jx = cross(jz, jy);
    }
#endif
    if(mode == 3){ // if doing an angular grid, need some fixed coordinates to orient it, but want to switch em up each time to avoid artifacts
        get_random_orthonormal_basis(P[i].ID_generation, jx, jy, jz);
    }
#ifdef SINK_WIND_SPAWN_SET_JET_PRECESSION /* rotate the jet angle according to the explicitly-included precession parameters */
    double degree = All.Sink_jet_precess_degree, period = All.Sink_jet_precess_period/UNIT_TIME_IN_GYR; Vec3<double> new_dir;
    new_dir[0]= jx[0]*cos(degree/180.*M_PI)-jx[2]*sin(degree/180.*M_PI); new_dir[1]= 1.0*jx[1]; new_dir[2]= jx[0]*sin(degree/180.*M_PI)+jx[2]*cos(degree/180.*M_PI);
    jx[0]= new_dir[0]*cos(2.*M_PI/period*All.Time)-new_dir[1]*sin(2.*M_PI/period*All.Time); jx[1]= new_dir[0]*sin(2.*M_PI/period*All.Time)+new_dir[1]*cos(2.*M_PI/period*All.Time); jx[2]= new_dir[2];

    new_dir[0]= jy[0]*cos(degree/180.*M_PI)-jy[2]*sin(degree/180.*M_PI); new_dir[1]= 1.0*jy[1]; new_dir[2]= jy[0]*sin(degree/180.*M_PI)+jy[2]*cos(degree/180.*M_PI);
    jy[0]= new_dir[0]*cos(2.*M_PI/period*All.Time)-new_dir[1]*sin(2.*M_PI/period*All.Time); jy[1]= new_dir[0]*sin(2.*M_PI/period*All.Time)+new_dir[1]*cos(2.*M_PI/period*All.Time); jy[2]= new_dir[2];

    new_dir[0]= jz[0]*cos(degree/180.*M_PI)-jz[2]*sin(degree/180.*M_PI); new_dir[1]= 1.0*jz[1]; new_dir[2]= jz[0]*sin(degree/180.*M_PI)+jz[2]*cos(degree/180.*M_PI);
    jz[0]= new_dir[0]*cos(2.*M_PI/period*All.Time)-new_dir[1]*sin(2.*M_PI/period*All.Time); jz[1]= new_dir[0]*sin(2.*M_PI/period*All.Time)+new_dir[1]*cos(2.*M_PI/period*All.Time); jz[2]= new_dir[2];
#endif    

    /* create the  new particles to be added to the end of the particle list :
        i is the sink particle tag, j is the new "spawed" particle's location, dummy_cell_i_to_clone is a dummy gas cell's tag to be used to init the wind particle */
    int mode_default = mode, mode_prev = mode;
    double v_magnitude_physical_default = get_spawned_cell_launch_speed(i), v_magnitude_physical=v_magnitude_physical_default, v_magnitude_physical_prev=v_magnitude_physical; /* call subroutine for this velocity */
    
    for(j = NumPart + num_already_spawned; j < NumPart + num_already_spawned + n_particles_split; j++)
    {   /* first, clone the 'dummy' particle so various fields are set appropriately */
        P[j] = P[dummy_cell_i_to_clone]; CellP[j] = CellP[dummy_cell_i_to_clone]; /* set the pointers equal to one another -- all quantities get copied, we only have to modify what needs changing */

#if defined(SINK_TEST_WIND_MIXED_FASTSLOW) || defined(SINK_RIAF_SUBEDDINGTON_MODEL)
        if(P[i].Type==5) {
            double masscorrfac_fast = 100.; /* ratio of spawned jet cell mass to non-jet cell mass */
            double fraction_to_spawn_in_jet = 0.1; /* fraction of spawned cells by number in jet */
            double frac_clight_jet = 0.1; /* default fraction of C for jet speed */
#ifdef SINK_TEST_WIND_MIXED_FASTSLOW
            frac_clight_jet = (SINK_TEST_WIND_MIXED_FASTSLOW/UNIT_VEL_IN_KMS) / C_LIGHT_CODE;
#endif
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
            double frac_clight_jet_max = 0.10; //1.0; // don't let the jet be too fast, for physical (superluminal) or numerical (timestep) reasons
            double frac_clight_jet_min = 0.03; //0.1; // don't let the jet be too slow, or it won't behave like a jet; lower jet mass to compensate
            double Mdot_wind = P[i].Sink_Mdot_ROI - P[i].Sink_Mdot;
            if(Mdot_wind > 0)
            {
                double a_spin = 0.33;
                double HR = 0.33;
                double mdot = P[i].Sink_Mdot / (P[i].Sink_Mass / (5.e7/UNIT_TIME_IN_YR));
                if(mdot < 0.01) {HR = 1;}
                double eff_jet = 0.1 * pow(a_spin*HR,2);
                double Mdot_jet = (fraction_to_spawn_in_jet/masscorrfac_fast) * Mdot_wind;
                double eta_jet = Mdot_jet / P[i].Sink_Mdot;
                frac_clight_jet = sqrt(2.*eff_jet/eta_jet); /* scaling so that KE of jet = desired */
                if(frac_clight_jet > frac_clight_jet_max) { /* superluminal - need more mass in jet to make this make sense */
                    frac_clight_jet = frac_clight_jet_max; // cap this at luminal
                    masscorrfac_fast = frac_clight_jet*frac_clight_jet * (fraction_to_spawn_in_jet/(2.*eff_jet)) * (Mdot_wind / P[i].Sink_Mdot); // boost this term to make up the difference
                }
                if(frac_clight_jet < frac_clight_jet_min) {
                    frac_clight_jet = frac_clight_jet_min; // cap this at minimum
                    masscorrfac_fast = frac_clight_jet*frac_clight_jet * (fraction_to_spawn_in_jet/(2.*eff_jet)) * (Mdot_wind / P[i].Sink_Mdot); // boost this term to make up the difference
                }
            }
#endif
            double m0_newparticlemass_for_target_spawnedmass = mass_of_new_particle_default / (1. - fraction_to_spawn_in_jet * (1.-1./masscorrfac_fast));
            mode = mode_default; mass_of_new_particle=m0_newparticlemass_for_target_spawnedmass; v_magnitude_physical=v_magnitude_physical_default;
            if((j - (NumPart + num_already_spawned)) % 2) {mode=mode_prev; v_magnitude_physical=v_magnitude_physical_prev; mass_of_new_particle=mass_of_new_particle_prev;  /* for every-other particle, need to match previous for conservation */
            } else { /* collimated jet */
                if(get_random_number(j) < fraction_to_spawn_in_jet) {
                    mode=1; mass_of_new_particle=m0_newparticlemass_for_target_spawnedmass/masscorrfac_fast; v_magnitude_physical = frac_clight_jet * C_LIGHT_CODE;
                } else { /* isotropic/broad-angle wind */
                    mode=0; mass_of_new_particle=m0_newparticlemass_for_target_spawnedmass; v_magnitude_physical=v_magnitude_physical_default; /* isotropic slow wind */
                }
            }
        }
#endif
        v_magnitude_physical_prev = v_magnitude_physical; mode_prev = mode; mass_of_new_particle_prev=mass_of_new_particle;

        /* now we need to make sure everything is correctly placed in timebins for the tree */
        P[j].TimeBin = bin; // get the timebin, and put this particle into the appropriate timebin
        ActiveParticleList.push_back(j);
        NumForceUpdate++;
        TimeBinCount[bin]++; TimeBinCountGas[bin]++; PrevInTimeBin[j] = i0; /* likewise add it to the counters that register how many particles are in each timebin */
        NextInTimeBin[j] = NextInTimeBin[i0]; if(NextInTimeBin[i0] >= 0) {PrevInTimeBin[NextInTimeBin[i0]] = j;} NextInTimeBin[i0] = j; if(LastInTimeBin[bin] == i0) {LastInTimeBin[bin] = j;}
        P[j].Ti_begstep = All.Ti_Current; P[j].Ti_current = All.Ti_Current;
#ifdef WAKEUP /* note - you basically MUST have this flag on for this routine to work at all -- */
        P[j].dt_step = GET_INTEGERTIME_FROM_TIMEBIN(bin);
        P[j].wakeup = 1;
        NeedToWakeupParticles_local = 1;
#endif
        /* this is a giant pile of variables to zero out. dont need everything here because we cloned a valid particle, but handy anyways */
        P[j].Particle_DivVel = 0; CellP[j].DtInternalEnergy = 0; CellP[j].HydroAccel = {}; P[j].GravAccel = {};
        P[j].NumNgb=cbrt(All.DesNumNgb); // this gets cube rooted at the end of the density loop, so take cbrt here
#ifdef PMGRID
        P[j].GravPM = {};
#endif
#ifdef ENERGY_ENTROPY_SWITCH_IS_ACTIVE
        CellP[j].MaxKineticEnergyNgb = 0;
#endif
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP[j].dMass = 0; CellP[j].DtMass = 0; CellP[j].MassTrue = P[j].Mass; CellP[j].GravWorkTerm = {};
#endif
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
        P[j].AGS_zeta = 0;
#ifdef ADAPTIVE_GRAVSOFT_FORALL
        P[j].AGS_KernelRadius = P[j].KernelRadius;
#endif
#endif
#ifdef ADAPTIVE_TREEFORCE_UPDATE
        P[j].tdyn_step_for_treeforce = 0; P[j].time_since_last_treeforce = MAX_REAL_NUMBER; // make sure we get a new tree force right off the bat
#endif
#ifdef CONDUCTION
        CellP[j].Kappa_Conduction = 0;
#endif
#ifdef MHD_NON_IDEAL
        CellP[j].Eta_MHD_OhmicResistivity_Coeff = 0; CellP[j].Eta_MHD_HallEffect_Coeff = 0; CellP[j].Eta_MHD_AmbiPolarDiffusion_Coeff = 0;
#endif
#ifdef VISCOSITY
        CellP[j].Eta_ShearViscosity = 0; CellP[j].Zeta_BulkViscosity = 0;
#endif
#ifdef TURB_DIFFUSION
        CellP[j].TD_DiffCoeff = 0;
#endif
#if defined(GALSF_SUBGRID_WINDS)
#if (GALSF_SUBGRID_WIND_SCALING==1)
        CellP[j].HostHaloMass = 0;
#endif
#endif
#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
        CellP[j].DelayTimeHII = 0;
#endif
#ifdef GALSF_FB_TURNOFF_COOLING
        CellP[j].DelayTimeCoolingSNe = 0;
#endif
#ifdef GALSF
        CellP[j].Sfr = 0;
#endif
#ifdef SPHAV_CD10_VISCOSITY_SWITCH
        CellP[j].alpha = 0.0;
#endif
#if defined(SINK_THERMALFEEDBACK)
        CellP[j].Injected_Sink_Energy = 0;
#endif
#ifdef RADTRANSFER
        for(k=0;k<N_RT_FREQ_BINS;k++)
        {
            CellP[j].Rad_E_gamma[k] = 0;
#if defined(RT_EVOLVE_ENERGY)
            CellP[j].Rad_E_gamma_Pred[k] = 0; CellP[j].Dt_Rad_E_gamma[k] = 0;
#endif
#if defined(RT_EVOLVE_FLUX)
            int kdir; for(kdir=0;kdir<3;kdir++){CellP[j].Rad_Flux[k][kdir] = 0;}
#endif
        }
#endif
#if defined(GALSF)
        P[j].StellarAge = All.Time; // use this attibute to save the gas cell's formation time for possible subsequent checks for special behavior on its first timestep
#endif

        /* now set the real hydro variables. */
        /* set the particle ID */ // unsigned int bits; int SPLIT_GENERATIONS = 4; for(bits = 0; SPLIT_GENERATIONS > (1 << bits); bits++); /* the particle needs an ID: we give it a bit-flip from the original particle to signify the split */
        P[j].ID = All.SpawnedWindCellID; /* update:  We are using a fixed wind ID, to allow for trivial wind particle identification */
#if defined(SINGLE_STAR_SINK_DYNAMICS)
        if(mass_of_new_particle >= 0.5*P[i].Sink_Formation_Mass) {P[j].ID = All.SpawnedWindCellID + 1;} // this just has the nominal mass resolution, so no special treatment - this avoids the P[i].ID == All.SpawnedWindCellID checks throughout the code
#endif
        P[j].ID_child_number = P[i].ID_child_number + P[i].ID_generation; P[i].ID_generation++; P[j].ID_generation = P[i].ID; // this allows us to track spawned particles by giving them unique sub-IDs. Remember we MUST NEVER alter an existing particle ID OR ID_child_number!
        P[j].Mass = mass_of_new_particle; /* assign masses to both particles (so they sum correctly) */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        CellP[j].MassTrue = P[j].Mass;
#endif
        P[i].dp -= P[j].Mass * P[i].Vel; /* track momentum change from mass loss for tree node update */
        P[i].Mass -= P[j].Mass; /* make sure the operation is mass conserving! */
        P[i].unspawned_wind_mass -= P[j].Mass; /* remove the mass successfully spawned, to update the remaining unspawned mass */

#if defined(METALS) && (defined(SINGLE_STAR_FB_JETS) || defined(SINGLE_STAR_FB_WINDS) || defined(SINGLE_STAR_FB_SNE) || defined(SNE_NONSINK_SPAWN) || (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4))
        double yields[NUM_METAL_SPECIES+NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION]={0}; get_jet_yields(yields,i); // default to jet-type
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(SINGLE_STAR_FB_WINDS)
        if(P[i].Type==5) {if((P[i].ProtoStellarStage == 5) && (P[i].wind_mode==1)) {get_wind_yields(yields,i);}} // get abundances in wind
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION) && defined(SINGLE_STAR_FB_SNE)
        if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {double Msne; get_SNe_yields(yields,i,stellar_lifetime_in_Gyr(i),0,&Msne);}} // get sne yields
#endif
        for(k=0;k<NUM_METAL_SPECIES;k++) {P[j].Metallicity[k]=yields[k];}  // update metallicity of spawned cell modules
#endif
        
        // actually lay down position and velocities using coordinate basis
        get_wind_spawn_direction(i, j - (NumPart + num_already_spawned), mode, jy, jz, veldir, dpdir);
        P[j].Pos = P[i].Pos + dpdir*d_r; P[j].Vel = P[i].Vel + veldir*(v_magnitude_physical*All.cf_atime); CellP[j].VelPred = P[j].Vel; // convert to code (comoving) velocity units

        /* condition number, smoothing length, and density */
        CellP[j].ConditionNumber *= 100.0; /* boost the condition number to be conservative, so we don't trigger madness in the kernel */
        CellP[j].recent_refinement_flag = 1; /* tag the newly-created cell as recently-refined for all purposes */
#if defined(SINGLE_STAR_SINK_DYNAMICS)
        CellP[j].MaxSignalVel = 2.*DMAX(v_magnitude_physical, CellP[j].MaxSignalVel); // need this to satisfy the Courant condition in the first timestep after spawn; note here MaxSignalVel is now defined in physical code units
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
	    P[j].ProtoStellarStage = P[i].ProtoStellarStage; // inherit this from the spawning sink particle so we can use it in subroutines
        // need to initialize the gas density and search radius so that we get sensible CFL timesteps (which happens before density() is called and we recalculate these self-consistently)
        if(n_particles_split > All.DesNumNgb) { // we are spawning a whole "shell" together, so initialize search radii/densities assuming kernels are confined to the region of spawned material.
            CellP[j].Density = mass_of_new_particle / (4 * M_PI * d_r*d_r*d_r);
            P[j].KernelRadius = P[j].NumNgb * 2.32489404843 * d_r;
        } else { // we are spawning in the jet/wind piecemeal, so use the local density estimator around the star
            CellP[j].Density = P[i].DensityAroundParticle;
            P[j].KernelRadius = P[i].KernelRadius;
        }
#endif
#endif
        /* note, if you want to use this routine to inject magnetic flux or cosmic rays, do this below */
#if defined(SINK_WIND_SPAWN_SET_BFIELD_POLTOR)
        CellP[j].IniDen = -1. * CellP[j].Density; /* this is essentially acting like a bitflag, to signal to the code that the density needs to be recalculated because a spawn event just occurred */
#endif
#ifdef MAGNETIC
        get_wind_spawn_magnetic_field(j, mode, jy, jz, dpdir, d_r);
#endif
#ifdef COSMIC_RAY_FLUID
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
        CellP[j].DtCREgyNewInjectionFromShocks=0;
#endif
        int k_CRegy; for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) /* initialize CR energy and other related terms to nil */
        {
            CellP[j].CosmicRayEnergyPred[k_CRegy]=CellP[j].CosmicRayEnergy[k_CRegy]=CellP[j].DtCosmicRayEnergy[k_CRegy]=0;
#ifdef CRFLUID_EVOLVE_SPECTRUM
            CellP[j].CosmicRay_Number_in_Bin[k_CRegy]=CellP[j].DtCosmicRay_Number_in_Bin[k_CRegy]=0;
#endif
            CellP[j].CosmicRayFlux[k_CRegy] = {}; CellP[j].CosmicRayFluxPred[k_CRegy] = {};
#ifdef CRFLUID_EVOLVE_SCATTERINGWAVES
            CellP[j].CosmicRayAlfvenEnergy[k_CRegy] = {}; CellP[j].CosmicRayAlfvenEnergyPred[k_CRegy] = {}; CellP[j].DtCosmicRayAlfvenEnergy[k_CRegy] = {};
#endif
        } /* complete CR initialization to null */
#endif
        CellP[j].InternalEnergy = All.Sink_outflow_temperature / (  0.59 * (5./3.-1.) * U_TO_TEMP_UNITS ); /* internal energy, determined by desired wind temperature (assume fully ionized primordial gas with gamma=5/3) */
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        CellP[j].InternalEnergy = 0.01 * (0.5*v_magnitude_physical*v_magnitude_physical); /* set to be 1% of the kinetic energy of the ejecta, here */
#endif
#if defined(SINGLE_STAR_FB_SNE) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
        double sne_energy_fraction_in_thermal = 1.e-3;
        if(P[i].Type==5) {if(P[i].ProtoStellarStage == 6) {CellP[j].InternalEnergy = All.MinGasTemp / (  0.59 * (5./3.-1.) * U_TO_TEMP_UNITS ) + sne_energy_fraction_in_thermal/(1.-sne_energy_fraction_in_thermal) * pow(single_star_SN_velocity(i),2.0);}}
#endif
        CellP[j].InternalEnergyPred = CellP[j].InternalEnergy;

#if defined(COSMIC_RAY_FLUID) && defined(SINK_COSMIC_RAYS) /* inject cosmic rays alongside wind injection */
        double eps_cr = evaluate_sink_cosmicray_efficiency(P[i].Sink_Mdot,P[i].Sink_Mass,i);
        double fac_wind_corr = All.Sink_accreted_fraction / (1.-All.Sink_accreted_fraction);
        double dEcr = eps_cr * P[j].Mass * fac_wind_corr * C_LIGHT_CODE*C_LIGHT_CODE;
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        dEcr = eps_cr * 0.5 * v_magnitude_physical*v_magnitude_physical * P[j].Mass; /* in this case, eps_cr refers to fraction relative to KE going into jet spawns */
        if(mass_of_new_particle > 2.*mass_of_new_particle_default/masscorrfac_fast) {dEcr=0;} /* only the jet cells carry CR energy */
#endif
#if defined(SINK_TEST_WIND_MIXED_FASTSLOW)
        if(mass_of_new_particle < 2.*mass_of_new_particle_default/masscorrfac_fast) {dEcr*=masscorrfac_fast;} else {dEcr=0;}
#endif
#if defined(SINK_CR_INJECTION_AT_TERMINATION)
        CellP[j].Sink_CR_Energy_Available_For_Injection = dEcr;     /* store energy for later injection */
#else
        inject_cosmic_rays(dEcr, v_magnitude_physical, 5, j, veldir.data); /* inject directly */
#endif
#endif
        /* Note: New tree construction can be avoided because of  `force_add_element_to_tree()' */
        force_add_element_to_tree(i0, j);// (buggy) /* we solve this by only calling the merge/split algorithm when we're doing the new domain decomposition */
    }
    if(P[i].unspawned_wind_mass < 0) {P[i].unspawned_wind_mass=0;}
    return n_particles_split;
}
#endif



#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4)
/* routine for injection from sink boundary around 'special' particle types */
void special_rt_feedback_injection(void)
{
    double L0_cgs = 7.e45, MdotJetMsunYr=1., mspecial_tot=0; int iBH0=-1, k;
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES == 3)
    L0_cgs = 1.e43; MdotJetMsunYr = 1.e-3;
#endif
    for(k=0;k<SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM;k++) {mspecial_tot += All.Mass_of_SpecialParticle[k];}
    if(mspecial_tot <= 0) {return;}
    double delta_wt_sum = 0, delta_wt_sumsum, r_min = All.ForceSoftening[3] * All.cf_atime, r_max = 5. * r_min, dt = All.TimeStep, subgrid_lum = L0_cgs / (UNIT_ENERGY_IN_CGS/UNIT_TIME_IN_CGS), de_00 = subgrid_lum * dt; if(dt <= 0) {return;}
    int n_wt = 0, i; for(i=0;i<NumPart;i++) {
        if(is_particle_a_special_zoom_target(i)) {iBH0=i;}
        if(P[i].Type != 0) {continue;}
        Vec3<double> dp{All.cf_atime*(double)P[i].Pos[0], All.cf_atime*(double)P[i].Pos[1], All.cf_atime*(double)P[i].Pos[2]};
        double r2 = dp.norm_sq(), wt, wt_new=0, r;
        r = sqrt(r2); if(r < r_min || r >= r_max) {continue;}
        double vol = P[i].Mass / (CellP[i].Density*All.cf_a3inv), cos_t = dp[0] / r;
        wt = 1.e-5 * pow(fabs(cos_t),8) * vol * (r_max*r_max/(r*r)-1.); delta_wt_sum += wt;
    }
    MPI_Allreduce(&delta_wt_sum, &delta_wt_sumsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // collect the information on weight sums
    if(All.Time <= All.TimeBegin) {return;}
    if(delta_wt_sumsum <= 0) {return;}
    for(i=0;i<NumPart;i++) {
        if(P[i].Type != 0) {continue;}
        Vec3<double> dp{All.cf_atime*(double)P[i].Pos[0], All.cf_atime*(double)P[i].Pos[1], All.cf_atime*(double)P[i].Pos[2]};
        double r2 = dp.norm_sq(), wt, wt_new=0, r, de;
        r = sqrt(r2); if(r < r_min || r >= r_max) {continue;}
        double vol = P[i].Mass / (CellP[i].Density*All.cf_a3inv), cos_t = dp[0] / r;
        wt = 1.e-5 * pow(fabs(cos_t),8) * vol * (r_max*r_max/(r*r)-1.); de = de_00 * wt / delta_wt_sumsum;
        if(de <= 0 || !isfinite(de)) {continue;}
        k=RT_FREQ_BIN_INFRARED; double T00 = 1.e5, f0 = de * C_LIGHT_CODE / r;
        if(CellP[i].Radiation_Temperature > 0 && CellP[i].Rad_E_gamma[k] > 0) {CellP[i].Radiation_Temperature = (CellP[i].Rad_E_gamma[k] + de)/(CellP[i].Rad_E_gamma[k]/CellP[i].Radiation_Temperature + de/T00);} else {CellP[i].Radiation_Temperature = T00;}
        CellP[i].Rad_E_gamma[k] += de; CellP[i].Rad_E_gamma_Pred[k] += de;
        int j; for(j=0;j<3;j++) {CellP[i].Rad_Flux[k][j] += f0 * dp[j]; CellP[i].Rad_Flux_Pred[k][j] += f0 * dp[j];}
    }
    if(iBH0 >= 0) {
        P[iBH0].unspawned_wind_mass += MdotJetMsunYr * dt * (6.304e25 * UNIT_TIME_IN_CGS/UNIT_MASS_IN_CGS); // will sent to jets subroutine, for spawning, alongside radiation injection //
        double n_unspawned = P[iBH0].unspawned_wind_mass / ((SINK_WIND_SPAWN)*target_mass_for_wind_spawning(iBH0)); // number of spawned gas cells that can be made from the mass in the reservoir
        if(n_unspawned> Max_Unspawned_MassUnits_fromSink) {Max_Unspawned_MassUnits_fromSink = n_unspawned;} // track the maximum integer number of elements this sink could spawn
        P[iBH0].Sink_Specific_AngMom[0]=1; P[iBH0].Sink_Specific_AngMom[1]=0; P[iBH0].Sink_Specific_AngMom[2]=0; // HACK FOR NOW!!!! (will fix to desired direction in future)
    }
    return;
}
#endif


/* simple routine that evaluates the target cell mass for the spawning subroutine */
double target_mass_for_wind_spawning(int i)
{
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES == 3) // replace later as needed //
    if(is_particle_a_special_zoom_target(i)) {return 1.e-9/UNIT_MASS_IN_SOLAR;} //
#endif
#if (SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES >= 4) // replace later as needed //
    if(is_particle_a_special_zoom_target(i)) {return 1.e-6/UNIT_MASS_IN_SOLAR;} //
#endif

#if defined(SNE_NONSINK_SPAWN)
    if(P[i].Type==4) {return 0.5 / UNIT_MASS_IN_SOLAR;} // replace later as needed //
#endif
    
#ifdef SINK_WIND_SPAWN

#if defined(SINGLE_STAR_FB_WINDS) && defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    if(P[i].Type==5) {
#if defined(SINGLE_STAR_AND_SSP_HYBRID_MODEL) || defined(SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS) // we specify the value relative to Sink_Formation_Mass
        if((All.Cell_Spawn_Mass_ratio_MS>0.0)&&(P[i].ProtoStellarStage == 5)&&(P[i].wind_mode==1)) {return All.Cell_Spawn_Mass_ratio_MS * P[i].Sink_Formation_Mass;} //use different (probably lower) mass for winds than for jets (will also reduce it for MS jets, but that should be fine)
        else {return All.Sink_outflow_particlemass * P[i].Sink_Formation_Mass;}
#else // we specify the absolute value
        if((P[i].ProtoStellarStage == 5) && (P[i].wind_mode==1)) {return All.Cell_Spawn_Mass_ratio_MS;} // specified absolute mass resolution for stellar winds
        else if(P[i].ProtoStellarStage == 6) {return P[i].Sink_Formation_Mass;} // If supernova, use the nominal "average" mass resolution
#endif
    }
#endif // single-star if above 

#if defined(SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS)
    return All.Sink_outflow_particlemass * P[i].Sink_Formation_Mass;
#else
    return All.Sink_outflow_particlemass;
#endif

#endif // SINK_WIND_SPAWN clause
    return 0; // no well-defined answer, this shouldn't be called in this instance
}


#endif // top-level flag
