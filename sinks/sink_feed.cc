/*! \file sink_feed.c
*  \brief This is where particles are marked for gas accretion.
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


#define CORE_FUNCTION_NAME sink_feed_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(sink_isactive(i)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */


/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    int NodeList[NODELISTLENGTH]; Vec3<MyDouble> Pos; Vec3<MyFloat> Vel; MyFloat KernelRadius, Mass, Sink_Mass, Dt, Density, Mdot; MyIDType ID;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    Vec3<MyFloat> Jgas_in_Kernel;
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    MyFloat mass_to_swallow_edd;
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    MyFloat Sink_AccretionDeficit;
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    MyFloat SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    MyFloat AGS_KernelRadius;
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    MyFloat Sink_Mass_Reservoir;
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    int Sink_eligible_for_binary_merge_away;
#endif
}
*DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k, j_tempinfo; j_tempinfo=P[i].IndexMapToTempStruc; /* link to the location in the shared structure where this is stored */
    in->Pos=P[i].Pos; in->Vel=P[i].Vel; /* good example - always needed */
    in->KernelRadius = P[i].KernelRadius; in->Mass = P[i].Mass; in->Sink_Mass = P[i].Sink_Mass; in->ID = P[i].ID; in->Density = P[i].DensityAroundParticle; in->Mdot = P[i].Sink_Mdot;
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    in->SinkRadius = P[i].SinkRadius;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    in->AGS_KernelRadius = ForceSoftening_KernelRadius(i);
#endif
#ifdef SINK_ALPHADISK_ACCRETION
    in->Sink_Mass_Reservoir = P[i].Sink_Mass_Reservoir;
#endif
    in->Dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    in->Dt = P[i].dt_since_last_gas_search;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    in->Jgas_in_Kernel = P[i].Sink_Specific_AngMom;
#else
    for(k=0;k<3;k++) {in->Jgas_in_Kernel[k] = SinkTempInfo[j_tempinfo].Jgas_in_Kernel[k];}
#endif
#endif
#if defined(SINK_GRAVCAPTURE_GAS)
    in->mass_to_swallow_edd = SinkTempInfo[j_tempinfo].mass_to_swallow_edd;
#endif
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    in->Sink_AccretionDeficit = P[i].Sink_AccretionDeficit;
#endif
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
    in->Sink_eligible_for_binary_merge_away = is_star_eligible_for_binary_merge_away(i);
#endif
}


/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{ /* define variables below as e.g. "double X;" */
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    double Sink_angle_weighted_kernel_sum;
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    double Sink_PotentialMinimumOfNeighbors, Sink_PotentialMinimumOfNeighborsPos[3];
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

#define ASSIGN_ADD_PRESET(x,y,mode) (mode == 0 ? (x=y) : (x+=y))
/* this subroutine assigns the values to the variables that need to be sent -back to- the 'searching' element */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k, target; k=0; target = P[i].IndexMapToTempStruc;
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    ASSIGN_ADD_PRESET(SinkTempInfo[target].Sink_angle_weighted_kernel_sum, out->Sink_angle_weighted_kernel_sum, mode);
#endif
#ifdef SINK_REPOSITION_ON_POTMIN
    if(mode==0) {P[i].Sink_PotentialMinimumOfNeighbors=out->Sink_PotentialMinimumOfNeighbors; for(k=0;k<3;k++) {P[i].Sink_PotentialMinimumOfNeighborsPos[k]=out->Sink_PotentialMinimumOfNeighborsPos[k];}
        } else {if(out->Sink_PotentialMinimumOfNeighbors < P[i].Sink_PotentialMinimumOfNeighbors) {P[i].Sink_PotentialMinimumOfNeighbors=out->Sink_PotentialMinimumOfNeighbors; for(k=0;k<3;k++) {P[i].Sink_PotentialMinimumOfNeighborsPos[k]=out->Sink_PotentialMinimumOfNeighborsPos[k];}}}
#endif
}



/* do loop over neighbors to get quantities for accretion */
/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below. none of the modified values are read, so only the write block is protected. */
int sink_feed_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration);
int sink_feed_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    /* initialize variables before loop is started */
    int startnode, numngb, listindex = 0, j, k, n; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */
    double h_i = local.KernelRadius, wk, dwk, vrel, vesc, f_accreted; f_accreted=1; Vec3<double> dpos={}, dvel={};
    if((local.Mass<0)||(h_i<=0)) {return 0;}
    double w, p, r2, r, u, sink_radius=SinkParticle_GravityKernelRadius, h_i2 = h_i * h_i, hinv = 1 / h_i, hinv3 = hinv * hinv * hinv, ags_h_i = SinkParticle_GravityKernelRadius; p=0; w=0;
#ifdef SINK_REPOSITION_ON_POTMIN
    out.Sink_PotentialMinimumOfNeighbors = SINK_MINPOTVALUE_INIT;
#endif
#if (ADAPTIVE_GRAVSOFT_FORALL & 32)
    ags_h_i = local.AGS_KernelRadius;
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    double J_dir[3]; for(k=0;k<3;k++) {J_dir[k] = local.Jgas_in_Kernel[k];}
#endif
#if defined(SINK_GRAVCAPTURE_GAS) && defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
    double meddington = sink_eddington_mdot(local.Sink_Mass), medd_max_accretable = All.SinkEddingtonFactor * meddington * local.Dt, eddington_factor = local.mass_to_swallow_edd / medd_max_accretable;   /* if <1 no problem, if >1, need to not set some swallowIDs */
#endif
#if defined(SINK_SWALLOWGAS)
    double mass_markedswallow,sink_mass_withdisk; mass_markedswallow=0; sink_mass_withdisk=local.Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
    sink_mass_withdisk += local.Sink_Mass_Reservoir;
#endif
#endif
#if defined(SINK_WIND_KICK) && !defined(SINK_GRAVCAPTURE_GAS) /* DAA: increase the effective mass-loading of BAL winds to reach the desired momentum flux given the outflow velocity "All.Sink_outflow_velocity" chosen --> appropriate for cosmological simulations where particles are effectively kicked from ~kpc scales (i.e. we need lower velocity and higher mass outflow rates compared to accretion disk scales) - */
    f_accreted = All.Sink_accreted_fraction; if((All.SinkFeedbackFactor > 0) && (All.SinkFeedbackFactor != 1.)) {f_accreted /= All.SinkFeedbackFactor;} else {if(All.Sink_outflow_velocity > 0) {f_accreted = 1./(1. + fabs(1.*SINK_WIND_KICK)*All.SinkRadiativeEfficiency*C_LIGHT_CODE/(All.Sink_outflow_velocity));}}
#endif
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS)
    double norm=0; for(k=0;k<3;k++) {norm+=J_dir[k]*J_dir[k];}
    if(norm>0) {norm=1/sqrt(norm); for(k=0;k<3;k++) {J_dir[k]*=norm;}} else {J_dir[0]=J_dir[1]=0; J_dir[2]=1;}
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
    sink_radius = local.SinkRadius;
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
                if(P[j].Mass > 0)
                {
                    dpos = P[j].Pos - local.Pos; dvel = P[j].Vel - local.Vel;
                    nearest_xyz(dpos,-1); r2=dpos.norm_sq();
                    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,dvel,-1); /* wrap velocities for shearing boxes if needed */
                    double heff_j = DMAX( P[j].KernelRadius , ForceSoftening_KernelRadius(j) );
                    if(r2 < h_i2 || r2 < heff_j*heff_j)
                    {
                        vrel=dvel.norm_sq();
                        r=sqrt(r2); vrel=sqrt(vrel)/All.cf_atime;  /* relative velocity in physical units. do this once and use below */
#if defined(MAGNETIC) && defined(GRAIN_LORENTZFORCE) /* need to project grain velocities, shouldn't include gyro motion */
                        if((1<<P[j].Type) & GRAIN_PTYPES) {vrel=0; double bmag2=0; for(k=0;k<3;k++) {vrel+=dvel[k]*P[j].Gas_B[k]; bmag2+=P[j].Gas_B[k]*P[j].Gas_B[k];}
                            vrel = (fabs(vrel)/sqrt(bmag2)) / All.cf_atime;}
#endif
                        vesc=sink_vesc(j, local.Mass, r, ags_h_i);
                        
                        /* note that SwallowID is both read and potentially re-written below: we need to make sure this is done in a thread-safe manner */
                        MyIDType SwallowID_j;
                        #pragma omp atomic read
                        SwallowID_j = P[j].SwallowID; // ok got a clean read. -not- gauranteed two threads won't see this at the same time and compete over it [both think they get it here]. but only one will -actually- get it, and that's ok.
                        
#ifdef SINK_REPOSITION_ON_POTMIN
                        /* check if we've found a new potential minimum which is not moving too fast to 'jump' to */
                        double boundedness_function, potential_function; boundedness_function = P[j].Potential + 0.5 * vrel*vrel * All.cf_atime; potential_function = P[j].Potential;
                        if( boundedness_function < 0 )
                        {
                            double wt_rsoft = r / (3.*SinkParticle_GravityKernelRadius); // normalization arbitrary here, just using for convenience for function below
                            boundedness_function *= 1./(1. + wt_rsoft*wt_rsoft); // this down-weights particles which are very far away, relative to the user-defined force softening scale, which should define some 'confidence radius' of resolution around the BH particle
                        }
                        potential_function = boundedness_function; // jumps based on -most bound- particle, not just deepest potential (down-weights fast-movers)
                        if(potential_function < out.Sink_PotentialMinimumOfNeighbors)
                        if( (P[j].Type != 0) && (P[j].Type != 5) )   // allow stars or dark matter but exclude gas, it's too messy! also exclude BHs, since we don't want to over-merge them
                        {
                            out.Sink_PotentialMinimumOfNeighbors=potential_function; for(k=0;k<3;k++) {out.Sink_PotentialMinimumOfNeighborsPos[k] = P[j].Pos[k];}
                        }
#endif
			
                        
                        
                        if(P[j].Type == 5)  /* we may have a sink particle merger -- check below if allowed */
                        {
                            if(((local.ID != P[j].ID) || (r2>0)) && (SwallowID_j == 0) && (P[j].Sink_Mass < local.Sink_Mass)) /* we'll assume most massive BH swallows the other - simplifies analysis and ensures unique results */
                            {
#ifdef SINGLE_STAR_SINK_DYNAMICS
                                int allow_sink_merger = 1; /* flag here b/c we have different options */
                                if(r >= 1.0001*P[j].Min_Distance_to_Sink) {allow_sink_merger = 0;} // not the closest sink!
                                if(r >= heff_j) {allow_sink_merger = 0;} // beyond MAX[search/kernel/softening] radius: heff_j = DMAX( P[j].KernelRadius , ForceSoftening_KernelRadius(j) )
                                if(P[j].Mass > local.Mass) {allow_sink_merger = 0;} // always merge from more massive eating lower
                                if((P[j].Mass == local.Mass) && (P[j].ID > local.ID)) {allow_sink_merger = 0;} // randomly pick for equal masses which way the merger goes
                                double max_rmerge = 1.0*sink_radius; // default STARFORGE behavior: only merge away stuff that is within the softening radius; sink_radius=SinkParticle_GravityKernelRadius
                                double max_mmerge = 10.*P[j].Sink_Formation_Mass; // default STARFORGE behavior: only merge away stuff no more massive than a few gas cells
#ifdef SINGLE_STAR_MERGE_AWAY_CLOSE_BINARIES
                                if(local.Sink_eligible_for_binary_merge_away == 0) {allow_sink_merger = 0;}
                                if(is_star_eligible_for_binary_merge_away(j) == 0) {allow_sink_merger = 0;}
                                max_mmerge = 10.*P[j].Mass; // makes it so there is no mass limit - even extremely massive stars can be merged
                                max_rmerge = DMAX(max_rmerge, ForceSoftening_KernelRadius(j));
                                max_rmerge = DMAX(max_rmerge, DMIN(local.KernelRadius, P[j].KernelRadius));
                                max_rmerge = DMIN(max_rmerge, 10.*sink_radius);
                                double dt_min_orbit_yr = 100.; // 'target' minimum orbital time of the binary in yr
                                double rmax_dt = 0.000485/(All.cf_atime*UNIT_LENGTH_IN_PC) * pow(((P[j].Mass+local.Mass)*UNIT_MASS_IN_SOLAR/100.) * (dt_min_orbit_yr*dt_min_orbit_yr/(100.*100.)),1./3.); // ensures the binary period satisfies this
                                max_rmerge = DMAX( max_rmerge , rmax_dt);
#else
                                if(sink_check_boundedness(j,vrel,vesc,r,sink_radius) != 1) {allow_sink_merger = 0;} // stricter criterion
#endif
                                if(r >= max_rmerge) {allow_sink_merger = 0;} // beyond max radius (default sink)
                                if(P[j].Mass > max_mmerge) {allow_sink_merger = 0;} // beyond max mass (default few cells)
                                if(allow_sink_merger == 1) /* ok only if meet all the criteria above are we allowed to consider a BH-BH merger */
#endif
                                {
                                    if(vrel < vesc)
                                    {
                                        printf(" ..Sink-Sink Merger: P[j.]ID=%llu to be swallowed by id=%llu \n", (unsigned long long) P[j].ID, (unsigned long long) local.ID);
                                        SwallowID_j = local.ID;
                                    } else {
#if defined(SINK_OUTPUT_MOREINFO)     // DAA: BH merger info will be saved in a separate output file
                                        printf(" ..ThisTask=%d, time=%g: id=%llu would like to swallow %llu, but vrel=%g vesc=%g\n", ThisTask, All.Time, (unsigned long long)local.ID, (unsigned long long)P[j].ID, vrel, vesc);
#elif defined(OUTPUT_ADDITIONAL_RUNINFO)
                                        fprintf(FdSinksDetails, "Sink-Sink Merger Not Allowed: ThisTask=%d, time=%.16g: id=%llu would like to swallow %llu, but vrel=%g vesc=%g\n", ThisTask, All.Time, (unsigned long long)local.ID, (unsigned long long)P[j].ID, vrel, vesc); fflush(FdSinksDetails);
#endif
                                    }
                                } // if eligible for bh-bh mergers //
                            } // unique BH, merging from higher (swallowing lower) mass
                        } // type == 5
                        
                        
                        /* This is a similar loop to what we already did in sink_environment, but here we stochastically
                         reduce GRAVCAPT events in order to (statistically) obey the eddington limit */
#if defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_GRAVCAPTURE_NONGAS)
                        if((P[j].Type != 5) && (SwallowID_j < local.ID)) // we have a particle not already marked to swallow
                        {
#ifdef SINGLE_STAR_SINK_DYNAMICS
                            double eps = DMAX( r , DMAX(heff_j , ags_h_i) * KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER); // plummer-equivalent
			                if(eps*eps*eps /(P[j].Mass + local.Mass) <= P[j].SwallowTime)
#endif
#if defined(SINK_ALPHADISK_ACCRETION)
                            if(local.Sink_Mass_Reservoir < SINK_ALPHADISK_ACCRETION*local.Sink_Mass)
#endif
                            if((vrel < vesc))
                            { /* bound */
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
                                double spec_mom=dot(dvel,dpos); // delta_x.delta_v
                                spec_mom = (r2*vrel*vrel - spec_mom*spec_mom*All.cf_a2inv); // specific angular momentum^2 = r^2(delta_v)^2 - (delta_v.delta_x)^2;
				                if(spec_mom < All.G * (local.Mass + P[j].Mass) * sink_radius)  // check Bate 1995 angular momentum criterion (in addition to bounded-ness)
#endif
                                if( sink_check_boundedness(j,vrel,vesc,r,sink_radius)==1 ) /* bound and apocenter within target distance */
                                {
#ifdef SINK_GRAVCAPTURE_NONGAS        /* simply swallow non-gas particle if SINK_GRAVCAPTURE_NONGAS enabled */
                                    if(P[j].Type != 0) {SwallowID_j = local.ID;}
#endif
#if defined(SINK_GRAVCAPTURE_GAS)     /* now deal with gas */
                                    if(P[j].Type == 0)
                                    {
#if defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION) /* if Eddington-limited and NO alpha-disk, do this stochastically */
                                        p = 1. / eddington_factor;
#if defined(SINK_WIND_KICK)
                                        p /= All.Sink_accreted_fraction; // we need to accrete more, then remove the mass in winds
#endif
                                        w = get_random_number(P[j].ID);
                                        if(w < p)
                                        {
#ifdef SINK_OUTPUT_MOREINFO
                                            printf(" ..Sink-Food Marked: P[j.]ID=%llu to be swallowed by id=%llu \n", (unsigned long long) P[j].ID, (unsigned long long) local.ID);
#endif
                                            SwallowID_j = local.ID;
                                        }
#else //if defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
                                        SwallowID_j = local.ID; /* in other cases, just swallow the particle */
#endif //else defined(SINK_ENFORCE_EDDINGTON_LIMIT) && !defined(SINK_ALPHADISK_ACCRETION)
                                    } //if (P[j].Type == 0)
#endif //ifdef SINK_GRAVCAPTURE_GAS
                                } // if( apocenter in tolerance range )
                            } // if(vrel < vesc)
                        } //if(P[j].Type != 5)
#endif // defined(SINK_GRAVCAPTURE_GAS) || defined(SINK_GRAVCAPTURE_NONGAS)
                        
                        
                        
                        /* now is the more standard accretion only of gas, according to the mdot calculated before */
                        if(P[j].Type == 0) /* here we have a gas particle */
                        {
                            u=r*hinv; if(u<1) {kernel_main(u,hinv3,hinv*hinv3,&wk,&dwk,-1);} else {wk=dwk=0;}
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS) /* compute accretion probability, this below is only meaningful if !defined(SINK_GRAVCAPTURE_GAS)... */
                            if(SwallowID_j < local.ID)
                            {
                                //double dm_toacc = sink_mass_withdisk - (local.Mass + mass_markedswallow); -- old model, used total mass as 'target' which can be a problem
                                double dm_toacc = local.Sink_AccretionDeficit - mass_markedswallow; // amount of continuous accretion 'deficit' integrated for these BHs
                                if(dm_toacc>0) {p=dm_toacc*wk/local.Density;} else {p=0;}
#ifdef SINK_WIND_KICK /* DAA: for stochastic winds (SINK_WIND_KICK) we remove a fraction of mass from gas particles prior to kicking --> need to increase the probability here to balance sink particle growth */
                                if(f_accreted>0) {p /= f_accreted; if((sink_mass_withdisk - local.Mass) < 0) {p = ( (1-f_accreted)/f_accreted ) * local.Mdot * local.Dt * wk / local.Density;}} /* DAA: compute outflow probability when "sink_mass_withdisk < mass" - we don't need to enforce mass conservation in this case, relevant only in low-res sims where the BH seed mass is much lower than the gas particle mass */
#endif
                                w = get_random_number(P[j].ID);
                                if(w < p)
                                {
#ifdef SINK_OUTPUT_MOREINFO
                                    printf(" ..Sink-Food Marked: j %d w %g p %g TO_BE_SWALLOWED \n",j,w,p);
#endif
                                    SwallowID_j = local.ID;
                                    mass_markedswallow += P[j].Mass*f_accreted;
                                } // if(w < p)
                            } // swallowID < localID
#endif // SINK_SWALLOWGAS
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) /* calculate the angle-weighting for the photon momentum */
                            if((local.Dt>0)&&(r>0)&&(SwallowID_j==0)&&(P[j].Mass>0)&&(P[j].Type==0))
                            { /* cos_theta with respect to disk of BH is given by dot product of r and Jgas */
                                norm=0; for(k=0;k<3;k++) {norm+=(dpos[k]/r)*J_dir[k];}
                                out.Sink_angle_weighted_kernel_sum += sink_fb_angleweight_localcoupling(j,norm,r,h_i);
                            }
#endif
#ifdef SINK_THERMALFEEDBACK
                            double energy = sink_lum_bol(local.Mdot, local.Sink_Mass, -1) * local.Dt;
                            if(local.Density > 0) {
                                #pragma omp atomic
                                CellP[j].Injected_Sink_Energy += (wk/local.Density) * energy * P[j].Mass;
                            }
#endif                            
                        } // if(P[j].Type == 0)
                        
                        
                        /* ok, before exiting this loop, need to mark whether or not we actually designated a particle for accretion! */
                        if(SwallowID_j > 0)
                        {
                            #pragma omp atomic write
                            P[j].SwallowID = SwallowID_j;  // ok got a clean write. -not- gauranteed two threads won't see this at the same time and compete over it [both think they get it here]. but only one will -actually- get it, and that's ok.
                        }
                        
                    } // if(r2 < h_i2)
                } // if(P[j].Mass > 0)
            } // for(n = 0; n < numngb; n++)
        } // while(startnode >= 0)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    } // while(startnode >= 0) (outer of the double-loop)
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
} /* closes evaluate routine */



void sink_feed_loop(void)
{
#include "../system/code_block_xchange_perform_ops_malloc.h" /* this calls the large block of code which contains the memory allocations for the MPI/OPENMP/Pthreads parallelization block which must appear below */
#include "../system/code_block_xchange_perform_ops.h" /* this calls the large block of code which actually contains all the loops, MPI/OPENMP/Pthreads parallelization */
#include "../system/code_block_xchange_perform_ops_demalloc.h" /* this de-allocates the memory for the MPI/OPENMP/Pthreads parallelization block which must appear above */
CPU_Step[CPU_SINKS] += measure_time(); /* collect timings and reset clock for next timing */
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */


#endif // top-level flag
