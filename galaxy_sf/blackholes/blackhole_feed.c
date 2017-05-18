/*! \file blackhole_feed.c
 *  \brief This is where particles are marked for gas accretion.
 */
/*
 * This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 *   It was based on a similar file in GADGET3 by Volker Springel (volker.springel@h-its.org),
 *   but the physical modules for black hole accretion and feedback have been
 *   replaced, and the algorithm for their coupling is new to GIZMO.  This file was modified
 *   on 1/9/15 by Paul Torrey (ptorrey@mit.edu) for clarity by parsing the existing code into
 *   smaller files and routines. Some communication and black hole structures were modified
 *   to reduce memory usage. Cleanup, de-bugging, and consolidation of routines by Xiangcheng Ma
 *   (xchma@caltech.edu) followed on 05/15/15; re-integrated by PFH.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../allvars.h"
#include "../../proto.h"
#include "../../kernel.h"
#include "blackhole_local.h"


void blackhole_feed_loop(void)
{
    int i, j, k, ndone_flag, ndone;
    int ngrp, recvTask, place, nexport, nimport, dummy;
    MPI_Status status;
    
    /* allocate buffers to arrange communication */
    size_t MyBufferSize = All.BufferSize;
    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));
    All.BunchSize = (int) ((MyBufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                                             sizeof(struct blackholedata_in) +
                                                             sizeof(struct blackholedata_out) +
                                                             sizemax(sizeof(struct blackholedata_in),sizeof(struct blackholedata_out))));
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    
    /* Let's determine which particles may be swallowed by whom, and the weights for feedback */
    i = FirstActiveParticle;
    do
    {
        for(j = 0; j < NTask; j++)
        {
            Send_count[j] = 0;
            Exportflag[j] = -1;
        }
        
        /* do local particles and prepare export list */
        for(nexport = 0; i >= 0; i = NextActiveParticle[i])                      // DAA: can this be replaced by a loop over N_active_loc_BHs
            if(P[i].Type == 5)
                if(blackhole_feed_evaluate(i, 0, &nexport, Send_count) < 0)
                    break;
        
        MYSORT_DATAINDEX(DataIndexTable, nexport, sizeof(struct data_index), data_index_compare);
        MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
        for(j = 0, nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++)
        {
            nimport += Recv_count[j];
            if(j > 0)
            {
                Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
            }
        }
        BlackholeDataGet = (struct blackholedata_in *) mymalloc("BlackholeDataGet", nimport * sizeof(struct blackholedata_in));
        BlackholeDataIn = (struct blackholedata_in *) mymalloc("BlackholeDataIn", nexport * sizeof(struct blackholedata_in));
        
        for(j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;
            
            for(k = 0; k < 3; k++)
            {
                BlackholeDataIn[j].Pos[k] = P[place].Pos[k];
                BlackholeDataIn[j].Vel[k] = P[place].Vel[k];
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
                BlackholeDataIn[j].Jgas_in_Kernel[k] = P[place].GradRho[k];
#endif
            }
#if defined(BH_GRAVCAPTURE_GAS)
            BlackholeDataIn[j].mass_to_swallow_edd = BlackholeTempInfo[P[place].IndexMapToTempStruc].mass_to_swallow_edd;
#endif
            BlackholeDataIn[j].Hsml = PPP[place].Hsml;
            BlackholeDataIn[j].Mass = P[place].Mass;
            BlackholeDataIn[j].BH_Mass = BPP(place).BH_Mass;
#ifdef BH_ALPHADISK_ACCRETION
            BlackholeDataIn[j].BH_Mass_AlphaDisk = BPP(place).BH_Mass_AlphaDisk;
#endif
#if defined(BH_PHOTONMOMENTUM) 	|| defined(BH_BAL_WINDS)
            BlackholeDataIn[j].BH_disk_hr = P[place].BH_disk_hr;
#endif
            BlackholeDataIn[j].Density = BPP(place).DensAroundStar;
            BlackholeDataIn[j].Mdot = BPP(place).BH_Mdot;
#ifndef WAKEUP
            BlackholeDataIn[j].Dt = (P[place].TimeBin ? (1 << P[place].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
            BlackholeDataIn[j].Dt = P[place].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
            BlackholeDataIn[j].ID = P[place].ID;
            memcpy(BlackholeDataIn[j].NodeList,DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
        }
        
        /* exchange particle data */
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* get the particles */
                    MPI_Sendrecv(&BlackholeDataIn[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
                                 recvTask, TAG_BH_E,
                                 &BlackholeDataGet[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct blackholedata_in), MPI_BYTE,
                                 recvTask, TAG_BH_E, MPI_COMM_WORLD, &status);
                }
            }
        }
        myfree(BlackholeDataIn);
        BlackholeDataResult = (struct blackholedata_out *) mymalloc("BlackholeDataResult",nimport * sizeof(struct blackholedata_out));
        BlackholeDataOut = (struct blackholedata_out *) mymalloc("BlackholeDataOut", nexport * sizeof(struct blackholedata_out));
        
        /* now do the particles that were sent to us */
        for(j = 0; j < nimport; j++)
            blackhole_feed_evaluate(j, 1, &dummy, &dummy);
        
        
        if(i < 0)
            ndone_flag = 1;
        else
            ndone_flag = 0;
        
        MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        /* get the result */
        for(ngrp = 1; ngrp < (1 << PTask); ngrp++)
        {
            recvTask = ThisTask ^ ngrp;
            if(recvTask < NTask)
            {
                if(Send_count[recvTask] > 0 || Recv_count[recvTask] > 0)
                {
                    /* send the results */
                    MPI_Sendrecv(&BlackholeDataResult[Recv_offset[recvTask]],
                                 Recv_count[recvTask] * sizeof(struct blackholedata_out),
                                 MPI_BYTE, recvTask, TAG_BH_F,
                                 &BlackholeDataOut[Send_offset[recvTask]],
                                 Send_count[recvTask] * sizeof(struct blackholedata_out),
                                 MPI_BYTE, recvTask, TAG_BH_F, MPI_COMM_WORLD, &status);
                }
            }
        } // for(ngrp = 1; ngrp < (1 << PTask); ngrp++) //
        
        /* add the result to the particles */
        for(j = 0; j < nexport; j++)
        {
            place = DataIndexTable[j].Index;
#ifdef BH_REPOSITION_ON_POTMIN
            if(BPP(place).BH_MinPot > BlackholeDataOut[j].BH_MinPot)
            {
                BPP(place).BH_MinPot = BlackholeDataOut[j].BH_MinPot;
                for(k = 0; k < 3; k++)
                    BPP(place).BH_MinPotPos[k] = BlackholeDataOut[j].BH_MinPotPos[k];
            }
#endif
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
            BlackholeTempInfo[P[place].IndexMapToTempStruc].BH_angle_weighted_kernel_sum += BlackholeDataOut[j].BH_angle_weighted_kernel_sum;
#endif
        }
        
        myfree(BlackholeDataOut);
        myfree(BlackholeDataResult);
        myfree(BlackholeDataGet);
    }
    while(ndone < NTask);
    
    myfree(DataNodeList);
    myfree(DataIndexTable);
    myfree(Ngblist);
    
}







/* do loop over neighbors to get quantities for accretion */
int blackhole_feed_evaluate(int target, int mode, int *nexport, int *nSend_local)
{
    int startnode, numngb, j, k, n, listindex = 0;
    MyIDType id;
    MyFloat *pos, *velocity, h_i, dt, mdot, rho, mass, bh_mass;
    double h_i2, r2, r, u, hinv, hinv3, wk, dwk, vrel, vesc, dpos[3];
    
#if defined(BH_GRAVCAPTURE_GAS) && defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
    double meddington, medd_max_accretable, mass_to_swallow_edd, eddington_factor;
#endif
    
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
    double norm, theta, BH_disk_hr, *Jgas_in_Kernel;
    double BH_angle_weighted_kernel_sum=0;
#endif

#if defined(BH_BAL_KICK) && !defined(BH_GRAVCAPTURE_GAS)
    double f_accreted=0; 
#endif
    
#ifdef BH_THERMALFEEDBACK
    double energy;
#endif
#ifdef BH_REPOSITION_ON_POTMIN
    MyFloat minpotpos[3] = { 0, 0, 0 }, minpot = BHPOTVALUEINIT;
#endif
#ifdef BH_ALPHADISK_ACCRETION
    MyFloat bh_mass_alphadisk;
#endif
#if defined(BH_SWALLOWGAS)
    double w,p,mass_markedswallow,bh_mass_withdisk;
    w=0; p=0; mass_markedswallow=0; bh_mass_withdisk=0;
#endif
    
    
    /* these are the BH properties */
    if(mode == 0)
    {
        pos = P[target].Pos;
        rho = P[target].DensAroundStar;       // DAA: DensAroundStar is not defined in BHP->BPP...
        mdot = BPP(target).BH_Mdot;
#ifndef WAKEUP
        dt = (P[target].TimeBin ? (1 << P[target].TimeBin) : 0) * All.Timebase_interval / All.cf_hubble_a;
#else
        dt = P[target].dt_step * All.Timebase_interval / All.cf_hubble_a;
#endif
        h_i = PPP[target].Hsml;
        mass = P[target].Mass;
        bh_mass = BPP(target).BH_Mass;
#ifdef BH_ALPHADISK_ACCRETION
        bh_mass_alphadisk = BPP(target).BH_Mass_AlphaDisk;
#endif
        velocity = P[target].Vel;
        id = P[target].ID;
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
        Jgas_in_Kernel = P[target].GradRho;
        BH_disk_hr = P[target].BH_disk_hr;
#endif
#if defined(BH_GRAVCAPTURE_GAS) && defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
        mass_to_swallow_edd = BlackholeTempInfo[P[target].IndexMapToTempStruc].mass_to_swallow_edd;
#endif
    }
    else
    {
        pos = BlackholeDataGet[target].Pos;
        rho = BlackholeDataGet[target].Density;
        mdot = BlackholeDataGet[target].Mdot;
        dt = BlackholeDataGet[target].Dt;
        h_i = BlackholeDataGet[target].Hsml;
        mass = BlackholeDataGet[target].Mass;
        bh_mass = BlackholeDataGet[target].BH_Mass;
#ifdef BH_ALPHADISK_ACCRETION
        bh_mass_alphadisk = BlackholeDataGet[target].BH_Mass_AlphaDisk;
#endif
        velocity = BlackholeDataGet[target].Vel;
        id = BlackholeDataGet[target].ID;
#if defined(BH_PHOTONMOMENTUM)  || defined(BH_BAL_WINDS)
        Jgas_in_Kernel = BlackholeDataGet[target].Jgas_in_Kernel;
        BH_disk_hr = BlackholeDataGet[target].BH_disk_hr;
#endif
#if defined(BH_GRAVCAPTURE_GAS) && defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
        mass_to_swallow_edd = BlackholeDataGet[target].mass_to_swallow_edd;
#endif
    }
    
    if((mass<0)||(h_i<=0)) return -1;
    
    
    /* initialize variables before SPH loop is started */
    h_i2 = h_i * h_i;
    hinv = 1 / h_i;
    hinv3 = hinv * hinv * hinv;
#if defined(BH_GRAVCAPTURE_GAS) && defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
    meddington = bh_eddington_mdot(bh_mass);
    medd_max_accretable = All.BlackHoleEddingtonFactor * meddington * dt;
    eddington_factor = mass_to_swallow_edd / medd_max_accretable;   /* if <1 no problem, if >1, need to not set some swallowIDs */
#endif
    
#if defined(BH_SWALLOWGAS)
    bh_mass_withdisk = bh_mass;
#ifdef BH_ALPHADISK_ACCRETION
    bh_mass_withdisk += bh_mass_alphadisk;
#endif
#endif

#if defined(BH_BAL_KICK) && !defined(BH_GRAVCAPTURE_GAS)
#ifdef BH_BAL_KICK_MOMENTUM_FLUX
    /* DAA: increase the effective mass-loading of BAL winds to reach the desired momentum flux given the outflow velocity "All.BAL_v_outflow" chosen
       --> appropriate for cosmological simulations where particles are effectively kicked from ~kpc scales
           (i.e. we need lower velocity and higher mass outflow rates compared to accretion disk scales) - */
    if(All.BAL_v_outflow > 0){
        f_accreted = 1. / ( 1. + BH_BAL_KICK_MOMENTUM_FLUX * All.BlackHoleRadiativeEfficiency * (C / All.UnitVelocity_in_cm_per_s) / All.BAL_v_outflow );
    }else{
        f_accreted = All.BAL_f_accretion;
    }
#else
    f_accreted = All.BAL_f_accretion;
#endif
#endif //#ifdef BH_BAL_KICK
    
    /* Now start the actual SPH computation for this BH particle */
    if(mode == 0)
    {
        startnode = All.MaxPart;	/* root node */
    }
    else
    {
        startnode = BlackholeDataGet[target].NodeList[0];
        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    int particles_swallowed_this_bh_this_process = 0;
    int particles_swallowed_this_bh_this_process_max = 1;
    
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
    BH_angle_weighted_kernel_sum = 0;
#endif
    
    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb = ngb_treefind_variable_targeted(pos, h_i, target, &startnode, mode, nexport, nSend_local, BH_NEIGHBOR_BITFLAG); // BH_NEIGHBOR_BITFLAG defines which types of particles we search for
            if(numngb < 0) return -1;
            
            for(n = 0; n < numngb; n++)
            {
                j = Ngblist[n];
                if(P[j].Mass > 0)
                {
                    for(k=0;k<3;k++) dpos[k] = pos[k] - P[j].Pos[k]; 
#ifdef PERIODIC
                    NEAREST_XYZ(dpos[0],dpos[1],dpos[2],1);
#endif
                    r2=0; for(k=0;k<3;k++) r2+=dpos[k]*dpos[k];
                    
                    if(r2 < h_i2)
                    {                    
                        r = sqrt(r2);
                        vrel = 0;
                        for(k=0;k<3;k++) vrel += (P[j].Vel[k] - velocity[k])*(P[j].Vel[k] - velocity[k]);
                        vrel = sqrt(vrel) / All.cf_atime;       /* do this once and use below */
                        vesc = bh_vesc(j, mass, r);

#ifdef BH_REPOSITION_ON_POTMIN
                        /* check if we've found a new potential minimum which is not moving too fast to 'jump' to */
#if (BH_REPOSITION_ON_POTMIN == 1) 
                        if( (P[j].Potential < minpot) && (P[j].Type == 4) )   // DAA: only if it is a star particle
#else
                        if(P[j].Potential < minpot)
#endif
                        {
                            if(vrel <= vesc)
                            {
                                minpot = P[j].Potential;
                                for(k = 0; k < 3; k++) minpotpos[k] = P[j].Pos[k];
                            }
                        }
#endif

                        /* check_for_bh_merger.  Easy.  No Edd limit, just a pos and vel criteria. */
                        if((id != P[j].ID) && (P[j].Mass > 0) && (P[j].Type == 5))	/* we may have a black hole merger */
                        {
                            if(id != P[j].ID) /* check its not the same bh  (DAA: this is duplicated here...) */
                            {
                                if(vrel > BH_CSND_FRAC_BH_MERGE * vesc)
                                {
#ifndef IO_REDUCED_MODE
#ifdef BH_OUTPUT_MOREINFO           // DAA: BH merger info will be saved in a separate output file
                                    printf("ThisTask=%d, time=%g: id=%u would like to swallow %u, but vrel=%g vesc=%g\n",
                                            ThisTask, All.Time, id, P[j].ID, vrel, vesc);
#else
                                    fprintf(FdBlackHolesDetails,
                                            "ThisTask=%d, time=%g: id=%u would like to swallow %u, but vrel=%g vesc=%g\n",
                                            ThisTask, All.Time, id, P[j].ID, vrel, vesc);
#endif
#endif
                                }
                                else
                                {
#ifndef IO_REDUCED_MODE
                                    printf("MARKING_BH_MERGER: P[j.]ID=%llu to be swallowed by id=%llu \n", (unsigned long long) P[j].ID, (unsigned long long) id);
#endif
#ifndef SINGLE_STAR_FORMATION
                                    if((P[j].SwallowID == 0) && (BPP(j).BH_Mass < bh_mass)) {P[j].SwallowID = id;} // most massive BH swallows the other - simplifies analysis
#endif
                                }
                            }
                        } // if(P[j].Type == 5) //
                        
                        
                        
                        /* This is a similar loop to what we already did in blackhole_environment, but here we stochastically
                         reduce GRAVCAPT events in order to (statistically) obey the eddington limit */
#if defined(BH_GRAVCAPTURE_GAS) || defined(BH_GRAVCAPTURE_NONGAS)
                        if(P[j].Type != 5)
                        {
                            if((vrel < vesc) && (particles_swallowed_this_bh_this_process < particles_swallowed_this_bh_this_process_max))
                            { /* bound */
                                if( bh_check_boundedness(j,vrel,vesc,r)==1 )
                                { /* apocenter within target distance */        
#ifdef BH_GRAVCAPTURE_NONGAS
                                    /* simply swallow non-gas particle if BH_GRAVCAPTURE_NONGAS enabled */
                                    if((P[j].Type != 0) && (P[j].SwallowID < id)) P[j].SwallowID = id;
#endif
                                    
#ifdef BH_GRAVCAPTURE_GAS
                                    /* now deal with gas */
                                    if (P[j].Type == 0){
#if defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
                                        /* if Eddington-limited and NO alpha-disk, do this stochastically */
                                        p = 1/eddington_factor;
#if defined(BH_BAL_WINDS) || defined(BH_BAL_KICK)
                                        p /= All.BAL_f_accretion; // we need to accrete more, then remove the mass in winds
#endif
                                        w = get_random_number(P[j].ID);
                                        if(w < p)
                                        {
#ifndef IO_REDUCED_MODE
                                            printf("MARKING_BH_FOOD: P[j.]ID=%llu to be swallowed by id=%llu \n", (unsigned long long) P[j].ID, (unsigned long long) id);
#endif
                                            if(P[j].SwallowID < id) P[j].SwallowID = id;
                                        }
#else //if defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
                                        /* in other cases, just swallow the particle */
                                        if(P[j].SwallowID < id) 
                                        {
                                            P[j].SwallowID = id;
                                            particles_swallowed_this_bh_this_process++;
                                        }
#endif //else defined(BH_ENFORCE_EDDINGTON_LIMIT) && !defined(BH_ALPHADISK_ACCRETION)
                                    } //if (P[j].Type == 0)
#endif //ifdef BH_GRAVCAPTURE_GAS
                                    
                                } // if( apocenter in tolerance range )
                            } // if(vrel < vesc)
                        } //if(P[j].Type != 5)
#endif // if defined(BH_GRAVCAPTURE_GAS) || defined(BH_GRAVCAPTURE_NONGAS)
                        
                        
                        
                        
                        /* now is the more standard accretion only of gas, according to the mdot calculated before */
                        if(P[j].Type == 0)
                        {
                            /* here we have a gas particle */
                            u = r * hinv;
                            kernel_main(u,hinv3,hinv*hinv3,&wk,&dwk,-1);
                            
// DAA: this below is only meaningful if !defined(BH_GRAVCAPTURE_GAS)...
//#ifdef BH_SWALLOWGAS
#if defined(BH_SWALLOWGAS) && !defined(BH_GRAVCAPTURE_GAS)
                            /* compute accretion probability */
                            if((bh_mass_withdisk - (mass + mass_markedswallow))>0)
                                p = (bh_mass_withdisk - (mass + mass_markedswallow)) * wk / rho;
                            else
                                p = 0;
                            
/* DAA: for stochastic winds (BH_BAL_KICK) we remove a fraction of mass from gas particles prior to kicking
 * --> need to increase the probability here to balance black hole growth   
 */
#ifdef BH_BAL_KICK
                            if(f_accreted>0) 
                            {
                                p /= f_accreted;

                                /* DAA: compute outflow probability when "bh_mass_withdisk < mass"
                                    - we don't need to enforce mass conservation in this case 
                                    - relevant only in low-res sims where the BH seed mass is much lower than the gas particle mass 
                                    - TODO: replace mdot below by mdot_alphadisk if BH_ALPHADISK_ACCRETION? */
                                if((bh_mass_withdisk - mass) < 0)
                                    p = ( (1-f_accreted)/f_accreted ) * mdot * dt * wk / rho;
                            }
#endif

                            
                            w = get_random_number(P[j].ID);
                            if(w < p)
                            {
#ifndef IO_REDUCED_MODE
                                printf("MARKING_BH_FOOD: j %d w %g p %g TO_BE_SWALLOWED \n",j,w,p);
#endif
                                if(P[j].SwallowID < id)
                                {
                                   P[j].SwallowID = id;
#ifdef BH_BAL_KICK
                                   mass_markedswallow += P[j].Mass*f_accreted;
#else
                                   mass_markedswallow += P[j].Mass;
#endif
                                }
                            } // if(w < p)
#endif // BH_SWALLOWGAS

                            
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
                            /* calculate the angle-weighting for the photon momentum */
                            if((mdot>0)&&(dt>0)&&(r>0)&&(P[j].SwallowID==0))
                            {
                                /* cos_theta with respect to disk of BH is given by dot product of r and Jgas */
                                norm=0; for(k=0;k<3;k++) norm+=(dpos[k]/r)*Jgas_in_Kernel[k];
                                norm=fabs(norm); theta=acos(norm);
                                BH_angle_weighted_kernel_sum += bh_angleweight_localcoupling(j,BH_disk_hr,theta);
                            }
#endif
                            
#ifdef BH_THERMALFEEDBACK
#ifdef UNIFIED_FEEDBACK
                            meddington = bh_eddington_mdot(bh_mass);
                            if(mdot > All.RadioThreshold * meddington)
#endif
                            {
                                energy = bh_lum_bol(mdot, bh_mass, -1) * dt;
                                if(rho > 0)
                                    SphP[j].Injected_BH_Energy += (wk/rho) * energy * P[j].Mass;
                            }
#endif
                            
                        } // if(P[j].Type == 0)
                        
                        
                        
                        
                        
                        
                        
                        
                        
                    } // if(r2 < h_i2)
                } // if(P[j].Mass > 0)
            } // for(n = 0; n < numngb; n++)
        } // while(startnode >= 0)
        
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = BlackholeDataGet[target].NodeList[listindex];
                if(startnode >= 0)
                    startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        } // mode==1
    } // while(startnode >= 0) (outer of the double-loop)
    
    
    
    /* Now collect the result at the right place */
    if(mode == 0)
    {
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
        BlackholeTempInfo[P[target].IndexMapToTempStruc].BH_angle_weighted_kernel_sum += BH_angle_weighted_kernel_sum;  /* need to correct target index */
#endif
#ifdef BH_REPOSITION_ON_POTMIN
        BPP(target).BH_MinPot = minpot;
        for(k = 0; k < 3; k++)
            BPP(target).BH_MinPotPos[k] = minpotpos[k];
#endif
    }
    else
    {
#if defined(BH_PHOTONMOMENTUM) || defined(BH_BAL_WINDS)
        BlackholeDataResult[target].BH_angle_weighted_kernel_sum = BH_angle_weighted_kernel_sum;
#endif
#ifdef BH_REPOSITION_ON_POTMIN
        BlackholeDataResult[target].BH_MinPot = minpot;
        for(k = 0; k < 3; k++)
            BlackholeDataResult[target].BH_MinPotPos[k] = minpotpos[k];
#endif
    }
    return 0;
} /* closes bh_evaluate routine */
