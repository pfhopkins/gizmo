#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"



/*! \file dynamic_diffusion.c
 *  \brief calculate the dynamic smagorinsky coefficient for each gas particle
 *
 *  Following Piomelli et al 1994 we calculate the correlation between velocities
 *  at higher and lower resolutions in order to constrain the value of the Smag.
 *  coefficient, hence calculating it dynamically based on the local fluid 
 *  properties at the time.
 *
 *  For the velocity smoothing, the Monaghan 2011 SPH turbulence paper
 *  was used, specifically equation 2.17 in that paper.
 */
/*
 * This file was rewritten by Doug Rennehan (douglas.rennehan@gmail.com) for GIZMO, and was
 * copied with modifications from gradients.c, which was written by Phil Hopkins 
 * (phopkins@caltech.edu) for GIZMO.
 */

#ifdef TURB_DIFF_DYNAMIC

#define ASSIGN_ADD_PRESET(x,y,mode) (x+=y)
#define MINMAX_CHECK(x,xmin,xmax) ((x<xmin)?(xmin=x):((x>xmax)?(xmax=x):(1)))
#define SHOULD_I_USE_SPH_GRADIENTS(condition_number) ((condition_number > CONDITION_NUMBER_DANGER) ? (1):(0))
#define MAX_ADD(x,y,mode) ((y > x) ? (x = y) : (1)) // simpler definition now used
#define MIN_ADD(x,y,mode) ((y < x) ? (x = y) : (1))
#define NV_MYSIGN(x) (( x > 0 ) - ( x < 0 ))


struct Quantities_for_Smooth_Gradients {
    double Velocity_hat[3];
};

struct kernel_DynamicDiff {
    double dp[3], r, wk_i, wk_j, dwk_i, dwk_j, h_i;
};

struct DynamicDiffdata_in {
    Vec3<MyDouble> Pos;
    MyFloat Mass;
    MyFloat KernelRadius;
    MyDouble Density;
    int NodeList[NODELISTLENGTH];

    MyDouble VelShear_bar[3][3];
    MyDouble TD_DynDiffCoeff;
    MyDouble MagShear_bar;
    Vec3<MyDouble> Velocity_bar;
    Vec3<MyDouble> Velocity_hat;
    MyDouble Norm_hat;
    MyDouble Dynamic_numerator;
    MyDouble Dynamic_denominator;
#ifdef GALSF_SUBGRID_WINDS
    MyFloat DelayTime;
#endif
    MyFloat FilterWidth_bar;
}
*DynamicDiffDataIn, *DynamicDiffDataGet;

struct DynamicDiffdata_out {
    struct Quantities_for_Smooth_Gradients Gradients[3];
    struct Quantities_for_Smooth_Gradients Maxima;
    struct Quantities_for_Smooth_Gradients Minima;
    MyFloat FilterWidth_hat;
    MyDouble Dynamic_numerator_hat;
    MyDouble Dynamic_denominator_hat;
    MyDouble ProductVelocity_hat[3][3];
}
*DynamicDiffDataResult, *DynamicDiffDataOut;

struct DynamicDiffdata_out_iter {
    MyDouble dynamic_fac[3][3];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    MyDouble dynamic_fac_const[3][3];
#endif
}
*DynamicDiffDataResult_iter, *DynamicDiffDataOut_iter;

/* These functions will handle setting the calculated information */
static inline void particle2in_DynamicDiff(struct DynamicDiffdata_in *in, int i, int dynamic_iteration);
static inline void out2particle_DynamicDiff(struct DynamicDiffdata_out *out, int i, int mode, int dynamic_iteration);
static inline void out2particle_DynamicDiff_iter(struct DynamicDiffdata_out_iter *out, int i, int mode, int dynamic_iteration);

static struct temporary_data_dyndiff {
    struct Quantities_for_Smooth_Gradients Maxima;
    struct Quantities_for_Smooth_Gradients Minima;
    MyFloat FilterWidth_hat;
    MyDouble Dynamic_numerator_hat;
    MyDouble Dynamic_denominator_hat;
    MyDouble GradVelocity_hat[3][3];
    MyDouble dynamic_fac[3][3];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    MyDouble dynamic_fac_const[3][3];
#endif
    MyDouble ProductVelocity_hat[3][3];
}
*DynamicDiffDataPasser;

static inline void particle2in_DynamicDiff(struct DynamicDiffdata_in *in, int i, int dynamic_iteration) {
    int k, v;
    in->Pos = P[i].Pos;
    in->Velocity_bar = CellP[i].Velocity_bar;
    in->Velocity_hat = CellP[i].Velocity_hat;
    for (k = 0; k < 3; k++) {
        for (v = 0; v < 3; v++) {
            in->VelShear_bar[k][v] = CellP[i].VelShear_bar[k][v];
        }
    }

    in->TD_DynDiffCoeff = CellP[i].TD_DynDiffCoeff;
    in->Density = CellP[i].Density;
    in->MagShear_bar = CellP[i].MagShear_bar;
    in->KernelRadius = P[i].KernelRadius;
    in->Mass = P[i].Mass;
    in->Norm_hat = CellP[i].Norm_hat;
    in->FilterWidth_bar = CellP[i].FilterWidth_bar;
    in->Dynamic_numerator = CellP[i].Dynamic_numerator;
    in->Dynamic_denominator = CellP[i].Dynamic_denominator;

#ifdef GALSF_SUBGRID_WINDS
    in->DelayTime = CellP[i].DelayTime;
#endif

    if (SHOULD_I_USE_SPH_GRADIENTS(CellP[i].ConditionNumber)) {in->Mass *= -1;}
}




static inline void out2particle_DynamicDiff_iter(struct DynamicDiffdata_out_iter *out, int i, int mode, int dynamic_iteration) {
    int j, k;
    for (j = 0; j < 3; j++) {
        for (k = 0; k < 3; k++) {
            ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].dynamic_fac[j][k], out->dynamic_fac[j][k], mode);
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
            ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].dynamic_fac_const[j][k], out->dynamic_fac_const[j][k], mode);
#endif
        } 
    }
}

static inline void out2particle_DynamicDiff(struct DynamicDiffdata_out *out, int i, int mode, int dynamic_iteration) {
    if (dynamic_iteration == 0) {
        int j, k;
        MAX_ADD(DynamicDiffDataPasser[i].FilterWidth_hat, out->FilterWidth_hat, mode);
        ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].Dynamic_numerator_hat, out->Dynamic_numerator_hat, mode);
        ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].Dynamic_denominator_hat, out->Dynamic_denominator_hat, mode);

        for (j = 0; j < 3; j++) {
            MAX_ADD(DynamicDiffDataPasser[i].Maxima.Velocity_hat[j], out->Maxima.Velocity_hat[j], mode);
            MIN_ADD(DynamicDiffDataPasser[i].Minima.Velocity_hat[j], out->Minima.Velocity_hat[j], mode);

            for (k = 0; k < 3; k++) {
                ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].ProductVelocity_hat[j][k], out->ProductVelocity_hat[j][k], mode);
                ASSIGN_ADD_PRESET(DynamicDiffDataPasser[i].GradVelocity_hat[j][k], out->Gradients[k].Velocity_hat[j], mode);
            }
        }
    }  // dynamic_iteration == 0
}



/**
 *  Iterates over particles and calculates the large filtered quantities. Will do this
 *  (All.TurbDynamicDiffIterations + 2) times. Starts off by setting the filtered quantities
 *  to their current bar values times (1 / All.TurbDynamicDiffSmoothing), following Monghan 2011
 *  as later the factor of epsilon (All.TurbDynamicDiffSmoothing) will be multiplied through.
 *
 *  Finally in the subsequent iterations, the dynamic Piomelli 1994 value of Cs is calculated
 *  using the filtered values. ITERATING MORE THAN ONCE HAS NOT BEEN TESTED.
 *      - D. Rennehan
 */
void dynamic_diff_calc(void) {
    PRINT_STATUS("Start dynamic diffusion calculations...");
    CPU_Step[CPU_MISC] += measure_time();
    int i, j, k, v, u, ngrp, ndone, ndone_flag, dynamic_iteration;
    double shear_factor, dynamic_denominator, trace = 0, trace_dynamic_fac = 0, hhat2 = 0, leonardTensor[3][3], prefactor = 0;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    double trace_dynamic_fac_const = 0;
#endif
    double smoothInv = 1.0 / All.TurbDynamicDiffSmoothing;
    int recvTask, place;
    double timeall = 0, timecomp1 = 0, timecomp2 = 0, timecommsumm1 = 0, timecommsumm2 = 0, timewait1 = 0, timewait2 = 0, timewait3 = 0;
    double timecomp, timecomm, timewait, tstart, tend, t0, t1;
    int save_NextParticle;
    long long n_exported = 0;

    /* allocate buffers to arrange communication */
    long long NTaskTimesNumPart;
    DynamicDiffDataPasser = (struct temporary_data_dyndiff *) mymalloc("DynamicDiffDataPasser", N_gas * sizeof(struct temporary_data_dyndiff));
    NTaskTimesNumPart = maxThreads * NumPart;
    All.BunchSize = (long) ((All.BufferSize * 1024 * 1024) / (sizeof(struct data_index) + sizeof(struct data_nodelist) +
                                                             sizeof(struct DynamicDiffdata_in) +
                                                             sizeof(struct DynamicDiffdata_out) +
                                                             sizemax(sizeof(struct DynamicDiffdata_in),
                                                                     sizeof(struct DynamicDiffdata_out))));
    CPU_Step[CPU_DYNDIFFMISC] += measure_time();
    t0 = my_second();
    
    Ngblist.resize(NTaskTimesNumPart);
    DataIndexTable = (struct data_index *) mymalloc("DataIndexTable", All.BunchSize * sizeof(struct data_index));
    DataNodeList = (struct data_nodelist *) mymalloc("DataNodeList", All.BunchSize * sizeof(struct data_nodelist));
    PRINT_STATUS(" ..begin initializing smoothed quantities.");

    /* Because of smoothing operation, we don't zero these out, they get set to their current value */
    for (int i : ActiveParticleList) {
        if (P[i].Type == 0) {
            memset(&DynamicDiffDataPasser[i], 0, sizeof(struct temporary_data_dyndiff));

            /* A little optimization to save calculating this 9 times per active particle */
            prefactor = CellP[i].TD_DynDiffCoeff * CellP[i].FilterWidth_bar * CellP[i].FilterWidth_bar * CellP[i].MagShear_bar * smoothInv;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
            double prefactor_error = CellP[i].FilterWidth_bar * CellP[i].FilterWidth_bar * CellP[i].MagShear_bar * smoothInv;
#endif

            DynamicDiffDataPasser[i].Dynamic_numerator_hat = CellP[i].Dynamic_numerator * smoothInv;
            DynamicDiffDataPasser[i].Dynamic_denominator_hat = CellP[i].Dynamic_denominator * smoothInv;

            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++) {
                    DynamicDiffDataPasser[i].dynamic_fac[j][k] = prefactor * CellP[i].VelShear_bar[j][k];
                    DynamicDiffDataPasser[i].ProductVelocity_hat[j][k] = CellP[i].Velocity_bar[j] * CellP[i].Velocity_bar[k] * smoothInv;

#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                    DynamicDiffDataPasser[i].dynamic_fac_const[j][k] = prefactor_error * CellP[i].VelShear_bar[j][k];
#endif
                }
            }
        }
    }
    PRINT_STATUS(" ..entering iteration loop for the first time. # of iterations = %d", (All.TurbDynamicDiffIterations + 2));
    
    /* prepare to do the requisite number of sweeps over the particle distribution */
    for (dynamic_iteration = 0; dynamic_iteration < (All.TurbDynamicDiffIterations + 1); dynamic_iteration++) {      
        // now we actually begin the main gradient loop //
        NextParticle = 0;	/* begin with this index */
        PRINT_STATUS(" ..first loop over active particles (iter = %d)", dynamic_iteration);

        memset(ProcessedFlag, 0, All.MaxPart * sizeof(unsigned char));
        BufferCollisionFlag = 0; /* set to zero before operations begin */
        do {
            BufferFullFlag = 0;
            Nexport = 0;
            save_NextParticle = NextParticle;
            
            for (j = 0; j < NTask; j++) {
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
                DynamicDiff_evaluate_primary(&mainthreadid, dynamic_iteration);	/* do local particles and prepare export list */
            }
            
            tend = my_second();
            timecomp1 += timediff(tstart, tend);
            
            if (BufferFullFlag) {

                int last_nextparticle = NextParticle;
                int processed_particles = 0;
                int first_unprocessedparticle = -1;
                NextParticle = save_NextParticle; /* figure out where we are */
                while(NextParticle < (int)ActiveParticleList.size())
                {
                    if(NextParticle == last_nextparticle) {break;}
                    int pindex = ActiveParticleList[NextParticle];
#ifndef _OPENMP
                    if(ProcessedFlag[pindex] != 1) {break;}
#else
                    if(ProcessedFlag[pindex] == 0 && first_unprocessedparticle < 0) {first_unprocessedparticle = NextParticle;}
                    if(ProcessedFlag[pindex] == 1)
#endif
                    {
                        processed_particles++;
                        ProcessedFlag[pindex] = 2;
                    }
                    NextParticle++;
                }
#ifdef _OPENMP
                if(first_unprocessedparticle >= 0) {NextParticle = first_unprocessedparticle;} /* reset the neighbor list properly for the next group since we can get 'jumps' with openmp active */
                if(processed_particles == 0 && NextParticle == save_NextParticle && NextParticle < (int)ActiveParticleList.size()) {
                    BufferCollisionFlag++; if(BufferCollisionFlag < 2) {continue;}} /* we overflowed without processing a single particle, but this could be because of a collision, try once with the serialized approach, but if it fails then, we're truly stuck */
                else if(processed_particles && BufferCollisionFlag) {BufferCollisionFlag = 0;} /* we had a problem in a previous iteration but things worked, reset to normal operations */
#endif
                if(processed_particles <= 0 && NextParticle == save_NextParticle) {
                    /* in this case, the buffer is too small to process even a single particle */
                    endrun(113308);
                }
                
                int new_export = 0;
                
                for (j = 0, k = 0; j < Nexport; j++) {
                    if (ProcessedFlag[DataIndexTable[j].Index] != 2) {
                        if (k < j + 1) k = j + 1;
                        
                        for (; k < Nexport; k++) {
                            if (ProcessedFlag[DataIndexTable[k].Index] == 2) {
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
                    }
                    else {
                        new_export++;
                    }                
                }

                Nexport = new_export;       
            }
            
            n_exported += Nexport;
            
            for (j = 0; j < NTask; j++) Send_count[j] = 0;
            for (j = 0; j < Nexport; j++) Send_count[DataIndexTable[j].Task]++;
            
            mysort_dataindex(DataIndexTable, Nexport, sizeof(struct data_index), data_index_compare);
            
            tstart = my_second();
            
            MPI_Alltoall(Send_count, 1, MPI_INT, Recv_count, 1, MPI_INT, MPI_COMM_WORLD);
            
            tend = my_second();
            timewait1 += timediff(tstart, tend);
            
            for (j = 0, Nimport = 0, Recv_offset[0] = 0, Send_offset[0] = 0; j < NTask; j++) {
                Nimport += Recv_count[j];
                
                if (j > 0) {
                    Send_offset[j] = Send_offset[j - 1] + Send_count[j - 1];
                    Recv_offset[j] = Recv_offset[j - 1] + Recv_count[j - 1];
                }
            }
            
            DynamicDiffDataGet = (struct DynamicDiffdata_in *) mymalloc("DynamicDiffDataGet", Nimport * sizeof(struct DynamicDiffdata_in));
            DynamicDiffDataIn = (struct DynamicDiffdata_in *) mymalloc("DynamicDiffDataIn", Nexport * sizeof(struct DynamicDiffdata_in));
            
            /* prepare particle data for export */
            
            for (j = 0; j < Nexport; j++) {
                place = DataIndexTable[j].Index;
                particle2in_DynamicDiff(&DynamicDiffDataIn[j], place, dynamic_iteration);
                memcpy(DynamicDiffDataIn[j].NodeList, DataNodeList[DataIndexTable[j].IndexGet].NodeList, NODELISTLENGTH * sizeof(int));
            }
            
            /* exchange particle data */
            tstart = my_second();
            for (ngrp = 1; ngrp < (1 << PTask); ngrp++) {
                recvTask = ThisTask ^ ngrp;
                
                if (recvTask < NTask) {
                    if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0) {
                        MPI_Sendrecv(&DynamicDiffDataIn[Send_offset[recvTask]],
                                    Send_count[recvTask] * sizeof(struct DynamicDiffdata_in), MPI_BYTE,
                                    recvTask, TAG_DYNSMAGLOOP_A,
                                    &DynamicDiffDataGet[Recv_offset[recvTask]],
                                    Recv_count[recvTask] * sizeof(struct DynamicDiffdata_in), MPI_BYTE,
                                    recvTask, TAG_DYNSMAGLOOP_A, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }

            tend = my_second();
            timecommsumm1 += timediff(tstart, tend);
            
            myfree(DynamicDiffDataIn);

            if (dynamic_iteration == 0) {
                DynamicDiffDataResult = (struct DynamicDiffdata_out *) mymalloc("DynamicDiffDataResult", Nimport * sizeof(struct DynamicDiffdata_out));
                DynamicDiffDataOut = (struct DynamicDiffdata_out *) mymalloc("DynamicDiffDataOut", Nexport * sizeof(struct DynamicDiffdata_out));
            }
 
            DynamicDiffDataResult_iter = (struct DynamicDiffdata_out_iter *) mymalloc("DynamicDiffDataResult_iter", Nimport * sizeof(struct DynamicDiffdata_out_iter));
            DynamicDiffDataOut_iter = (struct DynamicDiffdata_out_iter *) mymalloc("DynamicDiffDataOut_iter", Nexport * sizeof(struct DynamicDiffdata_out_iter));
            
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
                DynamicDiff_evaluate_secondary(&mainthreadid, dynamic_iteration);
            }
            
            tend = my_second();
            timecomp2 += timediff(tstart, tend);
            
            if (NextParticle >= (int)ActiveParticleList.size()) {
                ndone_flag = 1;
            }
            else {
                ndone_flag = 0;
            }
            
            tstart = my_second();
            MPI_Allreduce(&ndone_flag, &ndone, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            tend = my_second();
            timewait2 += timediff(tstart, tend);
            
            /* get the result */
            tstart = my_second();
            for (ngrp = 1; ngrp < (1 << PTask); ngrp++) {
                recvTask = ThisTask ^ ngrp;
                if (recvTask < NTask) {
                    if (Send_count[recvTask] > 0 || Recv_count[recvTask] > 0) {
                        /* send the results */
                        if (dynamic_iteration == 0) {
                            MPI_Sendrecv(&DynamicDiffDataResult[Recv_offset[recvTask]],
                                        Recv_count[recvTask] * sizeof(struct DynamicDiffdata_out),
                                        MPI_BYTE, recvTask, TAG_DYNSMAGLOOP_B,
                                        &DynamicDiffDataOut[Send_offset[recvTask]],
                                        Send_count[recvTask] * sizeof(struct DynamicDiffdata_out),
                                        MPI_BYTE, recvTask, TAG_DYNSMAGLOOP_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } 
              
                        MPI_Sendrecv(&DynamicDiffDataResult_iter[Recv_offset[recvTask]],
                                        Recv_count[recvTask] * sizeof(struct DynamicDiffdata_out_iter),
                                        MPI_BYTE, recvTask, TAG_DYNSMAGLOOP_C,
                                        &DynamicDiffDataOut_iter[Send_offset[recvTask]],
                                        Send_count[recvTask] * sizeof(struct DynamicDiffdata_out_iter),
                                        MPI_BYTE, recvTask, TAG_DYNSMAGLOOP_C, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }

            tend = my_second();
            timecommsumm2 += timediff(tstart, tend);
            
            /* add the result to the local particles */
            tstart = my_second();
            for (j = 0; j < Nexport; j++) {
                place = DataIndexTable[j].Index;
                if (dynamic_iteration == 0) {
                    out2particle_DynamicDiff(&DynamicDiffDataOut[j], place, 1, dynamic_iteration);
                } 
        
                out2particle_DynamicDiff_iter(&DynamicDiffDataOut_iter[j], place, 1, dynamic_iteration);
            }

            tend = my_second();
            timecomp1 += timediff(tstart, tend);

            myfree(DynamicDiffDataOut_iter);
            myfree(DynamicDiffDataResult_iter);

            if (dynamic_iteration == 0) {
                myfree(DynamicDiffDataOut);
                myfree(DynamicDiffDataResult);
            }
      
            myfree(DynamicDiffDataGet);
        }
        while(ndone < NTask);
        PRINT_STATUS(" ..finished communication, beginning secondary calculations (iter = %d)", dynamic_iteration);

        /* The first two iterations were solely to calculate the hat quantities */ 
        { 
            /* Now that we have finished preliminaries, need to do the coefficient calculation */
            for (int i : ActiveParticleList) {
                if (P[i].Type == 0) {
#ifdef GALSF_SUBGRID_WINDS
                    if (CellP[i].DelayTime > 0) continue; /* Leave C_s alone for wind particles */
#endif
                    double VelShear_hat[3][3];

                    shear_factor = 0;
                    dynamic_denominator = 0;
                    CellP[i].Dynamic_numerator = 0;
                    CellP[i].Dynamic_denominator = 0;
                    trace = trace_dynamic_fac = 0;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                    CellP[i].TD_DynDiffCoeff_error = 0;
                    CellP[i].TD_DynDiffCoeff_error_default = 0;
                    trace_dynamic_fac_const = 0;
#endif
                    hhat2 = All.TurbDynamicDiffFac * All.TurbDynamicDiffFac * P[i].KernelRadius * P[i].KernelRadius;
          
                    /* We must construct grad(v_hat) before moving on */
                    if (dynamic_iteration == 0) {
                        double stol = 0.0;

                        double h_lim = DMAX(P[i].KernelRadius, CellP[i].MaxDistance_for_grad);
                        double a_limiter = 0.25;
                        if (CellP[i].ConditionNumber > 100) {
                            a_limiter = 2.0 * DMIN(0.5, 0.25 + 0.25 * (CellP[i].ConditionNumber - 100) / 100);
                        }
  
#if (SLOPE_LIMITER_TOLERANCE > 1)
                        h_lim = P[i].KernelRadius;
                        a_limiter *= 0.5;
                        stol = 0.125;
#endif
         
                        for (k = 0; k < 3; k++) {
                            construct_gradient(DynamicDiffDataPasser[i].GradVelocity_hat[k], i);
                            local_slopelimiter(DynamicDiffDataPasser[i].GradVelocity_hat[k], DynamicDiffDataPasser[i].Maxima.Velocity_hat[k], DynamicDiffDataPasser[i].Minima.Velocity_hat[k], a_limiter, h_lim, stol, 0,0,0);
                        }

                        /* Slope-limit the VelShear_hat tensor */
                        double shearfac_max = 0.5 * sqrt(CellP[i].Velocity_hat[0] * CellP[i].Velocity_hat[0] + CellP[i].Velocity_hat[1] * CellP[i].Velocity_hat[1]+CellP[i].Velocity_hat[2] * CellP[i].Velocity_hat[2]) / CellP[i].h_turb;

                        for (k = 0; k < 3; k++) {
                            for (v = 0; v < 3; v++) {
                                VelShear_hat[k][v] = 0.5 * (DynamicDiffDataPasser[i].GradVelocity_hat[k][v] + DynamicDiffDataPasser[i].GradVelocity_hat[v][k]);

                                if (VelShear_hat[k][v] < 0) {
                                    VelShear_hat[k][v] = DMAX(VelShear_hat[k][v], -shearfac_max);
                                }
                                else {
                                    VelShear_hat[k][v] = DMIN(VelShear_hat[k][v], shearfac_max);
                                }

                                if (k == v) {
                                    trace += VelShear_hat[k][k];
                                }
                            }
                        }

                        /* Don't zero the diagonal components if it was already trace-free */
                        if (trace != 0 && NUMDIMS > 1) {
                            for (k = 0; k < NUMDIMS; k++) {
                                VelShear_hat[k][k] -= 1.0 / NUMDIMS * trace;
                            }
                        }
                    }

                    trace = 0;

                    /* Calculates the denominator of equation for dynamic Smag. C */
                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            dynamic_denominator += VelShear_hat[k][v] * VelShear_hat[k][v];
                            leonardTensor[k][v] = All.TurbDynamicDiffSmoothing * DynamicDiffDataPasser[i].ProductVelocity_hat[k][v] - CellP[i].Velocity_hat[k] * CellP[i].Velocity_hat[v];

                            if (k == v) {
                                trace += leonardTensor[k][k];
                                trace_dynamic_fac += DynamicDiffDataPasser[i].dynamic_fac[k][k];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                                trace_dynamic_fac_const += DynamicDiffDataPasser[i].dynamic_fac_const[k][k];
#endif
                            }
                        }
                    }

                    shear_factor = sqrt(2.0 * dynamic_denominator);

                    /* Don't zero out the diagonal components if it was trace-free */
                    if (trace != 0 && NUMDIMS > 1) {
                        for (u = 0; u < NUMDIMS; u++) {
                            if (trace != 0) {
                                leonardTensor[u][u] -= (1.0 / NUMDIMS) * trace;
                            }

                            if (trace_dynamic_fac != 0) {
                                DynamicDiffDataPasser[i].dynamic_fac[u][u] -= (1.0 / NUMDIMS) * trace_dynamic_fac;
                            }

#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                            if (trace_dynamic_fac_const != 0) {
                                DynamicDiffDataPasser[i].dynamic_fac_const[u][u] -= (1.0 / NUMDIMS) * trace_dynamic_fac_const;
                            }
#endif
                        }
                    }

                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            CellP[i].Dynamic_numerator += (leonardTensor[k][v] - 2.0 * All.TurbDynamicDiffSmoothing * DynamicDiffDataPasser[i].dynamic_fac[k][v]) * VelShear_hat[k][v];
                        }
                    }

                    CellP[i].Dynamic_denominator = DynamicDiffDataPasser[i].FilterWidth_hat * DynamicDiffDataPasser[i].FilterWidth_hat * shear_factor * dynamic_denominator;

                    if (DynamicDiffDataPasser[i].Dynamic_denominator_hat != 0) {
                        /* There should be a factor of All.TurbDynamicDiffSmoothing in numerator and denominator, but it cancels out */
                        CellP[i].TD_DynDiffCoeff = -0.5 * DynamicDiffDataPasser[i].Dynamic_numerator_hat / DynamicDiffDataPasser[i].Dynamic_denominator_hat;
                    }
                    else {
                        CellP[i].TD_DynDiffCoeff = 0;
                    }

                    CellP[i].TD_DynDiffCoeff = DMIN(DMAX(0, CellP[i].TD_DynDiffCoeff), All.TurbDynamicDiffMax);

#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                    double error[3][3], trace_error = 0, defaultError[3][3], trace_defaultError = 0, leonardTensorMag = 0;

                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            error[k][v] = leonardTensor[k][v] - (-2.0 * CellP[i].TD_DynDiffCoeff * DynamicDiffDataPasser[i].FilterWidth_hat * DynamicDiffDataPasser[i].FilterWidth_hat * shear_factor * VelShear_hat[k][v] + 2.0 * All.TurbDynamicDiffSmoothing * DynamicDiffDataPasser[i].dynamic_fac[k][v]);
                            defaultError[k][v] = leonardTensor[k][v] - (-2.0 * 0.05 * DynamicDiffDataPasser[i].FilterWidth_hat * DynamicDiffDataPasser[i].FilterWidth_hat * shear_factor * VelShear_hat[k][v] + 2.0 * All.TurbDynamicDiffSmoothing * 0.05 * DynamicDiffDataPasser[i].dynamic_fac_const[k][v]);
                            leonardTensorMag += leonardTensor[k][v] * leonardTensor[k][v];

                            if (k == v) {
                                trace_error += error[k][k];
                                trace_defaultError += defaultError[k][k];
                            }
                        }
                    }

                    if (NUMDIMS > 1) {
                        for (u = 0; u < 3; u++) {
                            error[u][u] -= (1.0 / NUMDIMS) * trace_error;
                            defaultError[u][u] -= (1.0 / NUMDIMS) * trace_defaultError;
                        }
                    }

                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            if (k != v) {
                                CellP[i].TD_DynDiffCoeff_error += error[k][v] * error[k][v];
                                CellP[i].TD_DynDiffCoeff_error_default += defaultError[k][v] * defaultError[k][v];
                            }
                        }
                    }

                    CellP[i].TD_DynDiffCoeff_error = sqrt(CellP[i].TD_DynDiffCoeff_error / leonardTensorMag);
                    CellP[i].TD_DynDiffCoeff_error_default = sqrt(CellP[i].TD_DynDiffCoeff_error_default / leonardTensorMag);
#endif

                    /* Contains the actual eddy viscosity like estimate */
                    CellP[i].TD_DiffCoeff = All.TurbDiffusion_Coefficient * CellP[i].TD_DynDiffCoeff * (CellP[i].h_turb * CellP[i].h_turb * All.cf_atime * All.cf_atime) * (CellP[i].MagShear * All.cf_a2inv); // Physical
                    /* Have to update the other coefficients as well with the new value */
#ifdef TURB_DIFF_ENERGY
                    CellP[i].Kappa_Conduction = All.ConductionCoeff * CellP[i].TD_DiffCoeff * CellP[i].Density * All.cf_a3inv;
#endif
#ifdef TURB_DIFF_VELOCITY
                    CellP[i].Eta_ShearViscosity = All.ShearViscosityCoeff * CellP[i].TD_DiffCoeff * CellP[i].Density * All.cf_a3inv;
                    CellP[i].Zeta_BulkViscosity = All.BulkViscosityCoeff * CellP[i].TD_DiffCoeff * CellP[i].Density * All.cf_a3inv;
#endif

                    prefactor = CellP[i].TD_DynDiffCoeff * CellP[i].FilterWidth_bar * CellP[i].FilterWidth_bar * CellP[i].MagShear_bar * smoothInv;

                    /* Need to prepare this for the next iteration */
                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            /* smoothInv is in prefactor */
                            DynamicDiffDataPasser[i].dynamic_fac[k][v] = prefactor * CellP[i].VelShear_bar[k][v];
                        }
                    }
                } /* P[i].Type == 0 */
            } /* Active particle loop */
        } /* dynamic_iteration >= 0 */
        tstart = my_second();
        PRINT_STATUS(" ..waiting for tasks... (iter = %d)", dynamic_iteration);
        /* Must wait for ALL tasks for finish each iteration in order to converge */
        MPI_Barrier(MPI_COMM_WORLD);   

        tend = my_second();
        timewait3 += timediff(tstart, tend); 
    } // closes dynamic_iteration
    
    myfree(DataNodeList);
    myfree(DataIndexTable);
    
    myfree(DynamicDiffDataPasser);
 
    /* collect some timing information */
    t1 = WallclockTime = my_second();
    timeall = timediff(t0, t1);
    timecomp = timecomp1 + timecomp2;
    timewait = timewait1 + timewait2 + timewait3;
    timecomm = timecommsumm1 + timecommsumm2;
    
    CPU_Step[CPU_DYNDIFFCOMPUTE] += timecomp;
    CPU_Step[CPU_DYNDIFFWAIT] += timewait;
    CPU_Step[CPU_DYNDIFFCOMM] += timecomm;
    CPU_Step[CPU_DYNDIFFMISC] += timeall - (timecomp + timewait + timecomm);
    PRINT_STATUS(" ..dynamic diffusion calculations done.");
}


/**
 * Need to apply 2 filters, one at h and one at k*h, where k ~ 2, 
 * but can be changed in the parameterfile (TurbDynamicDiffFac).
 *  - D. Rennehan
 *
 */
/*!   -- this subroutine contains no writes to shared memory -- */
int DynamicDiff_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int dynamic_iteration) {
    int startnode, numngb, listindex = 0;
    int j, k, v, n;
    double hinv, hinv3, hinv4, r2, u;
    struct kernel_DynamicDiff kernel;
    struct DynamicDiffdata_in local;

    if (mode == 0) {
        particle2in_DynamicDiff(&local, target, dynamic_iteration);
    }
    else {
        local = DynamicDiffDataGet[target];
    }

    struct DynamicDiffdata_out out;
    struct DynamicDiffdata_out_iter out_iter;
    memset(&out, 0, sizeof(struct DynamicDiffdata_out));
    memset(&out_iter, 0, sizeof(struct DynamicDiffdata_out_iter));
    memset(&kernel, 0, sizeof(struct kernel_DynamicDiff));
    int sph_gradients_flag_i = 0;
    /* check if we should bother doing a neighbor loop */
    if (local.KernelRadius <= 0) return 0;
    if (local.Mass <= 0) return 0;
    if (local.Density <= 0) return 0;

    /* now set particle-i centric quantities so we don't do it inside the loop */
    kernel.h_i = All.TurbDynamicDiffFac * local.KernelRadius;
   
    double h2_i = kernel.h_i * kernel.h_i;

    kernel_hinv(kernel.h_i, &hinv, &hinv3, &hinv4);
    
    if (local.Mass < 0) {
        sph_gradients_flag_i = 1;
        local.Mass *= -1;
    }

    /* Just always return wk and dwk */
    int kernel_mode_i = 0;
    
    /* This is a bit of optimization to save calculating this 9 times for each neighbor */
    //double prefactor_i = local.Density_bar * local.KernelRadius * local.KernelRadius * local.TD_DynDiffCoeff * local.MagShear_bar;
    double prefactor_i = local.FilterWidth_bar * local.FilterWidth_bar * local.TD_DynDiffCoeff * local.MagShear_bar;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
    double prefactor_const_i = local.FilterWidth_bar * local.FilterWidth_bar * local.MagShear_bar;
#endif

    /* Now start the actual neighbor computation for this particle */ 
    if (mode == 0) {
        startnode = All.MaxPart;	/* root node */
    }
    else {
        startnode = DynamicDiffDataGet[target].NodeList[0];

        startnode = Nodes[startnode].u.d.nextnode;	/* open it */
    }
    
    while (startnode >= 0) {
        while (startnode >= 0) {
           
            numngb = ngb_treefind_pairs_threads(local.Pos, kernel.h_i, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb < 0) {return -2;}
            
            for (n = 0; n < numngb; n++) {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
#ifdef GALSF_SUBGRID_WINDS
                if (local.DelayTime == 0 && CellP[j].DelayTime > 0) continue;
                if (local.DelayTime > 0 && CellP[j].DelayTime == 0) continue;
#endif
                if (P[j].Mass <= 0) continue;
                if (CellP[j].Density <= 0) continue;
                
                kernel.dp[0] = local.Pos[0] - P[j].Pos[0];
                kernel.dp[1] = local.Pos[1] - P[j].Pos[1];
                kernel.dp[2] = local.Pos[2] - P[j].Pos[2];
                nearest_xyz(kernel.dp);
                r2 = kernel.dp[0] * kernel.dp[0] + kernel.dp[1] * kernel.dp[1] + kernel.dp[2] * kernel.dp[2];
                double h_j = All.TurbDynamicDiffFac * P[j].KernelRadius;
                double h_avg = 0.5 * (kernel.h_i + h_j);
                double mean_weight = 0.5 * (CellP[j].Norm_hat + local.Norm_hat) / (local.Norm_hat * CellP[j].Norm_hat);
                double V_j = P[j].Mass * mean_weight;
                if (r2 <= 0) {continue;}
                if ((r2 >= h2_i) && (r2 >= (h_j * h_j))) {continue;}
                kernel.r = sqrt(r2);
                if (kernel.r > out.FilterWidth_hat) {out.FilterWidth_hat = kernel.r;}

                /* In this case, W_ij = W_ji, so we only need wk_i and dwk_i */
                kernel_hinv(h_avg, &hinv, &hinv3, &hinv4);
                u = DMIN(kernel.r * hinv, 1.0);
                if(u<1) {kernel_main(u, hinv3, hinv4, &kernel.wk_i, &kernel.dwk_i, kernel_mode_i);} else {kernel.wk_i=kernel.dwk_i=0;}

                double weight_i = kernel.wk_i * V_j;
                if (dynamic_iteration == 0) {
                    /* Need to calculate the filtered velocity gradient for the filtered shear */
                    double dv_hat[3]; for (k=0;k<3;k++) {dv_hat[k] = CellP[j].Velocity_hat[k] - local.Velocity_hat[k];}
                    NGB_SHEARBOX_BOUNDARY_VELCORR_(local.Pos,P[j].Pos,dv_hat,-1); /* wrap velocities for shearing boxes if needed */
                    for (k=0;k<3;k++) {MINMAX_CHECK(dv_hat[k], out.Minima.Velocity_hat[k], out.Maxima.Velocity_hat[k]);}

                    double hinv_forgrad, hinv3_forgrad, hinv4_forgrad, u_forgrad, wk_i_forgrad, dwk_i_forgrad;
                    if (kernel.r < local.KernelRadius) {
                        kernel_hinv(local.KernelRadius, &hinv_forgrad, &hinv3_forgrad, &hinv4_forgrad);
                        u_forgrad = DMIN(kernel.r * hinv_forgrad, 1.0);
                        kernel_main(u_forgrad, hinv3_forgrad, hinv4_forgrad, &wk_i_forgrad, &dwk_i_forgrad, kernel_mode_i);

                        if(sph_gradients_flag_i) {wk_i_forgrad = -dwk_i_forgrad * P[j].Mass / kernel.r;}

                        for (k = 0; k < 3; k++) {
                            double grad_prefactor_i = -wk_i_forgrad * kernel.dp[k];
                            for (v = 0; v < 3; v++) {out.Gradients[k].Velocity_hat[v] += grad_prefactor_i * dv_hat[v];}
                        }
                    }
                } /* dynamic_iteration == 0 */

                /* This is a bit of optimization to save 9 times calculating these for each particle */
                double prefactor_j = CellP[j].FilterWidth_bar * CellP[j].FilterWidth_bar * CellP[j].TD_DynDiffCoeff * CellP[j].MagShear_bar;
                double dynamic_fac_diff[3][3], ProductVelocity_diff[3][3], Dynamic_numerator_diff, Dynamic_denominator_diff;

#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                double prefactor_const_j = CellP[j].FilterWidth_bar * CellP[j].FilterWidth_bar * CellP[j].MagShear_bar;
                double dynamic_fac_const_diff[3][3];
#endif

                /* Do particle i */
                if (kernel.r < h_avg) {
                    if (dynamic_iteration == 0) {
                        Dynamic_numerator_diff = CellP[j].Dynamic_numerator - local.Dynamic_numerator;
                        Dynamic_denominator_diff = CellP[j].Dynamic_denominator - local.Dynamic_denominator;

                        out.Dynamic_numerator_hat += Dynamic_numerator_diff * weight_i;
                        out.Dynamic_denominator_hat += Dynamic_denominator_diff * weight_i;
                    }

                    for (k = 0; k < 3; k++) {
                        for (v = 0; v < 3; v++) {
                            dynamic_fac_diff[k][v] = prefactor_j * CellP[j].VelShear_bar[k][v] - prefactor_i * local.VelShear_bar[k][v];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                            dynamic_fac_const_diff[k][v] = prefactor_const_j * CellP[j].VelShear_bar[k][v] - prefactor_const_i * local.VelShear_bar[k][v];
#endif

                            out_iter.dynamic_fac[k][v] += dynamic_fac_diff[k][v] * weight_i;
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                            out_iter.dynamic_fac_const[k][v] += dynamic_fac_const_diff[k][v] * weight_i;
#endif

                            if (dynamic_iteration == 0) {
                                ProductVelocity_diff[k][v] = CellP[j].Velocity_bar[k] * CellP[j].Velocity_bar[v] - local.Velocity_bar[k] * local.Velocity_bar[v];
                                out.ProductVelocity_hat[k][v] += ProductVelocity_diff[k][v] * weight_i;
                            }
                        }
                    }
                }
            } // numngb loop
        } // while(startnode)
        
        if (mode == 1) {
            listindex++;
            if (listindex < NODELISTLENGTH) {
                startnode = DynamicDiffDataGet[target].NodeList[listindex];
                if (startnode >= 0) startnode = Nodes[startnode].u.d.nextnode;	/* open it */
            }
        }
    }
    
    /* ------------------------------------------------------------------------------------------------ */
    /* Now collect the result at the right place */
    if (dynamic_iteration == 0) {
        if (mode == 0) {
            out2particle_DynamicDiff(&out, target, 0, dynamic_iteration);
        }
        else {
            DynamicDiffDataResult[target] = out;
        }
    }

    if (mode == 0) {
        out2particle_DynamicDiff_iter(&out_iter, target, 0, dynamic_iteration);
    }
    else {
        DynamicDiffDataResult_iter[target] = out_iter;
    }

    /* ------------------------------------------------------------------------------------------------ */
    
    return 0;
}


void *DynamicDiff_evaluate_primary(void *p, int dynamic_iteration) {
    int thread_id = *(int *) p;
    int i, j;
    int *exportflag, *exportnodecount, *exportindex, *ngblist;
    ngblist = Ngblist + thread_id * NumPart;
    exportflag = Exportflag + thread_id * NTask;
    exportnodecount = Exportnodecount + thread_id * NTask;
    exportindex = Exportindex + thread_id * NTask;
    /* Note: exportflag is local to each thread */
    for (j = 0; j < NTask; j++) exportflag[j] = -1;
#ifdef _OPENMP
    if(BufferCollisionFlag && thread_id) {return NULL;} /* force to serial for this subloop if threads simultaneously cross the Nexport bunchsize threshold */
#endif
    while (1) {
        int exitFlag = 0;
#ifdef _OPENMP
#pragma omp critical(_nexportdd_)
#endif
        {
            if (BufferFullFlag != 0 || NextParticle >= (int)ActiveParticleList.size()) {
                exitFlag = 1;
            }
            else {
                i = ActiveParticleList[NextParticle];
                NextParticle++;
            }
        }

        if(exitFlag) break;
        if(ProcessedFlag[i]) {continue;}

        if(P[i].Type == 0) {
	    if(DynamicDiff_evaluate(i, 0, exportflag, exportnodecount, exportindex, ngblist, dynamic_iteration) < 0) break;		/* export buffer has filled up */
        }

        ProcessedFlag[i] = 1; /* particle successfully finished */
    }

    return NULL;
}



void *DynamicDiff_evaluate_secondary(void *p, int dynamic_iteration) {
    int thread_id = *(int *) p;
    int j, dummy, *ngblist;
    ngblist = Ngblist + thread_id * NumPart;

    while (1) {
#ifdef _OPENMP
#pragma omp critical(_nextlistdd_)
#endif
        {
            j = NextJ;
            NextJ++;
        }
        
        if (j >= Nimport) break;

        DynamicDiff_evaluate(j, 1, &dummy, &dummy, &dummy, ngblist, dynamic_iteration);
    }

    return NULL;
}

#endif /* ends TURB_DIFF_DYNAMIC */
