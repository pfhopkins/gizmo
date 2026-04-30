#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
extern void dynamicdiff_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                     int, int *, int, int *, int *, int,
                                     void *, void *, void *, int);



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

#define SHOULD_I_USE_SPH_GRADIENTS(condition_number) ((condition_number > CONDITION_NUMBER_DANGER) ? (1):(0))


struct Quantities_for_Smooth_Gradients {
    double Velocity_hat[3];
};

/* Legacy CPU-tree scaffolding (struct kernel_DynamicDiff, DynamicDiffdata_in,
 * DynamicDiffdata_out, DynamicDiffdata_out_iter, particle2in/out2particle
 * forward decls + bodies, and the DynamicDiff_evaluate / _primary / _secondary
 * functions at the bottom of this file) was retired in Step 5 Phase D2.5-ext.
 * Modern path: dynamic_diff_calc() dispatches to dynamicdiff_evaluate_gpu()
 * with anonymous in/out structs gathered/scattered inline; no legacy ngb_treefind
 * tree walk is used. */

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
    DynamicDiffDataPasser = (struct temporary_data_dyndiff *) mymalloc("DynamicDiffDataPasser", N_gas * sizeof(struct temporary_data_dyndiff));
    CPU_Step[CPU_DYNDIFFMISC] += measure_time();
    t0 = my_second();
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
        PRINT_STATUS(" ..first loop over active particles (iter = %d)", dynamic_iteration);

        /* GPU/neighbor-list path: use cached symmetric CSR list with wider search radius.
           Follows the same pattern as gradient_evaluate_gpu multi-iteration. */
        {
            /* Build active index and gather per-particle input */
            int dd_num_active = 0;
            for(int ii : ActiveParticleList) { if(P[ii].Type == 0 && P[ii].Mass > 0) dd_num_active++; }
            int *dd_active = (int *) mymalloc("dd_active", (dd_num_active > 0 ? dd_num_active : 1) * sizeof(int));
            {int aa = 0; for(int ii : ActiveParticleList) { if(P[ii].Type == 0 && P[ii].Mass > 0) dd_active[aa++] = ii; }}

            /* Gather input struct per active particle */
            struct { double VelShear_bar[3][3]; double TD_DynDiffCoeff; double MagShear_bar;
                     double Velocity_bar[3]; double Velocity_hat[3]; double Norm_hat;
                     double Dynamic_numerator; double Dynamic_denominator; double FilterWidth_bar;
                     double KernelRadius; int sph_gradients_flag;
#ifdef GALSF_SUBGRID_WINDS
                     double DelayTime;
#endif
            } *dd_in;
            dd_in = (decltype(dd_in)) mymalloc("dd_in", (dd_num_active > 0 ? dd_num_active : 1) * sizeof(*dd_in));

            for(int aa = 0; aa < dd_num_active; aa++) {
                int ii = dd_active[aa];
                for(int kk=0; kk<3; kk++) for(int vv=0; vv<3; vv++) dd_in[aa].VelShear_bar[kk][vv] = CellP[ii].VelShear_bar[kk][vv];
                dd_in[aa].TD_DynDiffCoeff = CellP[ii].TD_DynDiffCoeff;
                dd_in[aa].MagShear_bar = CellP[ii].MagShear_bar;
                for(int kk=0; kk<3; kk++) { dd_in[aa].Velocity_bar[kk] = CellP[ii].Velocity_bar[kk]; dd_in[aa].Velocity_hat[kk] = CellP[ii].Velocity_hat[kk]; }
                dd_in[aa].Norm_hat = CellP[ii].Norm_hat;
                dd_in[aa].Dynamic_numerator = CellP[ii].Dynamic_numerator;
                dd_in[aa].Dynamic_denominator = CellP[ii].Dynamic_denominator;
                dd_in[aa].FilterWidth_bar = CellP[ii].FilterWidth_bar;
                dd_in[aa].KernelRadius = P[ii].KernelRadius;
                dd_in[aa].sph_gradients_flag = SHOULD_I_USE_SPH_GRADIENTS(CellP[ii].ConditionNumber);
#ifdef GALSF_SUBGRID_WINDS
                dd_in[aa].DelayTime = CellP[ii].DelayTime;
#endif
            }

            /* Allocate output structs */
            struct { double GradVelocity_hat[3][3]; double Maxima_Velocity_hat[3]; double Minima_Velocity_hat[3];
                     double FilterWidth_hat; double Dynamic_numerator_hat; double Dynamic_denominator_hat;
                     double ProductVelocity_hat[3][3]; } *dd_out0;
            dd_out0 = (decltype(dd_out0)) mymalloc("dd_out0", (dd_num_active > 0 ? dd_num_active : 1) * sizeof(*dd_out0));

            struct { double dynamic_fac[3][3];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                     double dynamic_fac_const[3][3];
#endif
            } *dd_out_iter;
            dd_out_iter = (decltype(dd_out_iter)) mymalloc("dd_out_iter", (dd_num_active > 0 ? dd_num_active : 1) * sizeof(*dd_out_iter));

            dynamicdiff_evaluate_gpu(P, CellP, NumPart, dd_active, dd_num_active,
                                     gizmo_sym_neighbor_list.offsets, gizmo_sym_neighbor_list.neighbors,
                                     gizmo_sym_neighbor_list.total_pairs,
                                     (void *)dd_in, (void *)dd_out0, (void *)dd_out_iter, dynamic_iteration);

            /* Scatter results */
            for(int aa = 0; aa < dd_num_active; aa++) {
                int ii = dd_active[aa];
                /* out_iter: always scatter (all iterations) */
                for(int kk=0; kk<3; kk++) for(int vv=0; vv<3; vv++) {
                    DynamicDiffDataPasser[ii].dynamic_fac[kk][vv] += dd_out_iter[aa].dynamic_fac[kk][vv];
#ifdef OUTPUT_TURB_DIFF_DYNAMIC_ERROR
                    DynamicDiffDataPasser[ii].dynamic_fac_const[kk][vv] += dd_out_iter[aa].dynamic_fac_const[kk][vv];
#endif
                }
                /* out0: only on iteration 0 */
                if(dynamic_iteration == 0) {
                    if(dd_out0[aa].FilterWidth_hat > DynamicDiffDataPasser[ii].FilterWidth_hat)
                        DynamicDiffDataPasser[ii].FilterWidth_hat = dd_out0[aa].FilterWidth_hat;
                    DynamicDiffDataPasser[ii].Dynamic_numerator_hat += dd_out0[aa].Dynamic_numerator_hat;
                    DynamicDiffDataPasser[ii].Dynamic_denominator_hat += dd_out0[aa].Dynamic_denominator_hat;
                    for(int kk=0; kk<3; kk++) {
                        if(dd_out0[aa].Maxima_Velocity_hat[kk] > DynamicDiffDataPasser[ii].Maxima.Velocity_hat[kk])
                            DynamicDiffDataPasser[ii].Maxima.Velocity_hat[kk] = dd_out0[aa].Maxima_Velocity_hat[kk];
                        if(dd_out0[aa].Minima_Velocity_hat[kk] < DynamicDiffDataPasser[ii].Minima.Velocity_hat[kk])
                            DynamicDiffDataPasser[ii].Minima.Velocity_hat[kk] = dd_out0[aa].Minima_Velocity_hat[kk];
                        for(int vv=0; vv<3; vv++) {
                            DynamicDiffDataPasser[ii].ProductVelocity_hat[kk][vv] += dd_out0[aa].ProductVelocity_hat[kk][vv];
                            DynamicDiffDataPasser[ii].GradVelocity_hat[kk][vv] += dd_out0[aa].GradVelocity_hat[kk][vv];
                        }
                    }
                }
            }

            myfree(dd_out_iter);
            myfree(dd_out0);
            myfree(dd_in);
            myfree(dd_active);
        }
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
                        double shearfac_max = 0.5 * sqrt(CellP[i].Velocity_hat.norm_sq()) / CellP[i].h_turb;

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
    
    myfree(DynamicDiffDataPasser);
 
    /* collect some timing information */
    t1 = WallclockTime = my_second();
    timeall = timediff(t0, t1);
    CPU_Step[CPU_DYNDIFFCOMPUTE] += timeall;
    PRINT_STATUS(" ..dynamic diffusion calculations done.");
}



#endif /* ends TURB_DIFF_DYNAMIC */
