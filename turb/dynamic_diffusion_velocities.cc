#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/*! \file dynamic_diffusion_velocities.c
 *  \brief need filtered velocity information to calculate filtered gradients
 *
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
#define NV_MYSIGN(x) (( x > 0 ) - ( x < 0 ))

struct kernel_DiffFilter {
    Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i;
};


/* GPU dispatcher (dynamic_diff_vel_calc below) calls difffilter_evaluate_gpu directly.
   Legacy CPU-tree scaffolding (code_block_xchange_initialize.h, INPUT_STRUCT_NAME,
   OUTPUT_STRUCT_NAME, particle2in_DiffFilter, out2particle_DiffFilter, DiffFilter_evaluate)
   was retired in Step 5 Phase D1. */

/* operations that need to be performed before entering the main loop */
void dynamic_diff_vel_calc_initial_operations_preloop(void);
void dynamic_diff_vel_calc_initial_operations_preloop(void)
{
    /* Because of the smoothing operation, need to set bar quantity to current fluid value first */
    for (int i : ActiveParticleList) {
        if (P[i].Type == 0) {
            CellP[i].Norm_hat = 0;
            CellP[i].h_turb = P[i].Get_Particle_Size(); // All.cf_atime unnecessary, will multiply later
            CellP[i].FilterWidth_bar = 0;
            CellP[i].MaxDistance_for_grad = 0;
            CellP[i].Velocity_bar = CellP[i].VelPred / All.TurbDynamicDiffSmoothing;
        }
    }
}


/**
 * primary routine being called for this calculation
 */
extern void difffilter_evaluate_gpu(struct particle_data *, struct gas_cell_data *,
                                    int, int *, int, void *);

void dynamic_diff_vel_calc(void) {
    CPU_Step[CPU_MISC] += measure_time(); double t00_truestart = my_second();
    PRINT_STATUS("Start velocity smoothing computation...");
    dynamic_diff_vel_calc_initial_operations_preloop(); /* any initial operations */

    /* GPU/neighbor-list path: build wider CSR list and run DiffFilter kernel */
    {
        /* Build active gas particle index */
        int num_active = 0;
        for(int ii : ActiveParticleList) { if(P[ii].Type == 0 && P[ii].TimeBin >= 0 && P[ii].Mass > 0) num_active++; }
        int *active_indices = (int *) mymalloc("difffilter_active", (num_active > 0 ? num_active : 1) * sizeof(int));
        int aa = 0;
        for(int ii : ActiveParticleList) { if(P[ii].Type == 0 && P[ii].TimeBin >= 0 && P[ii].Mass > 0) active_indices[aa++] = ii; }

        /* Output struct must match DiffFilter_out in difffilter_gpu.cc */
        struct { double Norm_hat; double Velocity_bar[3]; double FilterWidth_bar; double MaxDistance_for_grad; } *df_out;
        df_out = (decltype(df_out)) mymalloc("difffilter_out", (num_active > 0 ? num_active : 1) * sizeof(*df_out));

        difffilter_evaluate_gpu(P, CellP, NumPart, active_indices, num_active, (void *)df_out);

        /* Scatter results back to CellP (mode=0: direct assignment += for accumulation fields) */
        for(aa = 0; aa < num_active; aa++) {
            int ii = active_indices[aa];
            for(int k = 0; k < 3; k++) CellP[ii].Velocity_bar[k] += df_out[aa].Velocity_bar[k];
            if(df_out[aa].FilterWidth_bar > CellP[ii].FilterWidth_bar) CellP[ii].FilterWidth_bar = df_out[aa].FilterWidth_bar;
            if(df_out[aa].MaxDistance_for_grad > CellP[ii].MaxDistance_for_grad) CellP[ii].MaxDistance_for_grad = df_out[aa].MaxDistance_for_grad;
            CellP[ii].Norm_hat += df_out[aa].Norm_hat;
        }

        myfree(df_out);
        myfree(active_indices);
    }

    PRINT_STATUS(" ..velocity smoothing done.");
    double t1; t1 = WallclockTime = my_second();
    double timeall_local = timediff(t00_truestart, t1);
    CPU_Step[CPU_IMPROVDIFFCOMPUTE] += timeall_local;
}


#endif /* End TURB_DIFF_DYNAMIC */
