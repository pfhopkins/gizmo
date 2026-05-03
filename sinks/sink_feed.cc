/*! \file sink_feed.c
*  \brief This is where particles are marked for gas accretion.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "sink_functions.h"
/*
* This file is largely written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
* see notes in sink.c for details on code history.
*/

#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //





void sink_feed_loop(void)
{
    {
#include "../sinks/sinks_gpu_decls.h"
        /* Build LOCAL active-source list from ActiveParticleList; iterating
           NumPart here would include ghost imports and double-deposit. */
        int num_active = 0;
        for(int i : ActiveParticleList) { if(sink_isactive(i)) num_active++; }
        /* Count-first guard: skip mymalloc + populate + evaluate_gpu (which
         * itself does ghost_prep) when no rank has any active sink. */
        int global_num_active = num_active;
        MPI_Allreduce(&num_active, &global_num_active, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(global_num_active == 0) {
            CPU_Step[CPU_SINKS] += measure_time();
            return;
        }
        int *nl_active = (int *) mymalloc("sinkfeed_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("sinkfeed_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int i : ActiveParticleList) {
            if(sink_isactive(i)) {
                nl_active[aa] = i; nl_radii[aa] = (double)P[i].KernelRadius; aa++;
            }
        }}
        sink_feed_evaluate_gpu(P, CellP, NumPart, nl_active, num_active, nl_radii);
        myfree(nl_radii); myfree(nl_active);
        CPU_Step[CPU_SINKS] += measure_time();
    }
}


#endif // top-level flag
