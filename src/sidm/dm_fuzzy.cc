#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <map>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/ags_functions.h" /* do_dm_fuzzy_drift_kick_P */
#include "../mesh/kernel.h"
#include "../mesh/ghost_symlist_lifecycle.h"

/*! \file dm_fuzzy.c
 *  \brief routines needed for fuzzy-DM implementation
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef DM_FUZZY



/* do_dm_fuzzy_flux_computation / _old moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION so both the CPU tree-walk and the AGSForce
   GPU kernel can share a single body. */




/* kicks for fuzzy-dm integration: just put relevant drift-kick operators here to keep the code clean
 mode=0 -> 'kick', mode=1 -> 'drift' */
void do_dm_fuzzy_drift_kick(int i, double dt, int mode) { do_dm_fuzzy_drift_kick_P(i, dt, mode, P); }


/* initialize wavefunction values in the code ICs */
void do_dm_fuzzy_initialization(void)
{
#if (DM_FUZZY > 0)
    int i;
    for(i = 0; i < NumPart; i++)
    {
        double volume = P[i].AGS_Density / P[i].Mass, psimag = sqrt(P[i].AGS_Density), phase = 0;
        /* approximation for initial phase below is fine for slowly-varying k, otherwise not ideal */
        Vec3<double> pos_d = {(double)P[i].Pos[0], (double)P[i].Pos[1], (double)P[i].Pos[2]};
        Vec3<double> vel_d = {(double)P[i].Vel[0], (double)P[i].Vel[1], (double)P[i].Vel[2]};
        phase = dot(pos_d, vel_d) / All.ScalarField_hbar_over_mass;

        P[i].AGS_Psi_Re = psimag * volume * cos(phase); /* remember, we evolve the volume-integrated value of psi */
        P[i].AGS_Psi_Im = psimag * volume * sin(phase);

        P[i].AGS_Dt_Psi_Mass = 0; P[i].AGS_Dt_Psi_Re = 0; P[i].AGS_Dt_Psi_Im = 0;
        P[i].AGS_Psi_Re_Pred = P[i].AGS_Psi_Re; P[i].AGS_Psi_Im_Pred = P[i].AGS_Psi_Im;
    }
#endif
}



/* dm_fuzzy_reconstruct_and_slopelimit{,_sub} moved to sidm/dm_fuzzy_functions.h
   as KOKKOS_INLINE_FUNCTION. */

#endif // DM_FUZZY
