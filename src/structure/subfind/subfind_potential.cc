#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
#include "../../mesh/kernel.h"
/*!
 * This file was originally part of the GADGET3 code developed by Volker Springel.
 * It has been updated significantly by PFH for basic compatibility with GIZMO,
 * as well as code cleanups, and accommodating new GIZMO functionality for various
 * other operations. See notes in subfind.c and GIZMO User Guide for details.
 */

#ifdef SUBFIND

#include "../fof.h"
#include "subfind.h"

void subfind_potential_compute(int num, struct unbind_data *d, int phase, double weakly_bound_limit)
{
    int i;
    /* Each rank computes its local particles' potentials directly from the tree.
     * Foreign-rank mass is covered by the locally-installed LET nodes (LET is
     * mandatory; force_treebuild hard-stops on an incomplete LET), so no export/
     * import round-trip is needed. subfind_potential_compute is reached symmetrically
     * (only from subfind_col_unbind), so the bad-stop poll below is collective-safe. */
    for(i = 0; i < num; i++)
    {
        if(phase == 1) {if(P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit) {continue;}}
        if(subfind_force_treeevaluate_potential(d[i].index) != 0) {break;}    /* LET-incomplete guard tripped */
    }
    /* Drain a guard-tripped bad stop on every rank BEFORE any potential-derived
     * collective logic (the caller's Allgather over the per-rank minimum potential). */
    gizmo_exit_bad_stop_if_requested("subfind_potential_compute:retired_export_pseudo");

    for(i = 0; i < num; i++)
    {
        if(phase == 1) {if(P[d[i].index].v.DM_BindingEnergy <= weakly_bound_limit) {continue;}}
        int p = d[i].index; double h_grav = ForceSoftening_KernelRadius(p);
        P[p].u.DM_Potential -= P[p].Mass / h_grav * kernel_gravity(0,1,1,-1); // subtract self-contribution here
        P[p].u.DM_Potential *= All.G / All.cf_atime;
        if(All.TotN_gas > 0 && (FOF_SECONDARY_LINK_TYPES & 1) == 0 && (FOF_PRIMARY_LINK_TYPES & 1) == 0 && All.OmegaBaryon > 0) {P[p].u.DM_Potential *= All.OmegaMatter / (All.OmegaMatter - All.OmegaBaryon);}
    }
}

#endif

