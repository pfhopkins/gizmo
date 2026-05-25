#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/*! \file cbe_integrator.cc
 *  \brief startup initialization for the CBE integrator basis moments.
 *         The per-step drift-kick (do_cbe_drift_kick_kernel) and the per-
 *         active post-gravity finalization (do_cbe_postgravity_kernel) now
 *         live in cbe_integrator_functions.h as KOKKOS_INLINE_FUNCTION
 *         kernels, dispatched via sidm/cbe_integrator_gpu.cc.
 *
 *         Per-pair CBE flux (cbe_integrator_flux_compute_pair) is invoked
 *         from the AGSForce iterative neighbor loop (gravity/ags_force_loop.h).
 *
 *         This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef CBE_INTEGRATOR


// moment ordering convention: 0, x, y, z, xx, yy, zz, xy, xz, yz
//                             0, 1, 2, 3,  4,  5,  6,  7,  8,  9


/* variable initialization (called once from core/init.cc) */
void do_cbe_initialization(void)
{
    int i,j,k;
    for(i=0;i<NumPart;i++)
    {
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments_dt[j][k]=0;}} // no time derivatives //
        double v2=0, v0=0;
        v2 = P[i].Vel.norm_sq();
        if(v2>0) {v0=sqrt(v2);} else {v0=1.e-10;}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++)
        {
            for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++)
            {
                // zeros will be problematic, instead initialize a random distribution //
if(j > 1)
{

                if(k==0) {P[i].CBE_basis_moments[j][0] = 1.e-5 * P[i].Mass * (0.5 + 0.01 + get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j));}
                if((k>0)&&(k<4)) {P[i].CBE_basis_moments[j][k] = P[i].CBE_basis_moments[j][0] * 1.e-8 * (0*2.*P[i].Vel[k]*(get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j + 2)-0.5) + 1.e-5*v0*(get_random_number(P[i].ID + i + 343*ThisTask + 912*k + 781*j + 2)-0.5));}
                if(k>=4 && k<7) {P[i].CBE_basis_moments[j][k] = P[i].CBE_basis_moments[j][0] * 1.e-15;} //(P[i].Vel[k]*P[i].Vel[k] + 1.e-2*v0*v0 + 1.e-3*1.e-3);}
                if(k>=7) {P[i].CBE_basis_moments[j][k] = 0;}

} else {

 P[i].CBE_basis_moments[0][0] = 0.5*P[i].Mass;
 P[i].CBE_basis_moments[1][0] = 0.5*P[i].Mass;
 P[i].CBE_basis_moments[0][1] =  1.0*P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[1][1] = -1.0*P[i].CBE_basis_moments[1][0];
 P[i].CBE_basis_moments[0][2] = P[i].CBE_basis_moments[0][3] = 0;
 P[i].CBE_basis_moments[1][2] = P[i].CBE_basis_moments[1][3] = 0;
#if (CBE_INTEGRATOR_NMOMENTS > 4)
 P[i].CBE_basis_moments[0][4] = P[i].CBE_basis_moments[1][4] = 0.5 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][5] = P[i].CBE_basis_moments[1][5] = 0.1 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][6] = P[i].CBE_basis_moments[1][6] = 0.1 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][7] = P[i].CBE_basis_moments[1][7] = 0.0 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][8] = P[i].CBE_basis_moments[1][8] = 0.0 * P[i].CBE_basis_moments[0][0];
 P[i].CBE_basis_moments[0][9] = P[i].CBE_basis_moments[1][9] = 0.0 * P[i].CBE_basis_moments[0][0];
#endif

}

#if (NUMDIMS==1)
                if((k!=0)&&(k!=1)&&(k!=4)) {P[i].CBE_basis_moments[j][k] = 0;}
#endif
#if (NUMDIMS==2)
                if((k==3)||(k==6)||(k==8)||(k==9)) {P[i].CBE_basis_moments[j][k] = 0;}
#endif
            }
        }
        double mom_tot[CBE_INTEGRATOR_NMOMENTS]={0};
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]+=P[i].CBE_basis_moments[j][k];}}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {P[i].CBE_basis_moments[j][k] *= P[i].Mass / mom_tot[0];}}
        for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]=0;}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=0;k<CBE_INTEGRATOR_NMOMENTS;k++) {mom_tot[k]+=P[i].CBE_basis_moments[j][k];}}
        for(j=0;j<CBE_INTEGRATOR_NBASIS;j++) {for(k=1;k<4;k++) {P[i].CBE_basis_moments[j][k] += P[i].CBE_basis_moments[j][0]*(P[i].Vel[k-1]-mom_tot[k]/P[i].Mass);}}
    }
    return;
}


/* ---------------------------------------------------------------------------
 * Per-output-interval CBE diagnostic counter scaffold (Wave-CBE Commit 2,
 * 2026-05-24). Gated by CBE_INTEGRATOR_OUTPUT_MOREINFO (or the broader
 * OUTPUT_ADDITIONAL_RUNINFO) so production runs can purge the entire
 * diagnostic subsystem by disabling the flag.
 *
 * Host-side per-rank counters accumulated by future Wave-CBE commits
 * (3 = root-found v_F, 4 = gradient/reconstruction, 5 = SPD repair), then
 * MPI-reduced and emitted to FdCbeDiagnostics (cbe_diagnostics.txt; opened
 * in core/begrun.cc under the same gate). Reset at each emit so the values
 * represent the accumulated total since the previous output.
 *
 * Aggregation path note: per-pair updates from inside AGSForce/GPU
 * kernels must go through the existing AgsForceOut accumulator / merge /
 * writeback pattern, not via direct writes to these globals. The pair-loop
 * infrastructure runs reductions onto per-particle out-structs which get
 * merged onto host particles in the post-loop step; the CBE counter
 * updates will hook there. This commit only defines the host-side
 * holding/reduce/emit scaffold; counters stay at zero until populated.
 *
 * No call site is added in this commit (Commit 3 will add the hook from
 * energy_statistics or equivalent once the counter writes are real). The
 * file is opened (with a column-header comment) but stays header-only.
 * --------------------------------------------------------------------------- */
#if defined(OUTPUT_ADDITIONAL_RUNINFO) || defined(CBE_INTEGRATOR_OUTPUT_MOREINFO)
struct cbe_step_accumulators {
    /* Populated by Wave-CBE Commit 3 (root-found v_F): */
    double face_mass_flux_residual_max;   /* max |sum_basis F_m * A| seen on any face */
    double face_mass_flux_residual_sum;   /* sum of |sum_basis F_m * A| over faces */
    long long bracket_fail_count;         /* root-find bracket-widening failures */
    /* Populated by Wave-CBE Commit 4 (gradient/reconstruction): */
    long long recon_rho_clamp_count;      /* Q-face rho < eps clamps */
    long long recon_S_clamp_count;        /* Q-face Sxx < 0 clamps */
    /* Populated by Wave-CBE Commit 5 (SPD repair): */
    double repair_dP_sum;                 /* sum |dP| introduced by repair */
    double repair_dT_sum;                 /* sum |dT| introduced by repair */
};
static struct cbe_step_accumulators CbeStepAccum;


void cbe_step_diagnostics_reset(void)
{
    CbeStepAccum.face_mass_flux_residual_max = 0;
    CbeStepAccum.face_mass_flux_residual_sum = 0;
    CbeStepAccum.bracket_fail_count          = 0;
    CbeStepAccum.recon_rho_clamp_count       = 0;
    CbeStepAccum.recon_S_clamp_count         = 0;
    CbeStepAccum.repair_dP_sum               = 0;
    CbeStepAccum.repair_dT_sum               = 0;
}


void cbe_step_diagnostics_emit(void)
{
    double rmax_local = CbeStepAccum.face_mass_flux_residual_max;
    double rsum_local = CbeStepAccum.face_mass_flux_residual_sum;
    long long brk_local = CbeStepAccum.bracket_fail_count;
    long long rc_local  = CbeStepAccum.recon_rho_clamp_count;
    long long sc_local  = CbeStepAccum.recon_S_clamp_count;
    double dP_local = CbeStepAccum.repair_dP_sum;
    double dT_local = CbeStepAccum.repair_dT_sum;
    double rmax, rsum, dP, dT;
    long long brk, rc, sc;
    MPI_Reduce(&rmax_local, &rmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rsum_local, &rsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&brk_local,  &brk,  1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rc_local,   &rc,   1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sc_local,   &sc,   1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dP_local,   &dP,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dT_local,   &dT,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && FdCbeDiagnostics != NULL) {
        fprintf(FdCbeDiagnostics, "%.16g %.16g %.16g %lld %lld %lld %.16g %.16g\n",
                All.Time, rmax, rsum, brk, rc, sc, dP, dT);
        fflush(FdCbeDiagnostics);
    }
}
#endif


#endif
