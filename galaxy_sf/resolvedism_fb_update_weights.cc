/* resolvedism_fb_update_weights.cc — Dedicated SN kernel search
 *
 * Replicates gizmo2017 update_weights() exactly using code_block_xchange:
 *   1. Iterative bisection to find HsmlSN enclosing DesNumNgb neighbors
 *   2. Computes NumNgbSN = sum(NORM_COEFF * f(u) / hinv3) over gas neighbors
 *
 * Called from resolvedism_inject_sn_energy() before the thermal injection.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

#ifdef GALSF_RESOLVEDISM_FB

/* NORM_COEFF from gizmo2017: 4*pi/3 for 3D */
#define SN_NORM_COEFF (4.0 * M_PI / 3.0)

struct kernel_resolvedismFB_update_weights {double dp[3], r, wk, dwk, hinv, hinv3, hinv4;};

#define CORE_FUNCTION_NAME resolvedismFB_update_weights_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismFB_update_weights
#define OUTPUTFUNCTION_NAME out2particle_resolvedismFB_update_weights
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismFB_update_weights_active_check(i,loop_iteration))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], Hsml;
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismFB_update_weights(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->Hsml = P[i].HsmlSN;
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat Ngb;   /* weighted neighbor count: sum(NORM_COEFF * f(u) / hinv3) */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismFB_update_weights(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    ASSIGN_ADD(P[i].NumNgbSN, out->Ngb, mode);
}

int resolvedismFB_update_weights_active_check(int i, int loop_iteration);
int resolvedismFB_update_weights_active_check(int i, int loop_iteration)
{
    if(P[i].Type != 4) return 0;
    if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 4) return 0;
    if(P[i].HsmlSN <= 0) return 0;
    return 1;
}

/* core evaluate function — replicates gizmo2017 update_weight_evaluate() */
int resolvedismFB_update_weights_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int j, n, startnode, numngb_inbox, listindex = 0;
    double r2, u;
    struct kernel_resolvedismFB_update_weights kernel;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0)
        particle2in_resolvedismFB_update_weights(&local, target, loop_iteration);
    else
        local = DATAGET_NAME[target];

    double h_i = local.Hsml;
    double h2 = h_i * h_i;

    if(mode == 0) { startnode = All.MaxPart; }
    else { startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode; }

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads(local.Pos, h_i, target, &startnode, mode,
                exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) return -2;

            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;

                int k; for(k=0;k<3;k++) { kernel.dp[k] = local.Pos[k] - P[j].Pos[k]; }
                NEAREST_XYZ(kernel.dp[0], kernel.dp[1], kernel.dp[2], 1);
                r2 = kernel.dp[0]*kernel.dp[0] + kernel.dp[1]*kernel.dp[1] + kernel.dp[2]*kernel.dp[2];
                if(r2 >= h2 || r2 <= 0) continue;

                kernel.r = sqrt(r2);
                kernel_hinv(h_i, &kernel.hinv, &kernel.hinv3, &kernel.hinv4);
                u = kernel.r * kernel.hinv;
                kernel_main(u, kernel.hinv3, kernel.hinv4, &kernel.wk, &kernel.dwk, -1); /* mode -1: f(u) only */

                /* gizmo2017 line 700: out.Ngb += NORM_COEFF * kernel.wk_i / hinv3 */
                out.Ngb += (SN_NORM_COEFF * kernel.wk / kernel.hinv3);
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) { startnode = Nodes[startnode].u.d.nextnode; }
            }
        }
    }

    if(mode == 0)
        out2particle_resolvedismFB_update_weights(&out, target, 0, loop_iteration);
    else
        DATARESULT_NAME[target] = out;

    return 0;
}

/* code_block_xchange boilerplate — generates the calc() driver + subfunctions */
void resolvedismFB_update_weights_calc(void)
{
#include "../system/code_block_xchange_perform_ops_malloc.h"
#include "../system/code_block_xchange_perform_ops.h"
#include "../system/code_block_xchange_perform_ops_demalloc.h"
}

#include "../system/code_block_xchange_finalize.h"

/* Main driver: iterative bisection to find HsmlSN, same as gizmo2017 update_weights() */
void resolvedism_update_sn_weights(void)
{
    int i;
    double desnumngb = All.DesNumNgb; /* FacNumNgbSN = 1 → same as hydro */
    double desnumngbdev = 1.0;

    /* Initialize HsmlSN for all active SN/Ia stars */
    int nstars_local = 0;
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type != 4) continue;
        if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 4) continue;
        P[i].HsmlSN = P[i].KernelRadius; /* initial guess from density loop */
        P[i].NumNgbSN = 0;
        nstars_local++;
    }
    int nstars_total;
    MPI_Allreduce(&nstars_local, &nstars_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(nstars_total == 0) return;

    double *Left = (double *)mymalloc("Left_SN", NumPart * sizeof(double));
    double *Right = (double *)mymalloc("Right_SN", NumPart * sizeof(double));
    for(i = 0; i < NumPart; i++) { Left[i] = Right[i] = 0; }

    int iter = 0, max_iter = 60;
    long long npleft_total;

    do {
        /* Reset NumNgbSN before each iteration */
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 4) continue;
            P[i].NumNgbSN = 0;
        }

        /* Run the tree walk to count weighted neighbors */
        resolvedismFB_update_weights_calc();

        /* Bisection update — same logic as gizmo2017 update_weights lines 472-567 */
        int npleft = 0;
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 4) continue;
            if(P[i].TimeBin < 0) continue; /* marked as converged */

            int redo = 0;
            if(P[i].NumNgbSN < (desnumngb - desnumngbdev) || P[i].NumNgbSN > (desnumngb + desnumngbdev))
                redo = 1;

            if(redo) {
                npleft++;
                if(Left[i] > 0 && Right[i] > 0 && (Right[i] - Left[i]) < 1.0e-3 * Left[i]) {
                    npleft--;
                    P[i].TimeBin = -P[i].TimeBin - 1;
                    continue;
                }
                if(P[i].NumNgbSN < (desnumngb - desnumngbdev))
                    Left[i] = DMAX(P[i].HsmlSN, Left[i]);
                else {
                    if(Right[i] != 0) { if(P[i].HsmlSN < Right[i]) Right[i] = P[i].HsmlSN; }
                    else Right[i] = P[i].HsmlSN;
                }
                if(Right[i] > 0 && Left[i] > 0)
                    P[i].HsmlSN = pow(0.5 * (pow(Left[i], 3) + pow(Right[i], 3)), 1.0 / 3);
                else if(Right[i] == 0 && Left[i] > 0)
                    P[i].HsmlSN *= 1.26;
                else if(Right[i] > 0 && Left[i] == 0)
                    P[i].HsmlSN /= 1.26;
            } else {
                P[i].TimeBin = -P[i].TimeBin - 1; /* mark as converged */
            }
        }

        long long npleft_local = npleft;
        MPI_Allreduce(&npleft_local, &npleft_total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        iter++;
        if(iter > max_iter) {
            if(ThisTask == 0) printf("RESOLVEDISM SN WEIGHTS: failed to converge after %d iterations, %lld remaining\n", iter, npleft_total);
            break;
        }
    } while(npleft_total > 0);

    /* Restore TimeBin for stars that were marked inactive during bisection */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type != 4) continue;
        if(P[i].SNe_ThisTimeStep != 1 && P[i].SNe_ThisTimeStep != 4) continue;
        if(P[i].TimeBin < 0) P[i].TimeBin = -P[i].TimeBin - 1;
    }

    myfree(Right);
    myfree(Left);

    if(ThisTask == 0 && nstars_total > 0)
        printf("RESOLVEDISM SN WEIGHTS: %d stars, converged in %d iterations\n", nstars_total, iter);
}

#endif /* GALSF_RESOLVEDISM_FB */
