#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* Single-star IMF sampling: draws ONE stellar mass from a Kroupa IMF for each
 * new star particle. Each star particle represents exactly one star.
 *
 * Two cases for the mass difference between drawn stellar mass and gas cell:
 *   M_drawn > M_cell: borrow mass from gas neighbors, nearest-first,
 *                     consuming cells fully if needed (gizmo removes zero-mass cells).
 *   M_drawn < M_cell: return excess mass to gas neighbors, kernel-weighted.
 * Both cases are mass and momentum conserving.
 *
 * An IMFDonorFlag prevents the same gas cell from being consumed by two
 * different stars in the same sampling pass.
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks.
 * The search radius is pre-scaled for massive stars that need to borrow
 * from more cells than the standard kernel contains. */

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF

#if (N_STELLAR_MASS < 5)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF requires N_STELLAR_MASS >= 5 (indices 0-4 used)"
#endif
#if (GALSF_GENERATIONS > 1)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF requires GALSF_GENERATIONS=1 (full cell conversion, no fractional spawning)"
#endif

static const double M_max = 50.;
static const double M_min = 0.08;

/* Kroupa IMF: alpha_1=1.3 (M<0.5), alpha_2=2.3 (M>=0.5), normalization A=0.126512 */
static inline double drawMassFromIMF(double x)
{
    double A = 0.126512;
    if(x < 0.5) return 2.0 * A * pow(x, -1.3);
    return A * pow(x, -2.3);
}

/* Envelope function for rejection sampling (upper bound of IMF) */
static inline double envelope_function(double x)
{
    return 2.0 * 0.126512 * pow(x, -1.3);
}

#define MIN_IMF_NGBS 128

struct imf_ngb_entry { int index; double dist2; };

/* Sort by distance (nearest first) for borrowing case */
static int compare_by_dist(const void *a, const void *b)
{
    const struct imf_ngb_entry *ea = (const struct imf_ngb_entry *)a;
    const struct imf_ngb_entry *eb = (const struct imf_ngb_entry *)b;
    if(ea->dist2 < eb->dist2) return -1;
    if(ea->dist2 > eb->dist2) return  1;
    return 0;
}

/* Per-rank flag array: prevents the same gas cell from being consumed by
 * multiple stars in the same IMF sampling pass. Allocated in assign_stellar_masses(),
 * used in the evaluate function, freed after the xchange. */
static int *IMFDonorFlag = NULL;


/* =========================================================================== */
/*  Mass transfer via code_block_xchange framework.                            */
/*  Borrowing case: nearest-first consumption of gas cells.                    */
/*  Return case: kernel-weighted distribution to gas cells.                    */
/*  MstarSampleIMF[1..3] used as temporary storage for momentum gained.        */
/* =========================================================================== */

#define CORE_FUNCTION_NAME resolvedismIMF_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismIMF
#define OUTPUTFUNCTION_NAME out2particle_resolvedismIMF
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismIMF_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius;
    MyDouble delta_code;   /* mass difference in code units (positive = borrow from gas) */
    MyDouble star_vel[3];  /* star velocity (for returning mass at star velocity) */
    int max_ngbs;          /* dynamic buffer size based on mass ratio */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismIMF(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k]; in->star_vel[k]=P[i].Vel[k];}
    in->KernelRadius = P[i].KernelRadius;
    if(in->KernelRadius <= 0) in->KernelRadius = All.ForceSoftening[P[i].Type] * 5.0; /* fallback for new stars */
    in->delta_code = 0;
    in->max_ngbs = MIN_IMF_NGBS;
    if(P[i].MstarSampleIMF[0] <= 0 || P[i].sampled != 0) return;
    in->delta_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR - P[i].Mass;
    if(fabs(in->delta_code) < 1.0e-3 * P[i].Mass) {in->delta_code = 0; return;} /* within 0.1% of cell mass: treat as exact match */

    /* Subtract mass already transferred in a previous mode 0 pass (prevents over-borrowing
     * or over-returning when particle2in is called again to pack export data for mode 1).
     * MstarSampleIMF[4] tracks |mass already transferred|, works for both borrow and return. */
    if(P[i].MstarSampleIMF[4] > 0) {
        if(in->delta_code > 0) in->delta_code -= P[i].MstarSampleIMF[4];
        else                   in->delta_code += P[i].MstarSampleIMF[4];
        if(fabs(in->delta_code) < 1.0e-3 * P[i].Mass) {in->delta_code = 0; return;}
    }

    /* For borrowing case: pre-scale search radius so we enclose enough gas mass.
     * Uses REMAINING delta (after subtracting already-transferred mass) to size
     * the radius and buffer. In iterative passes the outer loop also bumps
     * KernelRadius by 10% each iteration, so radius grows until enough mass is found. */
    if(in->delta_code > 0) {
        double n_needed = in->delta_code / P[i].Mass; /* how many cell-masses still needed */
        in->max_ngbs = (int)(n_needed * 2.0) + MIN_IMF_NGBS; /* 2x safety margin */
        if(n_needed > All.DesNumNgb) {
            in->KernelRadius *= pow(n_needed / All.DesNumNgb, 1.0/3.0) * 1.5;
        }
    }
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat mom_gained[3]; /* momentum gained from gas (borrowing case) */
    MyFloat mass_borrowed;  /* actual mass taken from gas in this mode */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismIMF(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    /* Accumulate momentum gained into temporary storage MstarSampleIMF[1..3] */
    int k; for(k=0;k<3;k++) {ASSIGN_ADD(P[i].MstarSampleIMF[k+1], out->mom_gained[k], mode);}
    /* Accumulate actual mass borrowed into MstarSampleIMF[4] so that particle2in
     * can reduce delta_code for subsequent mode 1 exports (prevents over-borrowing) */
    ASSIGN_ADD(P[i].MstarSampleIMF[4], out->mass_borrowed, mode);
}

int resolvedismIMF_active_check(int i);
int resolvedismIMF_active_check(int i)
{
    if(P[i].Type != 4) {return 0;}
    if(P[i].sampled != 0) {return 0;} /* already sampled */
    if(P[i].MstarSampleIMF[0] <= 0) {return 0;} /* no drawn mass (pre-processing didn't run) */
    return 1;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
int resolvedismIMF_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double r2, h2;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismIMF(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.delta_code == 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;

    /* Phase 1: Tree walk — collect gas neighbors, record distance and kernel weight */
    int max_ngbs = local.max_ngbs;
    if(max_ngbs < MIN_IMF_NGBS) max_ngbs = MIN_IMF_NGBS;
    struct imf_ngb_entry *ngb_buf = (struct imf_ngb_entry *) malloc(max_ngbs * sizeof(struct imf_ngb_entry));
    if(!ngb_buf) {printf("IMF sampling: malloc failed for %d ngbs\n", max_ngbs); return 0;}
    int n_collected = 0;

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {free(ngb_buf); return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;
                double Mass_j;
                #pragma omp atomic read
                Mass_j = P[j].Mass;
                if(Mass_j <= 0) continue;

                double dp[3];
                for(k=0;k<3;k++) {dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(dp[0],dp[1],dp[2],1);
                r2=0; for(k=0;k<3;k++) {r2 += dp[k]*dp[k];}
                if(r2 <= 0 || r2 >= h2) continue;

                if(n_collected < max_ngbs) {
                    ngb_buf[n_collected].index = j;
                    ngb_buf[n_collected].dist2 = r2;
                    n_collected++;
                }
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    }

    if(n_collected == 0) {free(ngb_buf); return 0;}
    if(n_collected >= max_ngbs)
        printf("IMF sampling WARNING: neighbor buffer full (%d), some neighbors dropped\n", max_ngbs);

    /* Phase 2: Mass transfer — different strategies for borrowing vs returning */
    if(local.delta_code > 0)
    {
        /* === BORROW from gas: nearest-first consumption ===
         * Sort neighbors by distance, walk outward, consume cells fully
         * (mass -> 0, gizmo removes them) until enough mass is accumulated.
         * The last cell may be only partially consumed.
         * IMFDonorFlag prevents double-consumption by multiple stars. */
        qsort(ngb_buf, n_collected, sizeof(struct imf_ngb_entry), compare_by_dist);
        double mass_still_needed = local.delta_code;

        for(n = 0; n < n_collected && mass_still_needed > 0; n++)
        {
            j = ngb_buf[n].index;

            /* Atomically claim this gas cell via donor flag (prevents two stars
             * from consuming the same cell in the same sampling pass) */
            if(IMFDonorFlag != NULL) {
                int was_flagged;
                #pragma omp atomic capture
                { was_flagged = IMFDonorFlag[j]; IMFDonorFlag[j] = 1; }
                if(was_flagged) continue;
            }

            double Mass_j;
            #pragma omp atomic read
            Mass_j = P[j].Mass;
            if(Mass_j <= 0) continue;

            /* Take at most what this cell has, or what we still need */
            double take = fmin(mass_still_needed, Mass_j);

            /* Read gas velocity for momentum tracking */
            double Vel_j[3];
            for(k=0;k<3;k++) {
                #pragma omp atomic read
                Vel_j[k] = P[j].Vel[k];
            }

            /* Remove mass from gas cell */
            #pragma omp atomic
            P[j].Mass -= take;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            #pragma omp atomic
            CellP[j].MassTrue -= take;
#endif
            /* Accumulate momentum and mass carried to star */
            for(k=0;k<3;k++) out.mom_gained[k] += take * Vel_j[k];
            out.mass_borrowed += take;
            mass_still_needed -= take;
        }
    }
    else
    {
        /* === RETURN excess mass to gas: hand it to the first valid neighbor ===
         * Star is lighter than the original gas cell (within factor ~2 of cell mass).
         * Just attach the residual to the first neighbor found, momentum conserving:
         * v_gas_new = (M_gas * v_gas + give * v_star) / (M_gas + give) */
        double give = fabs(local.delta_code);
        for(n = 0; n < n_collected; n++)
        {
            j = ngb_buf[n].index;
            double Mass_j;
            #pragma omp atomic read
            Mass_j = P[j].Mass;
            if(Mass_j <= 0) continue;

            double Vel_j[3];
            for(k=0;k<3;k++) {
                #pragma omp atomic read
                Vel_j[k] = P[j].Vel[k];
            }

            double Mass_j_new = Mass_j + give;
            for(k=0;k<3;k++) {
                double dv = (give / Mass_j_new) * (local.star_vel[k] - Vel_j[k]);
                #pragma omp atomic
                P[j].Vel[k] += dv;
                #pragma omp atomic
                CellP[j].VelPred[k] += dv;
            }
            #pragma omp atomic
            P[j].Mass += give;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
            #pragma omp atomic
            CellP[j].MassTrue += give;
#endif
            out.mass_borrowed += give; /* track so mode 1 knows it's done */
            break; /* done — all excess mass handed to this one neighbor */
        }
    }

    free(ngb_buf);
    if(mode == 0) {out2particle_resolvedismIMF(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void assign_stellar_masses_xchange(void)
{
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}
#include "../system/code_block_xchange_finalize.h"


/* =========================================================================== */
/*  Top-level routine: pre-processing, xchange mass transfer, post-processing  */
/* =========================================================================== */
void assign_stellar_masses(void)
{
    int i, k;

    /* Count unsampled stars across all ranks */
    int nstars_local = 0;
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 4 && P[i].sampled == 0 && P[i].Mass > 0) nstars_local++;

    int nstars_total;
    MPI_Allreduce(&nstars_local, &nstars_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(nstars_total == 0) return;

    if(ThisTask == 0) printf("IMF sampling: %d unsampled single-star particles\n", nstars_total);

    /* === Pre-processing: draw one IMF mass per unsampled star particle === */
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 0 || P[i].Mass <= 0) continue;

        /* Restart safety: if MstarSampleIMF[0] > 0 but sampled == 0, this star was
         * already drawn in a previous run that got interrupted. Skip the redraw,
         * just zero the temporary storage so the xchange can run cleanly. */
        if(P[i].MstarSampleIMF[0] > 0) {
            int j; for(j = 1; j < N_STELLAR_MASS; j++) P[i].MstarSampleIMF[j] = 0;
            continue;
        }

        /* Draw ONE stellar mass from Kroupa IMF via rejection sampling */
        double M_drawn;
        do {
            M_drawn = pow((pow(M_max, -0.3) - pow(M_min, -0.3)) * gsl_rng_uniform(random_generator)
                          + pow(M_min, -0.3), -1.0 / 0.3);
        } while(gsl_rng_uniform(random_generator) > (drawMassFromIMF(M_drawn) / envelope_function(M_drawn)));

        /* Store drawn mass in MstarSampleIMF[0], zero the rest */
        int j; for(j = 0; j < N_STELLAR_MASS; j++) P[i].MstarSampleIMF[j] = 0;
        P[i].MstarSampleIMF[0] = M_drawn;
        /* MstarSampleIMF[1..3] = 0: temporary momentum storage for xchange */
    }

    /* === Iterative xchange: mass borrowing/returning ===
     * The initial search radius is pre-scaled in particle2in to enclose enough
     * gas for the drawn stellar mass. If that's not enough (e.g. star near void),
     * we bump KernelRadius by 10% and repeat until all stars have enough mass.
     * In practice the pre-scaling is generous (1.5x safety factor), so iteration
     * rarely triggers. */
    PRINT_STATUS(" ..IMF sampling: mass transfer for single stars");
    int iter = 0, max_iter = 50;
    int incomplete_total;
    do {
        IMFDonorFlag = (int *) mymalloc("IMFDonorFlag", NumPart * sizeof(int));
        memset(IMFDonorFlag, 0, NumPart * sizeof(int));

        assign_stellar_masses_xchange();

        myfree(IMFDonorFlag); IMFDonorFlag = NULL;

        /* Check for incomplete borrowing across all ranks */
        int incomplete_local = 0;
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;
            double m_new_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double delta_code = m_new_code - P[i].Mass;
            if(delta_code <= 1.0e-3 * P[i].Mass) continue; /* not borrowing or within tolerance */
            double remaining = delta_code - P[i].MstarSampleIMF[4];
            if(remaining > 1.0e-3 * P[i].Mass) {
                incomplete_local++;
                P[i].KernelRadius *= 1.1; /* expand search radius by 10% for next pass */
            }
        }
        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;
        if(ThisTask == 0 && incomplete_total > 0)
            printf("IMF sampling iter %d: %d stars still need more mass, expanding search radius\n", iter, incomplete_total);
    } while(incomplete_total > 0 && iter < max_iter);

    if(ThisTask == 0 && iter >= max_iter && incomplete_total > 0)
        printf("WARNING: IMF sampling did not converge after %d iterations, %d stars incomplete\n", max_iter, incomplete_total);

    /* === Post-processing: velocity update for momentum conservation, finalize === */
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;

        double M_drawn = P[i].MstarSampleIMF[0]; /* solar masses */
        double m_old_code = P[i].Mass; /* original gas cell mass in code units */
        double m_new_code = M_drawn / UNIT_MASS_IN_SOLAR;
        double delta_code = m_new_code - m_old_code;

        /* Update star velocity for momentum conservation (borrowing case only).
         * v_star_new = (m_old * v_old + mom_gained) / m_new
         * where mom_gained was accumulated in MstarSampleIMF[1..3] by the xchange */
        if(delta_code > 0 && m_new_code > 0) {
            for(k=0;k<3;k++) {
                P[i].Vel[k] = (m_old_code * P[i].Vel[k] + P[i].MstarSampleIMF[k+1]) / m_new_code;
            }
        }

        /* Set dynamical mass to drawn IMF mass */
        P[i].Mass = m_new_code;
        P[i].sampled = 1;

#ifdef GALSF_RESOLVEDISM_WINDS
        P[i].WindMassAccum = 0;
        P[i].WindMomentumAccum = 0;
        P[i].M_current_old = M_drawn; /* initial mass = ZAMS mass */
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
        P[i].M_drawn_Ia = 0; /* set when star dies as WD */
#endif

        /* Clear temporary storage (momentum [1..3] and mass_borrowed [4]) */
        P[i].MstarSampleIMF[1] = P[i].MstarSampleIMF[2] = P[i].MstarSampleIMF[3] = P[i].MstarSampleIMF[4] = 0;

        /* Compute UV luminosity from the single drawn star */
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        P[i].UV_luminosity = 0;
        P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_PHOTOION
        P[i].Lyman_photons_per_sec = 0;
#endif
        if(M_drawn > All.IMFSampleStellarMassCut) {
            double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
            if(star_age_yr < get_lifetime(M_drawn)) {
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
                double logM = log10(M_drawn);
                double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
                double log_age = log10(DMAX(star_age_yr, 100.0));
                P[i].UV_luminosity = pow(10., stellar_log_L_FUV_total(logM, logZ, log_age));
                P[i].LW_luminosity = pow(10., stellar_log_L_LW(logM, logZ, log_age));
#ifdef GALSF_RESOLVEDISM_PHOTOION
                P[i].Lyman_photons_per_sec = pow(10., stellar_log_Q_ion(logM, logZ, log_age));
#endif
#else
                P[i].UV_luminosity = pow(10., get_logL_pe(M_drawn));
                P[i].LW_luminosity = P[i].UV_luminosity; /* fallback: assume all FUV is LW */
#ifdef GALSF_RESOLVEDISM_PHOTOION
                P[i].Lyman_photons_per_sec = pow(10., get_logS_ly(M_drawn));
#endif
#endif
            }
        }
#endif

    }

    if(ThisTask == 0) printf("IMF sampling done (%d iterations).\n", iter);
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF */
