#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* LYRA-style single-star IMF sampling (Gutcke+ 2021).
 *
 * Each star formation event draws a stellar mass from the Kroupa IMF,
 * then collects that mass from an accretion region around the parent
 * gas cell.  Every gas cell within the accretion radius loses the same
 * fraction f_acc = M_star / M_enclosed of its mass.  No cell loses more
 * than 50%.  The star inherits CoM velocity and mass-weighted metallicity
 * from all donor cells.
 *
 * The accretion radius and f_acc are pre-computed from the local gas
 * density (known from the density loop), so only a single code_block_xchange
 * pass is needed per iteration.  Iteration is rare — only when the
 * density estimate was too far off (massive stars in patchy ISM).
 *
 * Runtime parameter IMFSampleMinMass controls the IMF lower bound:
 *   0.08 → full IMF  [0.08, 350] Msun
 *   4.0  → truncated  [4.0, 350] Msun  (LYRA-style)
 */

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF

#if (N_STELLAR_MASS < 5)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF requires N_STELLAR_MASS >= 5 (indices 0-4 used)"
#endif

static const double M_max = 350.;

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

/* Public function: draw a single stellar mass from the Kroupa IMF via rejection sampling.
 * Lower bound from All.IMFSampleMinMass (default 0.08), upper bound 350 Msun. */
double draw_one_mass_from_kroupa_IMF(void)
{
    double M_min = All.IMFSampleMinMass;
    if(M_min < 0.08) M_min = 0.08;
    if(M_min > M_max) M_min = M_max * 0.99; /* safety */
    double M_drawn;
    do {
        M_drawn = pow((pow(M_max, -0.3) - pow(M_min, -0.3)) * gsl_rng_uniform(random_generator)
                      + pow(M_min, -0.3), -1.0 / 0.3);
    } while(gsl_rng_uniform(random_generator) > (drawMassFromIMF(M_drawn) / envelope_function(M_drawn)));
    return M_drawn;
}

/* Public function: finalize a newly sampled star — set up luminosity, winds, Ia fields.
 * Called from assign_stellar_masses() after the accretion walk sets Mass and Vel.
 * Assumes P[i].Mass and P[i].MstarSampleIMF[0] are already set to the correct values. */
void finalize_sampled_star(int i, double M_drawn)
{
    P[i].sampled = 1;

    printf("IMF_DRAW: Task=%d ID=%llu M_drawn=%.3f[Msun] M_particle=%.4f\n",
        ThisTask, (unsigned long long)P[i].ID, M_drawn, P[i].Mass * UNIT_MASS_IN_SOLAR);
    if(!isfinite(P[i].Mass) || P[i].Mass <= 0)
        printf("NAN_CHECK_IMF: Task=%d ID=%llu Mass=%.6e after IMF sampling\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].Mass);

#ifdef GALSF_RESOLVEDISM_WINDS
    P[i].WindMassAccum = 0;
    P[i].WindMomentumAccum = 0;
    P[i].M_current_old = M_drawn; /* initial mass = ZAMS mass */
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    P[i].M_drawn_Ia = 0; /* set when star dies as WD */
#endif

    /* Clear temporary storage */
    P[i].MstarSampleIMF[1] = P[i].MstarSampleIMF[2] = P[i].MstarSampleIMF[3] = P[i].MstarSampleIMF[4] = 0;

    /* Compute luminosity from the single drawn star */
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
            P[i].LW_luminosity = P[i].UV_luminosity;
#ifdef GALSF_RESOLVEDISM_PHOTOION
            P[i].Lyman_photons_per_sec = pow(10., get_logS_ly(M_drawn));
#endif
#endif
        }
    }
    /* Sanity check luminosities */
    if(!isfinite(P[i].UV_luminosity) || P[i].UV_luminosity < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu UV_lum=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].UV_luminosity, M_drawn);
    if(!isfinite(P[i].LW_luminosity) || P[i].LW_luminosity < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu LW_lum=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].LW_luminosity, M_drawn);
#ifdef GALSF_RESOLVEDISM_PHOTOION
    if(!isfinite(P[i].Lyman_photons_per_sec) || P[i].Lyman_photons_per_sec < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu Q_ion=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].Lyman_photons_per_sec, M_drawn);
#endif
#endif
}


/* =========================================================================== */
/*  Accretion walk via code_block_xchange framework.                           */
/*  Equal-fraction mass removal from all gas cells within R_acc.               */
/*  f_acc and R_acc pre-computed in assign_stellar_masses(), stored on particle */
/*  MstarSampleIMF[2] = f_acc,  KernelRadius = R_acc.                         */
/* =========================================================================== */

/* Global accumulators — allocated in assign_stellar_masses(), indexed by particle index */
static double *IMF_MassAccreted = NULL;
static double *IMF_MomAccreted = NULL;   /* [NumPart * 3] */
#ifdef METALS
static double *IMF_MetalAccreted = NULL;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
static double *IMF_ElemAccreted = NULL;  /* [NumPart * NUM_RESOLVEDISM_ELEMENTS] */
#endif
/* Two-pass mode flag: 0 = count enclosed mass only (no removal), 1 = remove mass */
static int IMF_AccretionMode = 0;


#define CORE_FUNCTION_NAME resolvedismIMF_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismIMF
#define OUTPUTFUNCTION_NAME out2particle_resolvedismIMF
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismIMF_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius;
    MyDouble f_acc;   /* mass fraction to remove from each neighbor */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismIMF(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    if(in->KernelRadius <= 0) in->KernelRadius = All.ForceSoftening[P[i].Type] * 5.0;
    in->f_acc = P[i].MstarSampleIMF[2]; /* set in assign_stellar_masses() pre-processing */
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat mass_accreted;
    MyFloat mom_accreted[3];
#ifdef METALS
    MyFloat metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    MyFloat elem_accreted[NUM_RESOLVEDISM_ELEMENTS];
#endif
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismIMF(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k;
    /* Always use += : arrays are zeroed before each iteration,
     * mode 0 (local) is always called first, then mode 1 (remote) adds. */
    if(mode == 0) {
        IMF_MassAccreted[i] += out->mass_accreted;
        for(k=0;k<3;k++) IMF_MomAccreted[i*3+k] += out->mom_accreted[k];
#ifdef METALS
        IMF_MetalAccreted[i] += out->metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++)
            IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] += out->elem_accreted[k];
#endif
    } else {
        #pragma omp atomic
        IMF_MassAccreted[i] += out->mass_accreted;
        for(k=0;k<3;k++) {
            #pragma omp atomic
            IMF_MomAccreted[i*3+k] += out->mom_accreted[k];
        }
#ifdef METALS
        #pragma omp atomic
        IMF_MetalAccreted[i] += out->metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++) {
            #pragma omp atomic
            IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] += out->elem_accreted[k];
        }
#endif
    }
}

int resolvedismIMF_active_check(int i);
int resolvedismIMF_active_check(int i)
{
    if(P[i].Type != 4) return 0;
    if(P[i].sampled != 0) return 0;
    if(P[i].MstarSampleIMF[0] <= 0) return 0;
    if(P[i].MstarSampleIMF[2] <= 0) return 0; /* f_acc = 0 means converged or not active */
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
    if(local.f_acc <= 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) return -2;

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
                if(r2 < 0 || r2 >= h2) continue; /* allow r2==0: parent cell at same position */

                if(IMF_AccretionMode == 0)
                {
                    /* --- COUNTING PASS: sum neighbor mass, no removal --- */
                    out.mass_accreted += Mass_j;
                }
                else
                {
                    /* --- REMOVAL PASS: use correct f_acc --- */
                    double dM = local.f_acc * Mass_j;
                    if(dM > 0.5 * Mass_j) dM = 0.5 * Mass_j;
                    if(dM <= 0) continue;

                    double Vel_j[3];
                    for(k=0;k<3;k++) {
                        #pragma omp atomic read
                        Vel_j[k] = P[j].Vel[k];
                    }

#ifdef METALS
                    double Z_j;
                    #pragma omp atomic read
                    Z_j = P[j].Metallicity[0];
                    out.metal_accreted += dM * Z_j;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++) {
                        double Xk_j;
                        #pragma omp atomic read
                        Xk_j = P[j].ElementAbundance[k];
                        out.elem_accreted[k] += dM * Xk_j;
                    }
#endif
                    /* Rescale conserved MHD quantities proportional to mass removed,
                     * matching parent-side of split in merge_split.cc:517-525.
                     * Star doesn't inherit B-flux; removed portion is discarded. */
                    double frac = dM / Mass_j;
#ifdef MAGNETIC
                    for(k=0;k<3;k++) {
                        double dq;
                        #pragma omp atomic read
                        dq = CellP[j].B[k];
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].B[k] -= dq;
                        #pragma omp atomic read
                        dq = CellP[j].BPred[k];
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].BPred[k] -= dq;
                        #pragma omp atomic read
                        dq = CellP[j].DtB[k];
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].DtB[k] -= dq;
                    }
                    {
                        double dq;
                        #pragma omp atomic read
                        dq = CellP[j].divB;
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].divB -= dq;
                    }
#ifdef DIVBCLEANING_DEDNER
                    {
                        double dq;
                        #pragma omp atomic read
                        dq = CellP[j].Phi;
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].Phi -= dq;
                        #pragma omp atomic read
                        dq = CellP[j].PhiPred;
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].PhiPred -= dq;
                        #pragma omp atomic read
                        dq = CellP[j].DtPhi;
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].DtPhi -= dq;
                    }
                    for(k=0;k<3;k++) {
                        double dq;
                        #pragma omp atomic read
                        dq = CellP[j].DtB_PhiCorr[k];
                        dq *= frac;
                        #pragma omp atomic
                        CellP[j].DtB_PhiCorr[k] -= dq;
                    }
#endif
#endif
                    #pragma omp atomic
                    P[j].Mass -= dM;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                    #pragma omp atomic
                    CellP[j].MassTrue -= dM;
#endif
                    out.mass_accreted += dM;
                    for(k=0;k<3;k++) out.mom_accreted[k] += dM * Vel_j[k];
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
/*  Top-level routine: pre-compute f_acc, run accretion walk, finalize stars    */
/* =========================================================================== */
void assign_stellar_masses(void)
{
    int i, k;

    /* Count unsampled stars (spawned by sfr_eff.cc with sampled=0) */
    int nstars_local = 0;
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 4 && P[i].sampled == 0 && P[i].MstarSampleIMF[0] > 0) nstars_local++;

    int nstars_total;
    MPI_Allreduce(&nstars_local, &nstars_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(nstars_total == 0) return;

    if(ThisTask == 0) printf("IMF accretion: %d stars drawing mass from neighbors\n", nstars_total);
    PRINT_STATUS(" ..IMF accretion walk for single stars");

    /* Allocate accumulators (persist across iterations, zeroed each pass) */
    IMF_MassAccreted = (double *) mymalloc("IMF_MassAccreted", NumPart * sizeof(double));
    IMF_MomAccreted = (double *) mymalloc("IMF_MomAccreted", NumPart * 3 * sizeof(double));
#ifdef METALS
    IMF_MetalAccreted = (double *) mymalloc("IMF_MetalAccreted", NumPart * sizeof(double));
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    IMF_ElemAccreted = (double *) mymalloc("IMF_ElemAccreted", NumPart * NUM_RESOLVEDISM_ELEMENTS * sizeof(double));
#endif

    /* Zero accumulators before first pass */
    memset(IMF_MassAccreted, 0, NumPart * sizeof(double));
    memset(IMF_MomAccreted, 0, NumPart * 3 * sizeof(double));
#ifdef METALS
    memset(IMF_MetalAccreted, 0, NumPart * sizeof(double));
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    memset(IMF_ElemAccreted, 0, NumPart * NUM_RESOLVEDISM_ELEMENTS * sizeof(double));
#endif

    int iter = 0, max_iter = 10;
    int incomplete_total;

    do {
        /* --- Set R_acc for this iteration --- */
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;

            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double rho = P[i].FormationDensity;
            if(rho <= 0) rho = 1e-30;

            double h_parent = P[i].KernelRadius;
            if(h_parent <= 0) h_parent = All.ForceSoftening[P[i].Type] * 5.0;

            /* On iterations after the first, check if this star is already converged */
            if(iter > 0) {
                double deficit = M_star_code - IMF_MassAccreted[i];
                if(deficit <= 0.05 * M_star_code) {
                    P[i].MstarSampleIMF[2] = 0; /* converged — skip */
                    continue;
                }
                /* Expand radius for another attempt */
                h_parent *= 1.5;
            }

            /* Estimate R_acc from density: want M_enclosed >= 2 * M_star */
            double M_remaining = (iter > 0) ? (M_star_code - IMF_MassAccreted[i]) : M_star_code;
            double M_kernel = (4.0/3.0) * M_PI * h_parent * h_parent * h_parent * rho;
            double R_acc = h_parent;
            if(2.0 * M_remaining > M_kernel && M_kernel > 0) {
                R_acc = h_parent * pow(2.0 * M_remaining / M_kernel, 1.0/3.0);
            }

            P[i].KernelRadius = R_acc;
            P[i].MstarSampleIMF[2] = 1.0; /* dummy positive value so active_check passes */
        }

        /* --- PASS 1: Count actual enclosed mass --- */
        IMF_AccretionMode = 0;
        /* Use a temporary accumulator for enclosed mass (reuse MassAccreted buffer but save/restore) */
        double *IMF_MassEnclosed = (double *) mymalloc("IMF_MassEnclosed", NumPart * sizeof(double));
        memset(IMF_MassEnclosed, 0, NumPart * sizeof(double));
        /* Temporarily swap the accumulator pointer */
        double *save_MassAccreted = IMF_MassAccreted;
        IMF_MassAccreted = IMF_MassEnclosed;
        assign_stellar_masses_xchange();
        IMF_MassAccreted = save_MassAccreted; /* restore */

        /* --- Compute correct f_acc from actual enclosed mass --- */
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;
            if(P[i].MstarSampleIMF[2] <= 0) continue; /* already converged */

            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double M_remaining = (iter > 0) ? (M_star_code - save_MassAccreted[i]) : M_star_code;
            double M_enclosed = IMF_MassEnclosed[i];

            double f_acc;
            if(M_enclosed > 0) {
                f_acc = M_remaining / M_enclosed;
            } else {
                f_acc = 0; /* no neighbors found */
            }
            /* Cap at 0.5 per cell */
            if(f_acc > 0.5) f_acc = 0.5;

            P[i].MstarSampleIMF[2] = f_acc;

            if(ThisTask == 0 || P[i].ID == 6880060)
                printf("IMF_TWO_PASS: iter=%d ID=%llu R_acc=%.6f M_enclosed=%.6e M_remaining=%.6e f_acc=%.6e\n",
                    iter, (unsigned long long)P[i].ID, P[i].KernelRadius, M_enclosed, M_remaining, f_acc);
        }
        myfree(IMF_MassEnclosed);

        /* --- PASS 2: Remove mass with correct f_acc --- */
        IMF_AccretionMode = 1;
        /* Zero the per-pass portion of accumulators (MassAccreted keeps running total across iterations) */
        /* We need a temporary to capture THIS pass's accreted mass */
        double *IMF_PassMass = (double *) mymalloc("IMF_PassMass", NumPart * sizeof(double));
        memset(IMF_PassMass, 0, NumPart * sizeof(double));
        memset(IMF_MomAccreted, 0, NumPart * 3 * sizeof(double));
#ifdef METALS
        memset(IMF_MetalAccreted, 0, NumPart * sizeof(double));
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        memset(IMF_ElemAccreted, 0, NumPart * NUM_RESOLVEDISM_ELEMENTS * sizeof(double));
#endif
        double *save2 = IMF_MassAccreted;
        IMF_MassAccreted = IMF_PassMass;
        assign_stellar_masses_xchange();
        IMF_MassAccreted = save2;
        /* Add this pass's accreted mass to running total */
        for(i = 0; i < NumPart; i++) IMF_MassAccreted[i] += IMF_PassMass[i];
        myfree(IMF_PassMass);

        /* Check for incomplete accretion */
        int incomplete_local = 0;
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;
            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double deficit = M_star_code - IMF_MassAccreted[i];
            if(deficit > 0.05 * M_star_code) incomplete_local++;
        }
        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;

        if(ThisTask == 0 && incomplete_total > 0)
            printf("IMF accretion pass %d: %d stars with >5%% mass deficit, iterating\n", iter, incomplete_total);
    } while(incomplete_total > 0 && iter < max_iter);

    if(ThisTask == 0 && iter >= max_iter && incomplete_total > 0)
        printf("WARNING: IMF accretion did not converge after %d passes, %d stars incomplete\n", max_iter, incomplete_total);

    /* === Post-processing: set star mass, CoM velocity, metallicity, then finalize === */
    /* Allocate local IMF logging buffers */
    double *imf_x = (double *)mymalloc("imf_x", DMAX(nstars_local,1) * sizeof(double));
    double *imf_y = (double *)mymalloc("imf_y", DMAX(nstars_local,1) * sizeof(double));
    double *imf_z = (double *)mymalloc("imf_z", DMAX(nstars_local,1) * sizeof(double));
    double *imf_mdrawn = (double *)mymalloc("imf_mdrawn", DMAX(nstars_local,1) * sizeof(double));
    double *imf_macc = (double *)mymalloc("imf_macc", DMAX(nstars_local,1) * sizeof(double));
    double *imf_racc = (double *)mymalloc("imf_racc", DMAX(nstars_local,1) * sizeof(double));
    double *imf_facc = (double *)mymalloc("imf_facc", DMAX(nstars_local,1) * sizeof(double));
    double *imf_zbirth = (double *)mymalloc("imf_zbirth", DMAX(nstars_local,1) * sizeof(double));
    long long *imf_id = (long long *)mymalloc("imf_id", DMAX(nstars_local,1) * sizeof(long long));
    int nimf = 0;

    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;

        double M_drawn = P[i].MstarSampleIMF[0]; /* solar masses */
        double mass_acc = IMF_MassAccreted[i];

        if(mass_acc <= 0) {
            printf("WARNING: IMF accretion failed for star ID=%llu M_star=%.2f Msun, using tiny mass\n",
                (unsigned long long)P[i].ID, M_drawn);
            mass_acc = 1e-10;
        }

        /* Star mass = actual mass accreted (exact conservation) */
        P[i].Mass = mass_acc;

        /* CoM velocity from all donor cells */
        if(mass_acc > 0) {
            for(k=0;k<3;k++) P[i].Vel[k] = IMF_MomAccreted[i*3+k] / mass_acc;
        }

        /* Mass-weighted metallicity */
#ifdef METALS
        P[i].Metallicity[0] = (mass_acc > 0) ? IMF_MetalAccreted[i] / mass_acc : 0;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++)
            P[i].ElementAbundance[k] = (mass_acc > 0) ? IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] / mass_acc : 0;
#endif
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        P[i].BirthMetallicity = P[i].Metallicity[0];
#endif

        /* Log IMF accretion info */
        imf_x[nimf] = P[i].Pos[0];
        imf_y[nimf] = P[i].Pos[1];
        imf_z[nimf] = P[i].Pos[2];
        imf_mdrawn[nimf] = M_drawn;
        imf_macc[nimf] = mass_acc * UNIT_MASS_IN_SOLAR;
        imf_racc[nimf] = P[i].KernelRadius;
        imf_facc[nimf] = P[i].MstarSampleIMF[2];
        imf_zbirth[nimf] = P[i].Metallicity[0];
        imf_id[nimf] = (long long)P[i].ID;
        nimf++;

        finalize_sampled_star(i, M_drawn);

        /* Deferred tree insertion: now the star has its real mass */
        {
            int iparent = (int)P[i].MstarSampleIMF[1];
            force_add_element_to_tree(iparent, i);
            P[i].MstarSampleIMF[1] = 0; /* clear temporary */
        }
    }

    /* Gather IMF info and write to IMFinfo.txt on rank 0 */
    {
        int nimf_total = 0;
        MPI_Allreduce(&nimf, &nimf_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(nimf_total > 0)
        {
            int *recvcounts = NULL, *displs = NULL;
            double *all_x=NULL, *all_y=NULL, *all_z=NULL, *all_mdrawn=NULL, *all_macc=NULL;
            double *all_racc=NULL, *all_facc=NULL, *all_zbirth=NULL;
            long long *all_id = NULL;
            if(ThisTask == 0) {
                recvcounts = (int *)mymalloc("recvcounts_imf", NTask * sizeof(int));
                displs = (int *)mymalloc("displs_imf", NTask * sizeof(int));
            }
            MPI_Gather(&nimf, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                displs[0] = 0;
                for(i = 1; i < NTask; i++) {displs[i] = displs[i-1] + recvcounts[i-1];}
                all_x = (double *)mymalloc("aimf_x", nimf_total * sizeof(double));
                all_y = (double *)mymalloc("aimf_y", nimf_total * sizeof(double));
                all_z = (double *)mymalloc("aimf_z", nimf_total * sizeof(double));
                all_mdrawn = (double *)mymalloc("aimf_md", nimf_total * sizeof(double));
                all_macc = (double *)mymalloc("aimf_ma", nimf_total * sizeof(double));
                all_racc = (double *)mymalloc("aimf_ra", nimf_total * sizeof(double));
                all_facc = (double *)mymalloc("aimf_fa", nimf_total * sizeof(double));
                all_zbirth = (double *)mymalloc("aimf_zb", nimf_total * sizeof(double));
                all_id = (long long *)mymalloc("aimf_id", nimf_total * sizeof(long long));
            }
            MPI_Gatherv(imf_x, nimf, MPI_DOUBLE, all_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_y, nimf, MPI_DOUBLE, all_y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_z, nimf, MPI_DOUBLE, all_z, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_mdrawn, nimf, MPI_DOUBLE, all_mdrawn, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_macc, nimf, MPI_DOUBLE, all_macc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_racc, nimf, MPI_DOUBLE, all_racc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_facc, nimf, MPI_DOUBLE, all_facc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_zbirth, nimf, MPI_DOUBLE, all_zbirth, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_id, nimf, MPI_LONG_LONG, all_id, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                for(i = 0; i < nimf_total; i++) {
                    fprintf(FdIMFinfo, "%.16g %g %g %g %lld %.4f %.4f %g %g %.6e\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_id[i],
                        all_mdrawn[i], all_macc[i], all_racc[i], all_facc[i], all_zbirth[i]);
                }
                fflush(FdIMFinfo);
                myfree(all_id); myfree(all_zbirth); myfree(all_facc); myfree(all_racc);
                myfree(all_macc); myfree(all_mdrawn); myfree(all_z); myfree(all_y); myfree(all_x);
                myfree(displs); myfree(recvcounts);
            }
        }
    }

    /* Free IMF logging buffers (LIFO) */
    myfree(imf_id); myfree(imf_zbirth); myfree(imf_facc); myfree(imf_racc);
    myfree(imf_macc); myfree(imf_mdrawn); myfree(imf_z); myfree(imf_y); myfree(imf_x);

    /* Free accumulators (LIFO order for mymalloc) */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    myfree(IMF_ElemAccreted); IMF_ElemAccreted = NULL;
#endif
#ifdef METALS
    myfree(IMF_MetalAccreted); IMF_MetalAccreted = NULL;
#endif
    myfree(IMF_MomAccreted); IMF_MomAccreted = NULL;
    myfree(IMF_MassAccreted); IMF_MassAccreted = NULL;

    if(ThisTask == 0) printf("IMF accretion done (%d pass%s).\n", iter, iter==1?"":"es");
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF */
