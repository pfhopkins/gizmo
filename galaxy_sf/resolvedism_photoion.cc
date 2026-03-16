#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../gravity/healpix_utils.h"

/* Resolved-ISM photo-ionization for individually sampled single stars.
 * Each star particle represents ONE star drawn from the Kroupa IMF.
 * If the single star is massive enough (>=8 Msun) and still alive, it
 * ionizes surrounding gas using direction-dependent Stromgren radii via
 * HEALPix angular decomposition (NSIDE=1, 12 pixels). Each pixel gets
 * 1/12 of the ionizing photon budget and walks outward through cells.
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks. */

#ifdef GALSF_RESOLVEDISM_PHOTOION

#define HEALPIX_NSIDE 1
#define HEALPIX_NPIX  12
#define MAX_PI_NGBS 4096

/* Stromgren radius: Rs = (3 S_ly / (4 pi alpha_B n_H^2))^(1/3) */
static double compute_stromgren_radius_cgs(double S_ly, double nH_cgs)
{
    double alpha_B = 2.6e-13; /* case B recombination coefficient, cm^3/s */
    if(nH_cgs <= 0 || S_ly <= 0) return 0;
    return pow(3.0 * S_ly / (4.0 * M_PI * alpha_B * nH_cgs * nH_cgs), 1.0/3.0);
}

/* Neighbor entry for collecting and sorting by HEALPix pixel then distance */
struct pi_ngb_entry { int index; double dist2; int pixel; };

static int compare_pix_then_dist(const void *a, const void *b)
{
    const struct pi_ngb_entry *ea = (const struct pi_ngb_entry *)a;
    const struct pi_ngb_entry *eb = (const struct pi_ngb_entry *)b;
    if(ea->pixel != eb->pixel) return ea->pixel - eb->pixel;
    if(ea->dist2 < eb->dist2) return -1;
    if(ea->dist2 > eb->dist2) return  1;
    return 0;
}

/* Per-star search radius override and budget tracking for iterative expansion.
 * Allocated in resolvedism_photoionize(), used in particle2in/evaluate/out2particle. */
static double *PISearchRadius = NULL;
static double *PIPixelBudget = NULL;  /* per-star per-pixel remaining photon budget [NumPart * HEALPIX_NPIX] */

/* Compute S_ly for a single star (used in particle2in and pre-processing) */
static double compute_single_star_S_ly(int i)
{
    double S_ly = 0, Mstar = 0;
    double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
    if(star_age_yr <= 0) return 0;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    Mstar = P[i].Mstar;
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(!P[i].sampled) return 0;
    Mstar = P[i].MstarSampleIMF[0]; /* single star */
#endif
    if(Mstar < 8.0 || Mstar <= 0) return 0;
    if(star_age_yr >= get_lifetime(Mstar, P[i].BirthMetallicity)) return 0; /* star already dead */

    /* PI feedback delay: stars younger than min(t_KH, t_ff) do not ionize */
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
    if(P[i].FormationDensity > 0) {
        double t_KH = 1.5e7 * pow(Mstar, -2.1); /* Kelvin-Helmholtz time [yr] */
        double rho_form_cgs = P[i].FormationDensity * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
        double t_ff_yr = sqrt(3.0 * M_PI / (32.0 * GRAVITY_G_CGS * rho_form_cgs)) / SECONDS_PER_YEAR;
        double delay = fmin(t_KH, t_ff_yr);
        if(delay > 1.0e6) delay = 1.0e6; /* cap at 1 Myr */
        if(star_age_yr < delay) return 0;
    }
#endif

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    {
        double logM = log10(Mstar);
        double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
        double log_age = log10(DMAX(star_age_yr, 100.0));
        S_ly = pow(10., stellar_log_Q_ion(logM, logZ, log_age));
    }
#else
    S_ly = pow(10., get_logS_ly(Mstar));
#endif
    return S_ly;
}


/* =========================================================================== */
/*  HEALPix photo-ionization via code_block_xchange framework                 */
/* =========================================================================== */

#define CORE_FUNCTION_NAME resolvedismPI_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismPI
#define OUTPUTFUNCTION_NAME out2particle_resolvedismPI
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismPI_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], SearchRadius, S_ly;
    MyDouble budget_per_pixel[HEALPIX_NPIX]; /* per-pixel remaining photon budget, shared between mode 0 and mode 1 */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismPI(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->S_ly = compute_single_star_S_ly(i);

    /* Per-pixel photon budget: read from PIPixelBudget which is initialized to S_ly/NPIX
     * before mode 0, then updated with remaining budget after mode 0 for export packing */
    for(k = 0; k < HEALPIX_NPIX; k++)
        in->budget_per_pixel[k] = (PIPixelBudget != NULL) ? PIPixelBudget[i * HEALPIX_NPIX + k] : ((in->S_ly > 0) ? in->S_ly / HEALPIX_NPIX : 0);

    /* Use stored search radius from iterative expansion if available,
     * otherwise compute from 2x Stromgren radius estimate */
    if(PISearchRadius != NULL && PISearchRadius[i] > 0) {
        in->SearchRadius = PISearchRadius[i];
    } else {
        in->SearchRadius = P[i].KernelRadius;
        if(in->S_ly > 0) {
            double nH_local = 1.0; /* fallback: 1 cm^-3 */
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
            if(P[i].DensityAroundParticle > 0)
                nH_local = HYDROGEN_MASSFRAC * P[i].DensityAroundParticle * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
#endif
            double Rs_cgs = compute_stromgren_radius_cgs(in->S_ly, nH_local);
            double h_stromgren = 2.0 * Rs_cgs / (UNIT_LENGTH_IN_CGS * All.cf_atime);
            if(h_stromgren > in->SearchRadius) in->SearchRadius = h_stromgren;
        }
        if(in->SearchRadius <= 0) in->SearchRadius = All.ForceSoftening[P[i].Type];
        /* Cap at MaxPISearchRadius (set in parameter file, in kpc) */
        double max_r = All.MaxPISearchRadius / UNIT_LENGTH_IN_KPC;
        if(max_r > 0 && in->SearchRadius > max_r) in->SearchRadius = max_r;
        /* Store the initial estimate for potential iteration */
        if(PISearchRadius != NULL) PISearchRadius[i] = in->SearchRadius;
    }
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat consumed_per_pixel[HEALPIX_NPIX]; /* per-pixel consumed photon budget */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismPI(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(PIPixelBudget == NULL) return;
    int k;
    for(k = 0; k < HEALPIX_NPIX; k++)
        PIPixelBudget[i * HEALPIX_NPIX + k] -= out->consumed_per_pixel[k];
}

int resolvedismPI_active_check(int i);
int resolvedismPI_active_check(int i)
{
    if(P[i].Type != 4) {return 0;}
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
    if(!P[i].sampled) {return 0;}
    if(P[i].MstarSampleIMF[0] < 8.0) {return 0;} /* single star must be massive to ionize */
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
    if(P[i].Mstar < 8.0) {return 0;}
#endif
    return 1;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
int resolvedismPI_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismPI(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.S_ly <= 0) return 0;
    if(local.SearchRadius <= 0) return 0;

    double h2 = local.SearchRadius * local.SearchRadius;

    /* Phase 1: Collect all gas neighbors from tree walk */
    struct pi_ngb_entry ngb_buf[MAX_PI_NGBS];
    int n_collected = 0;

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_pairs_threads(local.Pos, local.SearchRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) {return -2;}
            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;
                if(P[j].Mass <= 0) continue;

                double dp[3];
                for(k=0;k<3;k++) {dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(dp[0],dp[1],dp[2],1);
                double r2 = dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2];
                if(r2 <= 0 || r2 >= h2) continue;

                /* Direction vector star -> cell, assign to HEALPix pixel */
                double vec[3] = {-dp[0], -dp[1], -dp[2]};
                long ipix;
                vec2pix_ring(HEALPIX_NSIDE, vec, &ipix);
                if(ipix < 0 || ipix >= HEALPIX_NPIX) ipix = 0;

                if(n_collected < MAX_PI_NGBS) {
                    ngb_buf[n_collected].index = j;
                    ngb_buf[n_collected].dist2 = r2;
                    ngb_buf[n_collected].pixel = (int)ipix;
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

    if(n_collected == 0) goto done;

    /* Phase 2: Sort by pixel then distance (nearest first per pixel) */
    qsort(ngb_buf, n_collected, sizeof(struct pi_ngb_entry), compare_pix_then_dist);

    /* Phase 3: Per-pixel ionization front walk.
     * budget_per_pixel carries remaining budget from mode 0 to mode 1,
     * preventing double-counting of photons at domain boundaries. */
    {
        double alpha_B = 2.6e-13; /* case B recombination coefficient */
        int idx = 0;
        for(k = 0; k < HEALPIX_NPIX; k++)
        {
            double budget = local.budget_per_pixel[k];
            double consumed = 0;
            int front_found = 0;
            if(budget <= 0) { /* no budget remaining (front found in previous mode) */
                out.consumed_per_pixel[k] = 0;
                while(idx < n_collected && ngb_buf[idx].pixel == k) idx++;
                continue;
            }
            while(idx < n_collected && ngb_buf[idx].pixel == k)
            {
                j = ngb_buf[idx].index;
                double rho_cgs = CellP[j].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
                double M_cgs   = P[j].Mass * UNIT_MASS_IN_CGS;
                double recomb  = alpha_B * HYDROGEN_MASSFRAC * HYDROGEN_MASSFRAC
                                 * rho_cgs * M_cgs / (PROTONMASS_CGS * PROTONMASS_CGS);
                budget -= recomb;
                consumed += recomb;
#ifdef GALSF_RESOLVEDISM_PHOTOION
                CellP[j].Ionized = 1;
#endif
                idx++;
                if(budget < 10.0 * recomb) {front_found = 1; break;} /* ionization front reached */
            }
            if(front_found)
                out.consumed_per_pixel[k] = local.budget_per_pixel[k]; /* consume all: front found */
            else
                out.consumed_per_pixel[k] = consumed; /* report actual consumption only */
            /* skip remaining cells in this pixel (beyond ionization front) */
            while(idx < n_collected && ngb_buf[idx].pixel == k) idx++;
        }
    }

done:
    if(mode == 0) {out2particle_resolvedismPI(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void resolvedism_photoionize(void)
{
    int i, j;

    /* Reset ionization flags for all gas cells */
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type == 0) {
#ifdef GALSF_RESOLVEDISM_PHOTOION
            CellP[j].Ionized = 0;
#endif
        }
    }

    /* Pre-compute Lyman photon rates for active stars (for use by other routines) */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && defined(GALSF_RESOLVEDISM_PHOTOION)
        P[i].Lyman_photons_per_sec = compute_single_star_S_ly(i);
#endif
    }

    /* Allocate per-star arrays for iterative search radius adjustment */
    PISearchRadius = (double *) mymalloc("PISearchRadius", NumPart * sizeof(double));
    PIPixelBudget = (double *) mymalloc("PIPixelBudget", (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    memset(PISearchRadius, 0, NumPart * sizeof(double));
    memset(PIPixelBudget, 0, (size_t)NumPart * HEALPIX_NPIX * sizeof(double));

    /* Initialize per-pixel budgets for active ionizing stars */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
        if(P[i].Type != 4) continue;
        double S_ly = compute_single_star_S_ly(i);
        if(S_ly <= 0) continue;
        double S_pix = S_ly / HEALPIX_NPIX;
        int k;
        for(k = 0; k < HEALPIX_NPIX; k++)
            PIPixelBudget[i * HEALPIX_NPIX + k] = S_pix;
    }

    /* Iterative xchange: expand search radius if any pixel's ionization front
     * extends beyond the current radius (budget remaining after all cells walked).
     * The initial 2x Stromgren estimate is usually sufficient; iteration is a safety net. */
    PRINT_STATUS(" ..photo-ionizing gas around single stars");
    int iter = 0, max_iter = 20;
    int incomplete_total;
    do {
        {
            #include "../system/code_block_xchange_perform_ops_malloc.h"
            #include "../system/code_block_xchange_perform_ops.h"
            #include "../system/code_block_xchange_perform_ops_demalloc.h"
        }

        /* Check for stars whose search radius was too small (any pixel has remaining budget) */
        int incomplete_local = 0;
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(PISearchRadius[i] <= 0) continue;
            int k; double budget_remaining = 0;
            for(k = 0; k < HEALPIX_NPIX; k++)
                if(PIPixelBudget[i * HEALPIX_NPIX + k] > 0) budget_remaining += PIPixelBudget[i * HEALPIX_NPIX + k];
            if(budget_remaining > 0) {
                PISearchRadius[i] *= 1.1; /* expand by 10% */
                /* Re-initialize per-pixel budget for re-run with larger search radius */
                double S_ly = compute_single_star_S_ly(i);
                double S_pix = (S_ly > 0) ? S_ly / HEALPIX_NPIX : 0;
                for(k = 0; k < HEALPIX_NPIX; k++)
                    PIPixelBudget[i * HEALPIX_NPIX + k] = S_pix;
                incomplete_local++;
            }
        }
        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;
        if(ThisTask == 0 && incomplete_total > 0)
            printf("PI iter %d: %d stars need larger search radius, expanding\n", iter, incomplete_total);
    } while(incomplete_total > 0 && iter < max_iter);

    /* Log per-star PI diagnostics to PIinfo.txt */
    {
        int npi_local = 0;
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(PISearchRadius[i] <= 0) continue;
            npi_local++;
        }
        int npi_total = 0;
        MPI_Allreduce(&npi_local, &npi_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if(ThisTask == 0)
            printf("ResolvedISM PI: %d ionizing stars this timestep at t=%g (%d iterations)\n", npi_total, All.Time, iter);

        if(npi_total > 0) {
            double *pi_x = (double *)mymalloc("pi_x", DMAX(npi_local,1) * sizeof(double));
            double *pi_y = (double *)mymalloc("pi_y", DMAX(npi_local,1) * sizeof(double));
            double *pi_z = (double *)mymalloc("pi_z", DMAX(npi_local,1) * sizeof(double));
            double *pi_mstar = (double *)mymalloc("pi_mstar", DMAX(npi_local,1) * sizeof(double));
            double *pi_sly = (double *)mymalloc("pi_sly", DMAX(npi_local,1) * sizeof(double));
            double *pi_rsearch = (double *)mymalloc("pi_rsearch", DMAX(npi_local,1) * sizeof(double));
            double *pi_frac = (double *)mymalloc("pi_frac", DMAX(npi_local,1) * sizeof(double));
            int *pi_npix = (int *)mymalloc("pi_npix", DMAX(npi_local,1) * sizeof(int));
            long long *pi_id = (long long *)mymalloc("pi_id", DMAX(npi_local,1) * sizeof(long long));
            int npi = 0;
            for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
                if(P[i].Type != 4) continue;
                if(PISearchRadius[i] <= 0) continue;
                double S_ly = compute_single_star_S_ly(i);
                double S_initial = S_ly; /* total budget at start */
                double budget_remaining = 0;
                int npix_complete = 0;
                int k;
                for(k = 0; k < HEALPIX_NPIX; k++) {
                    double rem = PIPixelBudget[i * HEALPIX_NPIX + k];
                    if(rem <= 0) npix_complete++;
                    else budget_remaining += rem;
                }
                double frac_consumed = (S_initial > 0) ? 1.0 - budget_remaining / S_initial : 0;
                double Mstar = 0;
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
                Mstar = P[i].MstarSampleIMF[0];
#endif
#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
                Mstar = P[i].Mstar;
#endif
                pi_x[npi] = P[i].Pos[0];
                pi_y[npi] = P[i].Pos[1];
                pi_z[npi] = P[i].Pos[2];
                pi_mstar[npi] = Mstar;
                pi_sly[npi] = (S_ly > 0) ? log10(S_ly) : 0;
                pi_rsearch[npi] = PISearchRadius[i];
                pi_frac[npi] = frac_consumed;
                pi_npix[npi] = npix_complete;
                pi_id[npi] = (long long)P[i].ID;
                npi++;
            }
            /* Gather on rank 0 */
            int *recvcounts = NULL, *displs = NULL;
            double *all_x=NULL, *all_y=NULL, *all_z=NULL, *all_mstar=NULL, *all_sly=NULL, *all_rsearch=NULL, *all_frac=NULL;
            int *all_npix = NULL;
            long long *all_id = NULL;
            if(ThisTask == 0) {
                recvcounts = (int *)mymalloc("recvcounts_pi", NTask * sizeof(int));
                displs = (int *)mymalloc("displs_pi", NTask * sizeof(int));
            }
            MPI_Gather(&npi, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                displs[0] = 0;
                for(i = 1; i < NTask; i++) {displs[i] = displs[i-1] + recvcounts[i-1];}
                all_x = (double *)mymalloc("api_x", npi_total * sizeof(double));
                all_y = (double *)mymalloc("api_y", npi_total * sizeof(double));
                all_z = (double *)mymalloc("api_z", npi_total * sizeof(double));
                all_mstar = (double *)mymalloc("api_ms", npi_total * sizeof(double));
                all_sly = (double *)mymalloc("api_sl", npi_total * sizeof(double));
                all_rsearch = (double *)mymalloc("api_rs", npi_total * sizeof(double));
                all_frac = (double *)mymalloc("api_fr", npi_total * sizeof(double));
                all_npix = (int *)mymalloc("api_np", npi_total * sizeof(int));
                all_id = (long long *)mymalloc("api_id", npi_total * sizeof(long long));
            }
            MPI_Gatherv(pi_x, npi, MPI_DOUBLE, all_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_y, npi, MPI_DOUBLE, all_y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_z, npi, MPI_DOUBLE, all_z, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_mstar, npi, MPI_DOUBLE, all_mstar, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_sly, npi, MPI_DOUBLE, all_sly, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_rsearch, npi, MPI_DOUBLE, all_rsearch, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_frac, npi, MPI_DOUBLE, all_frac, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_npix, npi, MPI_INT, all_npix, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Gatherv(pi_id, npi, MPI_LONG_LONG, all_id, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                for(i = 0; i < npi_total; i++) {
                    fprintf(FdPIinfo, "%.16g %g %g %g %lld %.2f %.3f %g %.4f %d\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_id[i],
                        all_mstar[i], all_sly[i], all_rsearch[i], all_frac[i], all_npix[i]);
                }
                fflush(FdPIinfo);
                myfree(all_id); myfree(all_npix); myfree(all_frac); myfree(all_rsearch);
                myfree(all_sly); myfree(all_mstar); myfree(all_z); myfree(all_y); myfree(all_x);
                myfree(displs); myfree(recvcounts);
            }
            myfree(pi_id); myfree(pi_npix); myfree(pi_frac); myfree(pi_rsearch);
            myfree(pi_sly); myfree(pi_mstar); myfree(pi_z); myfree(pi_y); myfree(pi_x);
        }
    }

    /* Free in LIFO order */
    myfree(PIPixelBudget); PIPixelBudget = NULL;
    myfree(PISearchRadius); PISearchRadius = NULL;
}
#include "../system/code_block_xchange_finalize.h"


#endif /* GALSF_RESOLVEDISM_PHOTOION */
