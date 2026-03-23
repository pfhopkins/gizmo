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
 * Ionization state machine (following gizmo2017 photoionize.c):
 *   Ionized=0 : not ionized
 *   Ionized=1 : confirmed ionized (persists from previous step)
 *   Ionized=2 : newly ionized THIS step (set by evaluate)
 * Global reset each step: 1->0, then evaluate marks 2, then 2->1.
 * This ensures dead/inactive stars' HII regions recombine naturally.
 *
 * Convergence tolerance (from gizmo2017): per-pixel budget walk stops
 * when remaining budget < tolerance, where tolerance scales with a
 * reference cell's recombination cost. This avoids chasing low-density
 * halo gas that is already UV-background ionized.
 *
 * Uses the code_block_xchange framework for MPI-parallel tree walks. */

#ifdef GALSF_RESOLVEDISM_PHOTOION

#define HEALPIX_NSIDE 1
#define HEALPIX_NPIX  12
#define MAX_PI_NGBS 4096

/* case B recombination coefficient at T=10^4 K [cm^3/s] */
#define PI_ALPHA_B 2.6e-13

/* Stromgren radius: Rs = (3 S_ly / (4 pi alpha_B n_H^2))^(1/3) */
static double compute_stromgren_radius_cgs(double S_ly, double nH_cgs)
{
    if(nH_cgs <= 0 || S_ly <= 0) return 0;
    return pow(3.0 * S_ly / (4.0 * M_PI * PI_ALPHA_B * nH_cgs * nH_cgs), 1.0/3.0);
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
static double *PIFrontRadius = NULL;  /* per-star per-pixel ionization front distance [code units, NumPart * HEALPIX_NPIX] */

/* Convergence tolerance: per-pixel budget walk stops when remaining budget
 * drops below this. Scaled to ~1000 reference cells' recombination cost
 * (following gizmo2017). Computed once per step in resolvedism_photoionize(). */
static double PITolerance = 0;

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
    if(Mstar < 5.0 || Mstar <= 0) return 0; /* early B-type and above have significant ionizing flux */
    if(star_age_yr >= get_lifetime(Mstar, P[i].BirthMetallicity)) return 0; /* star already dead */

    /* PI feedback delay: stars younger than min(t_KH, t_ff) do not ionize */
#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF)
    if(P[i].FormationDensity > 0) {
        double t_KH = 1.5e7 * pow(Mstar, -2.1); /* Kelvin-Helmholtz time [yr] */
        double rho_form_cgs = P[i].FormationDensity * UNIT_DENSITY_IN_CGS; /* already physical at formation */
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
     * otherwise compute from Stromgren radius estimate */
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
            double h_stromgren = 1.0 * Rs_cgs / (UNIT_LENGTH_IN_CGS * All.cf_atime);
            if(h_stromgren > in->SearchRadius) in->SearchRadius = h_stromgren;
        }
        if(in->SearchRadius <= 0) in->SearchRadius = All.ForceSoftening[P[i].Type];
        /* Cap at MaxPISearchRadius (set in parameter file, in kpc) */
        double max_r = All.MaxPISearchRadius / (UNIT_LENGTH_IN_KPC * All.cf_atime); /* physical kpc -> comoving code */
        if(max_r > 0 && in->SearchRadius > max_r) in->SearchRadius = max_r;
        /* Store the initial estimate for potential iteration */
        if(PISearchRadius != NULL) PISearchRadius[i] = in->SearchRadius;
    }
}

struct OUTPUT_STRUCT_NAME
{
    MyFloat consumed_per_pixel[HEALPIX_NPIX]; /* per-pixel consumed photon budget */
    MyFloat front_radius[HEALPIX_NPIX]; /* per-pixel distance to ionization front [code units] */
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismPI(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    if(PIPixelBudget == NULL) return;
    int k;
    for(k = 0; k < HEALPIX_NPIX; k++) {
        PIPixelBudget[i * HEALPIX_NPIX + k] -= out->consumed_per_pixel[k];
        if(PIFrontRadius != NULL && out->front_radius[k] > PIFrontRadius[i * HEALPIX_NPIX + k])
            PIFrontRadius[i * HEALPIX_NPIX + k] = out->front_radius[k]; /* take max across mode 0/1 */
    }
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
     * preventing double-counting of photons at domain boundaries.
     * Tolerance (from gizmo2017): stop when remaining budget is small
     * relative to a reference cell's recombination cost. This prevents
     * draining budget into low-density halo gas. */
    {
        double tolerance_per_pixel = PITolerance / HEALPIX_NPIX;
        int idx = 0;
        for(k = 0; k < HEALPIX_NPIX; k++)
        {
            double budget = local.budget_per_pixel[k];
            double consumed = 0;
            int front_found = 0;
            double last_r = 0;
            if(budget <= 0) { /* no budget remaining (front found in previous mode) */
                out.consumed_per_pixel[k] = 0;
                out.front_radius[k] = 0;
                while(idx < n_collected && ngb_buf[idx].pixel == k) idx++;
                continue;
            }
            while(idx < n_collected && ngb_buf[idx].pixel == k)
            {
                j = ngb_buf[idx].index;
                if(CellP[j].Ionized == 2) {idx++; continue;} /* already ionized this step by another star — skip */
                double rho_cgs = CellP[j].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
                double M_cgs   = P[j].Mass * UNIT_MASS_IN_CGS;
                double recomb  = PI_ALPHA_B * HYDROGEN_MASSFRAC * HYDROGEN_MASSFRAC
                                 * rho_cgs * M_cgs / (PROTONMASS_CGS * PROTONMASS_CGS);
                budget -= recomb;
                consumed += recomb;
                last_r = sqrt(ngb_buf[idx].dist2);
                CellP[j].Ionized = 2; /* mark as ionized THIS step */
                idx++;
                /* Ionization front found: either budget exhausted relative to
                 * next cell cost, or remaining budget below tolerance */
                if(budget < 10.0 * recomb || budget < tolerance_per_pixel) {front_found = 1; break;}
            }
            if(front_found)
                out.consumed_per_pixel[k] = local.budget_per_pixel[k]; /* consume all: front found */
            else
                out.consumed_per_pixel[k] = consumed; /* report actual consumption only */
            out.front_radius[k] = last_r;
            /* Skip remaining cells in this pixel beyond the front */
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

    /* === Global reset (gizmo2017 pattern) ===
     * Clear all previously-ionized cells. The evaluate will re-mark cells
     * inside active HII regions as Ionized=2. After the evaluate, we
     * transition 2->1. Cells not re-ionized (dead/inactive stars) revert
     * to Ionized=0 and recombine naturally via CHEMCOOL. */
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
        if(CellP[j].Ionized == 1) CellP[j].Ionized = 0;
    }

    /* Pre-compute Lyman photon rates for active stars (for use by other routines) */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
#if defined(GALSF_RESOLVEDISM_G0_VARIABLE) && defined(GALSF_RESOLVEDISM_PHOTOION)
        P[i].Lyman_photons_per_sec = compute_single_star_S_ly(i);
#endif
    }

    /* Compute convergence tolerance (gizmo2017 pattern):
     * tolerance = alpha_B * 1000 * N_H_per_reference_cell
     * where N_H = M_cell * X_H / m_p for the target cell mass.
     * This is generous at low density, preventing over-ionization of halo gas. */
    {
        double M_ref_cgs = All.MassTable[0] * UNIT_MASS_IN_CGS; /* reference cell mass from IC */
        PITolerance = PI_ALPHA_B * 1000.0 * (M_ref_cgs * HYDROGEN_MASSFRAC / PROTONMASS_CGS);
    }

    /* Allocate per-star arrays for iterative search radius adjustment */
    PISearchRadius = (double *) mymalloc("PISearchRadius", NumPart * sizeof(double));
    PIPixelBudget = (double *) mymalloc("PIPixelBudget", (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    PIFrontRadius = (double *) mymalloc("PIFrontRadius", (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    memset(PISearchRadius, 0, NumPart * sizeof(double));
    memset(PIPixelBudget, 0, (size_t)NumPart * HEALPIX_NPIX * sizeof(double));
    memset(PIFrontRadius, 0, (size_t)NumPart * HEALPIX_NPIX * sizeof(double));

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
     * Budget remaining below tolerance counts as converged (gizmo2017 pattern).
     * The initial Stromgren estimate is usually sufficient; iteration is a safety net. */
    PRINT_STATUS(" ..photo-ionizing gas around single stars");
    int iter = 0, max_iter = 20;
    int incomplete_total;
    do {
        {
            #include "../system/code_block_xchange_perform_ops_malloc.h"
            #include "../system/code_block_xchange_perform_ops.h"
            #include "../system/code_block_xchange_perform_ops_demalloc.h"
        }

        /* Check for stars whose search radius was too small (any pixel has remaining budget > tolerance) */
        int incomplete_local = 0;
        double tol_pix = PITolerance / HEALPIX_NPIX;
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(PISearchRadius[i] <= 0) continue;
            int k; int needs_expansion = 0;
            for(k = 0; k < HEALPIX_NPIX; k++)
                if(PIPixelBudget[i * HEALPIX_NPIX + k] > tol_pix) {needs_expansion = 1; break;}
            if(needs_expansion) {
                double max_r = All.MaxPISearchRadius / (UNIT_LENGTH_IN_KPC * All.cf_atime); /* physical kpc -> comoving code */
                if(max_r > 0 && PISearchRadius[i] >= max_r) {continue;} /* hit cap, stop expanding */
                PISearchRadius[i] *= 1.1; /* expand by 10% */
                if(max_r > 0 && PISearchRadius[i] > max_r) PISearchRadius[i] = max_r;
                /* Re-initialize per-pixel budget and front radii for re-run with larger search radius */
                double S_ly = compute_single_star_S_ly(i);
                double S_pix = (S_ly > 0) ? S_ly / HEALPIX_NPIX : 0;
                for(k = 0; k < HEALPIX_NPIX; k++) {
                    PIPixelBudget[i * HEALPIX_NPIX + k] = S_pix;
                    PIFrontRadius[i * HEALPIX_NPIX + k] = 0;
                }
                incomplete_local++;
            }
        }
        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;
    } while(incomplete_total > 0 && iter < max_iter);
    if(iter > 1) {
        double local_max_r = 0, local_min_r = MAX_REAL_NUMBER;
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type == 4 && PISearchRadius != NULL && PISearchRadius[i] > 0) {
                if(PISearchRadius[i] > local_max_r) local_max_r = PISearchRadius[i];
                if(PISearchRadius[i] < local_min_r) local_min_r = PISearchRadius[i];
            }
        }
        if(local_min_r >= MAX_REAL_NUMBER) local_min_r = 0;
        double glob_max_r = 0, glob_min_r = 0;
        MPI_Reduce(&local_max_r, &glob_max_r, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min_r, &glob_min_r, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("RESOLVEDISM PI: expanded search radius in %d iterations (R_min=%.1f pc, R_max=%.1f pc, cap=%.1f pc)\n",
                iter, glob_min_r * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_max_r * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, All.MaxPISearchRadius * 1e3);
    }

    /* HII region size summary: per-star min/max pixel front radius (asymmetry measure) */
    {
        int npi_local = 0;
        double local_rfront_min = MAX_REAL_NUMBER, local_rfront_max = 0;
        double local_asym_max = 0;
        for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i]) {
            if(P[i].Type != 4) continue;
            if(PISearchRadius[i] <= 0) continue;
            npi_local++;
            double star_rmin = MAX_REAL_NUMBER, star_rmax = 0;
            int k, npix_active = 0;
            for(k = 0; k < HEALPIX_NPIX; k++) {
                double rf = PIFrontRadius[i * HEALPIX_NPIX + k];
                if(rf > 0) {
                    npix_active++;
                    if(rf > star_rmax) star_rmax = rf;
                    if(rf < star_rmin && rf > 0) star_rmin = rf;
                }
            }
            if(npix_active > 0) {
                if(star_rmin < local_rfront_min) local_rfront_min = star_rmin;
                if(star_rmax > local_rfront_max) local_rfront_max = star_rmax;
                double asym = (star_rmin > 0) ? star_rmax / star_rmin : 0;
                if(asym > local_asym_max) local_asym_max = asym;
            }
        }
        if(local_rfront_min >= MAX_REAL_NUMBER) local_rfront_min = 0;

        int npi_total = 0;
        double glob_rfront_min = 0, glob_rfront_max = 0, glob_asym_max = 0;
        MPI_Allreduce(&npi_local, &npi_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Reduce(&local_rfront_min, &glob_rfront_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_rfront_max, &glob_rfront_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_asym_max, &glob_asym_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(ThisTask == 0 && npi_total > 0) {
            printf("RESOLVEDISM PI: %d ionizing stars at t=%g (%d iter)\n", npi_total, All.Time, iter);
            printf("RESOLVEDISM PI: HII front R_min=%.1f pc, R_max=%.1f pc, max asymmetry=%.1f:1\n",
                glob_rfront_min * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_rfront_max * All.cf_atime * UNIT_LENGTH_IN_KPC * 1e3, glob_asym_max);
        }
    }

    /* Post-evaluate ionization state transitions (gizmo2017 pattern):
     * The global reset already cleared 1->0 at the top.
     * Now transition 2->1 (confirm newly ionized) and wake them.
     * Cells not re-ionized remain at 0 and recombine via CHEMCOOL. */
    int sum_ionized = 0, n_woken = 0;
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
        if(CellP[j].Ionized == 2) {
            CellP[j].Ionized = 1;
            P[j].wakeup = 1;
            NeedToWakeupParticles_local = 1;
            n_woken++;
        }
        if(CellP[j].Ionized == 1)
            sum_ionized++;
    }
    {
        int total_ionized = 0, total_woken = 0;
        MPI_Reduce(&sum_ionized, &total_ionized, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&n_woken, &total_woken, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(ThisTask == 0)
            printf("RESOLVEDISM PI: %d ionized, %d woken\n", total_ionized, total_woken);
    }

    /* Free in LIFO order */
    myfree(PIFrontRadius); PIFrontRadius = NULL;
    myfree(PIPixelBudget); PIPixelBudget = NULL;
    myfree(PISearchRadius); PISearchRadius = NULL;
}
#include "../system/code_block_xchange_finalize.h"


#endif /* GALSF_RESOLVEDISM_PHOTOION */
