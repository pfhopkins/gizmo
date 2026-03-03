#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../gravity/healpix_utils.h"

/* Resolved-ISM photo-ionization: direction-dependent Stromgren radii via HEALPix
 * angular decomposition (NSIDE=1, 12 pixels). Each pixel gets 1/12 of the ionizing
 * photon budget, walks outward through cells accumulating recombinations, and marks
 * cells inside the per-pixel ionization front as ionized.
 * Includes PI delay: stars younger than min(t_KH, t_ff) do not ionize. */

#ifdef GALSF_RESOLVEDISM_PHOTOION

#define HEALPIX_NSIDE 1
#define HEALPIX_NPIX  12

/* Compute Stromgren radius for a given ionizing photon rate and local density.
 * Rs = (3 * S_ly / (4 * pi * alpha_B * n_H^2))^(1/3)
 * where alpha_B = 2.6e-13 cm^3/s (case B recombination at T=10^4 K) */
static double compute_stromgren_radius_cgs(double S_ly, double nH_cgs)
{
    double alpha_B = 2.6e-13; /* case B recombination coefficient, cm^3/s */
    if(nH_cgs <= 0 || S_ly <= 0) return 0;
    double Rs = pow(3.0 * S_ly / (4.0 * M_PI * alpha_B * nH_cgs * nH_cgs), 1.0/3.0);
    return Rs; /* in cm */
}

struct ngb_pixel_data {
    int index;     /* gas particle index */
    double dist2;  /* squared distance to star (code units) */
};

static int compare_by_distance(const void *a, const void *b)
{
    double da = ((const struct ngb_pixel_data *)a)->dist2;
    double db = ((const struct ngb_pixel_data *)b)->dist2;
    if(da < db) return -1;
    if(da > db) return  1;
    return 0;
}


/* Main photo-ionization routine. Called each timestep for active star particles. */
void resolvedism_photoionize(void)
{
    int i, j, n, k;
    double alpha_B = 2.6e-13; /* case B recombination coefficient, cm^3/s */

    /* Phase 0: reset ionization flags for all gas cells */
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
#ifdef CHEMCOOL
        CellP[j].Ionized = 0;
#endif
    }

    /* Allocate neighbor list buffer (standard GIZMO pattern) */
    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

    /* Loop over active star particles */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
        if(P[i].Mass <= 0) continue;

        /* --- Determine most massive star and apply early checks --- */
        double M_star_max = 0;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        M_star_max = P[i].Mstar;
        if(M_star_max < 8.0) continue;
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(!P[i].sampled) continue;
        M_star_max = P[i].MstarSampleIMF[0];
        if(M_star_max < 8.0) continue;
#endif

        /* --- PI feedback delay --- */
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;

        if(P[i].FormationDensity > 0) {
            double t_KH = 1.5e7 * pow(M_star_max, -2.1); /* Kelvin-Helmholtz time [yr] */
            double rho_form_cgs = P[i].FormationDensity * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
            double t_ff_yr = sqrt(3.0 * M_PI / (32.0 * GRAVITY_G_CGS * rho_form_cgs)) / SECONDS_PER_YEAR;
            double delay = fmin(t_KH, t_ff_yr);
            if(delay > 1.0e6) delay = 1.0e6; /* cap at 1 Myr */
            if(star_age_yr < delay) continue;
        }

        /* --- Compute ionizing photon rate S_ly [photons/s] --- */
        double S_ly = 0;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        {
            double lifetime_yr = get_lifetime(P[i].Mstar);
            if(star_age_yr > lifetime_yr) continue; /* star already exploded */
            S_ly = pow(10., get_logS_ly(P[i].Mstar));
        }
#endif
#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        {
            int jj;
            for(jj = 0; jj < N_STELLAR_MASS; jj++) {
                if(P[i].MstarSampleIMF[jj] < 8.0) continue;
                double lifetime_yr = get_lifetime(P[i].MstarSampleIMF[jj]);
                if(star_age_yr < lifetime_yr) {
                    S_ly += pow(10., get_logS_ly(P[i].MstarSampleIMF[jj]));
                }
            }
        }
#endif

#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        if(S_ly <= 0) continue;
        P[i].Lyman_photons_per_sec = S_ly;
#endif
        if(S_ly <= 0) continue;

        double S_pix = S_ly / HEALPIX_NPIX;

        /* --- Estimate initial search radius from local density --- */
        double nH_local = 1.0; /* fallback: 1 cm^-3 */
#ifdef DO_DENSITY_AROUND_NONGAS_PARTICLES
        if(P[i].DensityAroundParticle > 0)
            nH_local = HYDROGEN_MASSFRAC * P[i].DensityAroundParticle * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
#endif
        double Rs_cgs = compute_stromgren_radius_cgs(S_ly, nH_local);
        double h_search = 2.0 * Rs_cgs / (UNIT_LENGTH_IN_CGS * All.cf_atime);
        if(h_search <= 0) h_search = P[i].KernelRadius;
        if(h_search <= 0) h_search = All.ForceSoftening[0];

        /* --- Expanding neighbor search (tree-based) --- */
        int numngb = 0, NITER = 0;
        while(NITER < 10) {
            int startnode = All.MaxPart, dummy = 0;
            numngb = ngb_treefind_variable_targeted(P[i].Pos, h_search, -1,
                            &startnode, 0, &dummy, &dummy, 1); /* bitflag=1: gas */
            if(numngb >= 12) break;
            h_search *= 2.0;
            NITER++;
        }
        if(numngb <= 0) continue;

        /* --- Assign each neighbor to a HEALPix pixel --- */
        int pix_count[HEALPIX_NPIX];
        for(k = 0; k < HEALPIX_NPIX; k++) pix_count[k] = 0;

        /* First pass: count cells per pixel */
        for(n = 0; n < numngb; n++) {
            j = Ngblist[n];
            if(P[j].Mass <= 0) continue;
            double dx = P[i].Pos[0] - P[j].Pos[0];
            double dy = P[i].Pos[1] - P[j].Pos[1];
            double dz = P[i].Pos[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double vec[3] = {-dx, -dy, -dz}; /* direction: star -> cell */
            long ipix;
            vec2pix_ring(HEALPIX_NSIDE, vec, &ipix);
            if(ipix < 0 || ipix >= HEALPIX_NPIX) ipix = 0;
            pix_count[ipix]++;
        }

        /* Prefix sums for contiguous indexing */
        int pix_offset[HEALPIX_NPIX], total = 0;
        for(k = 0; k < HEALPIX_NPIX; k++) {
            pix_offset[k] = total;
            total += pix_count[k];
        }
        if(total <= 0) continue;

        /* Allocate contiguous pixel data array */
        struct ngb_pixel_data *pixdata = (struct ngb_pixel_data *) mymalloc("pixdata", total * sizeof(struct ngb_pixel_data));
        int pix_pos[HEALPIX_NPIX];
        for(k = 0; k < HEALPIX_NPIX; k++) pix_pos[k] = 0;

        /* Second pass: fill pixel data with index and distance */
        for(n = 0; n < numngb; n++) {
            j = Ngblist[n];
            if(P[j].Mass <= 0) continue;
            double dx = P[i].Pos[0] - P[j].Pos[0];
            double dy = P[i].Pos[1] - P[j].Pos[1];
            double dz = P[i].Pos[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double r2 = dx*dx + dy*dy + dz*dz;
            double vec[3] = {-dx, -dy, -dz};
            long ipix;
            vec2pix_ring(HEALPIX_NSIDE, vec, &ipix);
            if(ipix < 0 || ipix >= HEALPIX_NPIX) ipix = 0;
            int slot = pix_offset[ipix] + pix_pos[ipix];
            pixdata[slot].index = j;
            pixdata[slot].dist2 = r2;
            pix_pos[ipix]++;
        }

        /* Sort each pixel's cells by distance (nearest first) */
        for(k = 0; k < HEALPIX_NPIX; k++) {
            if(pix_count[k] > 1)
                qsort(&pixdata[pix_offset[k]], pix_count[k], sizeof(struct ngb_pixel_data), compare_by_distance);
        }

        /* --- Per-pixel ionization front walk --- */
        for(k = 0; k < HEALPIX_NPIX; k++) {
            double budget = S_pix;
            int m;
            for(m = 0; m < pix_count[k]; m++) {
                j = pixdata[pix_offset[k] + m].index;
                /* Recombination rate in this cell: alpha_B * X_H^2 * rho * M / m_p^2 */
                double rho_cgs = CellP[j].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
                double M_cgs   = P[j].Mass * UNIT_MASS_IN_CGS;
                double recomb  = alpha_B * HYDROGEN_MASSFRAC * HYDROGEN_MASSFRAC * rho_cgs * M_cgs
                                 / (PROTONMASS_CGS * PROTONMASS_CGS);
                budget -= recomb;
#ifdef CHEMCOOL
                CellP[j].Ionized = 1;
#endif
                if(budget < 0) break; /* ionization front is inside this cell */
            }
        }

        myfree(pixdata);
    }

    myfree(Ngblist);
}


#endif /* GALSF_RESOLVEDISM_PHOTOION */
