#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Resolved-ISM photo-ionization: computes Stromgren radii around massive stars
 * and marks gas cells within those radii as ionized.
 * Ported from gizmo2017/photoionize.c */

#ifdef GALSF_RESOLVEDISM_PHOTOION


/* Compute Stromgren radius for a given ionizing photon rate, local density, and electron fraction.
 * Rs = (3 * S_ly / (4 * pi * alpha_B * n_H^2))^(1/3)
 * where alpha_B = 2.6e-13 cm^3/s (case B recombination at T=10^4 K) */
static double compute_stromgren_radius_cgs(double S_ly, double nH_cgs)
{
    double alpha_B = 2.6e-13; /* case B recombination coefficient, cm^3/s */
    if(nH_cgs <= 0 || S_ly <= 0) return 0;
    double Rs = pow(3.0 * S_ly / (4.0 * M_PI * alpha_B * nH_cgs * nH_cgs), 1.0/3.0);
    return Rs; /* in cm */
}


/* Main photo-ionization routine. Called each timestep for active star particles. */
void resolvedism_photoionize(void)
{
    int i, j;

    /* First pass: reset ionization flags for all gas cells */
    for(j = 0; j < N_gas; j++) {
        if(P[j].Type != 0) continue;
#ifdef CHEMCOOL
        CellP[j].Ionized = 0;
#endif
    }

    /* Loop over star particles */
    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
        if(P[i].Mass <= 0) continue;

        double S_ly = 0; /* Lyman continuum photon rate [photons/s] */

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        double Mstar = P[i].Mstar;
        if(Mstar < 8.0) continue; /* only massive stars produce significant ionizing flux */
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        double lifetime_yr = get_lifetime(Mstar);
        if(star_age_yr > lifetime_yr) continue; /* star already exploded */
        S_ly = pow(10., get_logS_ly(Mstar));
#endif

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        if(!P[i].sampled) continue;
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        int jj;
        for(jj = 0; jj < N_STELLAR_MASS; jj++) {
            if(P[i].MstarSampleIMF[jj] < 8.0) continue;
            double lifetime_yr = get_lifetime(P[i].MstarSampleIMF[jj]);
            if(star_age_yr < lifetime_yr) {
                S_ly += pow(10., get_logS_ly(P[i].MstarSampleIMF[jj]));
            }
        }
#endif

#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        if(S_ly <= 0) continue;
        P[i].Lyman_photons_per_sec = S_ly;
#endif

        if(S_ly <= 0) continue;

        /* Estimate local density from nearest gas cell */
        double pos_i[3]; int k;
        for(k=0; k<3; k++) pos_i[k] = P[i].Pos[k];

        /* Compute Stromgren radius using a representative local density */
        double nH_local = 1.0; /* default fallback, cm^-3 */
        double min_r2 = MAX_REAL_NUMBER;
        int nearest_gas = -1;
        for(j = 0; j < N_gas; j++) {
            if(P[j].Type != 0 || P[j].Mass <= 0) continue;
            double dx = pos_i[0] - P[j].Pos[0];
            double dy = pos_i[1] - P[j].Pos[1];
            double dz = pos_i[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double r2 = dx*dx + dy*dy + dz*dz;
            if(r2 < min_r2) { min_r2 = r2; nearest_gas = j; }
        }
        if(nearest_gas >= 0) {
            nH_local = HYDROGEN_MASSFRAC * CellP[nearest_gas].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
        }

        double Rs_cgs = compute_stromgren_radius_cgs(S_ly, nH_local);
        double Rs_code = Rs_cgs / (UNIT_LENGTH_IN_CGS * All.cf_atime); /* convert to comoving code units */

        /* Mark gas cells within Stromgren radius as ionized */
        for(j = 0; j < N_gas; j++) {
            if(P[j].Type != 0 || P[j].Mass <= 0) continue;
            double dx = pos_i[0] - P[j].Pos[0];
            double dy = pos_i[1] - P[j].Pos[1];
            double dz = pos_i[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double r2 = dx*dx + dy*dy + dz*dz;
            if(r2 < Rs_code * Rs_code) {
#ifdef CHEMCOOL
                CellP[j].Ionized = 1;
#endif
            }
        }
    }
}


#endif /* GALSF_RESOLVEDISM_PHOTOION */
