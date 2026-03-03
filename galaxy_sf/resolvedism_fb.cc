#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* Resolved-ISM supernova feedback: checks star particle ages against stellar lifetimes,
 * flags SN events, and injects energy into neighboring gas cells.
 * Ported from gizmo2017/stellarfeedback.c */

#ifdef GALSF_RESOLVEDISM_FB


/* Check all active star particles for SN events. Called each timestep. */
void resolvedism_determine_SNe(void)
{
    if(All.Time <= 0) return;
    int i;
    double dt, star_age_yr, Mstar, lifetime_yr;
    int n_sne_local = 0, n_sne_total = 0;

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
        if(P[i].Mass <= 0) continue;
        dt = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(i);
        if(dt <= 0) continue;

        star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9; /* convert to yr */
        if(star_age_yr <= 0) continue;

#ifdef GALSF_RESOLVEDISM_STOCHASTIC_IMF
        Mstar = P[i].Mstar; /* single massive-star mass assigned at formation */
        if(Mstar < 8.0) continue; /* only massive stars produce core-collapse SNe */
        lifetime_yr = get_lifetime(Mstar);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
        if(All.Time < All.TimeInstantSN) {lifetime_yr = 0;} /* instant SN before TimeInstantSN */
#endif
        if(star_age_yr > lifetime_yr && P[i].SNe_ThisTimeStep == 0) {
            P[i].SNe_ThisTimeStep = 1;
            n_sne_local++;
        }
#endif

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF
        /* Check each sampled star for SN. Stars explode when age > lifetime. */
        if(P[i].sampled) {
            int j;
            for(j = 0; j < N_STELLAR_MASS; j++) {
                if(P[i].MstarSampleIMF[j] <= 0) continue; /* empty slot */
                if(P[i].MstarSampleIMF[j] < 8.0) continue; /* not massive enough */
                lifetime_yr = get_lifetime(P[i].MstarSampleIMF[j]);
#ifdef GALSF_RESOLVEDISM_INSTANT_SN
                if(All.Time < All.TimeInstantSN) {lifetime_yr = 0;} /* instant SN before TimeInstantSN */
#endif
                if(star_age_yr > lifetime_yr) {
                    P[i].SNe_ThisTimeStep++;
                    P[i].MstarSampleIMF[j] = 0; /* remove exploded star */
                    n_sne_local++;
                }
            }
        }
#endif
    }

    MPI_Reduce(&n_sne_local, &n_sne_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && n_sne_total > 0) {
        printf("ResolvedISM: %d SNe this timestep at t=%g\n", n_sne_total, All.Time);
        fflush(stdout);
    }
}


/* Inject SN energy into neighboring gas cells.
 * Simple kernel-weighted energy injection: E_SN = 10^51 erg per SN.
 * Uses the gas density kernel for weighting. */
void resolvedism_inject_sn_energy(void)
{
    int i, j;
    double E_SN_cgs = 1.0e51; /* erg per SN */
    double E_SN_code = E_SN_cgs / UNIT_ENERGY_IN_CGS; /* convert to code units */

    for(i = FirstActiveParticle; i >= 0; i = NextActiveParticle[i])
    {
        if(P[i].Type != 4) continue;
        if(P[i].SNe_ThisTimeStep <= 0) continue;

        double E_total = P[i].SNe_ThisTimeStep * E_SN_code;
        double pos_i[3]; int k;
        for(k=0; k<3; k++) pos_i[k] = P[i].Pos[k];

        /* Use the star's kernel radius to find neighbors */
        double h_i = P[i].KernelRadius;
        if(h_i <= 0) h_i = Get_Particle_Size(i) * All.cf_atime * 5.0; /* fallback */

        /* Loop over gas particles and inject energy kernel-weighted */
        double wt_sum = 0;
        for(j = 0; j < N_gas; j++) {
            if(P[j].Type != 0 || P[j].Mass <= 0) continue;
            double dx = pos_i[0] - P[j].Pos[0];
            double dy = pos_i[1] - P[j].Pos[1];
            double dz = pos_i[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double r2 = dx*dx + dy*dy + dz*dz;
            if(r2 >= h_i*h_i) continue;
            double r = sqrt(r2);
            double u = r / h_i;
            double hinv = 1.0/h_i, hinv3 = hinv*hinv*hinv;
            double wk = 0, dwk = 0;
            kernel_main(u, hinv3, hinv3*hinv, &wk, &dwk, 0);
            double wt = P[j].Mass * wk / CellP[j].Density;
            wt_sum += wt;
        }

        if(wt_sum <= 0) continue;

        /* Second pass: inject energy */
        for(j = 0; j < N_gas; j++) {
            if(P[j].Type != 0 || P[j].Mass <= 0) continue;
            double dx = pos_i[0] - P[j].Pos[0];
            double dy = pos_i[1] - P[j].Pos[1];
            double dz = pos_i[2] - P[j].Pos[2];
#ifdef BOX_PERIODIC
            NEAREST_XYZ(dx,dy,dz,1);
#endif
            double r2 = dx*dx + dy*dy + dz*dz;
            if(r2 >= h_i*h_i) continue;
            double r = sqrt(r2);
            double u = r / h_i;
            double hinv = 1.0/h_i, hinv3 = hinv*hinv*hinv;
            double wk = 0, dwk = 0;
            kernel_main(u, hinv3, hinv3*hinv, &wk, &dwk, 0);
            double wt = P[j].Mass * wk / CellP[j].Density;
            double dE = E_total * wt / wt_sum;
            CellP[j].InternalEnergy += dE / P[j].Mass;
            CellP[j].InternalEnergyPred = CellP[j].InternalEnergy;
        }

        P[i].SNe_ThisTimeStep = 0; /* reset flag */
    }
}


#endif /* GALSF_RESOLVEDISM_FB */
