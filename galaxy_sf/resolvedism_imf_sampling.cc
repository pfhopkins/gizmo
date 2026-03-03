#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Single-star IMF sampling: draws one stellar mass from a Kroupa IMF for each
 * new star particle. Mass difference from the drawn value is borrowed from or
 * returned to neighboring gas cells via tree neighbor search.
 * Mass and momentum conserving: search radius expands until enough gas is found,
 * and velocities are updated to conserve total momentum during mass transfer. */

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF

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


/* Main entry: each rank processes its own unsampled stars, using tree neighbor
 * search to borrow/return mass from/to nearby gas cells. Not OpenMP-parallel
 * (number of unsampled stars per timestep is small). */
void assign_stellar_masses(void)
{
    int i, j, n, k;

    /* Count unsampled stars across all ranks */
    int nstars_local = 0;
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 4 && P[i].sampled == 0) nstars_local++;

    int nstars_total;
    MPI_Allreduce(&nstars_local, &nstars_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(nstars_total == 0) return;

    if(ThisTask == 0) printf("IMF sampling: %d unsampled star particles\n", nstars_total);

    /* Allocate neighbor list buffer (same pattern as radfb_local.cc) */
    Ngblist = (int *) mymalloc("Ngblist", NumPart * sizeof(int));

    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 0) continue;
        if(P[i].Mass <= 0) continue;

        double M_particle_solar = P[i].Mass * UNIT_MASS_IN_SOLAR;
        double m_i_old_code = P[i].Mass; /* save original star mass in code units */

        /* === Phase 1: Draw ONE stellar mass from Kroupa IMF === */
        double M_drawn;
        do {
            M_drawn = pow((pow(M_max, -0.3) - pow(M_min, -0.3)) * gsl_rng_uniform(random_generator)
                          + pow(M_min, -0.3), -1.0 / 0.3);
        } while(gsl_rng_uniform(random_generator) > (drawMassFromIMF(M_drawn) / envelope_function(M_drawn)));

        /* Store drawn mass in MstarSampleIMF[0], zero the rest */
        for(j = 0; j < N_STELLAR_MASS; j++) P[i].MstarSampleIMF[j] = 0;
        P[i].MstarSampleIMF[0] = M_drawn;

        /* === Phase 2: Mass & momentum conserving transfer with gas neighbors === */
        double delta_solar = M_drawn - M_particle_solar;

        if(fabs(delta_solar) > All.MassTolerance)
        {
            double h_search = All.SearchingRadius;
            if(h_search <= 0) h_search = P[i].KernelRadius;
            if(h_search <= 0) h_search = All.ForceSoftening[0];

            int numngb = 0, NITER = 0, MAX_SEARCH_ITER = 30;
            int enough_mass = 0;

            /* Iteratively expand search radius until enough gas mass is found */
            while(!enough_mass && h_search > 0 && NITER < MAX_SEARCH_ITER)
            {
                int startnode = All.MaxPart, dummy = 0;
                numngb = ngb_treefind_variable_targeted(P[i].Pos, h_search, -1,
                                &startnode, 0, &dummy, &dummy, 1);

                if(numngb > 0)
                {
                    if(delta_solar < 0) {
                        enough_mass = 1; /* any neighbor works for returning mass */
                    } else {
                        double total_avail = 0;
                        for(n = 0; n < numngb; n++) {
                            j = Ngblist[n];
                            if(P[j].Mass > 0) total_avail += 0.99 * P[j].Mass * UNIT_MASS_IN_SOLAR;
                        }
                        if(total_avail >= delta_solar - All.MassTolerance) enough_mass = 1;
                    }
                }

                if(!enough_mass) {
                    h_search *= 2.0;
                    NITER++;
                }
            }

            if(NITER >= MAX_SEARCH_ITER && !enough_mass) {
                printf("Task %d: WARNING — IMF sampling for particle %d could not find enough gas after %d iterations (h=%g, delta=%g Msun)\n",
                       ThisTask, i, NITER, h_search, delta_solar);
            }

            if(enough_mass && numngb > 0)
            {
                /* Sort neighbors by distance (insertion sort) */
                double *ngb_dist2 = (double *) mymalloc("ngb_d2", numngb * sizeof(double));
                int *ngb_idx = (int *) mymalloc("ngb_idx", numngb * sizeof(int));

                for(n = 0; n < numngb; n++) {
                    ngb_idx[n] = Ngblist[n];
                    double dx[3];
                    for(k = 0; k < 3; k++) dx[k] = P[Ngblist[n]].Pos[k] - P[i].Pos[k];
                    NEAREST_XYZ(dx[0], dx[1], dx[2], 1);
                    ngb_dist2[n] = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
                }

                for(n = 1; n < numngb; n++) {
                    double kd = ngb_dist2[n]; int ki = ngb_idx[n];
                    int m = n - 1;
                    while(m >= 0 && ngb_dist2[m] > kd) {
                        ngb_dist2[m+1] = ngb_dist2[m];
                        ngb_idx[m+1] = ngb_idx[m];
                        m--;
                    }
                    ngb_dist2[m+1] = kd;
                    ngb_idx[m+1] = ki;
                }

                /* Track total momentum gained by star from gas (code units) */
                double star_mom_gained[3] = {0, 0, 0};

                if(delta_solar > 0)
                {
                    /* Overshoot: borrow mass from nearest gas first.
                     * Gas velocity unchanged (mass removed at gas velocity).
                     * Star accumulates momentum from borrowed mass. */
                    double remaining = delta_solar;
                    for(n = 0; n < numngb && remaining > All.MassTolerance; n++)
                    {
                        j = ngb_idx[n];
                        if(P[j].Mass <= 0) continue;
                        double gas_solar = P[j].Mass * UNIT_MASS_IN_SOLAR;
                        double available = 0.99 * gas_solar; /* 1% mass floor */
                        if(available <= 0) continue;
                        double take = DMIN(available, remaining);
                        double take_code = take / UNIT_MASS_IN_SOLAR;

                        /* Gas mass update */
                        P[j].Mass -= take_code;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                        CellP[j].MassTrue -= take_code;
                        if(CellP[j].MassTrue < 0) CellP[j].MassTrue = 0;
#endif
                        /* Gas velocity stays the same (mass leaves at gas velocity).
                         * Momentum carried to star = take_code * v_gas */
                        for(k = 0; k < 3; k++)
                            star_mom_gained[k] += take_code * P[j].Vel[k];

                        remaining -= take;
                    }
                }
                else
                {
                    /* Undershoot: return excess mass to nearest viable gas cell.
                     * Star velocity unchanged (mass leaves at star velocity).
                     * Gas velocity updated to conserve momentum. */
                    double to_return_code = (-delta_solar) / UNIT_MASS_IN_SOLAR;
                    for(n = 0; n < numngb; n++)
                    {
                        j = ngb_idx[n];
                        if(P[j].Mass <= 0) continue;

                        double m_j_new = P[j].Mass + to_return_code;
                        /* v_gas_new = (m_gas_old * v_gas + dm * v_star) / m_gas_new */
                        for(k = 0; k < 3; k++) {
                            double dv = (to_return_code / m_j_new) * (P[i].Vel[k] - P[j].Vel[k]);
                            P[j].Vel[k] += dv;
                            CellP[j].VelPred[k] += dv;
                        }
                        P[j].Mass = m_j_new;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                        CellP[j].MassTrue += to_return_code;
#endif
                        break; /* all returned to nearest neighbor */
                    }
                }

                myfree(ngb_idx);
                myfree(ngb_dist2);

                /* Update star velocity for momentum conservation (borrowing case).
                 * v_star_new = (m_old * v_old + sum(take_k * v_gas_k)) / m_new */
                if(delta_solar > 0 && (star_mom_gained[0] != 0 || star_mom_gained[1] != 0 || star_mom_gained[2] != 0))
                {
                    double m_i_new_code = M_drawn / UNIT_MASS_IN_SOLAR;
                    for(k = 0; k < 3; k++)
                        P[i].Vel[k] = (m_i_old_code * P[i].Vel[k] + star_mom_gained[k]) / m_i_new_code;
                }
            }
        }

        /* Set dynamical mass to drawn IMF mass and mark as sampled */
        P[i].Mass = M_drawn / UNIT_MASS_IN_SOLAR;
        P[i].sampled = 1;

        /* === Phase 3: Compute UV luminosity from drawn star === */
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
        P[i].UV_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_PHOTOION
        P[i].Lyman_photons_per_sec = 0;
#endif
        if(M_drawn > All.IMFSampleStellarMassCut) {
            double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
            if(star_age_yr < get_lifetime(M_drawn)) {
                P[i].UV_luminosity = pow(10., get_logL_pe(M_drawn));
#ifdef GALSF_RESOLVEDISM_PHOTOION
                P[i].Lyman_photons_per_sec = pow(10., get_logS_ly(M_drawn));
#endif
            }
        }
#endif
    } /* for i over all particles */

    myfree(Ngblist);

    if(ThisTask == 0) printf("IMF sampling done.\n");
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF */
