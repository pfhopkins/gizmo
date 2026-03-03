#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* Full IMF sampling: draws individual stellar masses from a Kroupa IMF for each star particle.
 * Uses MPI gather/scatter so sampling is done sequentially on rank 0.
 * Mass borrowing from nearby star particles handles overshoot.
 * TODO: the mass borrowing allows unphysical mass transfer — fix later.
 * Ported from gizmo2017/imf_sampling.c */

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

/* Sequential IMF sampling, called only on rank 0 after MPI_Gatherv.
 * Input: masses (solar), positions, output: stellar_masses array, flag array.
 * Stars above IMFSampleStellarMassCut are recorded in MstarSampleIMF slots. */
static void IMFsampling(double *Mass, double *Pos_x, double *Pos_y, double *Pos_z,
                        double *MstarSampleIMF, int *flag, int NumNewStars)
{
    int i, j, k;
    double r_search_2 = All.SearchingRadius * All.SearchingRadius;
    double totalSampledMass = 0;

    printf("IMF sampling started...\n");
    for(i = 0; i < NumNewStars; i++)
    {
        double M_clus = 0;
        j = 0;
        if(fabs(Mass[i]) < 1e-30) continue; /* mass zeroed by borrowing from a previous particle */

        while(1)
        {
            /* Rejection sampling: draw from proposal (power-law x^-0.3) and accept with IMF/envelope ratio */
            double m_proposed;
            do {
                m_proposed = pow((pow(M_max, -0.3) - pow(M_min, -0.3)) * gsl_rng_uniform(random_generator) + pow(M_min, -0.3), -1.0/0.3);
            } while(gsl_rng_uniform(random_generator) > (drawMassFromIMF(m_proposed) / envelope_function(m_proposed)));

            M_clus += m_proposed;
            totalSampledMass += m_proposed;

            /* Record massive stars above the cut */
            if(m_proposed > All.IMFSampleStellarMassCut) {
                if(j >= N_STELLAR_MASS) {
                    printf("Not enough space for IMF sampling, M_clus=%g\n", M_clus);
                    fflush(stdout);
                    endrun(807);
                }
                MstarSampleIMF[i * N_STELLAR_MASS + j] = m_proposed;
                j++;
            }

            /* Check convergence: did we match the particle mass? */
            if(fabs(M_clus - Mass[i]) <= All.MassTolerance) {
                flag[i] = 1; /* perfect match */
                break;
            }
            if(M_clus - Mass[i] > All.MassTolerance) {
                /* Overshoot: try to borrow mass from nearby star particles */
                double mass_sum = Mass[i];
                flag[i] = -1;
                int idx = 0;
                for(k = i + 1; k < NumNewStars; k++) {
                    double distance_2 = pow(Pos_x[i] - Pos_x[k], 2) + pow(Pos_y[i] - Pos_y[k], 2) + pow(Pos_z[i] - Pos_z[k], 2);
                    if(distance_2 < r_search_2) {
                        mass_sum += Mass[k];
                        flag[k] = -1;
                    }
                    if(mass_sum > M_clus) { idx = k; break; }
                }
                if(idx > 0) {
                    /* Found enough neighbors to borrow from */
                    Mass[i] = M_clus;
                    flag[i] = 1;
                    for(k = i + 1; k < idx; k++) {
                        if(flag[k] == -1) { Mass[k] = 0; flag[k] = 1; }
                    }
                    Mass[idx] = mass_sum - M_clus; /* leftover mass stays for later sampling */
                } else {
                    /* Not enough neighbors: undo last sample and wait for next round */
                    totalSampledMass -= m_proposed;
                    if(j > 0 && MstarSampleIMF[i * N_STELLAR_MASS + j - 1] > 0)
                        MstarSampleIMF[i * N_STELLAR_MASS + j - 1] = 0;
                    for(k = i + 1; k < NumNewStars; k++) {
                        if(flag[k] == -1) flag[k] = 0;
                    }
                }
                break;
            }
        } /* while loop */
    }
    printf("totalSampledMass=%g\n", totalSampledMass);
}


/* Main entry point: gather unsampled star particles to rank 0, sample IMF, scatter results back */
void assign_stellar_masses(void)
{
    int i, j;
    int nstars_per_proc = 0;

    if(ThisTask == 0) printf("assigning stellar masses to star particles...\n");

    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 4 && P[i].sampled == 0) nstars_per_proc++;

    double *mass = (double *) mymalloc("mass", DMAX(nstars_per_proc, 1) * sizeof(double));
    int *flag = (int *) mymalloc("flag", DMAX(nstars_per_proc, 1) * sizeof(int));
    double *xstars = (double *) mymalloc("xstars", DMAX(nstars_per_proc, 1) * sizeof(double));
    double *ystars = (double *) mymalloc("ystars", DMAX(nstars_per_proc, 1) * sizeof(double));
    double *zstars = (double *) mymalloc("zstars", DMAX(nstars_per_proc, 1) * sizeof(double));
    double *stellar_masses = (double *) mymalloc("stellar_masses", DMAX(nstars_per_proc * N_STELLAR_MASS, 1) * sizeof(double));

    j = 0;
    for(i = 0; i < NumPart; i++) {
        if(P[i].Type == 4 && P[i].sampled == 0) {
            mass[j] = P[i].Mass * UNIT_MASS_IN_SOLAR;
            xstars[j] = P[i].Pos[0];
            ystars[j] = P[i].Pos[1];
            zstars[j] = P[i].Pos[2];
            flag[j] = 0;
            j++;
        }
    }
    for(i = 0; i < nstars_per_proc * N_STELLAR_MASS; i++) stellar_masses[i] = 0;

    /* Gather counts to root */
    int *nstars_per_proc_global_list = NULL;
    if(ThisTask == 0) nstars_per_proc_global_list = (int *) mymalloc("nstars_global", NTask * sizeof(int));
    MPI_Gather(&nstars_per_proc, 1, MPI_INT, nstars_per_proc_global_list, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_nstars;
    MPI_Allreduce(&nstars_per_proc, &total_nstars, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int *displs = NULL;
    double *global_mass = NULL, *global_xstars = NULL, *global_ystars = NULL, *global_zstars = NULL, *global_stellar_masses = NULL;
    int *global_flag = NULL;

    if(ThisTask == 0) {
        displs = (int *) mymalloc("displs", NTask * sizeof(int));
        for(i = 1, displs[0] = 0; i < NTask; i++)
            displs[i] = displs[i - 1] + nstars_per_proc_global_list[i - 1];

        global_mass = (double *) mymalloc("global_mass", DMAX(total_nstars, 1) * sizeof(double));
        global_flag = (int *) mymalloc("global_flag", DMAX(total_nstars, 1) * sizeof(int));
        global_xstars = (double *) mymalloc("global_xstars", DMAX(total_nstars, 1) * sizeof(double));
        global_ystars = (double *) mymalloc("global_ystars", DMAX(total_nstars, 1) * sizeof(double));
        global_zstars = (double *) mymalloc("global_zstars", DMAX(total_nstars, 1) * sizeof(double));
        global_stellar_masses = (double *) mymalloc("global_smasses", DMAX(total_nstars * N_STELLAR_MASS, 1) * sizeof(double));
        for(j = 0; j < total_nstars * N_STELLAR_MASS; j++) global_stellar_masses[j] = 0;
    }

    if(total_nstars > 0)
    {
        /* Gather mass, positions, flags to root */
        MPI_Gatherv(mass, nstars_per_proc, MPI_DOUBLE, global_mass, nstars_per_proc_global_list, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(flag, nstars_per_proc, MPI_INT, global_flag, nstars_per_proc_global_list, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(xstars, nstars_per_proc, MPI_DOUBLE, global_xstars, nstars_per_proc_global_list, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(ystars, nstars_per_proc, MPI_DOUBLE, global_ystars, nstars_per_proc_global_list, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(zstars, nstars_per_proc, MPI_DOUBLE, global_zstars, nstars_per_proc_global_list, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Do the sampling on rank 0 */
        if(ThisTask == 0) {
            IMFsampling(global_mass, global_xstars, global_ystars, global_zstars, global_stellar_masses, global_flag, total_nstars);
            printf("sampling done.\n");
        }

        /* Scatter results back */
        MPI_Scatterv(global_mass, nstars_per_proc_global_list, displs, MPI_DOUBLE, mass, nstars_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(global_flag, nstars_per_proc_global_list, displs, MPI_INT, flag, nstars_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

        /* Scatter stellar_masses: multiply counts/displs by N_STELLAR_MASS */
        nstars_per_proc *= N_STELLAR_MASS;
        if(ThisTask == 0) {
            for(i = 0; i < NTask; i++) {
                nstars_per_proc_global_list[i] *= N_STELLAR_MASS;
                displs[i] *= N_STELLAR_MASS;
            }
        }
        MPI_Scatterv(global_stellar_masses, nstars_per_proc_global_list, displs, MPI_DOUBLE, stellar_masses, nstars_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nstars_per_proc /= N_STELLAR_MASS; /* undo the hack */
    }

    /* Assign results to actual particle data */
    int k = 0;
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 4 && P[i].sampled == 0)
        {
            if(flag[k] == 1)
            {
                P[i].sampled = 1;
                P[i].Mass = mass[k] / UNIT_MASS_IN_SOLAR; /* update dynamical mass (may have changed from borrowing) */

#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                P[i].UV_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_PHOTOION
                P[i].Lyman_photons_per_sec = 0;
#endif
#endif
                for(j = 0; j < N_STELLAR_MASS; j++)
                {
                    P[i].MstarSampleIMF[j] = stellar_masses[k * N_STELLAR_MASS + j];
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
                    double mass_j = P[i].MstarSampleIMF[j];
                    if(mass_j > 0) {
                        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
                        if(star_age_yr < get_lifetime(mass_j)) {
                            P[i].UV_luminosity += pow(10., get_logL_pe(mass_j));
#ifdef GALSF_RESOLVEDISM_PHOTOION
                            P[i].Lyman_photons_per_sec += pow(10., get_logS_ly(mass_j));
#endif
                        }
                    }
#endif
                }
            }
            k++;
        }
    }

    /* Free memory in reverse allocation order */
    if(ThisTask == 0) {
        myfree(global_stellar_masses);
        myfree(global_zstars);
        myfree(global_ystars);
        myfree(global_xstars);
        myfree(global_flag);
        myfree(global_mass);
        myfree(displs);
        myfree(nstars_per_proc_global_list);
    }
    myfree(stellar_masses);
    myfree(zstars);
    myfree(ystars);
    myfree(xstars);
    myfree(flag);
    myfree(mass);
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF */
