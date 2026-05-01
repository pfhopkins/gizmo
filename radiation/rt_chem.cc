/* Standard and Kokkos headers must come BEFORE global_data_all_struct.h.
 * macros.h (included by global_data_all_struct.h) defines #define terminate(x) {...}
 * which conflicts with std::terminate declared in <exception>.  Including Kokkos/stdlib
 * first ensures <exception> is processed before the macro is defined. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../core/proto.h"
/* Timestep functions — single source in timestep_functions.h */
#include "../core/timestep_functions.h"

#include "../declarations/gpu_numeric_macros.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../system/gpu_particles_arena.h"


#ifdef RT_CHEM_PHOTOION


/* Return gas temperature from internal energy for photo-ionization chemistry.
 * Takes compact arrays for GPU compatibility (same function for CPU and GPU). */
KOKKOS_FUNCTION
double rt_photoion_chem_return_temperature(int i, double internal_energy, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef RT_ILIEV_TEST1
    return 1e4; // use a fixed temp if this special flag for numerical testing is enabled
#endif
    double mol_wt = 4 / (1 + 3 * HYDROGEN_MASSFRAC + 4 * HYDROGEN_MASSFRAC * cell[i].Ne);
    return mol_wt * (cell[i].gamma_eos_value()-1) * internal_energy * U_TO_TEMP_UNITS;
}


/* Per-particle RT chemistry kernel — device-callable, handles both single-freq
 * and multi-freq paths via compile-time #ifdef RT_PHOTOION_MULTIFREQUENCY.
 * Operates on compact arrays pp[i] and cell[i] (not global P/CellP). */
KOKKOS_FUNCTION
void rt_update_chemistry_for_particle(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(pp[i].Type != 0) return;
    double dtime = get_particle_timestep_in_physical(i, pp);
    if(dtime <= 0 || cell[i].Mass <= 0) return;

    double fac = UNIT_TIME_IN_CGS / (UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS);
    double c_light_codeunits = C_LIGHT_CODE;
    double rho = cell[i].Density * All.cf_a3inv;
    double nH = HYDROGEN_MASSFRAC * rho / PROTONMASS_CGS * UNIT_MASS_IN_CGS;
    double temp = rt_photoion_chem_return_temperature(i, cell[i].InternalEnergyPred, pp, cell);

    /* collisional ionization rate */
    double gamma_HI = 5.85e-11 * sqrt(temp) * exp(-157809.1 / temp) / (1.0 + sqrt(temp / 1e5)) * fac;
    /* alpha_B recombination coefficient */
    double alpha_HII = 2.59e-13 * pow(temp / 1e4, -0.7) * fac;

    double A, B, CC, nHII;

#ifndef RT_PHOTOION_MULTIFREQUENCY
    /* single-frequency: use single ionization bin */
    double n_photons_vol = cell[i].rt_photon_number_density(RT_FREQ_BIN_H0);
    if(n_photons_vol < 0 || isnan(n_photons_vol))
    {
        printf("NEGATIVE n_photons_per_volume: %g %d\n", n_photons_vol, i);
        printf("Rad_E_gamma %g mass %g All.cf_a3inv %g\n", cell[i].Rad_E_gamma[0], pp[i].Mass, All.cf_a3inv);
        endrun(111);
        return;
    }
    A = dtime * gamma_HI * nH * cell[i].Ne;
    B = dtime * c_light_codeunits * n_photons_vol * All.rt_ion_sigma_HI[RT_FREQ_BIN_H0];
    CC = dtime * alpha_HII * nH * cell[i].Ne;
#else
    /* multi-frequency: sum photo-ionization rates over all frequency bins */
    double k_HI = 0.0;
#ifdef RT_CHEM_PHOTOION_HE
    double k_HeI = 0.0, k_HeII = 0.0;
#endif
    for(int j = 0; j < N_RT_FREQ_BINS; j++)
    {
        double n_photons_vol = cell[i].rt_photon_number_density(j);
        if(All.rt_ion_nu_min[j] >= 13.6) {k_HI += c_light_codeunits * All.rt_ion_sigma_HI[j] * n_photons_vol;}
#ifdef RT_CHEM_PHOTOION_HE
        if(All.rt_ion_nu_min[j] >= 24.6) {k_HeI += c_light_codeunits * All.rt_ion_sigma_HeI[j] * n_photons_vol;}
        if(All.rt_ion_nu_min[j] >= 54.4) {k_HeII += c_light_codeunits * All.rt_ion_sigma_HeII[j] * n_photons_vol;}
#endif
    }
    A = dtime * gamma_HI * nH * cell[i].Ne;
    B = dtime * k_HI;
    CC = dtime * alpha_HII * nH * cell[i].Ne;
#endif

    /* semi-implicit scheme for hydrogen ionization */
    nHII = cell[i].HII + B + A;
    nHII /= 1.0 + B + CC + A;
    if(nHII < 0 || nHII > 1 || isnan(nHII))
    {
        printf("ERROR nHII %g\n", nHII);
        endrun(333);
        return;
    }
    cell[i].Ne = nHII;
    cell[i].HII = nHII;
    cell[i].HI = 1.0 - nHII;

#ifdef RT_CHEM_PHOTOION_HE
    /* collisional ionization rate */
    double gamma_HeI = 2.38e-11 * sqrt(temp) * exp(-285335.4 / temp) / (1.0 + sqrt(temp / 1e5)) * fac;
    double gamma_HeII = 5.68e-12 * sqrt(temp) * exp(-631515 / temp) / (1.0 + sqrt(temp / 1e5)) * fac;
    /* alpha_B recombination coefficient */
    double alpha_HeII = 1.5e-10 * pow(temp, -0.6353) * fac;
    double alpha_HeIII = 3.36e-10 / sqrt(temp) * pow(temp / 1e3, -0.2) / (1.0 + pow(temp / 1e6, 0.7)) * fac;
    cell[i].Ne += cell[i].HeII + 2.0 * cell[i].HeIII;

    double D = dtime * gamma_HeII * nH * cell[i].Ne;
    double E = dtime * alpha_HeIII * nH * cell[i].Ne;
    double F = dtime * gamma_HeI * nH * cell[i].Ne;
    double J = dtime * alpha_HeII * nH * cell[i].Ne;
#ifndef RT_PHOTOION_MULTIFREQUENCY
    double G = 0.0;
    double L = 0.0;
#else
    double G = dtime * k_HeI;
    double L = dtime * k_HeII;
#endif
    double y_fac = (1.0-HYDROGEN_MASSFRAC)/4.0/HYDROGEN_MASSFRAC;

    double nHeII = cell[i].HeII / y_fac;
    double nHeIII = cell[i].HeIII / y_fac;
    nHeII = nHeII + G + F - ((G + F - E) / (1.0 + E)) * nHeIII;
    nHeII /= 1.0 + G + F + D + J + ((G + F - E) / (1.0 + E)) * (D + L);
    if(nHeII < 0 || nHeII > 1 || isnan(nHeII))
    {
        printf("ERROR nHeII %g\n", nHeII);
        endrun(333);
        return;
    }
    nHeIII = nHeIII + (D + L) * nHeII;
    nHeIII /= 1.0 + E;
    if(nHeIII < 0 || nHeIII > 1 || isnan(nHeIII))
    {
        printf("ERROR nHeIII %g\n", nHeIII);
        endrun(333);
        return;
    }
    nHeII *= y_fac;
    nHeIII *= y_fac;
    cell[i].Ne = cell[i].HII + nHeII + 2.0 * nHeIII;
    cell[i].HeII = nHeII;
    cell[i].HeIII = nHeIII;
    cell[i].HeI = y_fac - cell[i].HeII - cell[i].HeIII;
    if(cell[i].HeI < 0) {cell[i].HeI = 0.0;}
    if(cell[i].HeI > y_fac) {cell[i].HeI = y_fac;}
#endif
}


void rt_get_sigma(void)
{
    /* rt_get_sigma sets All.rt_ion_sigma_HI etc. — global initialization fields that must
       be written to the HOST All, not All_dev. Since rt_chem.cc is a GPU TU with
       #define All All_dev, we must temporarily restore the host All for this function.
       Otherwise the values go to All_dev, then gizmo_gpu_sync_all() overwrites All_dev
       with host zeros, and non-GPU code reads zero opacities. */
#pragma push_macro("All")
#undef All
    extern struct global_data_all_processes All;
    /* first initialize all bands, so non-ionizing bands don't cause problems */
    int k;
    for(k=0;k<N_RT_FREQ_BINS;k++)
    {
        All.rt_ion_nu_min[k] = 0; All.rt_nu_eff_eV[k] = All.rt_ion_nu_min[k];
        All.rt_ion_G_HI[k]=All.rt_ion_G_HeI[k]=All.rt_ion_G_HeII[k]=All.rt_ion_sigma_HI[k]=All.rt_ion_sigma_HeI[k]=All.rt_ion_sigma_HeII[k]=All.rt_ion_precalc_stellar_luminosity_fraction[k]=0;
    }
    double fac = 1.0 / (UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS);

#ifndef RT_PHOTOION_MULTIFREQUENCY
    /* just the hydrogen ionization bin */
    All.rt_ion_sigma_HI[RT_FREQ_BIN_H0] = 6.3e-18 * fac; // cross-section (blackbody-weighted) for photons
    All.rt_ion_nu_min[RT_FREQ_BIN_H0] = 13.6; // minimum frequency [in eV] of photons of interest
    All.rt_nu_eff_eV[RT_FREQ_BIN_H0] = 27.2; // typical blackbody-weighted frequency [in eV] of photons of interest: to convert energies to numbers
    All.rt_ion_G_HI[RT_FREQ_BIN_H0] = (All.rt_nu_eff_eV[RT_FREQ_BIN_H0]-13.6)*ELECTRONVOLT_IN_ERGS/UNIT_ENERGY_IN_CGS; // absorption cross-section weighted photon energy in code units
#else

    /* now we use the multi-bin spectral information */
#define N_BINS_FOR_IONIZATION 4
    double nu_vec[N_BINS_FOR_IONIZATION] = {13.6, 24.6, 54.4, 70.0};
    int i_vec[N_BINS_FOR_IONIZATION] = {RT_FREQ_BIN_H0, RT_FREQ_BIN_He0, RT_FREQ_BIN_He1, RT_FREQ_BIN_He2};

    int i, j, integral=10000;
    double e, d_nu, e_start, e_end, sum_HI_sigma=0, sum_HI_G=0, hc=C_LIGHT_CGS*6.6262e-27, I_nu, sig, f, fac_two, T_eff, sum_egy_allbands=0;
#if defined(GALSF)
    T_eff = 4.0e4;
#else
    T_eff = All.star_Teff;
#endif
    fac_two = ELECTRONVOLT_IN_ERGS/UNIT_ENERGY_IN_CGS;
#ifdef RT_CHEM_PHOTOION_HE
    double sum_HeI_sigma=0, sum_HeII_sigma=0, sum_HeI_G=0, sum_HeII_G=0;
#endif
    for(k = 0; k < N_BINS_FOR_IONIZATION; k++)
    {
        i = i_vec[k];
        e_start = All.rt_ion_nu_min[i] = nu_vec[k];
        if(k==N_BINS_FOR_IONIZATION-1) {e_end = 500.;} else {e_end = nu_vec[k+1];}
        d_nu = (e_end - e_start) / (float)(integral - 1);
        All.rt_ion_sigma_HI[i] = All.rt_ion_G_HI[i] = All.rt_nu_eff_eV[i] = 0.0;
        sum_HI_sigma = sum_HI_G = 0.0;
#ifdef RT_CHEM_PHOTOION_HE
        All.rt_ion_sigma_HeI[i] = All.rt_ion_sigma_HeII[i] = All.rt_ion_G_HeI[i] = All.rt_ion_G_HeII[i] = 0.0;
        sum_HeI_sigma = sum_HeII_sigma = sum_HeI_G = sum_HeII_G = 0.0;
#endif
        double n_photon_sum = 0.0, sum_energy = 0.0;
        for(j = 0; j < integral; j++)
        {
            e = e_start + j * d_nu;
            I_nu = 2.0 * pow(e * ELECTRONVOLT_IN_ERGS, 3) / (hc * hc) / (exp(e * ELECTRONVOLT_IN_ERGS / (BOLTZMANN_CGS * T_eff)) - 1.0);
            sum_energy += d_nu * I_nu;
            n_photon_sum += d_nu * I_nu / e;
            if(All.rt_ion_nu_min[i] >= 13.6)
            {
                f = sqrt((e / 13.6) - 1.0);
                if(e <= 13.6) {sig = 6.3e-18;} else {sig = 6.3e-18 * pow(13.6 / e, 4) * exp(4 - (4 * atan(f) / f)) / (1.0 - exp(-2 * M_PI / f));}
                All.rt_ion_sigma_HI[i] += d_nu * sig * I_nu / e;
                sum_HI_sigma += d_nu * I_nu / e;
                All.rt_ion_G_HI[i] += d_nu * sig * (e - 13.6) * I_nu / e;
                sum_HI_G += d_nu * sig * I_nu / e;
            }
#ifdef RT_CHEM_PHOTOION_HE
            if(All.rt_ion_nu_min[i] >= 24.6)
            {
                f = sqrt((e / 24.6) - 1.0);
                if(e <= 24.6) {sig = 7.83e-18;} else {sig = 7.83e-18 * pow(24.6 / e, 4) * exp(4 - (4 * atan(f) / f)) / (1.0 - exp(-2 * M_PI / f));}
                All.rt_ion_sigma_HeI[i] += d_nu * sig * I_nu / e;
                sum_HeI_sigma += d_nu * I_nu / e;
                All.rt_ion_G_HeI[i] += d_nu * sig * (e - 24.6) * I_nu / e;
                sum_HeI_G += d_nu * sig * I_nu / e;
            }
            if(All.rt_ion_nu_min[i] >= 54.4)
            {
                f = sqrt((e / 54.4) - 1.0);
                if(e <= 54.4) {sig = 1.58e-18;} else {sig = 1.58e-18 * pow(54.4 / e, 4) * exp(4 - (4 * atan(f) / f)) / (1.0 - exp(-2 * M_PI / f));}
                All.rt_ion_sigma_HeII[i] += d_nu * sig * I_nu / e;
                sum_HeII_sigma += d_nu * I_nu / e;
                All.rt_ion_G_HeII[i] += d_nu * sig * (e - 54.4) * I_nu / e;
                sum_HeII_G += d_nu * sig * I_nu / e;
            }
#endif
        }
        All.rt_nu_eff_eV[i] = sum_energy / n_photon_sum;
        All.rt_ion_precalc_stellar_luminosity_fraction[i] = sum_energy;

        if(All.rt_ion_nu_min[i] >= 13.6)
        {
            All.rt_ion_sigma_HI[i] *= fac / sum_HI_sigma;
            All.rt_ion_G_HI[i] *= fac_two / sum_HI_G;
        }
#ifdef RT_CHEM_PHOTOION_HE
        if(All.rt_ion_nu_min[i] >= 24.6)
        {
            All.rt_ion_sigma_HeI[i] *= fac / sum_HeI_sigma;
            All.rt_ion_G_HeI[i] *= fac_two / sum_HeI_G;
        }
        if(All.rt_ion_nu_min[i] >= 54.4)
        {
            All.rt_ion_sigma_HeII[i] *= fac / sum_HeII_sigma;
            All.rt_ion_G_HeII[i] *= fac_two / sum_HeII_G;
        }
#endif
         sum_egy_allbands += sum_energy;
    }

    for(i = 0; i < N_RT_FREQ_BINS; i++) {All.rt_ion_precalc_stellar_luminosity_fraction[i] /= sum_egy_allbands;}

    if(ThisTask == 0) {for(i = 0; i < N_RT_FREQ_BINS; i++) {printf("%g %g | %g %g | %g %g\n",All.rt_ion_sigma_HI[i]/fac, All.rt_ion_G_HI[i]/fac_two,All.rt_ion_sigma_HeI[i]/fac, All.rt_ion_G_HeI[i]/fac_two,All.rt_ion_sigma_HeII[i]/fac, All.rt_ion_G_HeII[i]/fac_two);}}
#endif
#pragma pop_macro("All")
}


/* Unified rt_update_chemistry: single-freq and multi-freq paths are both handled
 * by rt_update_chemistry_for_particle(). GPU path uses gather-dispatch-scatter;
 * CPU path uses compact arrays with OpenMP. */
void rt_update_chemistry(void)
{
    GIZMO_GPU_ENSURE_ALL_FRESH(rt_chem);
    /* Build index of active gas particles */
    int N_active = 0;
    for(int i : ActiveParticleList) {if(P[i].Type == 0) N_active++;}
    int *chem_indices = (int *) malloc(N_active * sizeof(int));
    {int j = 0; for(int i : ActiveParticleList) {if(P[i].Type == 0) chem_indices[j++] = i;}}

    if(N_active >= GPU_MIN_PARTICLES_FOR_OFFLOAD)
    {
        /* Gather into compact SharedSpace arrays */
        int batch_n = N_active;
        struct particle_data *compact_P    = (struct particle_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(batch_n * sizeof(struct particle_data));
        struct gas_cell_data *compact_Cell = (struct gas_cell_data *) Kokkos::kokkos_malloc<GIZMO_KOKKOS_SHARED_SPACE>(batch_n * sizeof(struct gas_cell_data));
        for(int j = 0; j < batch_n; j++)
        {
            compact_P[j]    = P[chem_indices[j]];
            compact_Cell[j] = CellP[chem_indices[j]];
        }

        /* Dispatch batch to GPU */
        {
            struct particle_data *kp = compact_P;
            struct gas_cell_data *kc = compact_Cell;
            gizmo_gpu_kernel_launch("rt_chem_loop", batch_n, KOKKOS_LAMBDA(int j) {
                rt_update_chemistry_for_particle(j, kp, kc);
            });
        }

        /* Scatter batch back */
        for(int j = 0; j < batch_n; j++)
        {
            int ii = chem_indices[j];
            CellP[ii] = compact_Cell[j];
            P[ii]     = compact_P[j];
        }
        gpu_particles_arena_invalidate(); /* host P/CellP scattered; arena stale */

        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_Cell);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);

    } else if(N_active > 0)
    { /* CPU path: OpenMP-parallel dispatch (fallback for small N on GPU builds) */
        struct particle_data *compact_P    = (struct particle_data *) malloc(N_active * sizeof(struct particle_data));
        struct gas_cell_data *compact_Cell = (struct gas_cell_data *) malloc(N_active * sizeof(struct gas_cell_data));
        for(int j = 0; j < N_active; j++)
        {
            compact_P[j]    = P[chem_indices[j]];
            compact_Cell[j] = CellP[chem_indices[j]];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int j = 0; j < N_active; j++)
        {
            rt_update_chemistry_for_particle(j, compact_P, compact_Cell);
        }
        for(int j = 0; j < N_active; j++)
        {
            int ii = chem_indices[j];
            CellP[ii] = compact_Cell[j];
            P[ii]     = compact_P[j];
        }
        free(compact_Cell);
        free(compact_P);
    } /* end CPU/GPU path selection */

    free(chem_indices);
}


void rt_write_chemistry_stats(void)
{
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    int i;
    double rho, total_nHI, total_V, total_nHI_all, total_V_all;
    total_nHI = 0.0; total_V = 0.0;
#ifndef RT_PHOTOION_MULTIFREQUENCY
    double total_ng, total_ng_all, n_photons_vol;
    total_ng = 0.0;
#endif
#ifdef RT_CHEM_PHOTOION_HE
    double total_nHeI, total_nHeI_all;
    double total_nHeII, total_nHeII_all;
    total_nHeI = total_nHeII = 0.0;
#endif

    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 0)
        {
            rho = CellP[i].Density * All.cf_a3inv;
#ifndef RT_PHOTOION_MULTIFREQUENCY
            n_photons_vol = CellP[i].rt_photon_number_density(RT_FREQ_BIN_H0);
            total_ng += n_photons_vol / 1e53 * P[i].Mass / rho;
#endif
#ifdef RT_CHEM_PHOTOION_HE
            total_nHeI += CellP[i].HeI * P[i].Mass / rho;
            total_nHeII += CellP[i].HeII * P[i].Mass / rho;
#endif
            total_nHI += CellP[i].HI * P[i].Mass / rho;
            total_V += P[i].Mass / rho;
        }
    MPI_Allreduce(&total_nHI, &total_nHI_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_V, &total_V_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#ifndef RT_PHOTOION_MULTIFREQUENCY
    MPI_Allreduce(&total_ng, &total_ng_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
#ifdef RT_CHEM_PHOTOION_HE
    MPI_Allreduce(&total_nHeI, &total_nHeI_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_nHeII, &total_nHeII_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    if(ThisTask == 0)
    {
        if(All.Time == All.TimeBegin)
        {
            fprintf(FdPhotoIonChemStats, "time, nHI");
#ifndef RT_PHOTOION_MULTIFREQUENCY
            fprintf(FdPhotoIonChemStats, ", N_photons");
#endif
#ifdef RT_CHEM_PHOTOION_HE
            fprintf(FdPhotoIonChemStats, ", nHeI, nHeII \n");
#else
            fprintf(FdPhotoIonChemStats, "\n");
#endif
        }
        fprintf(FdPhotoIonChemStats, "%.16g %g ", All.Time, total_nHI_all / total_V_all);
#ifndef RT_PHOTOION_MULTIFREQUENCY
        fprintf(FdPhotoIonChemStats, "%g ", total_ng_all / total_V_all);
#endif
#ifdef RT_CHEM_PHOTOION_HE
        fprintf(FdPhotoIonChemStats, "%g %g\n", total_nHeI_all / total_V_all, total_nHeII_all / total_V_all);
#else
        fprintf(FdPhotoIonChemStats, "\n");
#endif
        fflush(FdPhotoIonChemStats);
    }
#endif
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */
GPU_ALL_SYNC_FUNC(rt_chem)


#endif /* RT_CHEM_PHOTOION */
