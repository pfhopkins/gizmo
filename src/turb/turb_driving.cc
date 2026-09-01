/* Standard and Kokkos headers must precede global_data_all_struct.h
 * (its macros may conflict with stdlib names). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../declarations/gizmo_rng.h"
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"

/* This file contains the routines for driven turbulence/stirring; use for things
 like idealized turbulence tests, large-eddy simulations, and the like */
/*
 *  This code was originally written for GADGET3 by Andreas Bauer; it has been
 *   modified significantly by Phil Hopkins for GIZMO, and slightly by Mike Grudic
 */

#if defined(TURB_DRIVING)

/* block of global variables used specifically for the set of subroutines below, which need to be carried between timesteps */
double* StOUPhases; // random fluctuating component of the amplitudes
double* StAmpl; // relative amplitude for each k
double* StAka; // phases (real part)
double* StAkb; // phases (imag part)
double* StMode; // k vectors
int StNModes; // total number of modes
integertime StTPrev; // time of last update (to determine when next will be)
gizmo_rng_t StRng; // random number generator key


/* routine to initialize the different modes and their relative amplitudes and other global variables needed for the turbulence driving routines */
void init_turb(void)
{
    int ikx, iky, ikz; double kx,ky,kz,k, ampl;
    
    double kmin = All.TurbDriving_Global_DrivingScaleKMinVar, kmax = All.TurbDriving_Global_DrivingScaleKMaxVar;
    kmin = 2.*M_PI / All.TurbDriving_Global_DrivingScaleKMinVar; kmax = 2.*M_PI / All.TurbDriving_Global_DrivingScaleKMaxVar; // convert these from spatial lengths to wavenumbers in the new convention we are using
    
    int ikxmax = boxSize_X * kmax/2./M_PI, ikymax = 0, ikzmax = 0;
#if (NUMDIMS > 1)
    ikymax = boxSize_Y * kmax/2./M_PI;
#endif
#if (NUMDIMS > 2)
    ikzmax = boxSize_Z * kmax/2./M_PI;
#endif
    
    StNModes = 0;
    for(ikx = 0;ikx <= ikxmax; ikx++)
    {
        kx = 2.*M_PI*ikx/boxSize_X;
        for(iky = 0;iky <= ikymax; iky++)
        {
            ky = 2.*M_PI*iky/boxSize_Y;
            for(ikz = 0;ikz <= ikzmax; ikz++)
            {
                kz = 2.*M_PI*ikz/boxSize_Z;
                k = sqrt(kx*kx+ky*ky+kz*kz);
                if(k>=kmin && k<=kmax)
                {
#if NUMDIMS ==1
                    StNModes+=1;
#endif
#if NUMDIMS == 2
                    StNModes+=2;
#endif
#if NUMDIMS == 3
                    StNModes+=4;
#endif
                }
            }
        }
    }
    
    PRINT_STATUS("Initializing turbulent driving: max integer mode number ikx/iky/ikz = %d %d %d",ikxmax,ikymax,ikzmax);
    StMode = (double*) mymalloc_movable(&StMode,"StModes", StNModes * 3 * sizeof(double));
    StAka = (double*) mymalloc_movable(&StAka,"StAka", StNModes * 3 * sizeof(double));
    StAkb = (double*) mymalloc_movable(&StAkb,"StAkb", StNModes * 3 * sizeof(double));
    StAmpl = (double*) mymalloc_movable(&StAmpl,"StAmpl", StNModes * sizeof(double));
    StOUPhases = (double*) mymalloc_movable(StOUPhases,"StOUPhases", StNModes * 6 * sizeof(double));
    double kc = 0.5*(kmin+kmax), amin = 0., amplitude_integrated_allmodes = 0.;
    StNModes = 0;
    
    for(ikx = 0;ikx <= ikxmax; ikx++)
    {
        kx = 2.*M_PI*ikx/boxSize_X;
        
        for(iky = 0;iky <= ikymax; iky++)
        {
            ky = 2.*M_PI*iky/boxSize_Y;
            
            for(ikz = 0;ikz <= ikzmax; ikz++)
            {
                kz = 2.*M_PI*ikz/boxSize_Z;
                
                k = sqrt(kx*kx+ky*ky+kz*kz);
                if(k>=kmin && k<=kmax)
                {
                    if(All.TurbDriving_Global_DrivingSpectrumKey == 0)
                    {
                        //ampl = 1.; // uniform amplitude for all
                        ampl = pow(k/kmin, -1. + 2. - 0*0.5*NUMDIMS); // this should scale as the acceleration per eddy, ~v^2/L, crudely
                    }
                    else if(All.TurbDriving_Global_DrivingSpectrumKey ==  1)
                    {
                        ampl = 4.0*(amin-1.0)/((kmax-kmin)*(kmax-kmin))*((k-kc)*(k-kc))+1.0; // spike at kc = k_driving
                    }
                    else if(All.TurbDriving_Global_DrivingSpectrumKey == 2)
                    {
                        //ampl = pow(k/kmin, (1.-NUMDIMS)- 5./3. ); // because this is E[vector_k] for NUMDIMS, need extra NUMDIMS-1 power term here
                        ampl = pow(k/kmin, -5./3. + 2. - 0*0.5*NUMDIMS); // this should scale as the acceleration per eddy, ~v^2/L, crudely
                    }
                    else if(All.TurbDriving_Global_DrivingSpectrumKey == 3)
                    {
                        //ampl = pow(k/kmin, (1.-NUMDIMS)- 2. ); // because this is E[vector_k] for NUMDIMS, need extra NUMDIMS-1 power term here
                        ampl = pow(k/kmin, -2. + 2. - 0*0.5*NUMDIMS); // this should scale as the acceleration per eddy, ~v^2/L, crudely
                    }
                    else
                    {
                        if(ThisTask == 0) {printf("unknown spectral form (TurbDriving_Global_DrivingSpectrumKey=%d)\n", All.TurbDriving_Global_DrivingSpectrumKey); fflush(stdout);}
                        endrun(90001012);
                        gizmo_exit_bad_stop_if_requested("turb_driving:unknown_spectrum");  /* symmetric config: all ranks poll together and finalize before ampl is used */
                        ampl = 0;   /* defensive: keep ampl defined if the poll ever returns */
                    }
                    
                    StAmpl[StNModes] = ampl;
                    StMode[3*StNModes+0] = kx;
                    StMode[3*StNModes+1] = ky;
                    StMode[3*StNModes+2] = kz;
                    PRINT_STATUS("  Mode: %d, ikx=%d, iky=%d, ikz=%d, kx=%f, ky=%f, kz=%f, ampl=%f",StNModes,ikx,iky,ikz,StMode[3*StNModes+0],StMode[3*StNModes+1],StMode[3*StNModes+2],StAmpl[StNModes]);
                    amplitude_integrated_allmodes += ampl*ampl;
                    StNModes++;
                    
#if (NUMDIMS > 1)
                    if(ikx>0 || iky>0) // if both of these are zero, only non-degenerate modes are the +/- z modes [ensured below]
                    {
                        StAmpl[StNModes] = ampl;
                        if(iky!=0) {StMode[3*StNModes+0] = kx;} else {StMode[3*StNModes+0] = -kx;}
                        StMode[3*StNModes+1] = -ky;
                        StMode[3*StNModes+2] = kz;
                        PRINT_STATUS("  Mode: %d, ikx=%d, iky=%d, ikz=%d, kx=%f, ky=%f, kz=%f, ampl=%f",StNModes,ikx,-iky,ikz,StMode[3*StNModes+0],StMode[3*StNModes+1],StMode[3*StNModes+2],StAmpl[StNModes]);
                        amplitude_integrated_allmodes += ampl*ampl;
                        StNModes++;
                    }

#if (NUMDIMS > 2)
                    if((iky>0 || ikz>0) && (ikx>0 || ikz>0)) // if both of these are zero, only non-degenerate modes are the +/- x or +/- y modes [already ensured above]
                    {
                        StAmpl[StNModes] = ampl;
                        if(ikz!=0) {StMode[3*StNModes+0] = kx;} else {StMode[3*StNModes+0] = -kx;}
                        StMode[3*StNModes+1] = ky;
                        StMode[3*StNModes+2] = -kz;
                        PRINT_STATUS("  Mode: %d, ikx=%d, iky=%d, ikz=%d, kx=%f, ky=%f, kz=%f, ampl=%f",StNModes,ikx,iky,-ikz,StMode[3*StNModes+0],StMode[3*StNModes+1],StMode[3*StNModes+2],StAmpl[StNModes]);
                        amplitude_integrated_allmodes += ampl*ampl;
                        StNModes++;
                    }

                    if((ikx>0 || iky>0) && (ikx>0 || ikz>0) && (iky>0 || ikz>0)) // if both of these are zero, only non-degenerate modes are +/- z or +/- y or +/- x modes already handled above
                    {
                        StAmpl[StNModes] = ampl;
                        if(ikz==0 || iky==0) {StMode[3*StNModes+0] = -kx;} else {StMode[3*StNModes+0] = kx;}
                        StMode[3*StNModes+1] = -ky;
                        StMode[3*StNModes+2] = -kz;
                        PRINT_STATUS("  Mode: %d, ikx=%d, iky=%d, ikz=%d, kx=%f, ky=%f, kz=%f, ampl=%f",StNModes,ikx,-iky,-ikz,StMode[3*StNModes+0],StMode[3*StNModes+1],StMode[3*StNModes+2],StAmpl[StNModes]);
                        amplitude_integrated_allmodes += ampl*ampl;
                        StNModes++;
                    }
#endif
#endif
                }
            }
        }
    }
    int i; for(i=0; i<StNModes; i++) {StAmpl[i] *= sqrt(1./amplitude_integrated_allmodes);} // normalize total driving amplitude across all modes here
    StTPrev = -1; // mark some arbitrarily old time as last update of turb driving fields
    gizmo_rng_init(&StRng, (uint64_t)All.TurbDriving_Global_DrivingRandomNumberKey);
    int j; for(j=0;j<100;j++) {double tmp; tmp=st_turbdrive_get_gaussian_random_variable();} // cycle past initial seed
    st_turbdrive_init_ouseq(); // initialize variable for phases
    st_turbdrive_calc_phases(); // initialize phases
    set_turb_ampl(); // set initial amplitudes and calculate initial quantities needed for dissipation measures
    StTPrev = All.Ti_Current; // mark current time as last update of turb driving fields
}


/* initialize phase variables */
void st_turbdrive_init_ouseq(void)
{
    int i; for(i = 0;i<6*StNModes;i++) {StOUPhases[i] = st_turbdrive_get_gaussian_random_variable()*st_return_rms_acceleration();}
}


/* return the rms acceleration we expect, using either the 'dissipation rate' or 'turbulent velocity' conventions for our variables */
double st_return_rms_acceleration(void)
{
    return All.TurbDriving_Global_AccelerationPowerVariable / st_return_mode_correlation_time(); // new convention, hoping this is more clear re: meaning of variable
}


/* return the driving scale needed for scaling some other quantities below, corresponding to our global variable convention */
double st_return_driving_scale(void)
{
    return All.TurbDriving_Global_DrivingScaleKMinVar; // this is now spatial scale
}


/* return the coherence time of the driving scale modes. if global variable negative, it uses the eddy turnover time of the driving-scale modes */
double st_return_mode_correlation_time(void)
{
    if(All.TurbDriving_Global_DecayTime > 0) {return All.TurbDriving_Global_DecayTime;} else {return st_return_driving_scale() / All.TurbDriving_Global_AccelerationPowerVariable;}
}


/* return time interval between turbulent driving field updates based on global variable. if negative, default to small interval of coherence time by default. */
double st_return_dt_between_updates(void)
{
    if(All.TurbDriving_Global_DtTurbUpdates > 0) {return All.TurbDriving_Global_DtTurbUpdates;} else {return 0.01*st_return_mode_correlation_time();}
}


/* return factor needed to renormalize below based on fraction of power projected out in our solenoidal projection, to return the correct normalization for accelerations. */
double solenoidal_frac_total_weight_renormalization(void)
{
#if (NUMDIMS >= 3)
    return sqrt(3.0/3.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*All.TurbDriving_Global_SolenoidalFraction+3.0*All.TurbDriving_Global_SolenoidalFraction*All.TurbDriving_Global_SolenoidalFraction);
#endif
#if (NUMDIMS == 2)
    return sqrt(3.0/2.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*All.TurbDriving_Global_SolenoidalFraction+2.0*All.TurbDriving_Global_SolenoidalFraction*All.TurbDriving_Global_SolenoidalFraction);
#endif
#if (NUMDIMS == 1)
    return sqrt(3.0/1.0)*sqrt(3.0)*1.0/sqrt(1.0-2.0*All.TurbDriving_Global_SolenoidalFraction+1.0*All.TurbDriving_Global_SolenoidalFraction*All.TurbDriving_Global_SolenoidalFraction);
#endif
}


/* update the Markov random variable that is the dimensional multiplier for the acceleration field, which has a correlation time specified */
void st_update_ouseq(void)
{
    int i; double damping = exp( -st_return_dt_between_updates()/st_return_mode_correlation_time());
    for(i = 0;i<6*StNModes;i++) {StOUPhases[i] = StOUPhases[i] * damping + st_return_rms_acceleration() * sqrt(1.-damping*damping)*st_turbdrive_get_gaussian_random_variable();}
}


/* routine to return gaussian random number with zero mean and unity variance */
double st_turbdrive_get_gaussian_random_variable(void)
{
    double r0 = gizmo_rng_uniform(&StRng), r1 = gizmo_rng_uniform(&StRng);
    return sqrt(2. * log(1. / r0) ) * cos(2. * M_PI * r1);
}


/* routine to calculate the projected phases/acceleration field variables, using the fourier-space solenoidal/compressible projection */
void st_turbdrive_calc_phases(void)
{
    int i,j;
    for(i = 0; i < StNModes;i++)
    {
        double ka = 0., kb = 0., kk = 0.; int dim = NUMDIMS;
        for(j = 0; j<dim;j++)
        {
            kk += StMode[3*i+j]*StMode[3*i+j];
            ka += StMode[3*i+j]*StOUPhases[6*i+2*j+1];
            kb += StMode[3*i+j]*StOUPhases[6*i+2*j+0];
        }
        for(j = 0; j<dim;j++)
        {
            double diva = StMode[3*i+j]*ka/kk;
            double divb = StMode[3*i+j]*kb/kk;
            double curla = StOUPhases[6*i+2*j+0] - divb;
            double curlb = StOUPhases[6*i+2*j+1] - diva;
            
            StAka[3*i+j] = All.TurbDriving_Global_SolenoidalFraction*curla+(1.-All.TurbDriving_Global_SolenoidalFraction)*divb;
            StAkb[3*i+j] = All.TurbDriving_Global_SolenoidalFraction*curlb+(1.-All.TurbDriving_Global_SolenoidalFraction)*diva;
        }
    }
}

/* Check whether the phases of the turbulent driving force must be recomputed this timestep */
int new_turbforce_needed_this_timestep(void)
{
    double delta = (All.Ti_Current - StTPrev) * unit_integertime_in_physical(-1, P), Dt_Update=st_return_dt_between_updates();
    if(delta >= Dt_Update){return 1;} else {return 0;}
}

/* parent routine to initialize and update turbulent driving fields and to track different variables used for analyzing power spectra of dissipation, etc. */
void set_turb_ampl(void)
{
    if(new_turbforce_needed_this_timestep())
    {
        double delta = (All.Ti_Current - StTPrev) * unit_integertime_in_physical(-1, P);
        if(delta > 0)
        {
            int i; double e_diss_sum=0, e_drive_sum=0, glob_diss_sum=0, glob_drive_sum=0;
            PRINT_STATUS(" ..updating fields tracked for following injected energy and dissipation");
            for(i=0; i < NumPart; i++)
            {
                if(P[i].Type == 0)
                {
                    if(P[i].Mass > 0)
                    {
                        e_diss_sum += CellP[i].EgyDiss;
                        CellP[i].DuDt_diss = (CellP[i].EgyDiss / P[i].Mass) / delta;
                        CellP[i].EgyDiss = 0;
                        e_drive_sum += CellP[i].EgyDrive;
                        CellP[i].DuDt_drive = (CellP[i].EgyDrive / P[i].Mass) / delta;
                        CellP[i].EgyDrive = 0;
                    } else {
                        CellP[i].DuDt_diss = CellP[i].EgyDiss = CellP[i].DuDt_drive = CellP[i].EgyDrive = 0;
                    }
                }
            }
            MPI_Allreduce(&e_diss_sum, &glob_diss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&e_drive_sum, &glob_drive_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            All.TurbDissipatedEnergy += glob_diss_sum;
            All.TurbInjectedEnergy += glob_drive_sum;
        }
        PRINT_STATUS(" ..updating fourier-space phase information");
        st_update_ouseq();
        PRINT_STATUS(" ..calculating coefficients and phases following desired projection");
        st_turbdrive_calc_phases();
        StTPrev = StTPrev + st_return_dt_between_updates() / All.Timebase_interval;
        PRINT_STATUS(" ..updated turbulent stirring field at time %f", StTPrev * All.Timebase_interval);
    }
}


/* Per-particle turbulent acceleration kernel — device-callable.
 * Evaluates the Fourier-mode driving field at particle position.
 * Mode arrays (mode/aka/akb/ampl) are passed as arguments since the
 * host-side global pointers (StMode etc.) are not device-accessible. */
KOKKOS_FUNCTION
void add_turb_accel_for_particle(int i, struct particle_data *pp, struct gas_cell_data *cell,
                                  const double *mode, const double *aka, const double *akb,
                                  const double *ampl, int nmodes, double fac_sol)
{
    if(pp[i].Type != 0) return;
    double fx = 0, fy = 0, fz = 0;
    for(int m = 0; m < nmodes; m++)
    {
        double kdotx = mode[3*m+0]*pp[i].Pos[0] + mode[3*m+1]*pp[i].Pos[1] + mode[3*m+2]*pp[i].Pos[2];
        double a = ampl[m], realt = cos(kdotx), imagt = sin(kdotx);
        fx += a*(aka[3*m+0]*realt - akb[3*m+0]*imagt);
        fy += a*(aka[3*m+1]*realt - akb[3*m+1]*imagt);
        fz += a*(aka[3*m+2]*realt - akb[3*m+2]*imagt);
    }
    fx *= fac_sol; fy *= fac_sol; fz *= fac_sol;

    if(pp[i].Mass > 0.)
    {
        double acc[3];
        acc[0] = fx; acc[1] = acc[2] = 0;
#if (NUMDIMS > 1)
        acc[1] = fy;
#endif
#if (NUMDIMS > 2)
        acc[2] = fz;
#endif
        cell[i].TurbAccel = {acc[0], acc[1], acc[2]};
    } else {
        cell[i].TurbAccel = {};
    }
}


/* routine to actually calculate the turbulent acceleration 'driving field' force on every resolution element */
void add_turb_accel()
{
    set_turb_ampl();
    double fac_sol = 2.*solenoidal_frac_total_weight_renormalization();

    /* Build active gas particle index list */
    int N_active = 0;
    int *turb_indices = (int *) malloc(ActiveParticleList.size() * sizeof(int));
    for (int i : ActiveParticleList) {if(P[i].Type == 0) {turb_indices[N_active++] = i;}}
    if(N_active == 0) {free(turb_indices); PRINT_STATUS("Finished turbulence driving (acceleration) computation"); return;}

    if(N_active >= GPU_MIN_PARTICLES_FOR_OFFLOAD) {
        GIZMO_GPU_ENSURE_ALL_FRESH(); /* All-mirror dispatch-boundary belt; the turb_accel kernel itself reads no All.* but keep the call at the GPU dispatch point per the All-mirror convention */
        /* Copy mode arrays to GPU-accessible memory (read-only, small: ~few KB) */
        const size_t mode_bytes = (size_t) StNModes * 3 * sizeof(double);
        const size_t ampl_bytes = (size_t) StNModes * sizeof(double);
        const size_t compact_bytes = (size_t) N_active * (sizeof(struct particle_data)
                                                          + sizeof(struct gas_cell_data));
        double *gpu_mode = (double *) gizmo_gpu_alloc_shared(mode_bytes, NULL);
        double *gpu_aka  = (double *) gizmo_gpu_alloc_shared(mode_bytes, NULL);
        double *gpu_akb  = (double *) gizmo_gpu_alloc_shared(mode_bytes, NULL);
        double *gpu_ampl = (double *) gizmo_gpu_alloc_shared(ampl_bytes, NULL);
        /* Gather into compact SharedSpace arrays */
        struct particle_data *compact_P    = (struct particle_data *) gizmo_gpu_alloc_shared((size_t) N_active * sizeof(struct particle_data), NULL);
        struct gas_cell_data *compact_Cell = (struct gas_cell_data *) gizmo_gpu_alloc_shared((size_t) N_active * sizeof(struct gas_cell_data), NULL);
        if(!gpu_mode || !gpu_aka || !gpu_akb || !gpu_ampl || !compact_P || !compact_Cell) {
            if(compact_Cell) {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_Cell);}
            if(compact_P)    {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);}
            if(gpu_ampl)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_ampl);}
            if(gpu_akb)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_akb);}
            if(gpu_aka)      {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_aka);}
            if(gpu_mode)     {Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_mode);}
            free(turb_indices);
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "turbulent driving: could not stage %d gas particles and %d driving "
                     "modes (%.1f MB); no driving acceleration is applied",
                     N_active, StNModes,
                     (double)(3 * mode_bytes + ampl_bytes + compact_bytes) / (1024.0 * 1024.0));
            gizmo_request_controlled_stop(7715, msg, __FILE__, __LINE__, __FUNCTION__);
            return;
        }
        memcpy(gpu_mode, StMode, StNModes * 3 * sizeof(double));
        memcpy(gpu_aka,  StAka,  StNModes * 3 * sizeof(double));
        memcpy(gpu_akb,  StAkb,  StNModes * 3 * sizeof(double));
        memcpy(gpu_ampl, StAmpl, StNModes * sizeof(double));

        for(int j = 0; j < N_active; j++)
        {
            compact_P[j]    = P[turb_indices[j]];
            compact_Cell[j] = CellP[turb_indices[j]];
        }

        /* Dispatch to GPU */
        {
            struct particle_data *kp = compact_P;
            struct gas_cell_data *kc = compact_Cell;
            const double *km = gpu_mode, *ka = gpu_aka, *kb = gpu_akb, *kamp = gpu_ampl;
            int nm = StNModes; double fs = fac_sol;
            gizmo_gpu_kernel_launch("turb_accel", N_active, KOKKOS_LAMBDA(int j) {
                add_turb_accel_for_particle(j, kp, kc, km, ka, kb, kamp, nm, fs);
            });
        }

        /* Scatter back */
        for(int j = 0; j < N_active; j++)
        {
            int ii = turb_indices[j];
            CellP[ii] = compact_Cell[j];
        }
        gpu_particles_arena_invalidate();

        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_Cell);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(compact_P);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_ampl);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_akb);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_aka);
        Kokkos::kokkos_free<GIZMO_KOKKOS_SHARED_SPACE>(gpu_mode);

    } else
    { /* CPU path — tiny-N: run the SSOT kernel directly over the global
         P/CellP arrays via turb_indices, no compact gather/scatter, no GPU
         mirror sync (matches the grain_drag tiny-N cleanup philosophy). */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int j = 0; j < N_active; j++)
        {
            add_turb_accel_for_particle(turb_indices[j], P, CellP, StMode, StAka, StAkb, StAmpl, StNModes, fac_sol);
        }
    } /* end CPU/GPU path selection */

    free(turb_indices);
    PRINT_STATUS("Finished turbulence driving (acceleration) computation");
}


/* routine to integrate the turbulent driving forces, specifically the 'TurbAccel' variables that need to be drifted and kicked: note that we actually do drifting and kicking in the normal routines, this is just to integrate the dissipation rates etc used for our tracking */
void do_turb_driving_step_first_half(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    int i; integertime ti_step, tstart, tend; double dt_gravkick;
    for (int i : ActiveParticleList)
    {
        ti_step = P[i].integertime_step(); tstart = P[i].Ti_begstep; tend = P[i].Ti_begstep + ti_step / 2;	/* beginning / midpoint of step */
        dt_gravkick = get_gravkick_factor(tstart, tend, -1, 0);
        if(P[i].Type == 0)
        {
            double ekin0 = 0.5 * P[i].Mass * P[i].Vel.norm_sq();
            Vec3<double> dvel = CellP[i].TurbAccel * dt_gravkick; Vec3<double> vtmp = P[i].Vel + dvel;
            double ekin1 = 0.5 * P[i].Mass * vtmp.norm_sq();
            CellP[i].EgyDrive += ekin1 - ekin0;
        }
    }
    CPU_Step[CPU_DRIFT] += measure_time();
}


/* routine to integrate the turbulent driving forces, specifically the 'TurbAccel' variables that need to be drifted and kicked: note that we actually do drifting and kicking in the normal routines, this is just to integrate the dissipation rates etc used for our tracking */
void do_turb_driving_step_second_half(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    int i; integertime ti_step, tstart, tend; double dt_gravkick;
    for (int i : ActiveParticleList)
    {
        ti_step = P[i].integertime_step(); tstart = P[i].Ti_begstep + ti_step / 2; tend = P[i].Ti_begstep + ti_step;	/* midpoint/end of step */
        dt_gravkick = get_gravkick_factor(tstart, tend, -1, 0);
        if(P[i].Type == 0)
        {
            double ekin0 = 0.5 * P[i].Mass * P[i].Vel.norm_sq();
            Vec3<double> dvel = CellP[i].TurbAccel * dt_gravkick; Vec3<double> vtmp = P[i].Vel + dvel;
            double ekin1 = 0.5 * P[i].Mass * vtmp.norm_sq();
            CellP[i].EgyDrive += ekin1 - ekin0;
        }
    }
    CPU_Step[CPU_DRIFT] += measure_time();
}


/* routine to record and optionally write to output files various statistics of driven turbulence here (most relevant to idealized sims with a hard-coded adiabatic EOS */
void log_turb_temp(void)
{
#ifdef OUTPUT_ADDITIONAL_RUNINFO
    int i; double dudt_drive = 0, dudt_diss = 0, mass = 0, ekin = 0, ethermal = 0;
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type == 0)
        {
            dudt_drive += P[i].Mass * CellP[i].DuDt_drive;
            dudt_diss += P[i].Mass * CellP[i].DuDt_diss;
            ekin += 0.5 * P[i].Mass * P[i].Vel.norm_sq();
            ethermal += P[i].Mass * CellP[i].InternalEnergy;
            mass += P[i].Mass;
        }
    }
    double glob_mass, glob_dudt_drive, glob_dudt_diss, glob_ekin, glob_ethermal;
    MPI_Allreduce(&mass, &glob_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&dudt_drive, &glob_dudt_drive, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&dudt_diss, &glob_dudt_diss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ekin, &glob_ekin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ethermal, &glob_ethermal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mach = sqrt(2.*glob_ekin / (GAMMA_DEFAULT*(GAMMA_DEFAULT-1)*glob_ethermal));
    
    if(ThisTask == 0)
    {
        fprintf(FdTurb, "%.16g %g %g %g %g %g %g\n", All.Time, mach, (glob_ekin + glob_ethermal) / glob_mass, glob_dudt_drive / glob_mass,
                glob_dudt_diss / glob_mass, All.TurbInjectedEnergy / glob_mass, All.TurbDissipatedEnergy / glob_mass); fflush(FdTurb);
    }
#endif
}


/* Per-TU init function: sets this TU's All_ptr to the shared UVM allocation */


#endif
