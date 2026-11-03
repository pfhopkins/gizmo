/* Standard and Kokkos headers must precede global_data_all_struct.h
 * (its macros may conflict with stdlib names). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include "../declarations/gizmo_rng.h"
#include <Kokkos_Core.hpp>

/* GPU All mirror: per-TU managed pointer to shared UVM allocation. */
#include "../declarations/gpu_all_mirror.h"
#include "../declarations/allvars.h"
#include "../declarations/gpu_error_check.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "../core/proto.h"
#include "../system/gpu_particles_arena.h"
#include "../mesh/ghost_writeback.h"

/* The device copy of the driving amplitudes and phases (see turb_mode_lattice_on_device): stale whenever the phases change. */
static double *turb_modes_dev = NULL;
static int turb_modes_dev_current = 0;   /* 0 once the phases change; the next device call copies the table up */

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
integertime StTUpdatedAt; // time at which the driving phases last changed (StTPrev advances by whole update intervals, so it can lag this)
static int StTInitialFieldPending = 0; // set on a start without restart files: the first force evaluation gives every active gas cell the initial field
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
    StOUPhases = (double*) mymalloc_movable(&StOUPhases,"StOUPhases", StNModes * 6 * sizeof(double));
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
    StTUpdatedAt = All.Ti_Current;
    if(RestartFlag != 1) {StTInitialFieldPending = 1;} /* restart files carry every cell's TurbAccel */
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
    turb_modes_dev_current = 0;   /* the device copy of the mode table is re-copied on its next use */
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
            const int num_local = ghost_get_num_local(); /* this can run while imported ghosts are live past the local cells: they copy their owners' accumulators */
            for(i=0; i < num_local; i++)
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
        StTUpdatedAt = All.Ti_Current;
        StTPrev = StTPrev + st_return_dt_between_updates() / All.Timebase_interval;
        PRINT_STATUS(" ..updated turbulent stirring field at time %f", StTPrev * All.Timebase_interval);
    }
}


/* The driving modes as the field evaluation reads them.  Every wavevector is 2 pi (ix/Lx, iy/Ly, iz/Lz) for integers ix, iy, iz,
 * so exp(i k.x) is the product of one complex factor per axis, each a power of exp(2 pi i x/L).  The lattice of integer triples
 * spanned by the modes is walked in order with those factors carried as running products, so the sum needs six sines and
 * cosines per position instead of two per mode, and no storage per position.  start/mode list, lattice point by lattice point
 * (x slowest), the modes sitting there: more than one mode can share a point (the generator emits sign variants that
 * coincide when an index is zero), and each keeps its own amplitude and phases.  The walk visits every point of the box the
 * modes span, so where that box holds many more points than there are modes (a thin shell of large wavenumbers) the direct
 * sum over the modes is used instead, chosen once for the run and the same on every route.  The pointers are host or device
 * memory to match the side that evaluates. */
struct TurbModeLattice
{
    int use_lattice;            /* 1: walk the lattice; 0: the direct sum over the modes */
    int nmodes;
    int lo[3], n[3];            /* lowest integer index and number of indices spanned, per axis */
    double k1[3];               /* 2 pi / box length, per axis */
    const int *start, *mode;    /* the modes at lattice point p are mode[start[p] .. start[p+1]-1] */
    const double *kvec, *aka, *akb, *ampl;
};

/* The driving field at one position: the sum over the Fourier modes, scaled by the solenoidal weight.  Device-callable;
 * every route below evaluates the field through this one function. */
KOKKOS_INLINE_FUNCTION
void turb_accel_at_position(double x, double y, double z, const struct TurbModeLattice &lat, double fac_sol, double f[3])
{
    double fx = 0, fy = 0, fz = 0;
    if(!lat.use_lattice)
    {
        for(int m = 0; m < lat.nmodes; m++)
        {
            const double kdotx = lat.kvec[3*m+0]*x + lat.kvec[3*m+1]*y + lat.kvec[3*m+2]*z;
            const double amp = lat.ampl[m], realt = cos(kdotx), imagt = sin(kdotx);
            fx += amp*(lat.aka[3*m+0]*realt - lat.akb[3*m+0]*imagt);
            fy += amp*(lat.aka[3*m+1]*realt - lat.akb[3*m+1]*imagt);
            fz += amp*(lat.aka[3*m+2]*realt - lat.akb[3*m+2]*imagt);
        }
        f[0] = fx * fac_sol; f[1] = fy * fac_sol; f[2] = fz * fac_sol;
        return;
    }
    const double px = lat.k1[0]*x, py = lat.k1[1]*y, pz = lat.k1[2]*z;   /* phase of one step along each axis */
    const double sxr = cos(px), sxi = sin(px), syr = cos(py), syi = sin(py), szr = cos(pz), szi = sin(pz);
    double cxr = cos(lat.lo[0]*px), cxi = sin(lat.lo[0]*px);
    const double cy0r = cos(lat.lo[1]*py), cy0i = sin(lat.lo[1]*py), cz0r = cos(lat.lo[2]*pz), cz0i = sin(lat.lo[2]*pz);
    int p = 0;
    for(int a = 0; a < lat.n[0]; a++)
    {
        double cyr = cy0r, cyi = cy0i;
        for(int b = 0; b < lat.n[1]; b++)
        {
            const double cxyr = cxr*cyr - cxi*cyi, cxyi = cxr*cyi + cxi*cyr;
            double czr = cz0r, czi = cz0i;
            for(int c = 0; c < lat.n[2]; c++, p++)
            {
                const int m0 = lat.start[p], m1 = lat.start[p+1];
                if(m1 > m0)
                {
                    const double realt = cxyr*czr - cxyi*czi, imagt = cxyr*czi + cxyi*czr;   /* cos and sin of k.x */
                    for(int q = m0; q < m1; q++)
                    {
                        const int m = lat.mode[q];
                        const double amp = lat.ampl[m];
                        fx += amp*(lat.aka[3*m+0]*realt - lat.akb[3*m+0]*imagt);
                        fy += amp*(lat.aka[3*m+1]*realt - lat.akb[3*m+1]*imagt);
                        fz += amp*(lat.aka[3*m+2]*realt - lat.akb[3*m+2]*imagt);
                    }
                }
                const double tz = czr*szr - czi*szi; czi = czr*szi + czi*szr; czr = tz;
            }
            const double ty = cyr*syr - cyi*syi; cyi = cyr*syi + cyi*syr; cyr = ty;
        }
        const double tx = cxr*sxr - cxi*sxi; cxi = cxr*sxi + cxi*sxr; cxr = tx;
    }
    f[0] = fx * fac_sol; f[1] = fy * fac_sol; f[2] = fz * fac_sol;
}

/* The acceleration a gas cell takes from the field at its position: none without mass, and only the run's dimensions. */
KOKKOS_INLINE_FUNCTION
void turb_accel_assign(struct gas_cell_data *cell, double mass, const double f[3])
{
    if(mass > 0.)
    {
        double acc[3];
        acc[0] = f[0]; acc[1] = acc[2] = 0;
#if (NUMDIMS > 1)
        acc[1] = f[1];
#endif
#if (NUMDIMS > 2)
        acc[2] = f[2];
#endif
        cell->TurbAccel = {acc[0], acc[1], acc[2]};
    } else {
        cell->TurbAccel = {};
    }
}

/* Per-particle turbulent acceleration -- device-callable. */
KOKKOS_FUNCTION
void add_turb_accel_for_particle(int i, struct particle_data *pp, struct gas_cell_data *cell,
                                  const struct TurbModeLattice &lat, double fac_sol)
{
    if(pp[i].Type != 0) return;
    double f[3];
    turb_accel_at_position(pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2], lat, fac_sol, f);
    turb_accel_assign(&cell[i], pp[i].Mass, f);
}

/* The lattice index of the modes, built once from StMode (which does not change after start-up or a restart reload): each
 * mode's integer indices, its axis range, and the modes listed lattice point by lattice point in mode order. */
/* The lattice walk is used while the box of lattice points the modes span holds at most this many points per mode: each
 * point costs a complex multiply where each mode of the direct sum costs a sine and a cosine.  A starting value, to be
 * priced, not a measured crossover. */
#define TURB_LATTICE_MAX_POINTS_PER_MODE 8
static int turb_lattice_built = 0, turb_lattice_use = 0, turb_lattice_lo[3], turb_lattice_n[3];
static double turb_lattice_k1[3];
static std::vector<int> turb_lattice_start, turb_lattice_mode;
static void turb_mode_lattice_build(void)
{
    if(turb_lattice_built) {return;}
    turb_lattice_k1[0] = 2.*M_PI / boxSize_X; turb_lattice_k1[1] = 2.*M_PI / boxSize_Y; turb_lattice_k1[2] = 2.*M_PI / boxSize_Z;
    std::vector<int> index(3 * (size_t) StNModes);
    int hi[3] = {0, 0, 0};
    for(int d = 0; d < 3; d++) {turb_lattice_lo[d] = 0;}   /* both set from the first mode; stay 0 with no modes */
    for(int m = 0; m < StNModes; m++) {
        for(int d = 0; d < 3; d++) {
            const double q = StMode[3*m+d] / turb_lattice_k1[d];
            const long ik = lround(q);
            if(fabs(q - (double) ik) > 1.e-6) {
                char msg[200];
                snprintf(msg, sizeof(msg), "turbulent driving: mode %d has a wavevector component %g that is not a whole multiple of 2 pi / box", m, StMode[3*m+d]);
                gizmo_request_controlled_stop(7716, msg, __FILE__, __LINE__, __FUNCTION__);
            }
            index[3*m+d] = (int) ik;
            if(m == 0 || ik < turb_lattice_lo[d]) {turb_lattice_lo[d] = (int) ik;}
            if(m == 0 || ik > hi[d]) {hi[d] = (int) ik;}
        }
    }
    for(int d = 0; d < 3; d++) {turb_lattice_n[d] = (StNModes > 0) ? hi[d] - turb_lattice_lo[d] + 1 : 0;}
    /* The box's point count, multiplied up only while it stays within the limit, so it cannot overflow. */
    const long long max_points = (long long) TURB_LATTICE_MAX_POINTS_PER_MODE * StNModes;
    long long npoints_ll = (StNModes > 0) ? 1 : 0;
    for(int d = 0; d < 3 && npoints_ll > 0; d++) {
        if(npoints_ll > max_points / turb_lattice_n[d]) {npoints_ll = -1; break;}   /* more points than the limit allows */
        npoints_ll *= turb_lattice_n[d];
    }
    turb_lattice_use = (npoints_ll > 0 && npoints_ll <= max_points) ? 1 : 0;
    if(turb_lattice_use) {PRINT_STATUS("Turbulent driving: %d modes span %lld lattice points; the field is summed over the lattice", StNModes, npoints_ll);}
    else {PRINT_STATUS("Turbulent driving: %d modes span more than %lld lattice points; the field is summed directly over the modes", StNModes, max_points);}
    turb_lattice_built = 1;
    if(!turb_lattice_use) {return;}
    const size_t npoints = (size_t) npoints_ll;
    turb_lattice_start.assign(npoints + 1, 0);
    turb_lattice_mode.assign((size_t) StNModes, 0);
    std::vector<size_t> point((size_t) StNModes);
    for(int m = 0; m < StNModes; m++) {
        point[m] = ((size_t)(index[3*m+0] - turb_lattice_lo[0]) * turb_lattice_n[1] + (index[3*m+1] - turb_lattice_lo[1])) * turb_lattice_n[2]
                   + (index[3*m+2] - turb_lattice_lo[2]);
        turb_lattice_start[point[m] + 1]++;
    }
    for(size_t p = 0; p < npoints; p++) {turb_lattice_start[p + 1] += turb_lattice_start[p];}
    std::vector<int> fill(turb_lattice_start.begin(), turb_lattice_start.end() - 1);
    for(int m = 0; m < StNModes; m++) {turb_lattice_mode[fill[point[m]]++] = m;}
}

/* The lattice view over host memory, for the host loops. */
static struct TurbModeLattice turb_mode_lattice_host(void)
{
    turb_mode_lattice_build();
    struct TurbModeLattice lat;
    lat.use_lattice = turb_lattice_use; lat.nmodes = StNModes;
    for(int d = 0; d < 3; d++) {lat.lo[d] = turb_lattice_lo[d]; lat.n[d] = turb_lattice_n[d]; lat.k1[d] = turb_lattice_k1[d];}
    lat.start = turb_lattice_start.data(); lat.mode = turb_lattice_mode.data();
    lat.kvec = StMode; lat.aka = StAka; lat.akb = StAkb; lat.ampl = StAmpl;
    return lat;
}

/* The device routes below take their per-call buffers (index list, compact positions and fields) for the call and free
 * them after it, as the particle staging does: the allocation costs little across a run, and holding them would keep
 * memory on every rank that only the turbulence update uses.  Only the mode tables stay on the device -- the lattice index,
 * copied once when the lattice is walked, and the wavevectors, amplitudes and phases (10 doubles per mode), copied again on the first device call after the phases
 * change; the restart reload and the start-up normalisation both precede the first call -- so host-only and tiny steps
 * never touch device memory for them.  Returns the view over device memory, or one with a null ampl on failure. */
static int *turb_lattice_dev = NULL;
static struct TurbModeLattice turb_mode_lattice_on_device(void)
{
    struct TurbModeLattice lat = turb_mode_lattice_host();
    const size_t n = (size_t) StNModes, nstart = turb_lattice_start.size();
    lat.ampl = NULL;
    if(turb_lattice_use && !turb_lattice_dev)
    {
        turb_lattice_dev = (int *) gizmo_gpu_alloc_device((nstart + n) * sizeof(int), "turb_driving_mode_lattice");
        if(!turb_lattice_dev) {return lat;}
        gizmo_gpu_copy_to_device(turb_lattice_dev, turb_lattice_start.data(), nstart * sizeof(int));
        gizmo_gpu_copy_to_device(turb_lattice_dev + nstart, turb_lattice_mode.data(), n * sizeof(int));
    }
    if(!turb_modes_dev)
    {
        turb_modes_dev = (double *) gizmo_gpu_alloc_device(10 * n * sizeof(double), "turb_driving_modes");
        if(!turb_modes_dev) {return lat;}
        turb_modes_dev_current = 0;
    }
    if(!turb_modes_dev_current)
    {
        std::vector<double> table(10 * n);
        memcpy(table.data(),         StAka,  3 * n * sizeof(double));
        memcpy(table.data() + 3 * n, StAkb,  3 * n * sizeof(double));
        memcpy(table.data() + 6 * n, StMode, 3 * n * sizeof(double));
        memcpy(table.data() + 9 * n, StAmpl,     n * sizeof(double));
        gizmo_gpu_copy_to_device(turb_modes_dev, table.data(), 10 * n * sizeof(double));
        turb_modes_dev_current = 1;
    }
    lat.start = turb_lattice_use ? turb_lattice_dev : NULL; lat.mode = turb_lattice_use ? turb_lattice_dev + nstart : NULL;
    lat.aka = turb_modes_dev; lat.akb = turb_modes_dev + 3 * n; lat.kvec = turb_modes_dev + 6 * n; lat.ampl = turb_modes_dev + 9 * n;
    return lat;
}


/* Whether gas cell i needs its TurbAccel recomputed. Without TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE every active cell is.
   With it, a cell's TurbAccel is the driving field as of the start of its step, so it is recomputed only once the phases
   have changed after that time: cells on long steps that are not active when the phases change pick up the new field
   the next time they are, and the cost stays proportional to the active cells. A cell whose step began at the time the
   phases changed already has that field: the phases change in add_turb_accel before its cells are chosen, and the
   cells that start a step at that time without being active -- woken or spawned -- are refreshed afterwards. */
static int turb_accel_needs_update(int i, integertime ti_step_start)
{
    if(P[i].Type != 0) {return 0;}
#ifdef TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE
    return (StTInitialFieldPending || StTUpdatedAt > ti_step_start);
#else
    return 1;
#endif
}

/* A cell that starts a new step without a force evaluation -- woken, or newly spawned -- is brought up to the current
   driving field here (a spawned cell passes -1, before any phase time, so it is always refreshed: its cloned acceleration
   belongs to another cell and position). */
void turb_accel_refresh_for_new_step(int i, integertime ti_previous_step_start)
{
#ifdef TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE
    if(turb_accel_needs_update(i, ti_previous_step_start))
        {add_turb_accel_for_particle(i, P, CellP, turb_mode_lattice_host(), 2.*solenoidal_frac_total_weight_renormalization());}
#endif
}

/* routine to actually calculate the turbulent acceleration 'driving field' force on every resolution element */
void add_turb_accel()
{
    set_turb_ampl();
    double fac_sol = 2.*solenoidal_frac_total_weight_renormalization();

    /* Build active gas particle index list */
    int N_active = 0;
    int *turb_indices = (int *) malloc(ActiveParticleList.size() * sizeof(int));
    for (int i : ActiveParticleList) {if(turb_accel_needs_update(i, P[i].Ti_begstep)) {turb_indices[N_active++] = i;}}
    StTInitialFieldPending = 0; /* cleared even when nothing is selected: every particle is active at this first call, so an empty list means this rank has no gas, and cells arriving later carry their TurbAccel with them */
    if(N_active == 0) {free(turb_indices); PRINT_STATUS("Finished turbulence driving (acceleration) computation"); return;}

    if(gizmo_gpu_offload_takes(GPU_OFFLOAD_PARTICLE_LOOP, N_active)) {
        GIZMO_GPU_ENSURE_ALL_FRESH(); /* All-mirror dispatch-boundary belt; the turb_accel kernel itself reads no All.* but keep the call at the GPU dispatch point per the All-mirror convention */
        /* In place where the device reaches the particle records without moving their pages (coherent memory,
           device-preferred storage, builds without a GPU): only the index list crosses.  Elsewhere the records sit
           host-side across the fabric, and the field needs only a position and gives back three numbers, so those are
           staged compactly, 48 bytes a cell, instead of having the kernel pull every cell's pages over. */
        const int in_place = gizmo_particle_storage_device_reads_in_place();
        const size_t n = (size_t) N_active;
        const struct TurbModeLattice lat = turb_mode_lattice_on_device();
        const double *modes = lat.ampl;   /* null when the device tables could not be had */
        int *index_dev = NULL; double *pos_dev = NULL, *field_dev = NULL;
        if(modes) {
            if(in_place) {index_dev = (int *) gizmo_gpu_alloc_device(n * sizeof(int), "turb_driving_index");}
            else {pos_dev   = (double *) gizmo_gpu_alloc_device(3 * n * sizeof(double), "turb_driving_positions");
                  field_dev = (double *) gizmo_gpu_alloc_device(3 * n * sizeof(double), "turb_driving_fields");}
        }
        if(!modes || (in_place ? !index_dev : (!pos_dev || !field_dev))) {
            if(index_dev) {gizmo_gpu_free_device(index_dev);}
            if(pos_dev)   {gizmo_gpu_free_device(pos_dev);}
            if(field_dev) {gizmo_gpu_free_device(field_dev);}
            free(turb_indices);
            char msg[256];
            snprintf(msg, sizeof(msg), "turbulent driving: could not get device memory for %d driving modes and %d gas cells (%.1f MB); "
                     "no driving acceleration is applied", StNModes, N_active,
                     (double)(10 * (size_t) StNModes * sizeof(double) + (in_place ? n * sizeof(int) : 6 * n * sizeof(double))) / (1024.0 * 1024.0));
            gizmo_request_controlled_stop(7715, msg, __FILE__, __LINE__, __FUNCTION__);
            return;
        }
        const double fs = fac_sol;
        if(in_place)
        {
            gizmo_gpu_copy_to_device(index_dev, turb_indices, n * sizeof(int));
            struct particle_data *kp = P;   /* captured by value: a lambda reaching for the globals would read host memory silently */
            struct gas_cell_data *kc = CellP;
            const int *kidx = index_dev;
            gizmo_gpu_kernel_launch("turb_accel_inplace", N_active, KOKKOS_LAMBDA(int j) {
                add_turb_accel_for_particle(kidx[j], kp, kc, lat, fs);
            });
            gizmo_gpu_free_device(index_dev);
        }
        else
        {
            std::vector<double> pos_host(3 * n), field_host(3 * n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(gizmo_host_threads_take(HOST_THREAD_BOOKKEEPING, N_active))
#endif
            for(int j = 0; j < N_active; j++)
            {
                const int i = turb_indices[j];
                pos_host[3*j+0] = P[i].Pos[0]; pos_host[3*j+1] = P[i].Pos[1]; pos_host[3*j+2] = P[i].Pos[2];
            }
            gizmo_gpu_copy_to_device(pos_dev, pos_host.data(), 3 * n * sizeof(double));
            const double *kpos = pos_dev;
            double *kfield = field_dev;
            gizmo_gpu_kernel_launch("turb_accel_compact", N_active, KOKKOS_LAMBDA(int j) {
                turb_accel_at_position(kpos[3*j+0], kpos[3*j+1], kpos[3*j+2], lat, fs, &kfield[3*j]);
            });
            gizmo_gpu_copy_to_host(field_host.data(), field_dev, 3 * n * sizeof(double));
            gizmo_gpu_free_device(pos_dev); gizmo_gpu_free_device(field_dev);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(gizmo_host_threads_take(HOST_THREAD_BOOKKEEPING, N_active))
#endif
            for(int j = 0; j < N_active; j++)
            {
                const int i = turb_indices[j];
                turb_accel_assign(&CellP[i], P[i].Mass, &field_host[3*j]);
            }
        }
        gpu_particles_arena_invalidate();   /* CellP's TurbAccel written; arena stale */

    } else
    { /* CPU path — tiny-N: run the SSOT kernel directly over the global
         P/CellP arrays via turb_indices, no compact gather/scatter, no GPU
         mirror sync (matches the grain_drag tiny-N cleanup philosophy). */
        const struct TurbModeLattice lat = turb_mode_lattice_host();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for(int j = 0; j < N_active; j++)
        {
            add_turb_accel_for_particle(turb_indices[j], P, CellP, lat, fac_sol);
        }
    } /* end CPU/GPU path selection */

    free(turb_indices);
    PRINT_STATUS("Finished turbulence driving (acceleration) computation");
}


/* The energy the driving puts into each active cell over one half of its step, kicked from the step's beginning to its
 * midpoint (first half) or from the midpoint to its end.  Each active writes only its own EgyDrive, so the loop runs on
 * several threads with the same result once there are enough of them. */
static void turb_driving_energy_half_kick(int second_half)
{
    CPU_Step[CPU_MISC] += measure_time();
    const int n_act = (int) ActiveParticleList.size();
    const int *act = ActiveParticleList.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(gizmo_host_threads_take(HOST_THREAD_BOOKKEEPING, n_act))
#endif
    for(int k_act = 0; k_act < n_act; k_act++)
    {
        const int i = act[k_act];
        if(P[i].Type == 0)
        {
            const integertime ti_step = P[i].integertime_step();
            const integertime tstart = second_half ? P[i].Ti_begstep + ti_step / 2 : P[i].Ti_begstep;   /* beginning or midpoint of step */
            const integertime tend   = second_half ? P[i].Ti_begstep + ti_step     : P[i].Ti_begstep + ti_step / 2;   /* midpoint or end */
            const double dt_gravkick = get_gravkick_factor(tstart, tend, i, 0);   /* the same factor the kick applies TurbAccel with (kicks.cc), dilation included */
            double ekin0 = 0.5 * P[i].Mass * P[i].Vel.norm_sq();
            Vec3<double> dvel = CellP[i].TurbAccel * dt_gravkick; Vec3<double> vtmp = P[i].Vel + dvel;
            double ekin1 = 0.5 * P[i].Mass * vtmp.norm_sq();
            CellP[i].EgyDrive += ekin1 - ekin0;
        }
    }
    CPU_Step[CPU_DRIFT] += measure_time();
}
void do_turb_driving_step_first_half(void) {turb_driving_energy_half_kick(0);}
void do_turb_driving_step_second_half(void) {turb_driving_energy_half_kick(1);}


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
