#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"

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
gsl_rng* StRng; // random number generator key


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
                        terminate("unknown spectral form");
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
    StRng = gsl_rng_alloc(gsl_rng_ranlxd1); // allocate seed variables
    gsl_rng_set(StRng, All.TurbDriving_Global_DrivingRandomNumberKey); // initialize seed
    int j; for(j=0;j<100;j++) {double tmp; tmp=st_turbdrive_get_gaussian_random_variable();} // cycle past initial seed
    st_turbdrive_init_ouseq(); // initialize variable for phases
    st_turbdrive_calc_phases(); // initialize phases
#ifdef TURB_DRIVING_TARGET_SIGMA
    All.TurbDriving_DrivingNormFactor = 1.0; All.TurbDriving_PID_Integral = All.TurbDriving_PID_Derivative = All.TurbDriving_PID_LastError = 0;
    All.TurbDriving_PID_Started = 0;
#endif
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
    return All.TurbDriving_Global_AccelerationPowerVariable / st_return_mode_correlation_time();
}


#ifdef TURB_DRIVING_TARGET_SIGMA
/* mass-weighted rms velocity of gas about its mean velocity; if TURB_DRIVING_TARGET_SIGMA > 0, only gas
   within that radius of the box center: in code units, or with STARFORGE_GMC_TURBINIT in units of the nominal
   cloud radius. Returns the target if there is no gas to measure. */
static double st_compute_vrms(void)
{
    double sums[5] = {0}; /* m, m*v[3], m*v^2 */
#ifdef STARFORGE_GMC_TURBINIT
    const double aperture = (double)(TURB_DRIVING_TARGET_SIGMA) * STARFORGE_GMC_TURBINIT_CLOUD_RADIUS_IN_BOXSIZE * All.BoxSize;
#else
    const double aperture = (double)(TURB_DRIVING_TARGET_SIGMA);
#endif
    const double box_cx = 0.5*boxSize_X, box_cy = 0.5*boxSize_Y, box_cz = 0.5*boxSize_Z;
    for(int i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 0 || P[i].Mass <= 0) continue;
        if(aperture > 0) {
            double dx = P[i].Pos[0] - box_cx, dy = P[i].Pos[1] - box_cy, dz = P[i].Pos[2] - box_cz;
            if(dx*dx + dy*dy + dz*dz > aperture*aperture) continue;
        }
        sums[0] += P[i].Mass;
        for(int k=0; k<3; k++) {sums[1+k] += P[i].Mass * P[i].Vel[k];}
        sums[4] += P[i].Mass * P[i].Vel.norm_sq();
    }
    double glob[5];
    MPI_Allreduce(sums, glob, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(glob[0] <= 0) return All.TurbDriving_Global_AccelerationPowerVariable;
    double vmean2 = 0; for(int k=0; k<3; k++) {vmean2 += (glob[1+k]/glob[0]) * (glob[1+k]/glob[0]);}
    return sqrt(DMAX(glob[4]/glob[0] - vmean2, 0.));
}


/* PID loop setting the gain on the driving force from e = ln(v_target / v_rms), called every driving update.
   Working in ln(gain) keeps the gain positive and makes the response to over- and under-shoot symmetric.
   Gains are dimensionless, with times in units of t_eddy = TurbDrive_MaxWavelength / TurbDrive_ApproxRMSVturb. */
static void st_update_driving_gain(void)
{
    const double v_target = All.TurbDriving_Global_AccelerationPowerVariable, t_eddy = st_return_driving_scale() / v_target;
    const double dt = st_return_dt_between_updates();
    const double e_max = log(2.); /* error clamp: bounds the P term and the integration rate far from the target */
    const double lng_max = log(100.); /* gain limits, 1/100 to 100 */
    const double Kp = All.TurbDriving_PID_Kp, Ti = All.TurbDriving_PID_Ti * t_eddy, Td = All.TurbDriving_PID_Td * t_eddy, Tf = 0.1 * Td; /* Tf: derivative filter time */

    double v = st_compute_vrms();
    double e = log(v_target / DMAX(v, 1.e-3*v_target)); e = DMAX(-e_max, DMIN(e_max, e));
    int first_update = !All.TurbDriving_PID_Started;
    if(first_update) /* bumpless start: begin at gain 1, with no derivative kick */
        {All.TurbDriving_PID_LastError = e; All.TurbDriving_PID_Integral = -Kp * e; All.TurbDriving_PID_Started = 1;}

    All.TurbDriving_PID_Derivative = (Tf * All.TurbDriving_PID_Derivative + Kp * Td * (e - All.TurbDriving_PID_LastError)) / (Tf + dt);
    All.TurbDriving_PID_LastError = e;
    double p_term = Kp * e, integral = All.TurbDriving_PID_Integral;
    if(Ti > 0 && !first_update) {integral += Kp * e * dt / Ti;} /* else, with a long update interval, the first interval would not run at gain 1 */
    double lng = p_term + integral + All.TurbDriving_PID_Derivative;
    if(fabs(lng) < lng_max || lng * e < 0) {All.TurbDriving_PID_Integral = integral;} /* anti-windup: no integration that would push further into saturation */
    lng = p_term + All.TurbDriving_PID_Integral + All.TurbDriving_PID_Derivative;
    All.TurbDriving_DrivingNormFactor = exp(DMAX(-lng_max, DMIN(lng_max, lng)));

    if(ThisTask == 0) {
        printf("TURB_DRIVING_TARGET_SIGMA: t=%g v_rms=%g v_target=%g gain=%g P=%g I=%g D=%g\n", All.Time, v, v_target,
               All.TurbDriving_DrivingNormFactor, p_term, All.TurbDriving_PID_Integral, All.TurbDriving_PID_Derivative);
        fflush(stdout);
    }
}
#endif


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
    double r0 = gsl_rng_uniform(StRng), r1 = gsl_rng_uniform(StRng);
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
    double delta = (All.Ti_Current - StTPrev) * unit_integertime_in_physical(-1), Dt_Update=st_return_dt_between_updates();
    if(delta >= Dt_Update){return 1;} else {return 0;}
}

/* parent routine to initialize and update turbulent driving fields and to track different variables used for analyzing power spectra of dissipation, etc. */
void set_turb_ampl(void)
{
    if(new_turbforce_needed_this_timestep())
    {
        double delta = (All.Ti_Current - StTPrev) * unit_integertime_in_physical(-1);
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
#ifdef TURB_DRIVING_TARGET_SIGMA
        st_update_driving_gain();
#endif
        StTPrev = StTPrev + st_return_dt_between_updates() / All.Timebase_interval;
        PRINT_STATUS(" ..updated turbulent stirring field at time %f", StTPrev * All.Timebase_interval);
    }
}


/* sum the driving modes at the positions of the gas cells in 'cells' and store the result in their TurbAccel.
   The mode wavevectors lie on the integer lattice k = 2pi (nx/Lx, ny/Ly, nz/Lz), so exp(i k.x) is the product of
   three one-dimensional factors exp(i n theta_a), theta_a = 2pi x_a/L_a. These are tabulated per cell for every n
   the modes use, which takes the transcendental functions out of the sum over modes. Cells are processed in tiles of
   ST_TILE with the cell index innermost, so the sum vectorizes over cells and each pass over the (large) mode arrays
   serves a whole tile. */
#define ST_TILE 16
static void st_evaluate_driving_force(const std::vector<int> &cells, double fac_sol)
{
    const double L[3] = {boxSize_X, boxSize_Y, boxSize_Z};
    std::vector<int> nk[3]; std::vector<double> ca[3], cb[3]; int nmax[3] = {0, 0, 0};
    for(int a = 0; a < 3; a++) {nk[a].resize(StNModes); ca[a].resize(StNModes); cb[a].resize(StNModes);}
    for(int m = 0; m < StNModes; m++)
    {
        for(int a = 0; a < 3; a++)
        {
            nk[a][m] = (int) lround(StMode[3*m+a] * L[a] / (2.*M_PI));
            nmax[a] = std::max(nmax[a], std::abs(nk[a][m]));
            ca[a][m] = fac_sol * StAmpl[m] * StAka[3*m+a];
            cb[a][m] = fac_sol * StAmpl[m] * StAkb[3*m+a];
        }
    }
    for(int a = 0; a < 3; a++) {for(int m = 0; m < StNModes; m++) {nk[a][m] += nmax[a];}} /* offset into the tables below */
    const int ntiles = ((int)cells.size() + ST_TILE - 1) / ST_TILE;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> ctab[3], stab[3]; /* cos, sin(n theta_a) for n = -nmax..nmax, cell index fastest */
        for(int a = 0; a < 3; a++) {ctab[a].resize((2*nmax[a]+1)*ST_TILE); stab[a].resize((2*nmax[a]+1)*ST_TILE);}
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for(int tile = 0; tile < ntiles; tile++)
        {
            const int i0 = tile * ST_TILE, nt = std::min(ST_TILE, (int)cells.size() - i0);
            for(int a = 0; a < 3; a++)
            {
                for(int t = 0; t < ST_TILE; t++)
                {
                    double theta = (t < nt) ? 2.*M_PI * P[cells[i0+t]].Pos[a] / L[a] : 0; /* lanes past the last cell are padding */
                    for(int n = -nmax[a]; n <= nmax[a]; n++) {ctab[a][(n+nmax[a])*ST_TILE + t] = cos(n*theta); stab[a][(n+nmax[a])*ST_TILE + t] = sin(n*theta);}
                }
            }
            double fx[ST_TILE] = {0}, fy[ST_TILE] = {0}, fz[ST_TILE] = {0};
            for(int m = 0; m < StNModes; m++)
            {
                const double *cx = &ctab[0][nk[0][m]*ST_TILE], *sx = &stab[0][nk[0][m]*ST_TILE];
                const double *cy = &ctab[1][nk[1][m]*ST_TILE], *sy = &stab[1][nk[1][m]*ST_TILE];
                const double *cz = &ctab[2][nk[2][m]*ST_TILE], *sz = &stab[2][nk[2][m]*ST_TILE];
                const double cax = ca[0][m], cay = ca[1][m], caz = ca[2][m], cbx = cb[0][m], cby = cb[1][m], cbz = cb[2][m];
#ifdef _OPENMP
#pragma omp simd
#endif
                for(int t = 0; t < ST_TILE; t++)
                {
                    double rxy = cx[t]*cy[t] - sx[t]*sy[t], ixy = cx[t]*sy[t] + sx[t]*cy[t];
                    double re = rxy*cz[t] - ixy*sz[t], im = rxy*sz[t] + ixy*cz[t]; /* cos, sin(k.x) */
                    fx[t] += cax*re - cbx*im; fy[t] += cay*re - cby*im; fz[t] += caz*re - cbz*im;
                }
            }
            for(int t = 0; t < nt; t++)
            {
                int i = cells[i0+t];
                if(P[i].Mass > 0.)
                {
                    Vec3<double> acc = {fx[t], 0, 0};
#if (NUMDIMS > 1)
                    acc[1] = fy[t];
#endif
#if (NUMDIMS > 2)
                    acc[2] = fz[t];
#endif
                    CellP[i].TurbAccel = acc;
                } else {
                    CellP[i].TurbAccel = {};
                }
            }
        }
    }
}


/* routine to actually calculate the turbulent acceleration 'driving field' force on every resolution element */
void add_turb_accel()
{
    set_turb_ampl();
    double fac_sol = 2.*solenoidal_frac_total_weight_renormalization();
#ifdef TURB_DRIVING_TARGET_SIGMA
    fac_sol *= All.TurbDriving_DrivingNormFactor; /* applied to the force, not the OU noise amplitude, so it acts immediately */
#endif
    std::vector<int> cells;
#ifdef TURB_DRIVING_UPDATE_FORCE_ON_TURBUPDATE
    /* called only on driving updates, so refresh every cell: otherwise a cell keeps its old force (and gain) until it
       happens to be active on an update, which for coarse timebins can be many coherence times. Inactive cells are
       evaluated at their last-drifted position, at most one of their own steps old, well below the driving scale. */
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0) {cells.push_back(i);}}
#else
    for(int i : ActiveParticleList) {if(P[i].Type == 0) {cells.push_back(i);}}
#endif
    st_evaluate_driving_force(cells, fac_sol);
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


#endif
