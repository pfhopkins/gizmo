#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <hdf5.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "resolvedism_stellar_tables.h"

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES

struct StellarTables StellarTbl;

/* ========================================================================
 *  Index computation on regular log-uniform grids.
 *  Grid spacing is constant → index = (x - x_min) / dx.  O(1), no search.
 * ======================================================================== */

static inline void stbl_idx_Z(double logZ, int *i0, double *f)
{
    double x = (logZ - StellarTbl.log_Z_min) / StellarTbl.dlog_Z;
    if(x < 0) x = 0;
    if(x >= STBL_NZ - 1) { *i0 = STBL_NZ - 2; *f = 1.0; return; }
    *i0 = (int)x;
    *f  = x - *i0;
}

static inline void stbl_idx_M(double logM, int *i0, double *f)
{
    double x = (logM - StellarTbl.log_M_min) / StellarTbl.dlog_M;
    if(x < 0) x = 0;
    if(x >= STBL_NM - 1) { *i0 = STBL_NM - 2; *f = 1.0; return; }
    *i0 = (int)x;
    *f  = x - *i0;
}

static inline void stbl_idx_age(double log_age, int *i0, double *f)
{
    double x = (log_age - StellarTbl.log_age_min) / StellarTbl.dlog_age;
    if(x < 0) x = 0;
    if(x >= STBL_NAGE - 1) { *i0 = STBL_NAGE - 2; *f = 1.0; return; }
    *i0 = (int)x;
    *f  = x - *i0;
}

/* flat index helpers */
#define IDX2(iz,im)       ((iz) * STBL_NM + (im))
#define IDX3(iz,im,ia)    (((iz) * STBL_NM + (im)) * STBL_NAGE + (ia))
#define IDX4(iz,im,ia,ie) ((((iz) * STBL_NM + (im)) * STBL_NAGE + (ia)) * STBL_NELEM + (ie))

/* ========================================================================
 *  Bilinear interpolation on 2D [NZ x NM] arrays (stored flat)
 * ======================================================================== */
static inline double interp2d(const double *arr, double logM, double logZ)
{
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    double v00 = arr[IDX2(iz,   im  )];
    double v01 = arr[IDX2(iz,   im+1)];
    double v10 = arr[IDX2(iz+1, im  )];
    double v11 = arr[IDX2(iz+1, im+1)];
    return (1-fz)*(1-fm)*v00 + (1-fz)*fm*v01 + fz*(1-fm)*v10 + fz*fm*v11;
}

/* ========================================================================
 *  Trilinear interpolation on 3D [NZ x NM x NAGE] float arrays
 * ======================================================================== */
static inline double interp3d(const float *arr, double logM, double logZ, double log_age)
{
    int iz, im, ia; double fz, fm, fa;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    stbl_idx_age(log_age, &ia, &fa);

    double v000 = arr[IDX3(iz,   im,   ia  )];
    double v001 = arr[IDX3(iz,   im,   ia+1)];
    double v010 = arr[IDX3(iz,   im+1, ia  )];
    double v011 = arr[IDX3(iz,   im+1, ia+1)];
    double v100 = arr[IDX3(iz+1, im,   ia  )];
    double v101 = arr[IDX3(iz+1, im,   ia+1)];
    double v110 = arr[IDX3(iz+1, im+1, ia  )];
    double v111 = arr[IDX3(iz+1, im+1, ia+1)];

    double c00 = v000*(1-fa) + v001*fa;
    double c01 = v010*(1-fa) + v011*fa;
    double c10 = v100*(1-fa) + v101*fa;
    double c11 = v110*(1-fa) + v111*fa;
    double c0  = c00*(1-fm)  + c01*fm;
    double c1  = c10*(1-fm)  + c11*fm;
    return c0*(1-fz) + c1*fz;
}

/* ========================================================================
 *  HDF5 helper: read a dataset into a pre-allocated buffer
 * ======================================================================== */
static void read_hdf5_dataset_double(hid_t file, const char *name, double *buf, size_t n)
{
    hid_t ds = H5Dopen(file, name);
    if(ds < 0) { printf("ERROR: cannot open HDF5 dataset '%s'\n", name); endrun(1); }
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(ds);
}

/* Read double dataset into a float buffer (downcast to save memory) */
static void read_hdf5_dataset_as_float(hid_t file, const char *name, float *buf, size_t n)
{
    double *tmp = (double *)malloc(n * sizeof(double));
    if(!tmp) { printf("ERROR: malloc failed for dataset '%s' (%zu doubles)\n", name, n); endrun(1); }
    read_hdf5_dataset_double(file, name, tmp, n);
    for(size_t i = 0; i < n; i++) buf[i] = (float)tmp[i];
    free(tmp);
}

static void read_hdf5_dataset_int(hid_t file, const char *name, int *buf, size_t n)
{
    /* remnant_type is stored as int64 in HDF5, read via native long then cast */
    long *tmp = (long *)malloc(n * sizeof(long));
    if(!tmp) { printf("ERROR: malloc failed for dataset '%s'\n", name); endrun(1); }
    hid_t ds = H5Dopen(file, name);
    if(ds < 0) { printf("ERROR: cannot open HDF5 dataset '%s'\n", name); endrun(1); }
    H5Dread(ds, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp);
    H5Dclose(ds);
    for(size_t i = 0; i < n; i++) buf[i] = (int)tmp[i];
    free(tmp);
}

/* ========================================================================
 *  Load the unified stellar tables from HDF5
 * ======================================================================== */
void resolvedism_load_stellar_tables(void)
{
    if(ThisTask == 0) printf("ResolvedISM: loading stellar tables from %s\n", All.StellarTablesFile);

    size_t n3d = (size_t)STBL_NZ * STBL_NM * STBL_NAGE;

    /* Allocate 3D arrays (float to save memory: ~33 MB each, ~230 MB total for 7 arrays) */
    StellarTbl.M_current       = (float *)mymalloc("stbl_Mcur",    n3d * sizeof(float));
    StellarTbl.v_wind          = (float *)mymalloc("stbl_vwind",   n3d * sizeof(float));
    StellarTbl.log_Q_ion       = (float *)mymalloc("stbl_logQion", n3d * sizeof(float));
    StellarTbl.log_L_FUV       = (float *)mymalloc("stbl_logLFUV", n3d * sizeof(float));
    StellarTbl.log_L_LW        = (float *)mymalloc("stbl_logLLW",  n3d * sizeof(float));
    StellarTbl.log_L_FUV_total = (float *)mymalloc("stbl_logLFUVt",n3d * sizeof(float));
    StellarTbl.log_L_bol       = (float *)mymalloc("stbl_logLbol", n3d * sizeof(float));

    /* Ionizing sub-band luminosities for M1 RT */
#ifdef RADTRANSFER
    StellarTbl.log_L_ion_tot   = (float *)mymalloc("stbl_logLiont", n3d * sizeof(float));
    StellarTbl.log_L_ion_H0    = (float *)mymalloc("stbl_logLiH0",  n3d * sizeof(float));
    StellarTbl.log_L_ion_He0   = (float *)mymalloc("stbl_logLiHe0", n3d * sizeof(float));
    StellarTbl.log_L_ion_He1   = (float *)mymalloc("stbl_logLiHe1", n3d * sizeof(float));
    StellarTbl.log_L_ion_He2   = (float *)mymalloc("stbl_logLiHe2", n3d * sizeof(float));
#else
    StellarTbl.log_L_ion_tot = StellarTbl.log_L_ion_H0 = StellarTbl.log_L_ion_He0 = NULL;
    StellarTbl.log_L_ion_He1 = StellarTbl.log_L_ion_He2 = NULL;
#endif

    /* Surface abundances: only if winds enabled (~500 MB as float) */
    StellarTbl.surface_abundances = NULL;
#ifdef GALSF_RESOLVEDISM_WINDS
    size_t n4d = n3d * STBL_NELEM;
    StellarTbl.surface_abundances = (float *)mymalloc("stbl_surfab", n4d * sizeof(float));
#endif

    /* Rank 0 reads HDF5, then broadcasts */
    if(ThisTask == 0)
    {
        hid_t file = H5Fopen(All.StellarTablesFile, H5F_ACC_RDONLY, H5P_DEFAULT);
        if(file < 0) { printf("ERROR: cannot open stellar tables file '%s'\n", All.StellarTablesFile); endrun(1); }

        /* 1D grid axes */
        read_hdf5_dataset_double(file, "log_Z",      StellarTbl.log_Z,   STBL_NZ);
        read_hdf5_dataset_double(file, "Z",           StellarTbl.Z,       STBL_NZ);
        read_hdf5_dataset_double(file, "log_M_init",  StellarTbl.log_M,   STBL_NM);
        double M_tmp[STBL_NM];
        read_hdf5_dataset_double(file, "M_init",      M_tmp,              STBL_NM);
        for(int i = 0; i < STBL_NM; i++) StellarTbl.M[i] = M_tmp[i];
        read_hdf5_dataset_double(file, "log_age_yr",  StellarTbl.log_age, STBL_NAGE);

        /* 2D scalar datasets [NZ x NM] */
        size_t n2d = (size_t)STBL_NZ * STBL_NM;
        read_hdf5_dataset_double(file, "lifetime_yr",  StellarTbl.lifetime_yr,  n2d);
        read_hdf5_dataset_double(file, "remnant_mass", StellarTbl.remnant_mass, n2d);
        read_hdf5_dataset_int   (file, "remnant_type", StellarTbl.remnant_type, n2d);
        read_hdf5_dataset_double(file, "M_preSN",      StellarTbl.M_preSN,     n2d);
        read_hdf5_dataset_double(file, "M_CO_core",    StellarTbl.M_CO_core,   n2d);
        read_hdf5_dataset_double(file, "M_He_core",    StellarTbl.M_He_core,   n2d);

        /* 3D time-dependent datasets */
        read_hdf5_dataset_as_float(file, "M_current",      StellarTbl.M_current,       n3d);
        read_hdf5_dataset_as_float(file, "v_wind",         StellarTbl.v_wind,          n3d);
        read_hdf5_dataset_as_float(file, "log_Q_ion",      StellarTbl.log_Q_ion,       n3d);
        read_hdf5_dataset_as_float(file, "log_L_FUV",      StellarTbl.log_L_FUV,       n3d);
        read_hdf5_dataset_as_float(file, "log_L_LW",       StellarTbl.log_L_LW,        n3d);
        read_hdf5_dataset_as_float(file, "log_L_FUV_total",StellarTbl.log_L_FUV_total, n3d);
        read_hdf5_dataset_as_float(file, "log_L_bol",      StellarTbl.log_L_bol,       n3d);

        /* Ionizing sub-band luminosities (only for M1 RT) */
#ifdef RADTRANSFER
        read_hdf5_dataset_as_float(file, "log_L_ion_tot",  StellarTbl.log_L_ion_tot,   n3d);
        read_hdf5_dataset_as_float(file, "log_L_ion_H0",   StellarTbl.log_L_ion_H0,    n3d);
        read_hdf5_dataset_as_float(file, "log_L_ion_He0",  StellarTbl.log_L_ion_He0,   n3d);
        read_hdf5_dataset_as_float(file, "log_L_ion_He1",  StellarTbl.log_L_ion_He1,   n3d);
        read_hdf5_dataset_as_float(file, "log_L_ion_He2",  StellarTbl.log_L_ion_He2,   n3d);
#endif

        /* Surface abundances */
#ifdef GALSF_RESOLVEDISM_WINDS
        read_hdf5_dataset_as_float(file, "surface_abundances", StellarTbl.surface_abundances, n3d * STBL_NELEM);
#endif

        /* Net yields [NZ x NM x NELEM] */
        read_hdf5_dataset_double(file, "net_yields", StellarTbl.net_yields, (size_t)STBL_NZ * STBL_NM * STBL_NELEM);

        /* Type Ia yields [NELEM] */
#ifdef GALSF_RESOLVEDISM_TYPE_IA
        read_hdf5_dataset_double(file, "type_ia_yields", StellarTbl.type_ia_yields, STBL_NELEM);
#else
        memset(StellarTbl.type_ia_yields, 0, STBL_NELEM * sizeof(double));
#endif

        H5Fclose(file);
    }

    /* Broadcast everything from rank 0 */
    MPI_Bcast(StellarTbl.log_Z,   STBL_NZ,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.Z,       STBL_NZ,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_M,   STBL_NM,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.M,       STBL_NM,   MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_age, STBL_NAGE,  MPI_DOUBLE, 0, MPI_COMM_WORLD);

    size_t n2d = (size_t)STBL_NZ * STBL_NM;
    MPI_Bcast(StellarTbl.lifetime_yr,  n2d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.remnant_mass, n2d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.remnant_type, n2d, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.M_preSN,     n2d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.M_CO_core,   n2d, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.M_He_core,   n2d, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(StellarTbl.M_current,       n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.v_wind,          n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_Q_ion,       n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_FUV,       n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_LW,        n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_FUV_total, n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_bol,       n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef RADTRANSFER
    MPI_Bcast(StellarTbl.log_L_ion_tot,   n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_ion_H0,    n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_ion_He0,   n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_ion_He1,   n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.log_L_ion_He2,   n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

#ifdef GALSF_RESOLVEDISM_WINDS
    /* Surface abundances are large — broadcast in chunks to avoid MPI limits */
    size_t chunk = 100000000; /* 100M floats per chunk (~400 MB) */
    for(size_t offset = 0; offset < n4d; offset += chunk) {
        size_t this_chunk = (offset + chunk > n4d) ? (n4d - offset) : chunk;
        MPI_Bcast(StellarTbl.surface_abundances + offset, (int)this_chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
#endif

    MPI_Bcast(StellarTbl.net_yields,     STBL_NZ * STBL_NM * STBL_NELEM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.type_ia_yields, STBL_NELEM,                      MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Compute grid spacing for O(1) index lookup */
    StellarTbl.log_Z_min = StellarTbl.log_Z[0];
    StellarTbl.log_Z_max = StellarTbl.log_Z[STBL_NZ - 1];
    StellarTbl.dlog_Z    = (StellarTbl.log_Z_max - StellarTbl.log_Z_min) / (STBL_NZ - 1);

    StellarTbl.log_M_min = StellarTbl.log_M[0];
    StellarTbl.log_M_max = StellarTbl.log_M[STBL_NM - 1];
    StellarTbl.dlog_M    = (StellarTbl.log_M_max - StellarTbl.log_M_min) / (STBL_NM - 1);

    StellarTbl.log_age_min = StellarTbl.log_age[0];
    StellarTbl.log_age_max = StellarTbl.log_age[STBL_NAGE - 1];
    StellarTbl.dlog_age    = (StellarTbl.log_age_max - StellarTbl.log_age_min) / (STBL_NAGE - 1);

    StellarTbl.loaded = 1;

    if(ThisTask == 0) {
        printf("ResolvedISM: stellar tables loaded. Grid: %d Z x %d M x %d ages\n", STBL_NZ, STBL_NM, STBL_NAGE);
        printf("  logZ: [%.3f, %.3f], dlogZ=%.4f\n", StellarTbl.log_Z_min, StellarTbl.log_Z_max, StellarTbl.dlog_Z);
        printf("  logM: [%.3f, %.3f], dlogM=%.4f\n", StellarTbl.log_M_min, StellarTbl.log_M_max, StellarTbl.dlog_M);
        printf("  log_age: [%.3f, %.3f], dlog_age=%.4f\n", StellarTbl.log_age_min, StellarTbl.log_age_max, StellarTbl.dlog_age);
        printf("  Memory: %.1f MB (3D arrays) + %.1f MB (surface abundances)\n",
               12.0 * n3d * sizeof(float) / 1e6,
               StellarTbl.surface_abundances ? (double)(n3d * STBL_NELEM * sizeof(float)) / 1e6 : 0.0);

        /* Spot-check: print a few values for verification */
        double logM_test = log10(20.0), logZ_test = log10(0.014);
        printf("  Spot check (20 Msun, Z=0.014):\n");
        printf("    lifetime = %.3f Myr\n", stellar_lifetime(logM_test, logZ_test) / 1e6);
        printf("    remnant_mass = %.2f Msun, type = %d\n",
               stellar_remnant_mass(logM_test, logZ_test), stellar_remnant_type(logM_test, logZ_test));
        printf("    M_current(ZAMS) = %.2f Msun\n", stellar_M_current(logM_test, logZ_test, 2.0));
        printf("    log_Q_ion(ZAMS) = %.2f\n", stellar_log_Q_ion(logM_test, logZ_test, 2.0));
    }
}


void resolvedism_free_stellar_tables(void)
{
#ifdef GALSF_RESOLVEDISM_WINDS
    if(StellarTbl.surface_abundances) { myfree(StellarTbl.surface_abundances); StellarTbl.surface_abundances = NULL; }
#endif
#ifdef RADTRANSFER
    if(StellarTbl.log_L_ion_He2)   { myfree(StellarTbl.log_L_ion_He2);   StellarTbl.log_L_ion_He2 = NULL; }
    if(StellarTbl.log_L_ion_He1)   { myfree(StellarTbl.log_L_ion_He1);   StellarTbl.log_L_ion_He1 = NULL; }
    if(StellarTbl.log_L_ion_He0)   { myfree(StellarTbl.log_L_ion_He0);   StellarTbl.log_L_ion_He0 = NULL; }
    if(StellarTbl.log_L_ion_H0)    { myfree(StellarTbl.log_L_ion_H0);    StellarTbl.log_L_ion_H0 = NULL; }
    if(StellarTbl.log_L_ion_tot)   { myfree(StellarTbl.log_L_ion_tot);   StellarTbl.log_L_ion_tot = NULL; }
#endif
    if(StellarTbl.log_L_bol)       { myfree(StellarTbl.log_L_bol);       StellarTbl.log_L_bol = NULL; }
    if(StellarTbl.log_L_FUV_total) { myfree(StellarTbl.log_L_FUV_total); StellarTbl.log_L_FUV_total = NULL; }
    if(StellarTbl.log_L_LW)        { myfree(StellarTbl.log_L_LW);        StellarTbl.log_L_LW = NULL; }
    if(StellarTbl.log_L_FUV)       { myfree(StellarTbl.log_L_FUV);       StellarTbl.log_L_FUV = NULL; }
    if(StellarTbl.log_Q_ion)       { myfree(StellarTbl.log_Q_ion);       StellarTbl.log_Q_ion = NULL; }
    if(StellarTbl.v_wind)          { myfree(StellarTbl.v_wind);          StellarTbl.v_wind = NULL; }
    if(StellarTbl.M_current)       { myfree(StellarTbl.M_current);       StellarTbl.M_current = NULL; }
    StellarTbl.loaded = 0;
}


/* ========================================================================
 *  Public interpolation functions
 * ======================================================================== */

/* --- 2D bilinear (logM, logZ) for scalar quantities --- */

double stellar_lifetime(double logM, double logZ)
{
    return interp2d(StellarTbl.lifetime_yr, logM, logZ);
}

double stellar_remnant_mass(double logM, double logZ)
{
    return interp2d(StellarTbl.remnant_mass, logM, logZ);
}

int stellar_remnant_type(double logM, double logZ)
{
    /* Nearest-neighbor for discrete values */
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    int iz_near = (fz > 0.5) ? iz + 1 : iz;
    int im_near = (fm > 0.5) ? im + 1 : im;
    if(iz_near >= STBL_NZ) iz_near = STBL_NZ - 1;
    if(im_near >= STBL_NM) im_near = STBL_NM - 1;
    return StellarTbl.remnant_type[IDX2(iz_near, im_near)];
}

double stellar_M_preSN(double logM, double logZ)
{
    return interp2d(StellarTbl.M_preSN, logM, logZ);
}

/* --- 3D trilinear (logM, logZ, log_age) --- */

double stellar_M_current(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.M_current, logM, logZ, log_age);
}

double stellar_v_wind(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.v_wind, logM, logZ, log_age);
}

double stellar_log_Q_ion(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_Q_ion, logM, logZ, log_age);
}

double stellar_log_L_FUV(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_FUV, logM, logZ, log_age);
}

double stellar_log_L_LW(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_LW, logM, logZ, log_age);
}

double stellar_log_L_FUV_total(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_FUV_total, logM, logZ, log_age);
}

double stellar_log_L_bol(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_bol, logM, logZ, log_age);
}

double stellar_log_L_ion_tot(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_ion_tot, logM, logZ, log_age);
}

double stellar_log_L_ion_H0(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_ion_H0, logM, logZ, log_age);
}

double stellar_log_L_ion_He0(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_ion_He0, logM, logZ, log_age);
}

double stellar_log_L_ion_He1(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_ion_He1, logM, logZ, log_age);
}

double stellar_log_L_ion_He2(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.log_L_ion_He2, logM, logZ, log_age);
}

/* --- Surface abundances --- */

double stellar_surface_abundance(double logM, double logZ, double log_age, int elem)
{
    if(!StellarTbl.surface_abundances || elem < 0 || elem >= STBL_NELEM) return 0;

    int iz, im, ia; double fz, fm, fa;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    stbl_idx_age(log_age, &ia, &fa);

    /* 8-corner trilinear with fixed element index */
    double v000 = StellarTbl.surface_abundances[IDX4(iz,   im,   ia,   elem)];
    double v001 = StellarTbl.surface_abundances[IDX4(iz,   im,   ia+1, elem)];
    double v010 = StellarTbl.surface_abundances[IDX4(iz,   im+1, ia,   elem)];
    double v011 = StellarTbl.surface_abundances[IDX4(iz,   im+1, ia+1, elem)];
    double v100 = StellarTbl.surface_abundances[IDX4(iz+1, im,   ia,   elem)];
    double v101 = StellarTbl.surface_abundances[IDX4(iz+1, im,   ia+1, elem)];
    double v110 = StellarTbl.surface_abundances[IDX4(iz+1, im+1, ia,   elem)];
    double v111 = StellarTbl.surface_abundances[IDX4(iz+1, im+1, ia+1, elem)];

    double c00 = v000*(1-fa) + v001*fa;
    double c01 = v010*(1-fa) + v011*fa;
    double c10 = v100*(1-fa) + v101*fa;
    double c11 = v110*(1-fa) + v111*fa;
    double c0  = c00*(1-fm)  + c01*fm;
    double c1  = c10*(1-fm)  + c11*fm;
    return c0*(1-fz) + c1*fz;
}

/* --- Yields --- */

double stellar_net_yield(double logM, double logZ, int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;

    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);

    /* bilinear with fixed element index */
    int base00 = (IDX2(iz,   im  )) * STBL_NELEM + elem;
    int base01 = (IDX2(iz,   im+1)) * STBL_NELEM + elem;
    int base10 = (IDX2(iz+1, im  )) * STBL_NELEM + elem;
    int base11 = (IDX2(iz+1, im+1)) * STBL_NELEM + elem;

    double v00 = StellarTbl.net_yields[base00];
    double v01 = StellarTbl.net_yields[base01];
    double v10 = StellarTbl.net_yields[base10];
    double v11 = StellarTbl.net_yields[base11];
    return (1-fz)*(1-fm)*v00 + (1-fz)*fm*v01 + fz*(1-fm)*v10 + fz*fm*v11;
}

double stellar_type_ia_yield(int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;
    return StellarTbl.type_ia_yields[elem];
}


#endif /* GALSF_RESOLVEDISM_STELLAR_TABLES */


/* Accessor for turbulent diffusion: returns the resolved ISM passive scalar
   (ElementAbundance or Dust) at global index k, or -1 if k is not ours */
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || defined(GALSF_RESOLVEDISM_DUST)
double return_resolvedism_species_for_diffusion(int i, int k)
{
    k -= NUM_METAL_SPECIES;
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    k -= (NUM_ISMDUSTCHEM_ELEMENTS + NUM_ISMDUSTCHEM_SOURCES + NUM_ISMDUSTCHEM_SPECIES);
#endif
    if(k < 0) return -1;
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL)
    if(k < NUM_RESOLVEDISM_ELEMENTS) return P[i].ElementAbundance[k];
    k -= NUM_RESOLVEDISM_ELEMENTS;
#endif
#if defined(GALSF_RESOLVEDISM_DUST)
    if(k < NUM_RESOLVEDISM_DUST) return CellP[i].Dust[k];
#endif
    return -1;
}
#endif
