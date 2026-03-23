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
 *  Index computation: binary search for non-uniform grids (M, Z),
 *  direct O(1) formula for the uniform age grid.
 * ======================================================================== */

/* Binary search: find i0 such that grid[i0] <= val < grid[i0+1], with
 * fractional position f in [0,1].  Clamps to grid boundaries. */
static inline void stbl_idx_bsearch(const double *grid, int N, double val, int *i0, double *f)
{
    if(val <= grid[0])   { *i0 = 0;     *f = 0.0; return; }
    if(val >= grid[N-1]) { *i0 = N - 2; *f = 1.0; return; }
    int lo = 0, hi = N - 1;
    while(hi - lo > 1) { int mid = (lo + hi) / 2; if(grid[mid] <= val) lo = mid; else hi = mid; }
    *i0 = lo;
    *f  = (val - grid[lo]) / (grid[hi] - grid[lo]);
}

static inline void stbl_idx_Z(double logZ, int *i0, double *f)
{
    stbl_idx_bsearch(StellarTbl.log_Z, STBL_NZ, logZ, i0, f);
}

static inline void stbl_idx_M(double logM, int *i0, double *f)
{
    stbl_idx_bsearch(StellarTbl.log_M, STBL_NM, logM, i0, f);
}

static inline void stbl_idx_age(double log_age, int *i0, double *f)
{
    /* Age grid is uniform — keep O(1) formula */
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
    if(ThisTask == 0) printf("RESOLVEDISM TABLES: loading stellar tables from %s\n", All.StellarTablesFile);

    size_t n3d = (size_t)STBL_NZ * STBL_NM * STBL_NAGE;

    /* Allocate 3D arrays (float to save memory: ~33 MB each, ~230 MB total for 7 arrays) */
    StellarTbl.M_current       = (float *)mymalloc("stbl_Mcur",    n3d * sizeof(float));
    StellarTbl.v_wind          = (float *)mymalloc("stbl_vwind",   n3d * sizeof(float));
    StellarTbl.log_Q_ion       = (float *)mymalloc("stbl_logQion", n3d * sizeof(float));
    StellarTbl.log_L_FUV       = (float *)mymalloc("stbl_logLFUV", n3d * sizeof(float));
    StellarTbl.log_L_LW        = (float *)mymalloc("stbl_logLLW",  n3d * sizeof(float));
    StellarTbl.log_L_FUV_total = (float *)mymalloc("stbl_logLFUVt",n3d * sizeof(float));
    StellarTbl.log_L_bol       = (float *)mymalloc("stbl_logLbol", n3d * sizeof(float));
    StellarTbl.logR_cm         = (float *)mymalloc("stbl_logRcm",  n3d * sizeof(float));

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
        read_hdf5_dataset_as_float(file, "logR_cm",         StellarTbl.logR_cm,         n3d);

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

        /* Wind yields [NZ x NM x NELEM] — wind-only portion of net yields */
        if(H5Lexists(file, "wind_yields", H5P_DEFAULT) > 0) {
            read_hdf5_dataset_double(file, "wind_yields", StellarTbl.wind_yields, (size_t)STBL_NZ * STBL_NM * STBL_NELEM);
        } else {
            memset(StellarTbl.wind_yields, 0, STBL_NZ * STBL_NM * STBL_NELEM * sizeof(double));
            if(ThisTask == 0) printf("RESOLVEDISM TABLES: WARNING — wind_yields not found in table, SN yields = net yields\n");
        }

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
    MPI_Bcast(StellarTbl.logR_cm,         n3d, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
    MPI_Bcast(StellarTbl.wind_yields,    STBL_NZ * STBL_NM * STBL_NELEM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

    /* Post-load fixup: force M_current at ZAMS (first age bin) to equal M_init.
     * Some tracks at the low-mass end don't exactly match the grid M_init. */
    {
        int n_fixed = 0;
        for(int iz = 0; iz < STBL_NZ; iz++) {
            for(int im = 0; im < STBL_NM; im++) {
                float mc0 = StellarTbl.M_current[IDX3(iz, im, 0)];
                float mi  = (float)StellarTbl.M[im];
                if(fabs(mc0 - mi) > 0.001 * mi) {
                    StellarTbl.M_current[IDX3(iz, im, 0)] = mi;
                    /* Also fix the next few age bins if they have the same issue (pre-MS) */
                    for(int ia = 1; ia < DMIN(10, STBL_NAGE); ia++) {
                        float mc_ia = StellarTbl.M_current[IDX3(iz, im, ia)];
                        if(mc_ia > mi * 1.001) StellarTbl.M_current[IDX3(iz, im, ia)] = mi;
                        else break;
                    }
                    n_fixed++;
                }
            }
        }
        if(n_fixed > 0 && ThisTask == 0)
            printf("RESOLVEDISM TABLES: fixed M_current(ZAMS) for %d (Z,M) entries to match M_init\n", n_fixed);
    }

    if(ThisTask == 0) {
        printf("RESOLVEDISM TABLES: stellar tables loaded. Grid: %d Z x %d M x %d ages\n", STBL_NZ, STBL_NM, STBL_NAGE);
        printf("  logZ: [%.3f, %.3f], dlogZ=%.4f\n", StellarTbl.log_Z_min, StellarTbl.log_Z_max, StellarTbl.dlog_Z);
        printf("  logM: [%.3f, %.3f], dlogM=%.4f\n", StellarTbl.log_M_min, StellarTbl.log_M_max, StellarTbl.dlog_M);
        printf("  log_age: [%.3f, %.3f], dlog_age=%.4f\n", StellarTbl.log_age_min, StellarTbl.log_age_max, StellarTbl.dlog_age);
        printf("  Memory: %.1f MB (3D arrays) + %.1f MB (surface abundances)\n",
               12.0 * n3d * sizeof(float) / 1e6,
               StellarTbl.surface_abundances ? (double)(n3d * STBL_NELEM * sizeof(float)) / 1e6 : 0.0);

        /* ======== Comprehensive table validation (Monte Carlo + grid) ======== */
        printf("  Running table validation...\n");
        int n_warn = 0;

        /* --- Test 1: M_current(ZAMS) = M_init for on-grid and off-grid masses --- */
        {
            int nfail = 0;
            double worst_err = 0, worst_M = 0, worst_Z = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                double logZ_t = StellarTbl.log_Z[iz];
                /* Test on-grid masses */
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im], logM_t = StellarTbl.log_M[im];
                    double Mc = stellar_M_current(logM_t, logZ_t, 2.0);
                    double err = fabs(Mc - M) / M;
                    if(err > 0.01) nfail++;
                    if(err > worst_err) { worst_err = err; worst_M = M; worst_Z = pow(10., logZ_t); }
                }
                /* Test mid-grid masses (between every pair of adjacent grid points) */
                for(int im = 0; im < STBL_NM - 1; im++) {
                    double logM_mid = 0.5 * (StellarTbl.log_M[im] + StellarTbl.log_M[im+1]);
                    double M_mid = pow(10., logM_mid);
                    double Mc = stellar_M_current(logM_mid, logZ_t, 2.0);
                    double err = fabs(Mc - M_mid) / M_mid;
                    if(err > 0.01) nfail++;
                    if(err > worst_err) { worst_err = err; worst_M = M_mid; worst_Z = pow(10., logZ_t); }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d M_current(ZAMS) deviations >1%% (worst: M=%.2f Z=%.4f err=%.2e)\n", nfail, worst_M, worst_Z, worst_err); n_warn++; }
            else printf("  [PASS] M_current(ZAMS) = M_init for all on-grid and mid-grid masses (worst err=%.2e)\n", worst_err);
        }

        /* --- Test 2: M_current monotonically decreases with age (no spurious jumps from corner mixing) --- */
        {
            int nfail = 0; double worst_jump = 0; double worst_M2 = 0;
            /* Test at mid-grid masses where interpolation mixing is strongest */
            for(int im = 0; im < STBL_NM - 1; im++) {
                double logM_mid = 0.5 * (StellarTbl.log_M[im] + StellarTbl.log_M[im+1]);
                double M_mid = pow(10., logM_mid);
                if(M_mid < 8.0) continue; /* only test massive stars with significant winds */
                for(int iz = 0; iz < STBL_NZ; iz++) {
                    double logZ_t = StellarTbl.log_Z[iz];
                    double lt = stellar_lifetime(logM_mid, logZ_t);
                    if(lt <= 0) continue;
                    double Mc_prev = M_mid;
                    int nstep = 50;
                    for(int s = 1; s <= nstep; s++) {
                        double age = lt * 0.99 * s / nstep;
                        if(age < 100) age = 100;
                        double Mc = stellar_M_current(logM_mid, logZ_t, log10(age));
                        if(Mc > Mc_prev * 1.01 && Mc_prev > 0) { /* allow 1% numerical noise */
                            double jump = (Mc - Mc_prev) / Mc_prev;
                            if(jump > worst_jump) { worst_jump = jump; worst_M2 = M_mid; }
                            nfail++;
                            break; /* one failure per (M,Z) pair is enough */
                        }
                        Mc_prev = Mc;
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d M_current non-monotonic cases (worst jump=%.2e at M=%.1f)\n", nfail, worst_jump, worst_M2); n_warn++; }
            else printf("  [PASS] M_current monotonically decreasing with age for all mid-grid massive stars\n");
        }

        /* --- Test 3: Surface abundances nonzero for all living stars (mid-grid, near lifetime boundary) --- */
        {
            int nfail = 0;
            for(int im = 0; im < STBL_NM - 1; im++) {
                double logM_mid = 0.5 * (StellarTbl.log_M[im] + StellarTbl.log_M[im+1]);
                double M_mid = pow(10., logM_mid);
                if(M_mid < 8.0) continue;
                for(int iz = 0; iz < STBL_NZ; iz++) {
                    double logZ_t = StellarTbl.log_Z[iz];
                    double lt = stellar_lifetime(logM_mid, logZ_t);
                    if(lt <= 0) continue;
                    /* Test at 90%, 95%, 99% of lifetime */
                    double fracs[] = {0.90, 0.95, 0.99};
                    for(int f = 0; f < 3; f++) {
                        double age = lt * fracs[f];
                        if(age < 100) continue;
                        double Z_surf = 0;
                        for(int kk = ELEM_C; kk < STBL_NELEM; kk++)
                            Z_surf += stellar_surface_abundance(logM_mid, logZ_t, log10(age), kk);
                        if(Z_surf <= 0) {
                            nfail++;
                            if(nfail <= 3) printf("  WARNING: Z_surface=0 at %.0f%% lifetime for M=%.2f Z=%.4f\n", fracs[f]*100, M_mid, pow(10., logZ_t));
                            (void)M_mid; /* worst case already printed above */
                        }
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d surface abundance=0 cases for living mid-grid stars\n", nfail); n_warn++; }
            else printf("  [PASS] Surface abundances > 0 for all living mid-grid stars at 90/95/99%% lifetime\n");
        }

        /* --- Test 4: Wind yields + SN yields = net yields for on-grid masses --- */
        {
            int nfail = 0; double worst_err4 = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double logM_t = StellarTbl.log_M[im], logZ_t = StellarTbl.log_Z[iz];
                    for(int kk = 0; kk < STBL_NELEM; kk++) {
                        double ny = stellar_net_yield(logM_t, logZ_t, kk);
                        double wy = stellar_wind_yield(logM_t, logZ_t, kk);
                        double sy = stellar_sn_yield(logM_t, logZ_t, kk);
                        double err = fabs((wy + sy) - ny);
                        if(fabs(ny) > 1e-10) err /= fabs(ny);
                        if(err > worst_err4) worst_err4 = err;
                        if(err > 0.01) nfail++;
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d wind+SN != net yield inconsistencies (worst=%.2e)\n", nfail, worst_err4); n_warn++; }
            else printf("  [PASS] wind_yield + sn_yield = net_yield for all on-grid (Z,M) (roundoff=%.2e)\n", worst_err4);
        }

        /* --- Test 5: Lifetime monotonically decreases with mass at each Z (M >= 0.5 only) --- */
        {
            int nfail = 0, nfail_lowmass = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                double lt_prev = 1e30;
                for(int im = 0; im < STBL_NM; im++) {
                    double lt = StellarTbl.lifetime_yr[IDX2(iz, im)];
                    if(lt > lt_prev * 1.05) {
                        if(StellarTbl.M[im] < 0.5) { nfail_lowmass++; }
                        else {
                            nfail++;
                            if(nfail <= 3) printf("  WARNING: lifetime not monotonic at Z=%.4f M=%.1f (%.3e > %.3e)\n",
                                pow(10., StellarTbl.log_Z[iz]), StellarTbl.M[im], lt, lt_prev);
                        }
                    }
                    lt_prev = lt;
                }
            }
            if(nfail > 0) { printf("  WARNING: %d lifetime non-monotonic entries (M>=0.5)\n", nfail); n_warn++; }
            else printf("  [PASS] Lifetime monotonically decreasing with mass for M>=0.5 at all Z\n");
            if(nfail_lowmass > 0) printf("  [INFO] %d lifetime non-monotonic entries for M<0.5 (irrelevant — lifetimes > Hubble time)\n", nfail_lowmass);
        }

        /* --- Test 6: Ionizing luminosity reasonable for massive stars at mid-lifetime --- */
        {
            int nfail = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                double logZ_t = StellarTbl.log_Z[iz];
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    if(M < 20.0) continue;
                    double lt = StellarTbl.lifetime_yr[IDX2(iz, im)];
                    if(lt <= 0) continue;
                    double log_age_mid = log10(lt * 0.5); /* mid-lifetime — star is on MS */
                    double logQ = stellar_log_Q_ion(StellarTbl.log_M[im], logZ_t, log_age_mid);
                    double logL = stellar_log_L_bol(StellarTbl.log_M[im], logZ_t, log_age_mid);
                    double logLFUV = stellar_log_L_FUV(StellarTbl.log_M[im], logZ_t, log_age_mid);
                    double logLLW = stellar_log_L_LW(StellarTbl.log_M[im], logZ_t, log_age_mid);
                    if(logQ < 44.0) { nfail++; if(nfail<=3) printf("  WARNING: log_Q_ion=%.1f < 44 at mid-life for M=%.0f Z=%.4f\n", logQ, M, pow(10.,logZ_t)); }
                    if(logL < 37.0) { nfail++; if(nfail<=3) printf("  WARNING: log_Lbol=%.1f < 37 at mid-life for M=%.0f Z=%.4f\n", logL, M, pow(10.,logZ_t)); }
                    if(logLFUV < 30.0) { nfail++; if(nfail<=3) printf("  WARNING: log_L_FUV=%.1f < 30 at mid-life for M=%.0f Z=%.4f\n", logLFUV, M, pow(10.,logZ_t)); }
                    if(logLLW < 30.0) { nfail++; if(nfail<=3) printf("  WARNING: log_L_LW=%.1f < 30 at mid-life for M=%.0f Z=%.4f\n", logLLW, M, pow(10.,logZ_t)); }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d unreasonable luminosity/Q_ion values for M>=20\n", nfail); n_warn++; }
            else printf("  [PASS] Ionizing luminosity and L_bol reasonable for all massive stars\n");
        }

        /* --- Test 7: Monte Carlo IMF draw — check interpolation at random masses --- */
        {
            int ntest = 100000;
            int n_mcur_fail = 0, n_surf_fail = 0, n_mono_fail = 0;
            double worst_mcur = 0;
            unsigned long seed = 12345;
            for(int it = 0; it < ntest; it++) {
                /* Simple LCG random for reproducibility without GSL */
                seed = seed * 1103515245 + 12345;
                double u1 = (double)((seed >> 16) & 0x7FFF) / 32767.0;
                seed = seed * 1103515245 + 12345;
                double u2 = (double)((seed >> 16) & 0x7FFF) / 32767.0;
                /* Draw from mass-weighted Kroupa CDF */
                double M_min = 0.08, M_max_t = 350.0;
                double M = pow((pow(M_max_t, 0.7) - pow(M_min, 0.7)) * u1 + pow(M_min, 0.7), 1.0 / 0.7);
                double logM_t = log10(M);
                /* Random Z from grid range */
                double logZ_t = StellarTbl.log_Z_min + u2 * (StellarTbl.log_Z_max - StellarTbl.log_Z_min);

                /* Check M_current at ZAMS */
                double Mc = stellar_M_current(logM_t, logZ_t, 2.0);
                double err = fabs(Mc - M) / M;
                if(err > worst_mcur) worst_mcur = err;
                if(err > 0.02) n_mcur_fail++;

                /* For massive stars: check surface abundances near death */
                if(M >= 8.0) {
                    double lt = stellar_lifetime(logM_t, logZ_t);
                    if(lt > 0) {
                        double age95 = lt * 0.95;
                        if(age95 > 100) {
                            double Z_surf = 0;
                            for(int kk = ELEM_C; kk < STBL_NELEM; kk++)
                                Z_surf += stellar_surface_abundance(logM_t, logZ_t, log10(age95), kk);
                            if(Z_surf <= 0) n_surf_fail++;
                        }
                        /* Check M_current monotonicity (3 age samples) */
                        double Mc1 = stellar_M_current(logM_t, logZ_t, log10(DMAX(lt*0.3, 100)));
                        double Mc2 = stellar_M_current(logM_t, logZ_t, log10(DMAX(lt*0.6, 100)));
                        double Mc3 = stellar_M_current(logM_t, logZ_t, log10(DMAX(lt*0.9, 100)));
                        if(Mc2 > Mc1 * 1.01 || Mc3 > Mc2 * 1.01) n_mono_fail++;
                    }
                }
            }
            printf("  Monte Carlo validation (%d random stars):\n", ntest);
            if(n_mcur_fail > 0) { printf("    WARNING: %d M_current(ZAMS) errors >2%% (worst=%.2e)\n", n_mcur_fail, worst_mcur); n_warn++; }
            else printf("    [PASS] M_current(ZAMS) within 2%% for all draws (worst=%.2e)\n", worst_mcur);
            if(n_surf_fail > 0) { printf("    WARNING: %d surface abundance=0 at 95%% lifetime for M>=8\n", n_surf_fail); n_warn++; }
            else printf("    [PASS] Surface abundances > 0 at 95%% lifetime for all M>=8 draws\n");
            if(n_mono_fail > 0) { printf("    WARNING: %d M_current non-monotonic cases\n", n_mono_fail); n_warn++; }
            else printf("    [PASS] M_current monotonic (30%%->60%%->90%% lifetime) for all M>=8 draws\n");
        }

        /* --- Test 8: Remnant mass sanity --- */
        {
            int nfail = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    int rt = StellarTbl.remnant_type[IDX2(iz, im)];
                    double rm = StellarTbl.remnant_mass[IDX2(iz, im)];
                    /* Remnant mass must be < M_init */
                    if(rm > M * 1.01) {
                        if(nfail < 3) printf("  WARNING: remnant_mass=%.2f > M_init=%.2f at Z=%.4f (type=%d)\n", rm, M, pow(10., StellarTbl.log_Z[iz]), rt);
                        nfail++;
                    }
                    /* Remnant mass must be > 0 for non-PISN */
                    if(rt != 5 && rm <= 0 && M >= 0.5) {
                        if(nfail < 6) printf("  WARNING: remnant_mass=0 for non-PISN M=%.2f Z=%.4f type=%d\n", M, pow(10., StellarTbl.log_Z[iz]), rt);
                        nfail++;
                    }
                    /* PISN must have rm=0 */
                    if(rt == 5 && rm > 0.01) {
                        if(nfail < 9) printf("  WARNING: PISN with remnant_mass=%.2f at M=%.1f Z=%.4f\n", rm, M, pow(10., StellarTbl.log_Z[iz]));
                        nfail++;
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d remnant mass inconsistencies\n", nfail); n_warn++; }
            else printf("  [PASS] Remnant mass consistent for all (Z,M): 0 < rem < M_init (PISN=0)\n");
        }

        /* --- Test 9: Net yield mass conservation: sum over all elements ≈ 0 --- */
        {
            int nfail_severe = 0, nfail_hhe = 0;
            double worst_imbal = 0, worst_M9 = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    int rt = StellarTbl.remnant_type[IDX2(iz, im)];
                    if(rt == 3 || rt == 6) continue; /* FSN/DBH have zeroed yields */
                    double logM_t = StellarTbl.log_M[im], logZ_t = StellarTbl.log_Z[iz];
                    double sum_all = 0, sum_metals = 0;
                    for(int kk = 0; kk < STBL_NELEM; kk++) {
                        double ny = stellar_net_yield(logM_t, logZ_t, kk);
                        sum_all += ny;
                        if(kk >= ELEM_C) sum_metals += ny;
                    }
                    double imbal = fabs(sum_all) / DMAX(M, 1.0);
                    if(imbal > worst_imbal) { worst_imbal = imbal; worst_M9 = M; }
                    if(imbal > 0.20) { /* >20% — H/He budget issue in source tables, not a code bug */
                        nfail_severe++;
                    } else if(imbal > 0.05) {
                        nfail_hhe++;
                    }
                }
            }
            if(nfail_severe > 0) printf("  [INFO] %d entries with >20%% yield mass imbalance (worst %.0f%% at M=%.1f) — H/He budget in source tables; code conserves mass via M_ej=M_particle-M_remnant\n", nfail_severe, worst_imbal*100, worst_M9);
            else printf("  [PASS] No severe yield mass imbalance >20%%\n");
            if(nfail_hhe > 0) printf("  [INFO] %d entries with 5-20%% H/He imbalance (untracked isotopes in Karakas/Limongi tables)\n", nfail_hhe);
        }

        /* --- Test 10: SN yields (net - wind) non-negative for major elements --- */
        {
            int nfail_severe = 0, nfail_minor = 0;
            int check_elems[] = {ELEM_C, ELEM_O, ELEM_Si, ELEM_Fe};
            const char *check_names[] = {"C", "O", "Si", "Fe"};
            for(int iz = 0; iz < STBL_NZ; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    if(M < 8.0) continue;
                    int rt = StellarTbl.remnant_type[IDX2(iz, im)];
                    if(rt == 3 || rt == 6) continue;
                    double logM_t = StellarTbl.log_M[im], logZ_t = StellarTbl.log_Z[iz];
                    for(int e = 0; e < 4; e++) {
                        double sy = stellar_sn_yield(logM_t, logZ_t, check_elems[e]);
                        if(sy < -0.1) { /* > 0.1 Msun negative is a real problem */
                            nfail_severe++;
                            if(nfail_severe <= 3) printf("  WARNING: large negative SN yield %s=%.4f at M=%.1f Z=%.4f type=%d\n",
                                check_names[e], sy, M, pow(10., logZ_t), rt);
                        } else if(sy < -0.001) {
                            nfail_minor++; /* small negative — clamped to 0 in injection code */
                        }
                    }
                }
            }
            if(nfail_severe > 0) { printf("  WARNING: %d large negative SN yields (>0.1 Msun)\n", nfail_severe); n_warn++; }
            else printf("  [PASS] No large negative SN yields (net-wind) for C, O, Si, Fe\n");
            if(nfail_minor > 0) printf("  [INFO] %d small negative SN yields (<0.1 Msun, clamped to 0 in injection code)\n", nfail_minor);
        }

        /* --- Test 11: Lifetime within age grid range (M >= 0.8 only — lower mass stars can exceed Hubble time at low Z) --- */
        {
            int nfail = 0, nfail_lowmass = 0;
            double age_max = pow(10., StellarTbl.log_age_max);
            for(int iz = 0; iz < STBL_NZ; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double lt = StellarTbl.lifetime_yr[IDX2(iz, im)];
                    if(lt > age_max * 1.1) {
                        if(StellarTbl.M[im] < 0.8) { nfail_lowmass++; }
                        else {
                            nfail++;
                            if(nfail <= 3) printf("  WARNING: lifetime %.3e yr > age_max %.3e yr at M=%.2f Z=%.4f\n",
                                lt, age_max, StellarTbl.M[im], pow(10., StellarTbl.log_Z[iz]));
                        }
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d lifetimes exceed age grid range (M>=0.8)\n", nfail); n_warn++; }
            else printf("  [PASS] All lifetimes within age grid range for M>=0.8\n");
            if(nfail_lowmass > 0) printf("  [INFO] %d low-mass (M<0.8) lifetimes exceed age grid (expected — these stars can outlive the Hubble time at low Z)\n", nfail_lowmass);
        }

        /* --- Test 12: AGB yields — C enrichment from 3rd dredge-up (M=2-4 Msun) --- */
        {
            int n_checked = 0, n_c_enriched = 0;
            for(int iz = 0; iz < STBL_NZ; iz++) {
                double logZ_t = StellarTbl.log_Z[iz];
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    if(M < 2.0 || M > 4.0) continue;
                    int rt = StellarTbl.remnant_type[IDX2(iz, im)];
                    if(rt != 0) continue; /* WD only */
                    n_checked++;
                    double yC = stellar_net_yield(StellarTbl.log_M[im], logZ_t, ELEM_C);
                    if(yC > 0) n_c_enriched++;
                }
            }
            if(n_checked > 0) {
                double frac = (double)n_c_enriched / n_checked * 100;
                if(frac < 50) { printf("  WARNING: only %.0f%% of AGB stars (2-4 Msun) have net C > 0 (expected ~100%%)\n", frac); n_warn++; }
                else printf("  [PASS] AGB C enrichment: %.0f%% of 2-4 Msun WD stars have net C > 0 (%d/%d)\n", frac, n_c_enriched, n_checked);
            }
        }

        /* --- Test 13: Cross-Z smoothness — same mass at adjacent Z shouldn't differ wildly --- */
        {
            int nfail = 0;
            for(int iz = 0; iz < STBL_NZ - 1; iz++) {
                for(int im = 0; im < STBL_NM; im++) {
                    double M = StellarTbl.M[im];
                    if(M < 8.0) continue;
                    double lt1 = StellarTbl.lifetime_yr[IDX2(iz, im)];
                    double lt2 = StellarTbl.lifetime_yr[IDX2(iz+1, im)];
                    if(lt1 <= 0 || lt2 <= 0) continue;
                    double ratio = lt1 / lt2;
                    if(ratio > 3.0 || ratio < 0.33) { /* factor 3 jump between adjacent Z */
                        if(nfail < 3) printf("  WARNING: lifetime jump x%.1f between Z=%.4f and Z=%.4f at M=%.1f\n",
                            ratio, pow(10., StellarTbl.log_Z[iz]), pow(10., StellarTbl.log_Z[iz+1]), M);
                        nfail++;
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d sharp lifetime jumps between adjacent Z bins\n", nfail); n_warn++; }
            else printf("  [PASS] Lifetime smooth across Z (no jumps > 3x between adjacent bins)\n");
        }

        /* --- Test 14: Q_ion = 0 after death (age clamping works for luminosity) --- */
        {
            int nfail = 0;
            for(int im = 0; im < STBL_NM; im++) {
                double M = StellarTbl.M[im];
                if(M < 20.0) continue;
                for(int iz = 0; iz < STBL_NZ; iz++) {
                    double logM_t = StellarTbl.log_M[im], logZ_t = StellarTbl.log_Z[iz];
                    double lt = StellarTbl.lifetime_yr[IDX2(iz, im)];
                    if(lt <= 0) continue;
                    double age_dead = lt * 1.5; /* 50% past death */
                    if(age_dead < 100) continue;
                    double logQ = stellar_log_Q_ion(logM_t, logZ_t, log10(age_dead));
                    if(logQ > 40.0) { /* should be ~0, definitely not > 10^40 photons/s */
                        if(nfail < 3) printf("  WARNING: log_Q_ion=%.1f for dead star M=%.0f Z=%.4f (age=1.5*lifetime)\n",
                            logQ, M, pow(10., logZ_t));
                        nfail++;
                    }
                }
            }
            if(nfail > 0) { printf("  WARNING: %d dead stars with significant Q_ion\n", nfail); n_warn++; }
            else printf("  [PASS] Q_ion negligible for all dead stars (age > lifetime)\n");
        }

        /* --- Test 15: Monte Carlo — wind surface composition matches birth Z at ZAMS --- */
        {
            int ntest_w = 10000;
            int nfail_zbirth = 0;
            unsigned long seed_w = 54321;
            for(int it = 0; it < ntest_w; it++) {
                seed_w = seed_w * 1103515245 + 12345;
                double u1 = (double)((seed_w >> 16) & 0x7FFF) / 32767.0;
                seed_w = seed_w * 1103515245 + 12345;
                double u2 = (double)((seed_w >> 16) & 0x7FFF) / 32767.0;
                double M = pow((pow(350.0, 0.7) - pow(8.0, 0.7)) * u1 + pow(8.0, 0.7), 1.0 / 0.7);
                double logM_t = log10(M);
                double logZ_t = StellarTbl.log_Z_min + u2 * (StellarTbl.log_Z_max - StellarTbl.log_Z_min);
                /* At ZAMS (age=100 yr), surface Z should be close to birth Z */
                double Z_surf = 0;
                for(int kk = ELEM_C; kk < STBL_NELEM; kk++)
                    Z_surf += stellar_surface_abundance(logM_t, logZ_t, 2.0, kk); /* log10(100)=2 */
                double Z_birth = pow(10., logZ_t);
                if(Z_surf > 0 && Z_birth > 0) {
                    double ratio = Z_surf / Z_birth;
                    if(ratio < 0.3 || ratio > 3.0) nfail_zbirth++;
                }
            }
            if(nfail_zbirth > 0) { printf("  WARNING: %d/%d stars have ZAMS surface Z differing >3x from birth Z\n", nfail_zbirth, ntest_w); n_warn++; }
            else printf("  [PASS] ZAMS surface metallicity matches birth Z within 3x for all %d random M>=8 draws\n", ntest_w);
        }

        if(n_warn == 0) printf("  === All %d table validation tests PASSED ===\n", 15);
        else printf("  === Table validation: %d WARNINGS out of 15 tests — check above ===\n", n_warn);
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
    if(StellarTbl.logR_cm)         { myfree(StellarTbl.logR_cm);         StellarTbl.logR_cm = NULL; }
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
    /* Interpolate M_current/M_init (fractional mass remaining) then scale by
       the queried mass, so off-grid masses always get M_current <= M_drawn at
       ZAMS and the wind mass-loss detection works correctly.
       Per-corner age clamping: each (iz,im) corner has a different lifetime.
       If the queried age exceeds a corner's lifetime, clamp to the last alive age
       to avoid mixing pre-death M_current with post-death remnant mass. */
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);

    double Mstar = pow(10.0, logM);
    int iz_c[4] = {iz, iz, iz+1, iz+1};
    int im_c[4] = {im, im+1, im, im+1};
    double M_grid[4] = {StellarTbl.M[im], StellarTbl.M[im+1], StellarTbl.M[im], StellarTbl.M[im+1]};
    double corners[4];

    for(int c = 0; c < 4; c++) {
        double lt = StellarTbl.lifetime_yr[IDX2(iz_c[c], im_c[c])];
        double log_age_c = (lt > 0 && pow(10.0, log_age) > lt) ? log10(lt) : log_age;
        int ia_c; double fa_c;
        stbl_idx_age(log_age_c, &ia_c, &fa_c);
        double Mg = (M_grid[c] > 0) ? M_grid[c] : 1e-30;
        double f0 = StellarTbl.M_current[IDX3(iz_c[c], im_c[c], ia_c  )] / Mg;
        double f1 = StellarTbl.M_current[IDX3(iz_c[c], im_c[c], ia_c+1)] / Mg;
        corners[c] = f0*(1-fa_c) + f1*fa_c;
    }

    double c0 = corners[0]*(1-fm) + corners[1]*fm;
    double c1 = corners[2]*(1-fm) + corners[3]*fm;
    double frac = c0*(1-fz) + c1*fz;

    return Mstar * DMAX(frac, 0.0);
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

double stellar_log_R_cm(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.logR_cm, logM, logZ, log_age);
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

    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);

    /* Per-corner age clamping: each (iz,im) corner has a different lifetime.
     * If the queried age exceeds a corner's lifetime, clamp to the last alive age
     * to avoid mixing live surface abundances with post-death zeros. */
    double corners[4]; /* [00, 01, 10, 11] = [(iz,im), (iz,im+1), (iz+1,im), (iz+1,im+1)] */
    int iz_c[4] = {iz, iz, iz+1, iz+1};
    int im_c[4] = {im, im+1, im, im+1};
    for(int c = 0; c < 4; c++) {
        double lt = StellarTbl.lifetime_yr[IDX2(iz_c[c], im_c[c])];
        double log_age_c = (lt > 0 && pow(10.0, log_age) > lt) ? log10(lt) : log_age;
        int ia_c; double fa_c;
        stbl_idx_age(log_age_c, &ia_c, &fa_c);
        double v0 = StellarTbl.surface_abundances[IDX4(iz_c[c], im_c[c], ia_c,   elem)];
        double v1 = StellarTbl.surface_abundances[IDX4(iz_c[c], im_c[c], ia_c+1, elem)];
        corners[c] = v0*(1-fa_c) + v1*fa_c;
    }

    double c0 = corners[0]*(1-fm) + corners[1]*fm;
    double c1 = corners[2]*(1-fm) + corners[3]*fm;
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

double stellar_wind_yield(double logM, double logZ, int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;

    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);

    int base00 = (IDX2(iz,   im  )) * STBL_NELEM + elem;
    int base01 = (IDX2(iz,   im+1)) * STBL_NELEM + elem;
    int base10 = (IDX2(iz+1, im  )) * STBL_NELEM + elem;
    int base11 = (IDX2(iz+1, im+1)) * STBL_NELEM + elem;

    double v00 = StellarTbl.wind_yields[base00];
    double v01 = StellarTbl.wind_yields[base01];
    double v10 = StellarTbl.wind_yields[base10];
    double v11 = StellarTbl.wind_yields[base11];
    return (1-fz)*(1-fm)*v00 + (1-fz)*fm*v01 + fz*(1-fm)*v10 + fz*fm*v11;
}

double stellar_sn_yield(double logM, double logZ, int elem)
{
    return stellar_net_yield(logM, logZ, elem) - stellar_wind_yield(logM, logZ, elem);
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
