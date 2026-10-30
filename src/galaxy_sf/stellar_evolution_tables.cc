/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <string>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "stellar_evolution_tables.h"
#include <mpi.h>

/* Single-star stellar-evolution tables (stella-tables, github.com/Ulli1991/stella-tables): lifetimes, tracks, band
   luminosities, winds, yields, remnants. Installed outside the repository by scripts/get_stella_tables.sh; the
   parameter StellarTablesFile names the main table, the wind table is read from the same directory. Both must have the
   content hashes compiled in (stellar_tables_check.cc). */

#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES

/* the wind table sits next to the main table */
static std::string stellar_wind_tables_path(void)
{
    std::string p = All.StellarTablesFile;
    size_t s = p.find_last_of('/');
    return ((s == std::string::npos) ? std::string("") : p.substr(0, s + 1)) + "stellar_wind_cumulative.hdf5";
}

/* rank 0 checks both files; a mismatch is a controlled stop at start-up */
void stellar_evolution_tables_init(void)
{
    if(ThisTask != 0) {return;}
    double t0 = MPI_Wtime();
    std::vector<std::string> paths = {All.StellarTablesFile, stellar_wind_tables_path()};
    int bad = stellar_tables_verify(paths);
    printf("STELLAR TABLES: hash check took %.2f s\n", MPI_Wtime() - t0); fflush(stdout);
    if(bad) {
        gizmo_request_controlled_stop(7801, "stellar evolution tables missing or not the version these sources expect; install with scripts/get_stella_tables.sh", __FILE__, __LINE__, __FUNCTION__);
    }
}


/* Interpolation and accessors: as the reference implementation (stella_dev galaxy_sf/resolvedism_stellar_tables.cc) */
struct StellarEvolutionTables StellarTbl;

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
    /* Age grid is non-uniform (adaptive) — use binary search */
    stbl_idx_bsearch(StellarTbl.log_age, STBL_NAGE, log_age, i0, f);
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

/* Boundary-aware bilinear: if the four corners of the (logM, logZ) cell share
 * the same remnant_type, do standard bilinear. Otherwise the cell straddles a
 * rem_type discontinuity (e.g. WD↔ECSN around 7-8 Msun), and the underlying
 * physics is discontinuous — fall back to the nearest corner (same selection
 * stellar_remnant_type uses). This guarantees that any quantity whose meaning
 * is rem_type-conditional (M_preSN, rem_mass, per-element ejecta) stays
 * self-consistent with the chosen rem_type, preserving the closure invariant
 * M_preSN − rem − Σ ej = 0 at every interpolation point. */
static inline double interp2d_rt_aware(const double *arr, double logM, double logZ)
{
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    int rt00 = StellarTbl.remnant_type[IDX2(iz,   im  )];
    int rt01 = StellarTbl.remnant_type[IDX2(iz,   im+1)];
    int rt10 = StellarTbl.remnant_type[IDX2(iz+1, im  )];
    int rt11 = StellarTbl.remnant_type[IDX2(iz+1, im+1)];
    if(rt00 == rt01 && rt00 == rt10 && rt00 == rt11) {
        double v00 = arr[IDX2(iz,   im  )];
        double v01 = arr[IDX2(iz,   im+1)];
        double v10 = arr[IDX2(iz+1, im  )];
        double v11 = arr[IDX2(iz+1, im+1)];
        return (1-fz)*(1-fm)*v00 + (1-fz)*fm*v01 + fz*(1-fm)*v10 + fz*fm*v11;
    }
    int iz_near = (fz > 0.5) ? iz + 1 : iz;
    int im_near = (fm > 0.5) ? im + 1 : im;
    return arr[IDX2(iz_near, im_near)];
}

/* Boundary-aware variant for per-element ejecta arrays (extra elem stride). */
static inline double interp2d_elem_rt_aware(const double *arr, double logM, double logZ, int elem)
{
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    int rt00 = StellarTbl.remnant_type[IDX2(iz,   im  )];
    int rt01 = StellarTbl.remnant_type[IDX2(iz,   im+1)];
    int rt10 = StellarTbl.remnant_type[IDX2(iz+1, im  )];
    int rt11 = StellarTbl.remnant_type[IDX2(iz+1, im+1)];
    int b00 = (IDX2(iz,   im  )) * STBL_NELEM + elem;
    int b01 = (IDX2(iz,   im+1)) * STBL_NELEM + elem;
    int b10 = (IDX2(iz+1, im  )) * STBL_NELEM + elem;
    int b11 = (IDX2(iz+1, im+1)) * STBL_NELEM + elem;
    if(rt00 == rt01 && rt00 == rt10 && rt00 == rt11) {
        return (1-fz)*(1-fm)*arr[b00] + (1-fz)*fm*arr[b01]
             + fz*(1-fm)*arr[b10]     + fz*fm*arr[b11];
    }
    int iz_near = (fz > 0.5) ? iz + 1 : iz;
    int im_near = (fm > 0.5) ? im + 1 : im;
    return arr[(IDX2(iz_near, im_near)) * STBL_NELEM + elem];
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

    /* corners flagged dead (-99 in the log tables) are left out and the rest renormalised; all dead gives -99 */
    const double v[8] = {
        arr[IDX3(iz,   im,   ia  )], arr[IDX3(iz,   im,   ia+1)],
        arr[IDX3(iz,   im+1, ia  )], arr[IDX3(iz,   im+1, ia+1)],
        arr[IDX3(iz+1, im,   ia  )], arr[IDX3(iz+1, im,   ia+1)],
        arr[IDX3(iz+1, im+1, ia  )], arr[IDX3(iz+1, im+1, ia+1)] };
    const double w[8] = {
        (1-fz)*(1-fm)*(1-fa), (1-fz)*(1-fm)*fa,
        (1-fz)*fm*(1-fa),     (1-fz)*fm*fa,
        fz*(1-fm)*(1-fa),     fz*(1-fm)*fa,
        fz*fm*(1-fa),         fz*fm*fa };
    double vsum = 0, wsum = 0; int k;
    for(k = 0; k < 8; k++) { if(v[k] > -90.0) { vsum += w[k]*v[k]; wsum += w[k]; } }
    if(wsum <= 0) return -99.0;   /* every corner dead -> dead star */
    return vsum / wsum;
}

/* ========================================================================
 *  Loading: world rank 0 reads; 1D/2D tables are broadcast to every rank, the 3D/4D tables to one rank per node,
 *  which holds them in MPI shared memory for the other ranks of its node.
 * ======================================================================== */
static int stbl_read(hid_t file, const char *name, hid_t memtype, void *buf, size_t n)
{
    hid_t ds = H5Dopen(file, name); /* 1.6 API (H5_USE_16_API) */
    if(ds < 0) {printf("ERROR: stellar evolution table: no dataset '%s'\n", name); return 1;}
    hid_t sp = H5Dget_space(ds); hssize_t npts = H5Sget_simple_extent_npoints(sp); H5Sclose(sp);
    int err = (npts < 0 || (size_t)npts != n);
    if(err) {printf("ERROR: stellar evolution table: dataset '%s' has %lld elements, expected %zu (grid does not match STBL_*)\n", name, (long long)npts, n);}
    else if(H5Dread(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf) < 0) {printf("ERROR: stellar evolution table: reading '%s' failed\n", name); err = 1;}
    H5Dclose(ds);
    return err;
}

static MPI_Win stbl_win = MPI_WIN_NULL;

/* the two broadcasts below send the plain-data head of the struct and the double arrays after the pointers as blocks */
static_assert(offsetof(struct StellarEvolutionTables, v_wind) < offsetof(struct StellarEvolutionTables, elem_ej_SN_mass), "table struct layout");
static_assert(offsetof(struct StellarEvolutionTables, elem_ej_AGB_mass) == offsetof(struct StellarEvolutionTables, elem_ej_SN_mass) + sizeof(StellarTbl.elem_ej_SN_mass), "table struct layout");
static_assert(offsetof(struct StellarEvolutionTables, type_ia_yields) == offsetof(struct StellarEvolutionTables, elem_ej_AGB_mass) + sizeof(StellarTbl.elem_ej_AGB_mass), "table struct layout");

void stellar_evolution_tables_free(void)
{
    if(stbl_win != MPI_WIN_NULL) {MPI_Win_free(&stbl_win);}
}

void stellar_evolution_tables_load(void)
{
    const size_t n2d = (size_t)STBL_NZ * STBL_NM, n3d = n2d * STBL_NAGE, n4d = n3d * STBL_NELEM;
    float **f3[1] = {&StellarTbl.v_wind};
    const char *n3[1] = {"v_wind"};
    const size_t nfloat = 1 * n3d + 2 * n4d;

    /* one shared segment per node, owned by node rank 0 */
    gizmo_node_comm_init();
    float *seg = NULL;
    MPI_Aint segsize = (GizmoNodeRankOfTask == 0) ? (MPI_Aint)(nfloat * sizeof(float)) : 0;
    MPI_Win_allocate_shared(segsize, sizeof(float), MPI_INFO_NULL, GizmoNodeComm, &seg, &stbl_win);
    gizmo_mem_account_add(GIZMO_MEM_SINGLE_STARS, (long long) segsize + (long long) sizeof(StellarTbl)); /* the segment on its node's lead rank; the plain tables on every rank */
    { MPI_Aint sz; int du; MPI_Win_shared_query(stbl_win, 0, &sz, &du, &seg); }
    for(int k = 0; k < 1; k++) {*f3[k] = seg + k * n3d;}
    StellarTbl.surface_abundances = seg + 1 * n3d;
    StellarTbl.elem_ej_wind_cumulative = seg + 1 * n3d + n4d;

    int err = 0;
    if(ThisTask == 0)
    {
        std::string wind = stellar_wind_tables_path();
        printf("Stellar evolution tables: loading %s and %s\n", All.StellarTablesFile, wind.c_str());
        hid_t f = H5Fopen(All.StellarTablesFile, H5F_ACC_RDONLY, H5P_DEFAULT);
        if(f < 0) {printf("ERROR: stellar evolution table: cannot open %s\n", All.StellarTablesFile); err = 1;}
        if(f >= 0) {
        err |= stbl_read(f, "log_Z", H5T_NATIVE_DOUBLE, StellarTbl.log_Z, STBL_NZ);
        err |= stbl_read(f, "Z", H5T_NATIVE_DOUBLE, StellarTbl.Z, STBL_NZ);
        err |= stbl_read(f, "log_M_init", H5T_NATIVE_DOUBLE, StellarTbl.log_M, STBL_NM);
        err |= stbl_read(f, "M_init", H5T_NATIVE_DOUBLE, StellarTbl.M, STBL_NM);
        err |= stbl_read(f, "log_age_yr", H5T_NATIVE_DOUBLE, StellarTbl.log_age, STBL_NAGE);
        err |= stbl_read(f, "lifetime_yr", H5T_NATIVE_DOUBLE, StellarTbl.lifetime_yr, n2d);
        err |= stbl_read(f, "remnant_mass", H5T_NATIVE_DOUBLE, StellarTbl.remnant_mass, n2d);
        err |= stbl_read(f, "remnant_type", H5T_NATIVE_INT, StellarTbl.remnant_type, n2d);
        err |= stbl_read(f, "t_PMS", H5T_NATIVE_DOUBLE, StellarTbl.t_PMS, n2d);
        for(int k = 0; k < 1; k++) {err |= stbl_read(f, n3[k], H5T_NATIVE_FLOAT, *f3[k], n3d);}
        err |= stbl_read(f, "surface_abundances", H5T_NATIVE_FLOAT, StellarTbl.surface_abundances, n4d);
        err |= stbl_read(f, "elem_ej_SN_mass", H5T_NATIVE_DOUBLE, StellarTbl.elem_ej_SN_mass, n2d * STBL_NELEM);
        err |= stbl_read(f, "elem_ej_AGB_mass", H5T_NATIVE_DOUBLE, StellarTbl.elem_ej_AGB_mass, n2d * STBL_NELEM);
        err |= stbl_read(f, "type_ia_yields", H5T_NATIVE_DOUBLE, StellarTbl.type_ia_yields, STBL_NELEM);
        H5Fclose(f);
        }
        hid_t w = H5Fopen(wind.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if(w < 0) {printf("ERROR: stellar evolution table: cannot open %s\n", wind.c_str()); err = 1;}
        else {err |= stbl_read(w, "elem_ej_wind_cumulative", H5T_NATIVE_FLOAT, StellarTbl.elem_ej_wind_cumulative, n4d); H5Fclose(w);}
    }
    MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(err) {gizmo_request_controlled_stop(7802, "stellar evolution tables: a dataset is missing or has the wrong size", __FILE__, __LINE__, __FUNCTION__); return;}
    /* everything before the pointers is plain data: one broadcast to all ranks */
    MPI_Bcast(&StellarTbl, (int)offsetof(struct StellarEvolutionTables, v_wind), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(StellarTbl.elem_ej_SN_mass, (int)(2 * n2d * STBL_NELEM + STBL_NELEM), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* the shared segment: from world rank 0 to the other node leaders, in chunks below INT_MAX */
    MPI_Comm leaders;
    MPI_Comm_split(MPI_COMM_WORLD, (GizmoNodeRankOfTask == 0) ? 0 : MPI_UNDEFINED, ThisTask, &leaders);
    if(leaders != MPI_COMM_NULL) {
        for(size_t off = 0; off < nfloat; off += 100000000) {MPI_Bcast(seg + off, (int)DMIN(100000000, nfloat - off), MPI_FLOAT, 0, leaders);}
        MPI_Comm_free(&leaders);
    }
    MPI_Win_fence(0, stbl_win); MPI_Barrier(GizmoNodeComm); MPI_Win_fence(0, stbl_win);
    StellarTbl.loaded = 1;
    if(ThisTask == 0) {printf("Stellar evolution tables: %d Z x %d M x %d ages, %.0f MB per node\n", STBL_NZ, STBL_NM, STBL_NAGE, nfloat * sizeof(float) / 1e6);}
}

double stellar_lifetime(double logM, double logZ)
{
    return interp2d(StellarTbl.lifetime_yr, logM, logZ);
}

double stellar_remnant_mass(double logM, double logZ)
{
    return interp2d_rt_aware(StellarTbl.remnant_mass, logM, logZ);
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

/* --- PMS duration / feedback delay (2D bilinear) --- */

double stellar_t_PMS(double logM, double logZ)
{
    return interp2d(StellarTbl.t_PMS, logM, logZ);
}

/* --- 3D trilinear (logM, logZ, log_age) --- */

double stellar_v_wind(double logM, double logZ, double log_age)
{
    return interp3d(StellarTbl.v_wind, logM, logZ, log_age);
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

double stellar_type_ia_yield(int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;
    return StellarTbl.type_ia_yields[elem];
}

/* Absolute element ejecta tables [Msun] — X_init baked in by table builder.
 * These are the runtime quantities for SN/AGB metal injection. */
double stellar_elem_ej_SN(double logM, double logZ, int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;
    return interp2d_elem_rt_aware(StellarTbl.elem_ej_SN_mass, logM, logZ, elem);
}

double stellar_elem_ej_AGB(double logM, double logZ, int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;
    return interp2d_elem_rt_aware(StellarTbl.elem_ej_AGB_mass, logM, logZ, elem);
}

/* Cumulative wind ejecta of element `elem` [Msun] from birth up to log_age.
 * Per-corner age clamp at lifetime: cum stays at end-of-life value past death. */
double stellar_elem_ej_wind_cumulative(double logM, double logZ, double log_age, int elem)
{
    if(!StellarTbl.elem_ej_wind_cumulative || elem < 0 || elem >= STBL_NELEM) return 0;
    int iz, im; double fz, fm;
    stbl_idx_Z(logZ, &iz, &fz);
    stbl_idx_M(logM, &im, &fm);
    double corners[4];
    int iz_c[4] = {iz, iz, iz+1, iz+1};
    int im_c[4] = {im, im+1, im, im+1};
    for(int c = 0; c < 4; c++) {
        double lt = StellarTbl.lifetime_yr[IDX2(iz_c[c], im_c[c])];
        double log_age_c = (lt > 0 && pow(10.0, log_age) > lt) ? log10(lt) : log_age;
        int ia_c; double fa_c;
        stbl_idx_age(log_age_c, &ia_c, &fa_c);
        double v0 = StellarTbl.elem_ej_wind_cumulative[IDX4(iz_c[c], im_c[c], ia_c,   elem)];
        double v1 = StellarTbl.elem_ej_wind_cumulative[IDX4(iz_c[c], im_c[c], ia_c+1, elem)];
        corners[c] = v0*(1-fa_c) + v1*fa_c;
    }
    double c0 = corners[0]*(1-fm) + corners[1]*fm;
    double c1 = corners[2]*(1-fm) + corners[3]*fm;
    return c0*(1-fz) + c1*fz;
}




#endif /* GALSF_SINGLE_STARS_EVOLUTION_TABLES */
