/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <string>
#include <vector>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h (a Kokkos TU: the device copy below) */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "stellar_evolution_tables.h"

/* Single-star stellar-evolution tables (stella-tables, github.com/Ulli1991/stella-tables): lifetimes, tracks, band
   luminosities, winds, yields, remnants. Installed outside the repository by scripts/get_stella_deps.sh; the
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
        gizmo_request_controlled_stop(7801, "stellar evolution tables missing or not the version these sources expect; run scripts/get_stella_deps.sh", __FILE__, __LINE__, __FUNCTION__);
    }
}


/* Interpolation and accessors */
struct StellarEvolutionTables StellarTbl;

/* the lookups are KOKKOS_INLINE_FUNCTIONs in stellar_evolution_tables.h, on a StellarTablesView; the host accessors below
   call them with this view of StellarTbl */
#ifdef GALSF_SINGLE_STARS_TYPE_IA
static double IaFdead[STBL_NZ * IA_FDEAD_NT]; /* f_dead(t, Z), [iz * IA_FDEAD_NT + it] */
#endif
struct StellarTablesView stellar_tables_host_view(void)
{
    struct StellarTablesView v; memset(&v, 0, sizeof(v));
    const struct StellarEvolutionTables &T = StellarTbl;
    v.log_Z = T.log_Z; v.log_M = T.log_M; v.log_age = T.log_age; v.lifetime_yr = T.lifetime_yr; v.remnant_mass = T.remnant_mass; v.t_PMS = T.t_PMS;
    v.elem_ej_SN_mass = T.elem_ej_SN_mass; v.elem_ej_AGB_mass = T.elem_ej_AGB_mass; v.type_ia_yields = T.type_ia_yields; v.remnant_type = T.remnant_type;
#ifdef GALSF_SINGLE_STARS_TYPE_IA
    v.ia_fdead = IaFdead;
#endif
    v.v_wind = T.v_wind; v.log_L_LW = T.log_L_LW; v.log_L_FUV_total = T.log_L_FUV_total; v.log_L_FUV_M1 = T.log_L_FUV_M1; v.log_L_NUV = T.log_L_NUV;
    v.log_L_OPT_NIR = T.log_L_OPT_NIR; v.log_L_bol = T.log_L_bol; v.log_Q_ion = T.log_Q_ion; v.log_L_ion_tot = T.log_L_ion_tot; v.log_L_ion_H0 = T.log_L_ion_H0;
    v.log_L_ion_He0 = T.log_L_ion_He0; v.log_L_ion_He1 = T.log_L_ion_He1; v.log_L_ion_He2 = T.log_L_ion_He2;
#ifdef TREECOL_SHELLS_ION_HELIUM
    v.log_Q_ion_H0 = T.log_Q_ion_H0;
#endif
    v.surface_abundances = T.surface_abundances; v.elem_ej_wind_cumulative = T.elem_ej_wind_cumulative;
    return v;
}
static struct StellarTablesView stbl_host; /* set once the tables are loaded */

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
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
static_assert(offsetof(struct StellarEvolutionTables, logR_cm) > offsetof(struct StellarEvolutionTables, v_wind), "table struct layout: logR_cm is a per-rank pointer, outside the broadcast block");
#endif
static_assert(offsetof(struct StellarEvolutionTables, elem_ej_AGB_mass) == offsetof(struct StellarEvolutionTables, elem_ej_SN_mass) + sizeof(StellarTbl.elem_ej_SN_mass), "table struct layout");
static_assert(offsetof(struct StellarEvolutionTables, type_ia_yields) == offsetof(struct StellarEvolutionTables, elem_ej_AGB_mass) + sizeof(StellarTbl.elem_ej_AGB_mass), "table struct layout");

/* The device view (All.StellarTablesDev): where the default execution space reads host memory (host backends; APUs whose
   Kokkos build says so) it is the host view itself, the node-shared segment included; otherwise each table is copied once into
   device memory (about 340 MB per rank with the full grid), kept until stellar_evolution_tables_free, which runs before Kokkos
   is finalised. The tables do not change after the load, so the copy never goes stale. */
static std::vector<void *> stbl_device_copies;
static long long stbl_device_bytes = 0;
template<typename T> static const T *stbl_to_device(const T *src, size_t n, const char *label)
{
    if(!src || n == 0) {return NULL;}
    const size_t bytes = n * sizeof(T);
    void *d = gizmo_gpu_alloc_device(bytes, label);
    if(!d) {printf("Stellar evolution tables: rank %d has no device memory for %s (%zu bytes)\n", ThisTask, label, bytes); fflush(stdout); endrun(7803);}
    gizmo_gpu_copy_to_device(d, src, bytes);
    stbl_device_copies.push_back(d); stbl_device_bytes += (long long) bytes;
    return (const T *) d;
}
static void stellar_tables_device_build(void)
{
    const struct StellarTablesView h = stbl_host;
    if(Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>::accessible) {All.StellarTablesDev = h; return;}
    const size_t n2d = (size_t)STBL_NZ * STBL_NM, n3d = n2d * STBL_NAGE, n4d = n3d * STBL_NELEM;
    struct StellarTablesView d; memset(&d, 0, sizeof(d));
    d.log_Z = stbl_to_device(h.log_Z, STBL_NZ, "stbl_log_Z"); d.log_M = stbl_to_device(h.log_M, STBL_NM, "stbl_log_M"); d.log_age = stbl_to_device(h.log_age, STBL_NAGE, "stbl_log_age");
    d.lifetime_yr = stbl_to_device(h.lifetime_yr, n2d, "stbl_lifetime"); d.remnant_mass = stbl_to_device(h.remnant_mass, n2d, "stbl_remnant_mass");
    d.t_PMS = stbl_to_device(h.t_PMS, n2d, "stbl_t_PMS"); d.remnant_type = stbl_to_device(h.remnant_type, n2d, "stbl_remnant_type");
    d.elem_ej_SN_mass = stbl_to_device(h.elem_ej_SN_mass, n2d * STBL_NELEM, "stbl_ej_SN"); d.elem_ej_AGB_mass = stbl_to_device(h.elem_ej_AGB_mass, n2d * STBL_NELEM, "stbl_ej_AGB");
    d.type_ia_yields = stbl_to_device(h.type_ia_yields, STBL_NELEM, "stbl_ia_yields");
    d.ia_fdead = stbl_to_device(h.ia_fdead, (size_t) STBL_NZ * IA_FDEAD_NT, "stbl_ia_fdead");
    const float *const *src3[] = {&h.v_wind, &h.log_L_LW, &h.log_L_FUV_total, &h.log_L_FUV_M1, &h.log_L_NUV, &h.log_L_OPT_NIR, &h.log_L_bol, &h.log_Q_ion, &h.log_L_ion_tot, &h.log_L_ion_H0, &h.log_L_ion_He0, &h.log_L_ion_He1, &h.log_L_ion_He2, &h.log_Q_ion_H0};
    const float **dst3[] = {&d.v_wind, &d.log_L_LW, &d.log_L_FUV_total, &d.log_L_FUV_M1, &d.log_L_NUV, &d.log_L_OPT_NIR, &d.log_L_bol, &d.log_Q_ion, &d.log_L_ion_tot, &d.log_L_ion_H0, &d.log_L_ion_He0, &d.log_L_ion_He1, &d.log_L_ion_He2, &d.log_Q_ion_H0};
    for(size_t k = 0; k < sizeof(src3) / sizeof(src3[0]); k++) {*dst3[k] = stbl_to_device(*src3[k], n3d, "stbl_3d");}
    d.surface_abundances = stbl_to_device(h.surface_abundances, n4d, "stbl_surface_abundances");
    d.elem_ej_wind_cumulative = stbl_to_device(h.elem_ej_wind_cumulative, n4d, "stbl_wind_cumulative");
    Kokkos::fence();
    All.StellarTablesDev = d;
    if(ThisTask == 0) {printf("Stellar evolution tables: device copy %.0f MB per rank\n", stbl_device_bytes / 1e6);}
}

void stellar_evolution_tables_free(void)
{
    for(void *d : stbl_device_copies) {gizmo_gpu_free_device(d);} /* before Kokkos is finalised */
    stbl_device_copies.clear(); stbl_device_bytes = 0;
    memset(&All.StellarTablesDev, 0, sizeof(All.StellarTablesDev));
    if(stbl_win != MPI_WIN_NULL) {MPI_Win_free(&stbl_win);}
}

void stellar_evolution_tables_load(void)
{
    const size_t n2d = (size_t)STBL_NZ * STBL_NM, n3d = n2d * STBL_NAGE, n4d = n3d * STBL_NELEM;
#ifdef TREECOL_SHELLS_ION_HELIUM
    enum {N3_HE = 1};
#else
    enum {N3_HE = 0};
#endif
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
    enum {N3 = 14 + N3_HE};
#else
    enum {N3 = 13 + N3_HE};
#endif
    float **f3[N3] = {&StellarTbl.v_wind, &StellarTbl.log_L_LW, &StellarTbl.log_L_FUV_total, &StellarTbl.log_L_FUV_M1, &StellarTbl.log_L_NUV, &StellarTbl.log_L_OPT_NIR, &StellarTbl.log_L_bol, &StellarTbl.log_Q_ion, &StellarTbl.log_L_ion_tot, &StellarTbl.log_L_ion_H0, &StellarTbl.log_L_ion_He0, &StellarTbl.log_L_ion_He1, &StellarTbl.log_L_ion_He2
#ifdef TREECOL_SHELLS_ION_HELIUM
        , &StellarTbl.log_Q_ion_H0
#endif
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
        , &StellarTbl.logR_cm
#endif
    };
    const char *n3[N3] = {"v_wind", "log_L_LW", "log_L_FUV_total", "log_L_FUV_M1", "log_L_NUV", "log_L_OPT_NIR", "log_L_bol", "log_Q_ion", "log_L_ion_tot", "log_L_ion_H0", "log_L_ion_He0", "log_L_ion_He1", "log_L_ion_He2"
#ifdef TREECOL_SHELLS_ION_HELIUM
        , "log_Q_ion_H0"
#endif
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
        , "logR_cm"
#endif
    };
    const size_t nfloat = N3 * n3d + 2 * n4d;

    /* one shared segment per node, owned by node rank 0 */
    gizmo_node_comm_init();
    float *seg = NULL;
    MPI_Aint segsize = (GizmoNodeRankOfTask == 0) ? (MPI_Aint)(nfloat * sizeof(float)) : 0;
    MPI_Win_allocate_shared(segsize, sizeof(float), MPI_INFO_NULL, GizmoNodeComm, &seg, &stbl_win);
    gizmo_mem_account_add(GIZMO_MEM_SINGLE_STARS, (long long) segsize + (long long) sizeof(StellarTbl)); /* the segment on its node's lead rank; the plain tables on every rank */
    { MPI_Aint sz; int du; MPI_Win_shared_query(stbl_win, 0, &sz, &du, &seg); }
    for(int k = 0; k < N3; k++) {*f3[k] = seg + k * n3d;}
    StellarTbl.surface_abundances = seg + N3 * n3d;
    StellarTbl.elem_ej_wind_cumulative = seg + N3 * n3d + n4d;

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
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
        err |= stbl_read(f, "phase_transitions/t_MS_start", H5T_NATIVE_DOUBLE, StellarTbl.t_MS_start, n2d);
        err |= stbl_read(f, "phase_transitions/t_MS_end", H5T_NATIVE_DOUBLE, StellarTbl.t_MS_end, n2d);
#endif
        for(int k = 0; k < N3; k++) {err |= stbl_read(f, n3[k], H5T_NATIVE_FLOAT, *f3[k], n3d);}
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
    stbl_host = stellar_tables_host_view();
#ifdef GALSF_SINGLE_STARS_TYPE_IA
    stellar_ia_fdead_init();
#endif
    stellar_tables_device_build();
    if(ThisTask == 0) {printf("Stellar evolution tables: %d Z x %d M x %d ages, %.0f MB per node\n", STBL_NZ, STBL_NM, STBL_NAGE, nfloat * sizeof(float) / 1e6);}
}

double stellar_lifetime(double logM, double logZ)
{
    return stbl_interp2d(stbl_host, StellarTbl.lifetime_yr, logM, logZ);
}

double stellar_remnant_mass(double logM, double logZ)
{
    return stbl_interp2d_rt_aware(stbl_host, StellarTbl.remnant_mass, logM, logZ, -1);
}

int stellar_remnant_type(double logM, double logZ) {return stbl_remnant_type(stbl_host, logM, logZ);} /* nearest corner: discrete values */

/* --- PMS duration / feedback delay (2D bilinear) --- */

double stellar_t_PMS(double logM, double logZ)
{
    return stbl_interp2d(stbl_host, StellarTbl.t_PMS, logM, logZ);
}

#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
double stellar_t_MS_start(double logM, double logZ) {return stbl_interp2d(stbl_host, StellarTbl.t_MS_start, logM, logZ);}
double stellar_t_MS_end(double logM, double logZ) {return stbl_interp2d(stbl_host, StellarTbl.t_MS_end, logM, logZ);}
#endif

/* --- 3D trilinear (logM, logZ, log_age) --- */

double stellar_v_wind(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.v_wind, logM, logZ, log_age);
}

double stellar_log_L_LW(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_LW, logM, logZ, log_age);
}

double stellar_log_L_FUV_total(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_FUV_total, logM, logZ, log_age);
}

double stellar_log_L_FUV_M1(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_FUV_M1, logM, logZ, log_age);
}

double stellar_log_L_NUV(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_NUV, logM, logZ, log_age);
}

double stellar_log_L_OPT_NIR(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_OPT_NIR, logM, logZ, log_age);
}

#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
double stellar_log_R_cm(double logM, double logZ, double log_age) {return stbl_interp3d(stbl_host, StellarTbl.logR_cm, logM, logZ, log_age);}
#endif
double stellar_log_L_bol(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_bol, logM, logZ, log_age);
}

double stellar_log_Q_ion(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_Q_ion, logM, logZ, log_age);
}

double stellar_log_L_ion_tot(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_ion_tot, logM, logZ, log_age);
}

double stellar_log_L_ion_H0(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_ion_H0, logM, logZ, log_age);
}

double stellar_log_L_ion_He0(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_ion_He0, logM, logZ, log_age);
}

double stellar_log_L_ion_He1(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_ion_He1, logM, logZ, log_age);
}

double stellar_log_L_ion_He2(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_L_ion_He2, logM, logZ, log_age);
}

#ifdef TREECOL_SHELLS_ION_HELIUM
double stellar_log_Q_ion_H0(double logM, double logZ, double log_age)
{
    return stbl_interp3d(stbl_host, StellarTbl.log_Q_ion_H0, logM, logZ, log_age);
}
#endif

/* --- Surface abundances --- */

double stellar_surface_abundance(double logM, double logZ, double log_age, int elem)
{
    return stbl_interp4d_clamped(stbl_host, StellarTbl.surface_abundances, logM, logZ, log_age, elem);
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
    return stbl_interp2d_rt_aware(stbl_host, StellarTbl.elem_ej_SN_mass, logM, logZ, elem);
}

double stellar_elem_ej_AGB(double logM, double logZ, int elem)
{
    if(elem < 0 || elem >= STBL_NELEM) return 0;
    return stbl_interp2d_rt_aware(stbl_host, StellarTbl.elem_ej_AGB_mass, logM, logZ, elem);
}

double stellar_death_ejecta(double logM, double logZ, double M_death, double *m_k) {return stbl_death_ejecta(stbl_host, logM, logZ, M_death, m_k);}

/* Cumulative wind ejecta of element `elem` [Msun] from birth up to log_age.
 * Per-corner age clamp at lifetime: cum stays at end-of-life value past death. */
double stellar_elem_ej_wind_cumulative(double logM, double logZ, double log_age, int elem)
{
    return stbl_interp4d_clamped(stbl_host, StellarTbl.elem_ej_wind_cumulative, logM, logZ, log_age, elem);
}

#ifdef GALSF_SINGLE_STARS_TYPE_IA
/* f_dead(t, Z): fraction of the formed mass (Kroupa, IMF lower end .. 350 Msun) in white-dwarf progenitors that have died
   by age t; normalises the per-white-dwarf Type Ia hazard (as the reference implementation) */

void stellar_ia_fdead_init(void)
{
    const int NM = 4000;
#ifdef GALSF_SINGLE_STARS_COLLECT_GAS_FROM_NEIGHBORS
    double Mmin = DMAX(All.IMFSampleMinMass, 0.08), Mmax = 350.;
#else
    double Mmin = 0.08, Mmax = 350.;
#endif
    double lmin = log10(Mmin), dl = (log10(Mmax) - lmin) / NM;
    std::vector<double> tdie(NM), w(NM);
    for(int iz = 0; iz < STBL_NZ; iz++) {
        double logZ = StellarTbl.log_Z[iz], wtot = 0;
        for(int im = 0; im < NM; im++) {
            double lM = lmin + (im + 0.5) * dl, M = pow(10., lM);
            w[im] = ((M < 0.5) ? 2.0 * pow(M, -0.3) : pow(M, -1.3)) * M * dl * log(10.); /* (M dN/dM) dM */
            wtot += w[im];
            tdie[im] = (stellar_remnant_type(lM, logZ) == REM_WD) ? stellar_t_PMS(lM, logZ) + stellar_lifetime(lM, logZ) : -1;
        }
        for(int it = 0; it < IA_FDEAD_NT; it++) {
            double t = pow(10., IA_FDEAD_LOGT_MIN + it * (IA_FDEAD_LOGT_MAX - IA_FDEAD_LOGT_MIN) / (IA_FDEAD_NT - 1)), wd = 0;
            for(int im = 0; im < NM; im++) {if(tdie[im] > 0 && tdie[im] <= t) {wd += w[im];}}
            IaFdead[iz * IA_FDEAD_NT + it] = wd / wtot;
        }
    }
}

double stellar_ia_fdead(double t_yr, double logZ) {return stbl_ia_fdead(stbl_host, t_yr, logZ);}
#endif

#endif /* GALSF_SINGLE_STARS_EVOLUTION_TABLES */
