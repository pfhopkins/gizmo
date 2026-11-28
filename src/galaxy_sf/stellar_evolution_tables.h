/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Single-star stellar-evolution tables (stella-tables): installed outside the repository, checked at start-up. */
#ifndef STELLAR_EVOLUTION_TABLES_H
#define STELLAR_EVOLUTION_TABLES_H

#if defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES) || defined(KETJU_PN_REMNANT_TAG)
#include <math.h>
#include <string>
#include <vector>
#define STBL_NZ    15     /* table grid: metallicities */
#define STBL_NM    110    /* initial masses */
#define STBL_NAGE  768    /* ages (non-uniform) */
#define STBL_NELEM 27     /* elements H..Zn */

/* element order of the tables (HDF5 'elements') */
enum StellarElement {
    ELEM_H  = 0,  ELEM_He = 1,  ELEM_C  = 2,  ELEM_N  = 3,  ELEM_O  = 4,
    ELEM_F  = 5,  ELEM_Ne = 6,  ELEM_Na = 7,  ELEM_Mg = 8,  ELEM_Al = 9,
    ELEM_Si = 10, ELEM_P  = 11, ELEM_S  = 12, ELEM_Cl = 13, ELEM_Ar = 14,
    ELEM_K  = 15, ELEM_Ca = 16, ELEM_Sc = 17, ELEM_Ti = 18, ELEM_V  = 19,
    ELEM_Cr = 20, ELEM_Mn = 21, ELEM_Fe = 22, ELEM_Co = 23, ELEM_Ni = 24,
    ELEM_Cu = 25, ELEM_Zn = 26
};

/* remnant classes (HDF5 'remnant_type') */
enum StellarRemnantType {
    REM_WD    = 0,  /* white dwarf */
    REM_ECSN  = 1,  /* electron-capture SN (NS) */
    REM_CCSN  = 2,  /* core-collapse SN (NS or BH) */
    REM_FSN   = 3,  /* failed SN (direct BH, no explosion) */
    REM_PPISN = 4,  /* pulsational pair-instability (BH) */
    REM_PISN  = 5,  /* pair-instability SN (complete disruption) */
    REM_DBH   = 6   /* direct BH (no explosion) */
};

/* 1D/2D tables are copied on every rank; the 3D (age-resolved) and 4D (age x element) tables live once per node in
   MPI shared memory */
struct StellarEvolutionTables {
    double log_Z[STBL_NZ], log_M[STBL_NM], log_age[STBL_NAGE], Z[STBL_NZ], M[STBL_NM];
    /* [NZ * NM] */
    double lifetime_yr[STBL_NZ * STBL_NM], remnant_mass[STBL_NZ * STBL_NM];
    int    remnant_type[STBL_NZ * STBL_NM];
    double t_PMS[STBL_NZ * STBL_NM];   /* pre-main-sequence duration (feedback delay) [yr] */
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
    double t_MS_start[STBL_NZ * STBL_NM], t_MS_end[STBL_NZ * STBL_NM]; /* table age of the main sequence [yr] */
#endif
    /* [NZ * NM * NAGE], node-shared.  Pointers into each rank's own mapping of the node segment: all of them from
       v_wind on, since the plain data above is broadcast from rank 0 as one block that ends at v_wind */
    float *v_wind, *log_L_LW, *log_L_FUV_total, *log_L_FUV_M1, *log_L_NUV, *log_L_OPT_NIR, *log_L_bol, *log_Q_ion, *log_L_ion_tot, *log_L_ion_H0, *log_L_ion_He0, *log_L_ion_He1, *log_L_ion_He2;
#ifdef TREECOL_SHELLS_ION_HELIUM
    float *log_Q_ion_H0; /* 13.6-24.6 eV photons */
#endif
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
    float *logR_cm;      /* log10 stellar radius [cm] */
#endif
    /* [NZ * NM * NAGE * NELEM], node-shared */
    float *surface_abundances, *elem_ej_wind_cumulative; /* mass fraction; cumulative wind ejecta [Msun] */
    /* [NZ * NM * NELEM]: absolute SN and AGB ejecta [Msun] with the table's birth composition */
    double elem_ej_SN_mass[STBL_NZ * STBL_NM * STBL_NELEM], elem_ej_AGB_mass[STBL_NZ * STBL_NM * STBL_NELEM];
    double type_ia_yields[STBL_NELEM];
    int loaded;
};
extern struct StellarEvolutionTables StellarTbl;

#define IA_FDEAD_NT 96 /* Type Ia f_dead(t, Z) table: ages */
#define IA_FDEAD_LOGT_MIN 6.5
#define IA_FDEAD_LOGT_MAX 10.3 /* log10 yr */

/* The tables as one memory space holds them: plain pointers, captured by value into a kernel. The host view points into
   StellarTbl (rank copy and node-shared segment); the device view (All.StellarTablesDev, set by stellar_evolution_tables_load
   and kept across a restart) points into Kokkos Views in the device space, or at the host tables where the default execution
   space reads host memory (host backends, APUs). Every lookup below is one KOKKOS_INLINE_FUNCTION used on both: the host
   accessors further down call it with the host view. */
struct StellarTablesView {
    const double *log_Z, *log_M, *log_age, *lifetime_yr, *remnant_mass, *t_PMS, *elem_ej_SN_mass, *elem_ej_AGB_mass, *type_ia_yields, *ia_fdead;
    const int *remnant_type;
    const float *v_wind, *log_L_LW, *log_L_FUV_total, *log_L_FUV_M1, *log_L_NUV, *log_L_OPT_NIR, *log_L_bol, *log_Q_ion, *log_L_ion_tot, *log_L_ion_H0, *log_L_ion_He0, *log_L_ion_He1, *log_L_ion_He2, *log_Q_ion_H0;
    const float *surface_abundances, *elem_ej_wind_cumulative;
};

/* the star-record columns the lookups of a star read (SingleStars, galaxy_sf/single_star_record.h; slot = StarIdx - 1). The
   records live in Kokkos shared space; All.SingleStarRecDev holds their current addresses (single_star_record_publish). */
struct SingleStarRecordView {const double *M_drawn, *Z_birth; const float *rho_birth;};

/* flat index helpers */
#define STBL_IDX2(iz,im)       ((iz) * STBL_NM + (im))
#define STBL_IDX3(iz,im,ia)    (((iz) * STBL_NM + (im)) * STBL_NAGE + (ia))
#define STBL_IDX4(iz,im,ia,ie) ((((iz) * STBL_NM + (im)) * STBL_NAGE + (ia)) * STBL_NELEM + (ie))

/* Binary search: i0 with grid[i0] <= val < grid[i0+1] and the fractional position f in [0,1], clamped to the grid */
KOKKOS_INLINE_FUNCTION void stbl_idx_bsearch(const double *grid, int N, double val, int *i0, double *f)
{
    if(val <= grid[0])   { *i0 = 0;     *f = 0.0; return; }
    if(val >= grid[N-1]) { *i0 = N - 2; *f = 1.0; return; }
    int lo = 0, hi = N - 1;
    while(hi - lo > 1) { int mid = (lo + hi) / 2; if(grid[mid] <= val) lo = mid; else hi = mid; }
    *i0 = lo;
    *f  = (val - grid[lo]) / (grid[hi] - grid[lo]);
}
KOKKOS_INLINE_FUNCTION void stbl_idx_ZM(const StellarTablesView &T, double logM, double logZ, int *iz, double *fz, int *im, double *fm)
{
    stbl_idx_bsearch(T.log_Z, STBL_NZ, logZ, iz, fz);
    stbl_idx_bsearch(T.log_M, STBL_NM, logM, im, fm);
}

/* bilinear on a [NZ x NM] array */
KOKKOS_INLINE_FUNCTION double stbl_interp2d(const StellarTablesView &T, const double *arr, double logM, double logZ)
{
    int iz, im; double fz, fm; stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    double v00 = arr[STBL_IDX2(iz,   im  )];
    double v01 = arr[STBL_IDX2(iz,   im+1)];
    double v10 = arr[STBL_IDX2(iz+1, im  )];
    double v11 = arr[STBL_IDX2(iz+1, im+1)];
    return (1-fz)*(1-fm)*v00 + (1-fz)*fm*v01 + fz*(1-fm)*v10 + fz*fm*v11;
}

/* Boundary-aware bilinear: if the four corners of the (logM, logZ) cell share the same remnant_type, standard bilinear;
 * otherwise the cell straddles a rem_type discontinuity (e.g. WD<->ECSN around 7-8 Msun) and the nearest corner is taken
 * (the selection stellar_remnant_type uses), so rem_type-conditional quantities (M_preSN, rem_mass, per-element ejecta) stay
 * consistent with the chosen rem_type: M_preSN - rem - sum ej = 0 at every interpolation point. elem < 0: no element
 * stride (a [NZ x NM] array), else the element of a [NZ x NM x NELEM] array. */
KOKKOS_INLINE_FUNCTION double stbl_interp2d_rt_aware(const StellarTablesView &T, const double *arr, double logM, double logZ, int elem)
{
    int iz, im; double fz, fm; stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    const int st = (elem < 0) ? 1 : STBL_NELEM, e = (elem < 0) ? 0 : elem;
    int rt00 = T.remnant_type[STBL_IDX2(iz,   im  )];
    int rt01 = T.remnant_type[STBL_IDX2(iz,   im+1)];
    int rt10 = T.remnant_type[STBL_IDX2(iz+1, im  )];
    int rt11 = T.remnant_type[STBL_IDX2(iz+1, im+1)];
    if(rt00 == rt01 && rt00 == rt10 && rt00 == rt11) {
        return (1-fz)*(1-fm)*arr[STBL_IDX2(iz, im) * st + e] + (1-fz)*fm*arr[STBL_IDX2(iz, im+1) * st + e]
             + fz*(1-fm)*arr[STBL_IDX2(iz+1, im) * st + e]   + fz*fm*arr[STBL_IDX2(iz+1, im+1) * st + e];
    }
    int iz_near = (fz > 0.5) ? iz + 1 : iz;
    int im_near = (fm > 0.5) ? im + 1 : im;
    return arr[STBL_IDX2(iz_near, im_near) * st + e];
}

/* trilinear on a [NZ x NM x NAGE] float array; corners flagged dead (-99 in the log tables) are left out and the rest
   renormalised; all dead gives -99 */
KOKKOS_INLINE_FUNCTION double stbl_interp3d(const StellarTablesView &T, const float *arr, double logM, double logZ, double log_age)
{
    int iz, im, ia; double fz, fm, fa;
    stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    stbl_idx_bsearch(T.log_age, STBL_NAGE, log_age, &ia, &fa); /* the age grid is non-uniform */
    const double v[8] = {
        arr[STBL_IDX3(iz,   im,   ia  )], arr[STBL_IDX3(iz,   im,   ia+1)],
        arr[STBL_IDX3(iz,   im+1, ia  )], arr[STBL_IDX3(iz,   im+1, ia+1)],
        arr[STBL_IDX3(iz+1, im,   ia  )], arr[STBL_IDX3(iz+1, im,   ia+1)],
        arr[STBL_IDX3(iz+1, im+1, ia  )], arr[STBL_IDX3(iz+1, im+1, ia+1)] };
    const double w[8] = {
        (1-fz)*(1-fm)*(1-fa), (1-fz)*(1-fm)*fa,
        (1-fz)*fm*(1-fa),     (1-fz)*fm*fa,
        fz*(1-fm)*(1-fa),     fz*(1-fm)*fa,
        fz*fm*(1-fa),         fz*fm*fa };
    double vsum = 0, wsum = 0;
    for(int k = 0; k < 8; k++) { if(v[k] > -90.0) { vsum += w[k]*v[k]; wsum += w[k]; } }
    if(wsum <= 0) return -99.0;   /* every corner dead -> dead star */
    return vsum / wsum;
}

/* nearest-corner remnant type */
KOKKOS_INLINE_FUNCTION int stbl_remnant_type(const StellarTablesView &T, double logM, double logZ)
{
    int iz, im; double fz, fm; stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    int iz_near = (fz > 0.5) ? iz + 1 : iz;
    int im_near = (fm > 0.5) ? im + 1 : im;
    if(iz_near >= STBL_NZ) iz_near = STBL_NZ - 1;
    if(im_near >= STBL_NM) im_near = STBL_NM - 1;
    return T.remnant_type[STBL_IDX2(iz_near, im_near)];
}

/* a [NZ x NM x NAGE x NELEM] table (surface abundances, cumulative wind ejecta) at one element: per-corner age clamp at the
   corner's lifetime (each corner dies at its own age: no mixing of live values with post-death ones), then bilinear */
KOKKOS_INLINE_FUNCTION double stbl_interp4d_clamped(const StellarTablesView &T, const float *arr, double logM, double logZ, double log_age, int elem)
{
    if(!arr || elem < 0 || elem >= STBL_NELEM) return 0;
    int iz, im; double fz, fm; stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    double corners[4]; /* [00, 01, 10, 11] = [(iz,im), (iz,im+1), (iz+1,im), (iz+1,im+1)] */
    const int iz_c[4] = {iz, iz, iz+1, iz+1};
    const int im_c[4] = {im, im+1, im, im+1};
    for(int c = 0; c < 4; c++) {
        double lt = T.lifetime_yr[STBL_IDX2(iz_c[c], im_c[c])];
        double log_age_c = (lt > 0 && pow(10.0, log_age) > lt) ? log10(lt) : log_age;
        int ia_c; double fa_c;
        stbl_idx_bsearch(T.log_age, STBL_NAGE, log_age_c, &ia_c, &fa_c);
        double v0 = arr[STBL_IDX4(iz_c[c], im_c[c], ia_c,   elem)];
        double v1 = arr[STBL_IDX4(iz_c[c], im_c[c], ia_c+1, elem)];
        corners[c] = v0*(1-fa_c) + v1*fa_c;
    }
    double c0 = corners[0]*(1-fm) + corners[1]*fm;
    double c1 = corners[2]*(1-fm) + corners[3]*fm;
    return c0*(1-fz) + c1*fz;
}

/* Death ejecta of element k at one grid star as a fraction of its mass at death: ej_k / (rem + sum ej), from the AGB table for
 * white dwarfs and the SN table for the rest; complete disruption (PISN) has rem = 0 */
KOKKOS_INLINE_FUNCTION void stbl_death_fractions(const StellarTablesView &T, int inode, double *e)
{
    const int rt = T.remnant_type[inode];
    const double *ej = ((rt == REM_WD) ? T.elem_ej_AGB_mass : T.elem_ej_SN_mass) + (size_t)inode * STBL_NELEM;
    double sum = 0;
    for(int k = 0; k < STBL_NELEM; k++) {e[k] = (ej[k] > 0) ? ej[k] : 0; sum += e[k];}
    const double D = sum + ((rt == REM_PISN) ? 0 : ((T.remnant_mass[inode] > 0) ? T.remnant_mass[inode] : 0));
    for(int k = 0; k < STBL_NELEM; k++) {e[k] = (D > 0) ? e[k] / D : 0;}
}

/* Element ejecta m_k [Msun] of a star holding M_death [Msun] at death; returns the remnant mass, M_death - sum m_k. The fractions
 * of the four corners are interpolated, not the ejecta and the remnant mass one by one, so that ejecta and remnant add up to the
 * star's mass at any (M, Z) and remnant types are not mixed (the nearest corner where the types differ, as the rt-aware bilinear). */
KOKKOS_INLINE_FUNCTION double stbl_death_ejecta(const StellarTablesView &T, double logM, double logZ, double M_death, double *m_k)
{
    int iz, im; double fz, fm, sum = 0, Mej = 0;
    stbl_idx_ZM(T, logM, logZ, &iz, &fz, &im, &fm);
    const int c[4] = {STBL_IDX2(iz, im), STBL_IDX2(iz, im+1), STBL_IDX2(iz+1, im), STBL_IDX2(iz+1, im+1)};
    const int *rtc = T.remnant_type;
    if(rtc[c[0]] == rtc[c[1]] && rtc[c[0]] == rtc[c[2]] && rtc[c[0]] == rtc[c[3]]) {
        double e[4][STBL_NELEM];
        for(int n = 0; n < 4; n++) {stbl_death_fractions(T, c[n], e[n]);}
        for(int k = 0; k < STBL_NELEM; k++) {m_k[k] = (1-fz)*(1-fm)*e[0][k] + (1-fz)*fm*e[1][k] + fz*(1-fm)*e[2][k] + fz*fm*e[3][k];}
    } else {stbl_death_fractions(T, c[2 * (fz > 0.5) + (fm > 0.5)], m_k);}
    for(int k = 0; k < STBL_NELEM; k++) {m_k[k] = (m_k[k] > 0) ? m_k[k] : 0; sum += m_k[k];}
    const int rt = stbl_remnant_type(T, logM, logZ);
    const double Md = (M_death > 0) ? M_death : 0;
    const double norm = (rt == REM_PISN || sum > 1) ? ((sum > 0) ? 1 / sum : 0) : 1; /* PISN: complete disruption */
    for(int k = 0; k < STBL_NELEM; k++) {m_k[k] *= norm * Md; Mej += m_k[k];}
    return (rt == REM_PISN) ? 0 : Md - Mej;
}

/* f_dead(t, Z): fraction of the formed mass in white-dwarf progenitors that have died by age t [yr] (Type Ia normalisation) */
KOKKOS_INLINE_FUNCTION double stbl_ia_fdead(const StellarTablesView &T, double t_yr, double logZ)
{
    if(t_yr <= 0 || !T.ia_fdead) {return 0;}
    int iz; double fz; stbl_idx_bsearch(T.log_Z, STBL_NZ, logZ, &iz, &fz);
    double x = (log10(t_yr) - IA_FDEAD_LOGT_MIN) / (IA_FDEAD_LOGT_MAX - IA_FDEAD_LOGT_MIN) * (IA_FDEAD_NT - 1);
    x = DMAX(0, DMIN(x, IA_FDEAD_NT - 1));
    int it = (int)x; if(it > IA_FDEAD_NT - 2) {it = IA_FDEAD_NT - 2;}
    double ft = x - it;
    int iz1 = (iz < STBL_NZ - 1) ? iz + 1 : iz;
    const double *F = T.ia_fdead;
    return (1 - fz) * ((1 - ft) * F[iz * IA_FDEAD_NT + it] + ft * F[iz * IA_FDEAD_NT + it + 1]) + fz * ((1 - ft) * F[iz1 * IA_FDEAD_NT + it] + ft * F[iz1 * IA_FDEAD_NT + it + 1]);
}

struct StellarTablesView stellar_tables_host_view(void); /* the host tables, for the host accessors and as the source of the device copy */

void stellar_evolution_tables_init(void);  /* start-up check (rank 0) */
int  stellar_tables_verify(const std::vector<std::string> &paths); /* stellar_tables_check.cc */
void stellar_evolution_tables_load(void);  /* collective: read on rank 0, share per node */
void stellar_evolution_tables_free(void);  /* releases the node-shared segment */

/* (logM, logZ): log10 of initial mass [Msun] and of metallicity (mass fraction) */
double stellar_lifetime(double logM, double logZ);          /* [yr] */
double stellar_remnant_mass(double logM, double logZ);      /* [Msun] */
int    stellar_remnant_type(double logM, double logZ);      /* enum StellarRemnantType */
double stellar_t_PMS(double logM, double logZ);             /* [yr] */
#if defined(KETJU_MERGE_STARS) || defined(STARS_INTEGRATED_BY_RUMI)
double stellar_t_MS_start(double logM, double logZ);        /* [yr] */
double stellar_t_MS_end(double logM, double logZ);          /* [yr] */
double stellar_log_R_cm(double logM, double logZ, double log_age);
#endif
/* (logM, logZ, log10 age [yr]) */
double stellar_v_wind(double logM, double logZ, double log_age);     /* [km/s] */
double stellar_log_L_LW(double logM, double logZ, double log_age);
double stellar_log_L_FUV_total(double logM, double logZ, double log_age);
double stellar_log_L_FUV_M1(double logM, double logZ, double log_age);
double stellar_log_L_NUV(double logM, double logZ, double log_age);
double stellar_log_L_OPT_NIR(double logM, double logZ, double log_age);
double stellar_log_L_bol(double logM, double logZ, double log_age);
double stellar_log_Q_ion(double logM, double logZ, double log_age);
double stellar_log_L_ion_tot(double logM, double logZ, double log_age);
double stellar_log_L_ion_H0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He1(double logM, double logZ, double log_age);
double stellar_log_L_ion_He2(double logM, double logZ, double log_age);
#ifdef TREECOL_SHELLS_ION_HELIUM
double stellar_log_Q_ion_H0(double logM, double logZ, double log_age);
#endif
double stellar_surface_abundance(double logM, double logZ, double log_age, int elem);
double stellar_elem_ej_wind_cumulative(double logM, double logZ, double log_age, int elem); /* [Msun] from birth */
double stellar_elem_ej_SN(double logM, double logZ, int elem);   /* [Msun], table birth composition */
double stellar_elem_ej_AGB(double logM, double logZ, int elem);  /* [Msun], table birth composition */
double stellar_death_ejecta(double logM, double logZ, double M_death, double *m_k); /* [Msun]: element ejecta of a death, returns the remnant mass */
double stellar_type_ia_yield(int elem);                          /* [Msun] per Ia */
#ifdef GALSF_SINGLE_STARS_TYPE_IA
void   stellar_ia_fdead_init(void);
double stellar_ia_fdead(double t_yr, double logZ);              /* dead white-dwarf-progenitor mass fraction at age t */
#endif
#endif

#endif
