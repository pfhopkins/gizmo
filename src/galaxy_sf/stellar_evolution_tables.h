/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Single-star stellar-evolution tables (stella-tables): installed outside the repository, checked at start-up. */
#ifndef STELLAR_EVOLUTION_TABLES_H
#define STELLAR_EVOLUTION_TABLES_H

#ifdef GALSF_SINGLE_STARS_EVOLUTION_TABLES
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
    /* [NZ * NM * NAGE], node-shared */
    float *v_wind;
    /* [NZ * NM * NAGE * NELEM], node-shared */
    float *surface_abundances, *elem_ej_wind_cumulative; /* mass fraction; cumulative wind ejecta [Msun] */
    /* [NZ * NM * NELEM]: absolute SN and AGB ejecta [Msun] with the table's birth composition */
    double elem_ej_SN_mass[STBL_NZ * STBL_NM * STBL_NELEM], elem_ej_AGB_mass[STBL_NZ * STBL_NM * STBL_NELEM];
    double type_ia_yields[STBL_NELEM];
    int loaded;
};
extern struct StellarEvolutionTables StellarTbl;

void stellar_evolution_tables_init(void);  /* start-up check (rank 0) */
int  stellar_tables_verify(const std::vector<std::string> &paths); /* stellar_tables_check.cc */
void stellar_evolution_tables_load(void);  /* collective: read on rank 0, share per node */
void stellar_evolution_tables_free(void);  /* releases the node-shared segment */

/* (logM, logZ): log10 of initial mass [Msun] and of metallicity (mass fraction) */
double stellar_lifetime(double logM, double logZ);          /* [yr] */
double stellar_remnant_mass(double logM, double logZ);      /* [Msun] */
int    stellar_remnant_type(double logM, double logZ);      /* enum StellarRemnantType */
double stellar_t_PMS(double logM, double logZ);             /* [yr] */
/* (logM, logZ, log10 age [yr]) */
double stellar_v_wind(double logM, double logZ, double log_age);     /* [km/s] */
double stellar_surface_abundance(double logM, double logZ, double log_age, int elem);
double stellar_elem_ej_wind_cumulative(double logM, double logZ, double log_age, int elem); /* [Msun] from birth */
double stellar_elem_ej_SN(double logM, double logZ, int elem);   /* [Msun], table birth composition */
double stellar_elem_ej_AGB(double logM, double logZ, int elem);  /* [Msun], table birth composition */
double stellar_type_ia_yield(int elem);                          /* [Msun] per Ia */
#ifdef GALSF_SINGLE_STARS_TYPE_IA
void   stellar_ia_fdead_init(void);
double stellar_ia_fdead(double t_yr, double logZ);              /* dead white-dwarf-progenitor mass fraction at age t */
#endif
#endif

#endif
