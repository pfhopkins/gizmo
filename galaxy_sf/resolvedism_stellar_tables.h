#ifndef RESOLVEDISM_STELLAR_TABLES_H
#define RESOLVEDISM_STELLAR_TABLES_H

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES

#define STBL_NZ    15
#define STBL_NM    110
#define STBL_NAGE  512
#define STBL_NELEM 15

/* Element index mapping (matches HDF5 'elements' dataset order) */
enum StellarElement {
    ELEM_H  = 0, ELEM_He = 1, ELEM_C  = 2, ELEM_N  = 3, ELEM_O  = 4,
    ELEM_F  = 5, ELEM_Ne = 6, ELEM_Na = 7, ELEM_Mg = 8, ELEM_Al = 9,
    ELEM_Si = 10, ELEM_S = 11, ELEM_Ca = 12, ELEM_Ti = 13, ELEM_Fe = 14
};

/* Remnant type codes (matches HDF5 'remnant_type' dataset) */
enum StellarRemnantType {
    REM_WD    = 0,  /* White dwarf */
    REM_ECSN  = 1,  /* Electron-capture SN (NS) */
    REM_CCSN  = 2,  /* Core-collapse SN (NS or BH) */
    REM_FSN   = 3,  /* Failed SN (direct BH, no explosion) */
    REM_PPISN = 4,  /* Pulsational pair-instability (BH) */
    REM_PISN  = 5,  /* Pair-instability SN (complete disruption) */
    REM_DBH   = 6   /* Direct BH (no explosion) */
};

struct StellarTables {
    /* 1D grid axes */
    double log_Z[STBL_NZ];
    double log_M[STBL_NM];
    double log_age[STBL_NAGE];
    double Z[STBL_NZ];
    double M[STBL_NM];

    /* Grid spacing for O(1) index computation */
    double log_Z_min, log_Z_max, dlog_Z;
    double log_M_min, log_M_max, dlog_M;
    double log_age_min, log_age_max, dlog_age;

    /* 2D scalar datasets — flat [NZ * NM] */
    double lifetime_yr[STBL_NZ * STBL_NM];
    double remnant_mass[STBL_NZ * STBL_NM];
    int    remnant_type[STBL_NZ * STBL_NM];
    double M_preSN[STBL_NZ * STBL_NM];
    double M_CO_core[STBL_NZ * STBL_NM];
    double M_He_core[STBL_NZ * STBL_NM];

    /* 3D time-dependent datasets — dynamically allocated [NZ * NM * NAGE] */
    float *M_current;      /* Msun */
    float *v_wind;         /* km/s */
    float *log_Q_ion;      /* log10(photons/s) */
    float *log_L_FUV;      /* log10(erg/s), 6-11.2 eV photoelectric */
    float *log_L_LW;       /* log10(erg/s), 11.2-13.6 eV Lyman-Werner */
    float *log_L_FUV_total;/* log10(erg/s), 6-13.6 eV total FUV */
    float *log_L_bol;      /* log10(erg/s), bolometric */

    /* Ionizing sub-band luminosities for M1 RT [NZ * NM * NAGE] */
    float *log_L_ion_tot;  /* log10(erg/s), total ionizing (>13.6 eV) */
    float *log_L_ion_H0;   /* log10(erg/s), 13.6-24.6 eV */
    float *log_L_ion_He0;  /* log10(erg/s), 24.6-54.4 eV */
    float *log_L_ion_He1;  /* log10(erg/s), 54.4-70 eV */
    float *log_L_ion_He2;  /* log10(erg/s), >70 eV */

    /* 4D surface abundances [NZ * NM * NAGE * NELEM] — only if winds enabled */
    float *surface_abundances;

    /* 3D yield datasets — flat [NZ * NM * NELEM] */
    double net_yields[STBL_NZ * STBL_NM * STBL_NELEM];

    /* Type Ia yields [NELEM] */
    double type_ia_yields[STBL_NELEM];

    int loaded;
};

extern struct StellarTables StellarTbl;

/* ---- Loading ---- */
void resolvedism_load_stellar_tables(void);
void resolvedism_free_stellar_tables(void);

/* ---- 2D bilinear interpolation (logM, logZ) for scalar quantities ---- */
double stellar_lifetime(double logM, double logZ);
double stellar_remnant_mass(double logM, double logZ);
int    stellar_remnant_type(double logM, double logZ);
double stellar_M_preSN(double logM, double logZ);

/* ---- 3D trilinear interpolation (logM, logZ, log_age) ---- */
double stellar_M_current(double logM, double logZ, double log_age);
double stellar_v_wind(double logM, double logZ, double log_age);
double stellar_log_Q_ion(double logM, double logZ, double log_age);
double stellar_log_L_FUV(double logM, double logZ, double log_age);
double stellar_log_L_LW(double logM, double logZ, double log_age);
double stellar_log_L_FUV_total(double logM, double logZ, double log_age);
double stellar_log_L_bol(double logM, double logZ, double log_age);
double stellar_log_L_ion_tot(double logM, double logZ, double log_age);
double stellar_log_L_ion_H0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He1(double logM, double logZ, double log_age);
double stellar_log_L_ion_He2(double logM, double logZ, double log_age);

/* ---- Surface abundances (3D + element index) ---- */
double stellar_surface_abundance(double logM, double logZ, double log_age, int elem);

/* ---- Yields (2D + element index) ---- */
double stellar_net_yield(double logM, double logZ, int elem);
double stellar_type_ia_yield(int elem);

#endif /* GALSF_RESOLVEDISM_STELLAR_TABLES */
#endif /* RESOLVEDISM_STELLAR_TABLES_H */
