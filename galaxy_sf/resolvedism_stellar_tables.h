#ifndef RESOLVEDISM_STELLAR_TABLES_H
#define RESOLVEDISM_STELLAR_TABLES_H

#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES

#define STBL_NZ    15
#define STBL_NM    110
#define STBL_NAGE  768
#define STBL_NELEM 27     /* element-summed yields in the table */
#define STBL_NISO  40     /* isotope-resolved yields in the table */

/* Element index mapping (matches HDF5 'elements' dataset order).
 * Table stores 27 elements + 40 isotopes. Code evolves NUM_RESOLVEDISM_ELEMENTS. */
enum StellarElement {
    ELEM_H  = 0,  ELEM_He = 1,  ELEM_C  = 2,  ELEM_N  = 3,  ELEM_O  = 4,
    ELEM_F  = 5,  ELEM_Ne = 6,  ELEM_Na = 7,  ELEM_Mg = 8,  ELEM_Al = 9,
    ELEM_Si = 10, ELEM_P  = 11, ELEM_S  = 12, ELEM_Cl = 13, ELEM_Ar = 14,
    ELEM_K  = 15, ELEM_Ca = 16, ELEM_Sc = 17, ELEM_Ti = 18, ELEM_V  = 19,
    ELEM_Cr = 20, ELEM_Mn = 21, ELEM_Fe = 22, ELEM_Co = 23, ELEM_Ni = 24,
    ELEM_Cu = 25, ELEM_Zn = 26
};

/* Isotope index mapping (matches HDF5 'isotopes' dataset order) */
enum StellarIsotope {
    ISO_H   = 0,  ISO_He3 = 1,  ISO_He4 = 2,  ISO_Li7 = 3,  ISO_Be7 = 4,
    ISO_C12 = 5,  ISO_C13 = 6,  ISO_N14 = 7,  ISO_N15 = 8,
    ISO_O16 = 9,  ISO_O17 = 10, ISO_O18 = 11, ISO_F19 = 12,
    ISO_Ne20= 13, ISO_Ne21= 14, ISO_Ne22= 15, ISO_Na23= 16,
    ISO_Mg24= 17, ISO_Mg25= 18, ISO_Mg26= 19, ISO_Al26= 20, ISO_Al27= 21,
    ISO_Si28= 22, ISO_Si29= 23,
    ISO_P   = 24, ISO_S   = 25, ISO_Cl  = 26, ISO_Ar  = 27, ISO_K   = 28,
    ISO_Ca  = 29, ISO_Sc  = 30, ISO_Ti  = 31, ISO_V   = 32, ISO_Cr  = 33,
    ISO_Mn  = 34, ISO_Fe  = 35, ISO_Co  = 36, ISO_Ni  = 37, ISO_Cu  = 38,
    ISO_Zn  = 39
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

    /* PMS duration (feedback delay) — flat [NZ * NM], in years */
    double t_PMS[STBL_NZ * STBL_NM];

    /* 2D phase transition times — flat [NZ * NM], in years.  Zero = phase not reached. */
    double t_MS_start[STBL_NZ * STBL_NM];
    double t_MS_end[STBL_NZ * STBL_NM];
    double t_RGB_start[STBL_NZ * STBL_NM];
    double t_CHeB_start[STBL_NZ * STBL_NM];
    double t_AGB_start[STBL_NZ * STBL_NM];
    double t_AGB_end[STBL_NZ * STBL_NM];
    double t_postAGB_start[STBL_NZ * STBL_NM];
    double t_WR_start[STBL_NZ * STBL_NM];

    /* 3D time-dependent datasets — dynamically allocated [NZ * NM * NAGE] */
    float *M_current;      /* Msun */
    float *v_wind;         /* km/s */
    float *log_Q_ion;      /* log10(photons/s) */
    float *log_L_FUV;      /* log10(erg/s), 6-11.2 eV photoelectric */
    float *log_L_LW;       /* log10(erg/s), 11.2-13.6 eV Lyman-Werner */
    float *log_L_FUV_total;/* log10(erg/s), 6-13.6 eV total FUV */
    float *log_L_bol;      /* log10(erg/s), bolometric */
    float *logR_cm;        /* log10(R [cm]), stellar radius */

    /* Ionizing sub-band luminosities for M1 RT [NZ * NM * NAGE] */
    float *log_L_ion_tot;  /* log10(erg/s), total ionizing (>13.6 eV) */
    float *log_L_ion_H0;   /* log10(erg/s), 13.6-24.6 eV */
    float *log_L_ion_He0;  /* log10(erg/s), 24.6-54.4 eV */
    float *log_L_ion_He1;  /* log10(erg/s), 54.4-70 eV */
    float *log_L_ion_He2;  /* log10(erg/s), >70 eV */

    /* Non-ionizing band luminosities for M1 RT [NZ * NM * NAGE] */
    float *log_L_NUV;      /* log10(erg/s), near-UV (~3.4-6 eV) */
    float *log_L_OPT_NIR;  /* log10(erg/s), optical+near-IR (~0.4-3.4 eV) */

    /* 4D surface abundances [NZ * NM * NAGE * NELEM] — only if winds enabled */
    float *surface_abundances;

    /* 3D yield datasets — flat [NZ * NM * NELEM] */
    double net_yields[STBL_NZ * STBL_NM * STBL_NELEM];
    double wind_yields[STBL_NZ * STBL_NM * STBL_NELEM];

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

/* ---- PMS duration / feedback delay (2D bilinear: logM, logZ) ---- */
double stellar_t_PMS(double logM, double logZ);

/* ---- Phase transition times (2D bilinear: logM, logZ), in years ---- */
double stellar_t_MS_start(double logM, double logZ);
double stellar_t_MS_end(double logM, double logZ);
double stellar_t_RGB_start(double logM, double logZ);
double stellar_t_CHeB_start(double logM, double logZ);
double stellar_t_AGB_start(double logM, double logZ);
double stellar_t_AGB_end(double logM, double logZ);
double stellar_t_postAGB_start(double logM, double logZ);
double stellar_t_WR_start(double logM, double logZ);
int    stellar_phase_at_age(double logM, double logZ, double age_yr);

/* ---- 3D trilinear interpolation (logM, logZ, log_age) ---- */
double stellar_M_current(double logM, double logZ, double log_age);
double stellar_v_wind(double logM, double logZ, double log_age);
double stellar_log_Q_ion(double logM, double logZ, double log_age);
double stellar_log_L_FUV(double logM, double logZ, double log_age);
double stellar_log_L_LW(double logM, double logZ, double log_age);
double stellar_log_L_FUV_total(double logM, double logZ, double log_age);
double stellar_log_L_bol(double logM, double logZ, double log_age);
double stellar_log_R_cm(double logM, double logZ, double log_age);
double stellar_log_L_ion_tot(double logM, double logZ, double log_age);
double stellar_log_L_ion_H0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He0(double logM, double logZ, double log_age);
double stellar_log_L_ion_He1(double logM, double logZ, double log_age);
double stellar_log_L_ion_He2(double logM, double logZ, double log_age);
double stellar_log_L_NUV(double logM, double logZ, double log_age);
double stellar_log_L_OPT_NIR(double logM, double logZ, double log_age);

/* ---- Surface abundances (3D + element index) ---- */
double stellar_surface_abundance(double logM, double logZ, double log_age, int elem);

/* ---- Yields (2D + element index) ---- */
double stellar_net_yield(double logM, double logZ, int elem);
double stellar_wind_yield(double logM, double logZ, int elem);
double stellar_sn_yield(double logM, double logZ, int elem); /* net_yield - wind_yield */
double stellar_type_ia_yield(int elem);

#endif /* GALSF_RESOLVEDISM_STELLAR_TABLES */
#endif /* RESOLVEDISM_STELLAR_TABLES_H */
