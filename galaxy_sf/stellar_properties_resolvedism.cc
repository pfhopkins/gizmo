#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) || defined(GALSF_RESOLVEDISM_STOCHASTIC_IMF) || defined(GALSF_RESOLVEDISM_G0_VARIABLE)

/* Stellar lifetime as a function of mass [yr] */
double get_lifetime(double mass) {
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    /* Use tabulated Z-dependent lifetime; assume solar Z if no birth metallicity available */
    double logM = log10(DMAX(mass, 0.09));
    double logZ = log10(0.014); /* solar default; callers with metallicity should use stellar_lifetime() directly */
    return stellar_lifetime(logM, logZ);
#else
    double A, B;
    if(mass < 3.0) {
        A = -2.926; B = 9.892;
    } else if(mass < 7.0) {
        A = -2.405; B = 9.641;
    } else if(mass < 15.0) {
        A = -1.765; B = 9.105;
    } else {
        A = -0.808; B = 7.954;
    }
    double logAge = A * log10(mass) + B;
    return pow(10., logAge); /* [yr] */
#endif
}


/* Lookup table: stellar mass [Msun] -> PE/UV/ionizing luminosities */
static double tbl_Mass[21] = {
    0.8, 0.9, 1., 1.1, 1.25, 1.35, 1.5, 1.7, 2., 2.5, 3., 4.,
    5., 6.999998, 8.999978, 11.999839, 14.999568, 19.999581, 24.999077, 31.998098, 39.996651
};

/* Photoelectric heating luminosity log10(L_pe) [erg/s] at solar metallicity */
static double tbl_logL_pe[21] = {
    27.96209322, 28.85707938, 29.59881618, 30.21651755,
    30.82357317, 31.32654796, 32.12136499, 33.04425441,
    33.99946029, 34.819052,   35.22220551, 35.84163201,
    36.319291,   36.98056909, 37.34013383, 37.68326336,
    37.93895345, 38.17665699, 38.40573288, 38.67544519,
    38.84857662
};

/* Lyman continuum photon rate log10(S_ly) [photons/s] at solar metallicity */
static double tbl_logS_ly[21] = {
    24.41647,  25.683735, 26.734005, 27.608646,
    28.468212, 29.180407, 30.305836, 32.479298,
    35.61226,  38.042862, 39.484432, 41.623093,
    43.022007, 44.548347, 45.449894, 46.48047,
    47.296062, 48.106155, 48.593216, 49.006676,
    49.291897
};


/* Binary search for table index */
int get_index(double search) {
    int n = 21, first = 0, last = n, middle;
    while(first < last) {
        middle = first + (last - first) / 2;
        if(tbl_Mass[middle] == search) {return middle;}
        else if(tbl_Mass[middle] < search) {first = middle + 1;}
        else {last = middle;}
    }
    return last - 1;
}


/* Interpolate log10(L_pe) from mass table */
double get_logL_pe(double mass) {
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    /* Use tabulated FUV luminosity (6-13.6 eV, photoelectric); assume ZAMS age */
    double logM = log10(DMAX(mass, 0.09));
    double logZ = log10(0.014);
    return stellar_log_L_FUV_total(logM, logZ, 2.0); /* log_age=2.0 ~ 100 yr ~ ZAMS */
#else
    double a, y;
    int idx = get_index(mass);
    if(mass < tbl_Mass[0]) {y = tbl_logL_pe[0];}
    else if(mass > tbl_Mass[20]) {y = tbl_logL_pe[20];}
    else {
        a = (mass - tbl_Mass[idx]) / (tbl_Mass[idx+1] - tbl_Mass[idx]);
        y = (1.0 - a) * tbl_logL_pe[idx] + a * tbl_logL_pe[idx+1];
    }
    return y;
#endif
}

/* Interpolate log10(S_ly) from mass table */
double get_logS_ly(double mass) {
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
    /* Use tabulated ionizing photon rate */
    double logM = log10(DMAX(mass, 0.09));
    double logZ = log10(0.014);
    return stellar_log_Q_ion(logM, logZ, 2.0); /* log_age=2.0 ~ ZAMS */
#else
    double a, y;
    int idx = get_index(mass);
    if(mass < tbl_Mass[0]) {y = tbl_logS_ly[0];}
    else if(mass > tbl_Mass[20]) {y = tbl_logS_ly[20];}
    else {
        a = (mass - tbl_Mass[idx]) / (tbl_Mass[idx+1] - tbl_Mass[idx]);
        y = (1.0 - a) * tbl_logS_ly[idx] + a * tbl_logS_ly[idx+1];
    }
    return y;
#endif
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF || GALSF_RESOLVEDISM_STOCHASTIC_IMF || GALSF_RESOLVEDISM_G0_VARIABLE */
