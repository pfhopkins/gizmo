/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Solar composition of the stellar-evolution tables (Asplund+09 present day, Z = 0.014), elements H..Zn. */
#ifndef STELLAR_SOLAR_ABUNDANCES_H
#define STELLAR_SOLAR_ABUNDANCES_H

#define STELLAR_SOLAR_Z 0.014 /* total metals C..Zn */
#define STELLAR_SOLAR_NELEM 27

/* mass fractions, H..Zn (the tables' initial composition at Z = 0.014) */
static const double stellar_solar_massfrac[STELLAR_SOLAR_NELEM] = {
    7.3750000000e-01, 2.4850000000e-01, 2.4765351227e-03, 7.2545495820e-04, 6.0028592850e-03, 5.2841265250e-07, /* H He C N O F */
    1.3157897125e-03, 3.0605839462e-05, 7.4124358876e-04, 5.8255839166e-05, 6.9620919184e-04, 6.0990559675e-06, /* Ne Na Mg Al Si P */
    3.2376421296e-04, 8.5877929305e-06, 7.6870616805e-05, 3.2093705781e-06, 6.7169384016e-05, 4.8646655463e-08, /* S Cl Ar K Ca Sc */
    3.2681466346e-06, 3.3215539907e-07, 1.7387424601e-05, 1.1327606908e-05, 1.3528499188e-03, 4.4118818564e-06, /* Ti V Cr Mn Fe Co */
    7.4619379442e-05, 7.5396909229e-07, 1.8184871682e-06};                                                     /* Ni Cu Zn */

/* element (H = 0 .. Zn = 26) of each FIRE metal slot 1..10; slot 0 is total Z */
static const int stellar_fire_slot_elem[11] = {-1, 1, 2, 3, 4, 6, 8, 10, 12, 16, 22};

/* the 16 elements outside the FIRE slots (F Na Al P Cl Ar K Sc Ti V Cr Mn Co Ni Cu Zn), in extra-slot order */
#define STELLAR_EXTRA_NELEM 16
static const int stellar_extra_slot_elem[STELLAR_EXTRA_NELEM] = {5, 7, 9, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26};

/* SolarAbundances in the FIRE layout from the tables' solar composition */
static inline void stellar_solar_abundances_fire_slots(double *solar, int n_cooling_species)
{
    solar[0] = STELLAR_SOLAR_Z;
    if(n_cooling_species >= 10) {for(int k = 1; k <= 10; k++) {solar[k] = stellar_solar_massfrac[stellar_fire_slot_elem[k]];}}
}

#ifdef GALSF_SINGLE_STARS_ELEMENT_ABUNDANCE_BLOCK
/* the 27 element mass fractions (H..Zn) of a particle's metal slots, H = 1 minus the rest; and back (H is implicit, slot 0 = metals C..Zn) */
static inline void stellar_element_abundance_from_slots(const MyFloat *Z, double *x)
{
    double sum = 0;
    for(int k = 0; k < STELLAR_SOLAR_NELEM; k++) {x[k] = 0;}
    for(int s = 1; s <= 10; s++) {x[stellar_fire_slot_elem[s]] = Z[s];}
    for(int e = 0; e < STELLAR_EXTRA_NELEM; e++) {x[stellar_extra_slot_elem[e]] = Z[METALS_EXTRA_PASSIVE_SPECIES_OFFSET + e];}
    for(int k = 1; k < STELLAR_SOLAR_NELEM; k++) {sum += x[k];}
    x[0] = 1.0 - sum;
}
static inline void stellar_element_abundance_to_slots(const double *x, MyFloat *Z)
{
    double sum = 0;
    for(int k = 2; k < STELLAR_SOLAR_NELEM; k++) {sum += x[k];}
    Z[0] = sum;
    for(int s = 1; s <= 10; s++) {Z[s] = x[stellar_fire_slot_elem[s]];}
    for(int e = 0; e < STELLAR_EXTRA_NELEM; e++) {Z[METALS_EXTRA_PASSIVE_SPECIES_OFFSET + e] = x[stellar_extra_slot_elem[e]];}
}
#endif

#endif
