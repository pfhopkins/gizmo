/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
#ifndef SINGLE_STAR_CHECKS_H
#define SINGLE_STAR_CHECKS_H
/* TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS: run-time invariant checks of the single-star model, host code only.
   Log lines: "SINGLE_STAR_CHECK <tag> FAIL Task=<rank> ..." when one breaks, "SINGLE_STAR_CHECK <tag> RAN Task=<rank> n=<count>"
   at the first and every 1000th call of a site (count = cases checked so far). Nothing is compiled without the flag. */
#ifdef TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS
#include <stdio.h>
#include <atomic>
#include <float.h>
#include <math.h>
#define SINGLE_STAR_CHECK(tag, ok, fmt, ...) do { \
    if(!(ok)) {printf("SINGLE_STAR_CHECK %s FAIL Task=%d " fmt "\n", tag, ThisTask, ##__VA_ARGS__); fflush(stdout);} \
} while(0)
#define SINGLE_STAR_CHECK_RAN(tag, n) do { \
    static std::atomic<long> ssc_calls_{0}, ssc_n_{0}; long ssc_k_ = (long)(n); \
    if(ssc_k_ > 0) {long ssc_s_ = (ssc_n_ += ssc_k_), ssc_c_ = ++ssc_calls_; \
        if(ssc_c_ == 1 || ssc_c_ % 1000 == 0) {printf("SINGLE_STAR_CHECK %s RAN Task=%d n=%ld\n", tag, ThisTask, ssc_s_); fflush(stdout);}} \
} while(0)
#else
#define SINGLE_STAR_CHECK(tag, ok, fmt, ...) do {} while(0)
#define SINGLE_STAR_CHECK_RAN(tag, n) do {} while(0)
#endif

#if defined(TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS) && defined(METALS)
/* sum of the C..Fe slots 2-10 does not exceed total Z (slot 0) */
static inline int single_star_check_slots_below_z(const MyFloat *Z)
{
    if(NUM_LIVE_SPECIES_FOR_COOLTABLES < 10) {return 1;}
    double sub = 0; for(int k = 2; k <= 10; k++) {sub += Z[k];}
    return sub <= Z[0];
}
/* a cell after a feedback deposit: each slot gained exactly what was sent, the C..Fe sum stays below Z if it was */
static inline void single_star_check_receiver(int j, double m0, double mf, const double *Z_before, const double *Z_injected)
{
    for(int k = 0; k < NUM_METAL_SPECIES; k++) {
        double X = P[j].Metallicity[k], dm = mf * X - m0 * Z_before[k], want = Z_injected[k];
        double tol = 8 * DBL_EPSILON * (fabs(m0 * Z_before[k]) + fabs(want) + fabs(mf * X));
        SINGLE_STAR_CHECK("fire_slot_deposit", fabs(dm - want) <= tol, "cell=%llu slot=%d dm=%.17g want=%.17g tol=%.3g", (unsigned long long)P[j].ID, k, dm, want, tol);
    }
    MyFloat Zb[NUM_METAL_SPECIES]; for(int k = 0; k < NUM_METAL_SPECIES; k++) {Zb[k] = Z_before[k];}
    SINGLE_STAR_CHECK("fire_slot_invariant_cell", !single_star_check_slots_below_z(Zb) || single_star_check_slots_below_z(P[j].Metallicity), "cell=%llu Z=%.17g", (unsigned long long)P[j].ID, (double)P[j].Metallicity[0]);
    SINGLE_STAR_CHECK_RAN("fire_slot_deposit", 1);
}
/* after the feedback: no gas slot negative, C..Fe below Z, CellP.Mass equal to P.Mass */
static inline void single_star_check_gas_after_feedback(void)
{
    int nbad = 0, nmass = 0, ngas = 0; long long first = -1;
    for(int j = 0; j < N_gas; j++) {
        if(P[j].Type != 0 || P[j].Mass <= 0) {continue;}
        int neg = 0; ngas++;
        for(int k = 0; k < NUM_METAL_SPECIES; k++) {if(P[j].Metallicity[k] < 0) {neg = 1;}}
        if(neg || !single_star_check_slots_below_z(P[j].Metallicity)) {nbad++; if(first < 0) {first = (long long)P[j].ID;}}
        if(fabs((double)P[j].Mass - (double)CellP[j].Mass) > 1.0e-10 * fabs((double)P[j].Mass) && ++nmass <= 20) {
            SINGLE_STAR_CHECK("mass_sync", 0, "ID=%llu P.Mass=%.10e CellP.Mass=%.10e", (unsigned long long)P[j].ID, (double)P[j].Mass, (double)CellP[j].Mass);}
    }
    SINGLE_STAR_CHECK("fire_slot_invariant", nbad == 0, "t=%.10g %d of %d cells, first ID=%lld", All.Time, nbad, ngas, first);
    SINGLE_STAR_CHECK_RAN("fire_slot_invariant", ngas);
    SINGLE_STAR_CHECK_RAN("mass_sync", ngas);
}
#endif
#endif
