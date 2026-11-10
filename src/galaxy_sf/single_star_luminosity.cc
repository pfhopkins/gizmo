/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* Band luminosities of IMF-sampled single stars from the stellar-evolution tables at the star's current age, in the
   disjoint bands PE 8-13.6, LW 11.2-13.6 (inside PE), 6-8, NUV 3.444-6, OPT 0.4133-3.444 eV: as gravity-tree radiation
   sources (GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) and for the radiation-pressure budget at the star
   (GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS). Host only. */
#include <math.h>
#include <vector>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "single_star_record.h"
#include "stellar_evolution_tables.h"
#ifdef SGCHEM
#include "../cooling/sgchem/sgchem_dust_to_gas.h"
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)

static inline double single_star_band_lum_cgs(double log_l) {double l = pow(10., log_l); return (isfinite(l) && l > 0) ? l : 0;}

/* L[SINGLE_STAR_BAND_*] [erg/s] and the bolometric luminosity; returns 0 (all zero) before the main sequence, after death
   without a record or at or below IMFSampleStellarMassCut. 6-8 eV = L(6-13.6) - L(8-13.6), unclipped */
int single_star_band_luminosities_cgs(int i, double *L, double *L_bol)
{
    for(int b = 0; b < SINGLE_STAR_NBANDS; b++) {L[b] = 0;}
    *L_bol = 0;
    if(P[i].StarIdx <= 0) {return 0;}
    int s = single_star_slot(P[i].StarIdx);
    if(SingleStars.M_drawn[s] <= All.IMFSampleStellarMassCut) {return 0;}
#ifdef TEST_SINGLE_STARS_LUMINOSITY_PINNED /* test only: constant L from birth, FUV 0.5 (in the 8-13.6 eV band), LW 0.15, NUV 0.2, optical 0.3, no ionising photons */
    L[SINGLE_STAR_BAND_PE] = 0.5 * All.TestFixedLbol; L[SINGLE_STAR_BAND_LW] = 0.15 * All.TestFixedLbol; L[SINGLE_STAR_BAND_NUV] = 0.2 * All.TestFixedLbol;
    L[SINGLE_STAR_BAND_OPT] = 0.3 * All.TestFixedLbol; *L_bol = All.TestFixedLbol;
    return 1;
#endif
#ifdef TEST_SINGLE_STARS_RADIATION_OFF
    return 0; /* test only: no radiation */
#endif
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double age_yr = evaluate_stellar_age_Gyr(i) * 1.e9, t_pms = stellar_t_PMS(logM, logZ);
    if(age_yr >= t_pms + stellar_lifetime(logM, logZ) || age_yr - t_pms <= 0) {return 0;}
    double la = log10(DMAX(age_yr - t_pms, 100.));
    double L_uv = single_star_band_lum_cgs(stellar_log_L_FUV_total(logM, logZ, la)); /* 6-13.6 eV */
    L[SINGLE_STAR_BAND_PE] = (L_uv > 0) ? single_star_band_lum_cgs(stellar_log_L_FUV_M1(logM, logZ, la)) : 0;
    L[SINGLE_STAR_BAND_68] = L_uv - L[SINGLE_STAR_BAND_PE];
    L[SINGLE_STAR_BAND_LW] = single_star_band_lum_cgs(stellar_log_L_LW(logM, logZ, la));
    L[SINGLE_STAR_BAND_NUV] = DMAX(0., single_star_band_lum_cgs(stellar_log_L_NUV(logM, logZ, la)) - L[SINGLE_STAR_BAND_68]); /* 3.444-6 eV */
    L[SINGLE_STAR_BAND_OPT] = single_star_band_lum_cgs(stellar_log_L_OPT_NIR(logM, logZ, la));
    *L_bol = single_star_band_lum_cgs(stellar_log_L_bol(logM, logZ, la));
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
    L[SINGLE_STAR_BAND_ION_H0] = single_star_band_lum_cgs(stellar_log_L_ion_H0(logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE0] = single_star_band_lum_cgs(stellar_log_L_ion_He0(logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE1] = single_star_band_lum_cgs(stellar_log_L_ion_He1(logM, logZ, la));
    L[SINGLE_STAR_BAND_ION_HE2] = single_star_band_lum_cgs(stellar_log_L_ion_He2(logM, logZ, la));
    L[SINGLE_STAR_BAND_Q] = single_star_band_lum_cgs(stellar_log_Q_ion(logM, logZ, la));
#endif
    return 1;
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
/* ionising energy [erg/s] and photon rate [1/s] of a shining star from the tables at its current age; with helium also
   above 24.6 eV: photons as the difference of two log tables, kept inside the band where that difference is rounding */
static void single_star_ionizing_rates(int i, double *L_ion, double *Q, double *L_he, double *Q_he)
{
    int s = single_star_slot(P[i].StarIdx);
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double la = log10(DMAX(evaluate_stellar_age_Gyr(i) * 1.e9 - stellar_t_PMS(logM, logZ), 100.));
#if defined(TEST_SINGLE_STARS_LUMINOSITY_PINNED) || defined(TEST_SINGLE_STARS_RADIATION_OFF)
    *L_ion = *Q = *L_he = *Q_he = 0; (void) la; return; /* test only */
#endif
    *L_ion = single_star_band_lum_cgs(stellar_log_L_ion_tot(logM, logZ, la)); *Q = single_star_band_lum_cgs(stellar_log_Q_ion(logM, logZ, la));
    *L_he = *Q_he = 0;
#ifdef TREECOL_SHELLS_ION_HELIUM
    *L_he = single_star_band_lum_cgs(stellar_log_L_ion_He0(logM, logZ, la));
    *Q_he = DMIN(DMAX(*Q - single_star_band_lum_cgs(stellar_log_Q_ion_H0(logM, logZ, la)), *L_he / (54.4 * ELECTRONVOLT_IN_ERGS)), *L_he / (24.6 * ELECTRONVOLT_IN_ERGS));
#endif
#ifdef TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestTreeRayFixedRate > 0)
    {
#ifdef TREECOL_SHELLS_ION_HELIUM
        double f = All.TestTreeRayFixedRate / *Q; *L_ion *= f; *L_he *= f; *Q_he *= f; *Q = All.TestTreeRayFixedRate; /* the star's spectrum */
#else
        *Q = All.TestTreeRayFixedRate; *L_ion = *Q * 27.2 * ELECTRONVOLT_IN_ERGS;
#endif
    }
#endif
}
#endif

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_SINGLE_STARS_RT_TABLE_BANDS)
/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star (pp is P here) */
int single_star_rt_band_luminosities(int i, int mode, double *lum, struct particle_data *pp)
{
    double L[SINGLE_STAR_NBANDS], L_bol; (void) pp;
#ifdef RT_STARBENCH_TEST
    if(mode >= 0) {lum[RT_FREQ_BIN_H0] = 1e49 * All.rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS * (C_LIGHT_CODE_REDUCED / C_LIGHT_CODE);}
    return 1;
#endif
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
    if(mode < 0) {return 1;}
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = L[SINGLE_STAR_BAND_PE] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_LYMAN_WERNER] = L[SINGLE_STAR_BAND_LW] / UNIT_LUM_IN_CGS;
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
    lum[RT_FREQ_BIN_NUV] = DMAX(0., L[SINGLE_STAR_BAND_68]) / UNIT_LUM_IN_CGS; /* 6-8 eV */
#ifdef RT_OPTICAL_NIR
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
#endif
#else
    lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = L[SINGLE_STAR_BAND_68] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_NUV] = L[SINGLE_STAR_BAND_NUV] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
    double L_ion, Q, L_he, Q_he; single_star_ionizing_rates(i, &L_ion, &Q, &L_he, &Q_he);
    lum[RT_FREQ_BIN_ION_ENERGY] = (L_ion - L_he) / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_ION_PHOTONS] = (Q - Q_he) * 13.6 * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS;
#ifdef TREECOL_SHELLS_ION_HELIUM
    lum[RT_FREQ_BIN_ION_HE_ENERGY] = L_he / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_ION_HE_PHOTONS] = Q_he * 13.6 * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS;
#endif
#endif
#endif
#ifdef GALSF_SINGLE_STARS_RT_TABLE_BANDS
#ifdef RT_CHEM_PHOTOION
#ifdef RT_PHOTOION_MULTIFREQUENCY
    lum[RT_FREQ_BIN_H0] = L[SINGLE_STAR_BAND_ION_H0] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He0] = L[SINGLE_STAR_BAND_ION_HE0] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He1] = L[SINGLE_STAR_BAND_ION_HE1] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_He2] = L[SINGLE_STAR_BAND_ION_HE2] / UNIT_LUM_IN_CGS;
#else
    lum[RT_FREQ_BIN_H0] = L[SINGLE_STAR_BAND_Q] * All.rt_nu_eff_eV[RT_FREQ_BIN_H0] * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS; /* the table's photons at the bin's photon energy */
#endif
#endif
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {lum[k] *= (C_LIGHT_CODE_REDUCED / C_LIGHT_CODE);} /* as rt_get_source_luminosity */
#endif
    return 1;
}
#endif

#ifdef GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS
/* photon momentum a single star hands to the gas this step [g cm/s]: band luminosities absorbed
   over the kernel column at the star, f_b = 1 - exp(-sigma_b DGR N_H) with sigma = 2.0 (PE + 6-8), 1.3 (NUV), 0.5 (OPT)
   x 1e-21 cm^2, LW not counted (inside PE), capped at L_bol; plus the IR boost tau_IR = 5 DGR Sigma [cm^2/g] on the
   absorbed light. GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE adds 6-8 eV to the NUV term.
   Stars below SINGLE_STAR_RADPRESSURE_MIN_MASS push nothing */
#define SINGLE_STAR_RADPRESSURE_MIN_MASS 2.0 /* [Msun] drawn mass */
double single_star_radpressure_momentum_cgs(int i, double dt_cgs, double *tau_IR)
{
    double L[SINGLE_STAR_NBANDS], L_bol;
    *tau_IR = 0;
    if(P[i].StarIdx <= 0 || SingleStars.M_drawn[single_star_slot(P[i].StarIdx)] < SINGLE_STAR_RADPRESSURE_MIN_MASS) {return 0;}
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
#ifdef SGCHEM
    double dgr = DMAX(sgchem_dust_to_gas_ratio(P[i].Metallicity[0]), 0.);
#else
    double dgr = DMAX(DMIN(P[i].Metallicity[0] / All.SolarAbundances[0], 10.), 0.);
#endif
    double rho = P[i].DensityAroundParticle * All.cf_a3inv, h = P[i].KernelRadius * All.cf_atime;
    double N_H = rho * UNIT_DENSITY_IN_NHCGS * h * UNIT_LENGTH_IN_CGS, Sigma = rho * UNIT_DENSITY_IN_CGS * h * UNIT_LENGTH_IN_CGS;
    double f_uv = 1.0 - exp(-2.0e-21 * dgr * N_H), f_nuv = 1.0 - exp(-1.3e-21 * dgr * N_H), f_opt = 1.0 - exp(-0.5e-21 * dgr * N_H);
    double L_nuv = L[SINGLE_STAR_BAND_NUV];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    L_nuv += L[SINGLE_STAR_BAND_68];
#endif
    double L_abs = DMIN(f_uv * (L[SINGLE_STAR_BAND_PE] + L[SINGLE_STAR_BAND_68]) + f_nuv * L_nuv + f_opt * L[SINGLE_STAR_BAND_OPT], L_bol);
    *tau_IR = 5.0 * dgr * Sigma;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    return L_abs * (*tau_IR) * dt_cgs / C_LIGHT_CGS; /* direct term: per cell, single_star_radpressure_per_cell */
#else
    return L_abs * (1.0 + *tau_IR) * dt_cgs / C_LIGHT_CGS;
#endif
}
#endif

#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
/* direct radiation pressure in the absorbing cell: L_abs = sum_b sigma_b F_b DGR N_H(cell) / 4 pi, pushed away from the pixel
   centres, all deposits scaled by L_emit / sum(L_abs) when above 1 (photon cap). The sum over cells is kept running:
   an active cell swaps its share, full recount on all-active steps. L_emit is the root node, so no star loop. One Allreduce per call. */
void single_star_radpressure_per_cell(double *dp_total_code)
{
    static double abs_sum_local = 0; static int counted = 0;
    *dp_total_code = 0;
    const double to_L = 1.0e-21 * All.cf_a2inv / (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS * 4.0 * M_PI); /* sum x DGR N_H -> code lum */
    const int all_active = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin);
    if(!counted || all_active)
    {
        abs_sum_local = 0; counted = 1;
        for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {abs_sum_local += CellP[i].RadpAbsorbedLum;}}
    }
    std::vector<double> lum_abs(ActiveParticleList.size(), 0.);
    for(size_t a = 0; a < ActiveParticleList.size(); a++)
    {
        const int i = ActiveParticleList[a];
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        const double dgr = (CellP[i].Temp > 1.0e5) ? 0 : DMAX(sgchem_dust_to_gas_ratio(P[i].Metallicity[0]), 0.);
        lum_abs[a] = to_L * dgr * P[i].Mass * UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
        const double l = lum_abs[a] * CellP[i].TreeColRadpFlux[3];
        abs_sum_local += l - CellP[i].RadpAbsorbedLum;
        CellP[i].RadpAbsorbedLum = (float) l;
    }
    const MyFloat *root = Nodes[All.TreeNodeIndexBase].stellar_lum;
    double L_emit = root[RT_FREQ_BIN_PHOTOELECTRIC] + root[RT_FREQ_BIN_GENERIC_USER_FREQ] + root[RT_FREQ_BIN_NUV] + root[RT_FREQ_BIN_OPTICAL_NIR];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    L_emit += root[RT_FREQ_BIN_LYMAN_WERNER] + root[RT_FREQ_BIN_GENERIC_USER_FREQ];
#endif
    double abs_sum = 0;
    MPI_Allreduce(&abs_sum_local, &abs_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double fcons = (abs_sum > L_emit && abs_sum > 0) ? L_emit / abs_sum : 1.0;
    const double to_p = fcons * UNIT_LUM_IN_CGS * UNIT_TIME_IN_CGS / (C_LIGHT_CGS * UNIT_MASS_IN_CGS * All.UnitVelocity_in_cm_per_s); /* code lum x code time -> code momentum */
    for(size_t a = 0; a < ActiveParticleList.size(); a++)
    {
        const int i = ActiveParticleList[a];
        const double dt = get_particle_timestep_in_physical(i, P);
        if(!(lum_abs[a] > 0) || !(dt > 0)) {continue;}
        const double dp = -lum_abs[a] * to_p * dt;
        double dpmag2 = 0;
        for(int k = 0; k < 3; k++)
        {
            const double dp_k = dp * CellP[i].TreeColRadpFlux[k], dv = dp_k * All.cf_atime / P[i].Mass;
            P[i].Vel[k] += dv; CellP[i].VelPred[k] += dv; P[i].dp[k] += dv * P[i].Mass;
            dpmag2 += dp_k * dp_k;
        }
        *dp_total_code += sqrt(dpmag2);
    }
}
#endif

#if defined(RT_USE_GRAVTREE) && defined(GALSF_SINGLE_STARS_FROM_IMF) && defined(GALSF_SINGLE_STARS_EVOLUTION_TABLES)
#define SINGLE_STAR_TREE_LUM_TOL 1.e-2 /* a star's band luminosity may drift this fraction of the root's before the nodes are refreshed */
/* before a gravity call: refresh the tree node moments (luminosities) if a single star started to shine, started to ionise or went
   dark since the last call, or its band luminosity moved by more than SINGLE_STAR_TREE_LUM_TOL of the root's since the nodes last
   saw it; reduced in gravity_tree, so no collective here */
void single_star_light_switch_check(void)
{
    static double t_last = -1; double t0 = t_last; t_last = All.Time;
    if(t0 < 0 || All.Time <= t0) {return;}
    if(!force_tree_is_allocated() || TreeReconstructFlag) {return;} /* a build is due: it sees every star as it is now */
    double dt_yr = evaluate_time_since_t_initial_in_Gyr(t0) * 1.e9;
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0 || P[i].Mass <= 0) {continue;}
        int s = single_star_slot(P[i].StarIdx);
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {if(fabs(lum[b] - SingleStars.tree_lum[s].L[b]) > SINGLE_STAR_TREE_LUM_TOL * Nodes[All.TreeNodeIndexBase].stellar_lum[b]) {TreeMomentsStaleFlag = 1; return;}}
        if(SingleStars.M_drawn[s] <= All.IMFSampleStellarMassCut) {continue;}
#if defined(GALSF_FB_MECHANICAL)
        if(SingleStars.event_bits[s] & (SINGLE_STAR_EVENT_SN_DONE | SINGLE_STAR_EVENT_AGB_DONE)) {continue;}
#endif
        double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
        double age = evaluate_stellar_age_Gyr(i) * 1.e9, t_pms = stellar_t_PMS(logM, logZ), t_on = single_star_ionizing_onset_yr(i);
        double t_switch[3] = {t_pms, t_pms + stellar_lifetime(logM, logZ), t_on};
        for(int k = 0; k < 3; k++) {if(t_switch[k] > 0 && age - dt_yr < t_switch[k] && t_switch[k] <= age) {TreeMomentsStaleFlag = 1; return;}}
    }
}

/* after a gravity call: if it rebuilt or refreshed the node luminosities (the root changed), they are the stars' current ones */
void single_star_tree_lum_store(const double *root_before)
{
    static int first = 1; int changed = first; first = 0;
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {if(Nodes[All.TreeNodeIndexBase].stellar_lum[b] != root_before[b]) {changed = 1;}}
    if(!changed) {return;}
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {SingleStars.tree_lum[single_star_slot(P[i].StarIdx)].L[b] = (float) lum[b];}
    }
}
#endif
