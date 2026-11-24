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
#include "single_star_checks.h"
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
    double age_yr = single_star_age_yr(i), t_pms = stellar_t_PMS(logM, logZ);
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
#if defined(TEST_SINGLE_STARS_PRINT_INVARIANT_CHECKS) && defined(GALSF_FB_MECHANICAL)
    SINGLE_STAR_CHECK("lifetime_consistent", !(SingleStars.event_bits[s] & (SINGLE_STAR_EVENT_SN_DONE | SINGLE_STAR_EVENT_AGB_DONE)), "ID=%llu M=%g radiates after its death", (unsigned long long)P[i].ID, SingleStars.M_drawn[s]);
    SINGLE_STAR_CHECK_RAN("lifetime_consistent", 1);
#endif
    return 1;
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
/* ionising energy L_b [erg/s] and photon rate Q_b [1/s] per band of a shining star from the tables at its current age. With
   helium the bands start at 13.6, 24.6, 54.4 and 70 eV: energies from the table's bands, photons above 54.4 eV at the mean
   energies of a 4e4 K blackbody, photons in 24.6-54.4 eV as the difference of two log tables, kept inside the band where that
   difference is rounding; energies and photons add up to the star's */
static void single_star_ionizing_rates(int i, double *L_b, double *Q_b)
{
    int s = single_star_slot(P[i].StarIdx);
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double la = log10(DMAX(single_star_age_yr(i) - stellar_t_PMS(logM, logZ), 100.));
#if defined(TEST_SINGLE_STARS_LUMINOSITY_PINNED) || defined(TEST_SINGLE_STARS_RADIATION_OFF)
    for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {L_b[b] = Q_b[b] = 0;} (void) la; return; /* test only */
#endif
    double L = single_star_band_lum_cgs(stellar_log_L_ion_tot(logM, logZ, la)), Q = single_star_band_lum_cgs(stellar_log_Q_ion(logM, logZ, la));
#ifdef TREECOL_SHELLS_ION_HELIUM
    L_b[1] = single_star_band_lum_cgs(stellar_log_L_ion_He0(logM, logZ, la));
    L_b[2] = single_star_band_lum_cgs(stellar_log_L_ion_He1(logM, logZ, la)); Q_b[2] = L_b[2] / (58.006 * ELECTRONVOLT_IN_ERGS);
    L_b[3] = single_star_band_lum_cgs(stellar_log_L_ion_He2(logM, logZ, la)); Q_b[3] = L_b[3] / (73.785 * ELECTRONVOLT_IN_ERGS);
    Q_b[1] = DMIN(DMAX(Q - single_star_band_lum_cgs(stellar_log_Q_ion_H0(logM, logZ, la)) - Q_b[2] - Q_b[3], L_b[1] / (54.4 * ELECTRONVOLT_IN_ERGS)), L_b[1] / (24.6 * ELECTRONVOLT_IN_ERGS));
    L_b[0] = DMAX(L - L_b[1] - L_b[2] - L_b[3], 0); Q_b[0] = DMAX(Q - Q_b[1] - Q_b[2] - Q_b[3], 0);
#else
    L_b[0] = L; Q_b[0] = Q;
#endif
#ifdef TEST_TREECOL_SHELLS_PHOTOION_FIXED_RATE_TWO_TEMPERATURE_GAS
    if(All.TestTreeRayFixedRate > 0)
    {
#ifdef TREECOL_SHELLS_ION_HELIUM
        for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {L_b[b] *= All.TestTreeRayFixedRate / Q; Q_b[b] *= All.TestTreeRayFixedRate / Q;} /* the star's spectrum */
#else
        Q_b[0] = All.TestTreeRayFixedRate; L_b[0] = Q_b[0] * 27.2 * ELECTRONVOLT_IN_ERGS;
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
    double L_ion[TREECOL_SHELLS_ION_NBANDS], Q_ion[TREECOL_SHELLS_ION_NBANDS]; single_star_ionizing_rates(i, L_ion, Q_ion);
    for(int b = 0; b < TREECOL_SHELLS_ION_NBANDS; b++) {
        lum[RT_FREQ_BIN_ION_ENERGY + 2 * b] = L_ion[b] / UNIT_LUM_IN_CGS;
        lum[RT_FREQ_BIN_ION_PHOTONS + 2 * b] = Q_ion[b] * 13.6 * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS;
    }
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

#ifdef SINGLE_STAR_ION_PHOTON_ENERGY_FROM_TABLES
/* mean photon energy of the one ionising bin = sum L_ion / sum Q over the living stars above 8 Msun, from the tables at
   their current ages: the stars' hardness instead of a fixed blackbody's. Emission and the cells' rates and heating read
   the same number, so the photon count is kept. Each star's share is renewed on its own active steps and kept in its
   record; the rank sums are not tied to the particles, so stars may move between ranks, and a step on which every
   particle is active counts anew (stars that merged or left are dropped then) */
static double ion_light_sum[2];
static int ion_light_sum_built = 0;

static void single_star_ion_light(int i, double *L, double *Q)
{
    double b[SINGLE_STAR_NBANDS], L_bol; *L = *Q = 0;
    int s = single_star_slot(P[i].StarIdx);
    if(SingleStars.M_drawn[s] <= 8. || !single_star_band_luminosities_cgs(i, b, &L_bol)) {return;}
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double l = single_star_band_lum_cgs(stellar_log_L_ion_tot(logM, logZ, log10(DMAX(evaluate_stellar_age_Gyr(i) * 1.e9 - stellar_t_PMS(logM, logZ), 100.))));
    if(l > 0 && b[SINGLE_STAR_BAND_Q] > 0) {*L = l; *Q = b[SINGLE_STAR_BAND_Q];}
}

/* the active stars' shares into the rank sums; 1 on the first call of a run: the bin's energy is not set yet */
int single_star_ion_light_count_active(void)
{
    int first = !ion_light_sum_built;
    if(first)
    {
        ion_light_sum[0] = ion_light_sum[1] = 0; ion_light_sum_built = 1; /* restart files carry the shares, other starts have all particles active */
        for(int s = 0; s < SingleStars.capacity; s++) {if(SingleStars.used[s]) {ion_light_sum[0] += SingleStars.L_ion_counted[s]; ion_light_sum[1] += SingleStars.Q_ion_counted[s];}}
    }
    if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin)
    {
        ion_light_sum[0] = ion_light_sum[1] = 0;
        for(int s = 0; s < SingleStars.capacity; s++) {if(SingleStars.used[s]) {SingleStars.L_ion_counted[s] = SingleStars.Q_ion_counted[s] = 0;}}
    }
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        int s = single_star_slot(P[i].StarIdx); double L, Q;
        single_star_ion_light(i, &L, &Q);
        ion_light_sum[0] += L - SingleStars.L_ion_counted[s]; ion_light_sum[1] += Q - SingleStars.Q_ion_counted[s];
        SingleStars.L_ion_counted[s] = L; SingleStars.Q_ion_counted[s] = Q;
    }
    return first;
}

/* collective: one reduction; without living ionising stars the energy stays */
void single_star_ion_photon_energy_update(void)
{
    double sum[2]; MPI_Allreduce(ion_light_sum, sum, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(!(sum[0] > 0 && sum[1] > 0)) {return;}
    double e = DMIN(DMAX(sum[0] / sum[1] / ELECTRONVOLT_IN_ERGS, 13.6), 100.);
    All.rt_nu_eff_eV[RT_FREQ_BIN_H0] = e;
    All.rt_ion_G_HI[RT_FREQ_BIN_H0] = (e - 13.6) * ELECTRONVOLT_IN_ERGS / UNIT_ENERGY_IN_CGS;
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
#ifdef SINGLE_STAR_RADPRESSURE_LIVE_DUST
#ifndef SGCHEM_LIVE_DUST_MILKY_WAY
#define SGCHEM_LIVE_DUST_MILKY_WAY 0.01 /* Milky Way dust mass fraction, without SGCHEM (sgchem_dust_to_gas.h) */
#endif
    double dgr = DMAX(DMIN(SingleStars.dust_ambient[single_star_slot(P[i].StarIdx)] / SGCHEM_LIVE_DUST_MILKY_WAY, 10.), 0.); /* kernel mean of the dust around the star */
#elif defined(SGCHEM)
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
   centres, all deposits scaled by L_emit / sum(L_abs) when above 1 (photon cap). The sum runs over every gas cell
   with its last flux, current mass and dust (as stella_dev). L_emit is the root node, so no star loop. The sum is reduced
   with the source count in radiation_pressure_winds_consolidated: no collective here. */
static std::vector<double> radp_lum_abs; /* per active cell, absorbed_local to per_cell */
static double radp_cell_dgr(int i) /* the cell's dust-to-gas ratio for the direct term; hot gas none */
{
#ifdef SINGLE_STAR_RADPRESSURE_LIVE_DUST
    return (CellP[i].Temp > 1.0e5) ? 0 : DMAX(sgchem_cell_dust_to_gas_ratio(P, CellP, i), 0.); /* the cell's live dust */
#else
    return (CellP[i].Temp > 1.0e5) ? 0 : DMAX(sgchem_dust_to_gas_ratio(P[i].Metallicity[0]), 0.);
#endif
}
double single_star_radpressure_absorbed_local(void)
{
    const double to_L = 1.0e-21 * All.cf_a2inv / (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS * 4.0 * M_PI); /* sum x DGR N_H -> code lum */
    double abs_sum_local = 0; /* every gas cell now: current mass and dust, last flux (as stella_dev) */
    for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {abs_sum_local += to_L * radp_cell_dgr(i) * P[i].Mass * UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS * CellP[i].TreeColRadpFlux[3];}}
    std::vector<double> &lum_abs = radp_lum_abs; lum_abs.assign(ActiveParticleList.size(), 0.);
    for(size_t a = 0; a < ActiveParticleList.size(); a++)
    {
        const int i = ActiveParticleList[a];
        if(P[i].Type != 0 || P[i].Mass <= 0) {continue;}
        lum_abs[a] = to_L * radp_cell_dgr(i) * P[i].Mass * UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
    }
#ifdef TEST_SINGLE_STARS_RADPRESSURE_PRINT_CAP_SUM
    { /* reference: every gas cell now, current mass and dust, last flux (stella_dev resolvedism_fb_momentum.cc photon cap) */
        double rc = 0; const int all_active = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin);
        for(int i = 0; i < NumPart; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {rc += to_L * radp_cell_dgr(i) * P[i].Mass * UNIT_MASS_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS * CellP[i].TreeColRadpFlux[3];}}
        printf("RADP_CAP_SUM: rank=%d t=%.10g all_active=%d n_active=%d used=%.17g recount=%.17g\n", ThisTask, All.Time, all_active, (int) ActiveParticleList.size(), abs_sum_local, rc);
    }
#endif
    return abs_sum_local;
}

void single_star_radpressure_per_cell(double abs_sum, double *dp_total_code)
{
    *dp_total_code = 0;
    const std::vector<double> &lum_abs = radp_lum_abs;
    const MyFloat *root = Nodes[All.TreeNodeIndexBase].stellar_lum;
    double L_emit = root[RT_FREQ_BIN_PHOTOELECTRIC] + root[RT_FREQ_BIN_GENERIC_USER_FREQ] + root[RT_FREQ_BIN_NUV] + root[RT_FREQ_BIN_OPTICAL_NIR];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    L_emit += root[RT_FREQ_BIN_LYMAN_WERNER] + root[RT_FREQ_BIN_GENERIC_USER_FREQ];
#endif
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
#if defined(RESOLVED_ISM_FB_AFTER_CLOSING_KICK) && defined(RESOLVED_ISM_RECOMPUTE_HYDRO_POST_FB)
        if(dpmag2 > 0) {FBHydroRedoReceivers++;} /* a cell the tree term pushed repeats the hydro pass */
#endif
    }
}
#endif

#ifdef GALSF_SINGLE_STARS_LIGHT_UPDATES_NODE_COPIES /* opt-in (Uli 2026-11-19, GKS-95 option c); default: bounded staleness */
#include "../gravity/forcetree.h"
#include "../gravity/gpu_pseudo_update.h"
#include "../gravity/gravtree_moment_kernel.h"   /* moment_light_centre_weight */
int let_light_wire_of(int R, int key);
int let_light_foreign_node(int sender, int wire);

/* one light change for one node copy: kind 0 = a top-level node (the same index on every rank; applied up its chain to the
   root), kind 1 = the receiver's imported copy of one of the sender's nodes or star leaves (wire index in the sender's import) */
struct single_star_light_change {int target, kind; double dl[N_RT_FREQ_BINS], x[3], v[3];};
static std::vector<int> light_pending; /* local stars whose light the nodes do not hold yet */
static double light_stats[4];          /* exchanges, changed stars, bytes sent, seconds (this rank, cumulative) */
static double light_scan_seconds;      /* the scans before every gravity call (this rank, cumulative) */

/* after a whole gravity-tree build: the nodes (and every rank's imported copies) hold each star's light as of now */
void single_star_tree_lum_set_from_build(void)
{
    light_pending.clear();
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {SingleStars.tree_lum[single_star_slot(P[i].StarIdx)].L[b] = (float) lum[b];}
    }
}

/* local stars whose band light differs from what the nodes hold by more than tol of their own in some band, or switched a band
   on or off (tol 0: any difference); returns their number. No collective. */
int single_star_light_exchange_scan(double tol)
{
    light_pending.clear();
    if(!force_tree_is_allocated()) {return 0;}
    const double t0 = MPI_Wtime();
    for(int i = 0; i < NumPart && i < All.TreeParticleSlots; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        int s = single_star_slot(P[i].StarIdx);
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const double l0 = SingleStars.tree_lum[s].L[b], lb = (float) lum[b];
            if(((l0 == 0) != (lb == 0)) || fabs(lb - l0) > tol * DMAX(fabs(lb), fabs(l0))) {light_pending.push_back(i); break;}
        }
    }
    light_scan_seconds += MPI_Wtime() - t0;
    return (int) light_pending.size();
}

/* add one change to node `no` (local, top-level or imported copy) at the current time: band luminosities, and the
   luminosity-weighted centre and velocity re-weighted with the star's position and velocity; the walk's mirror follows */
static void single_star_light_apply(int no, const struct single_star_light_change *c)
{
    if(Nodes[no].Ti_current != All.Ti_Current) {force_drift_node(no, All.Ti_Current);}
    const double L0 = moment_light_centre_weight(&Nodes[no].stellar_lum[0]);
    for(int b = 0; b < N_RT_FREQ_BINS; b++) {
        double l = (double) Nodes[no].stellar_lum[b] + c->dl[b];
        Nodes[no].stellar_lum[b] = (MyFloat) ((l > 0) ? l : 0);
    }
    const double L1 = moment_light_centre_weight(&Nodes[no].stellar_lum[0]);
#ifdef RT_SEPARATELY_TRACK_LUMPOS
    if(L1 > 0) {
        const double h = Nodes[no].len; /* guard against cancellation when the dominant source goes dark: stay near the node */
        for(int k = 0; k < 3; k++) {
            double sk = (L0 * (double) Nodes[no].rt_source_lum_s[k] + (L1 - L0) * c->x[k]) / L1;
            sk = DMIN(DMAX(sk, (double) Nodes[no].center[k] - h), (double) Nodes[no].center[k] + h);
            Nodes[no].rt_source_lum_s[k] = (MyFloat) sk;
            Extnodes[no].rt_source_lum_vs[k] = (MyFloat) ((L0 * (double) Extnodes[no].rt_source_lum_vs[k] + (L1 - L0) * c->v[k]) / L1);
        }
    } else {
        for(int k = 0; k < 3; k++) {Nodes[no].rt_source_lum_s[k] = Nodes[no].center[k]; Extnodes[no].rt_source_lum_vs[k] = 0;}
    }
#endif
    gpu_mirror_put_node_light(no);
}

static inline int single_star_top_level_node(int no) {return no >= All.TreeNodeIndexBase && no - All.TreeNodeIndexBase < NTopnodes;}

/* collective, on every rank when any rank has pending stars (or before a node-moment refresh): each pending star's light change
   is added to its rank's nodes below the top level, sent to the ranks holding imported copies of those nodes or of the star's
   leaf, and to every rank for the top-level nodes above it; all ranks apply the top-level changes in rank order, so their
   copies of the top tree stay identical. The nodes then hold every pending star's current light. */
void single_star_light_exchange(void)
{
    const double t0 = MPI_Wtime();
    std::vector<std::vector<struct single_star_light_change>> out(NTask);
    for(int p : light_pending) {
        int s = single_star_slot(P[p].StarIdx);
        struct single_star_light_change c; memset(&c, 0, sizeof(c));
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(p, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const float lb = (float) lum[b];
            c.dl[b] = (double) lb - (double) SingleStars.tree_lum[s].L[b];
            SingleStars.tree_lum[s].L[b] = lb;
        }
        const double dt = get_drift_factor(P[p].Ti_current, All.Ti_Current, p, 0);
        for(int k = 0; k < 3; k++) {c.x[k] = (double) P[p].Pos[k] + (double) P[p].Vel[k] * dt; c.v[k] = (double) P[p].Vel[k];}
        for(int R = 0; R < NTask; R++) {
            if(R == ThisTask) {continue;}
            const int j = let_light_wire_of(R, -1 - p);
            if(j >= 0) {c.kind = 1; c.target = j; out[R].push_back(c);}
        }
        int no = Father[p];
        while(no >= All.TreeNodeIndexBase && no < All.TreeNodeIndexBase + MaxNodes) {
            for(int R = 0; R < NTask; R++) {
                if(R == ThisTask) {continue;}
                const int j = let_light_wire_of(R, no);
                if(j >= 0) {c.kind = 1; c.target = j; out[R].push_back(c);}
            }
            if(single_star_top_level_node(no)) {c.kind = 0; c.target = no; for(int R = 0; R < NTask; R++) {out[R].push_back(c);} break;}
            single_star_light_apply(no, &c);
            no = Nodes[no].u.d.father;
        }
    }
    std::vector<int> scount(NTask), rcount(NTask), soff(NTask), roff(NTask);
    long long nsend = 0, nrecv = 0;
    for(int R = 0; R < NTask; R++) {scount[R] = (int) (out[R].size() * sizeof(struct single_star_light_change)); soff[R] = (int) nsend; nsend += scount[R];}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for(int R = 0; R < NTask; R++) {roff[R] = (int) nrecv; nrecv += rcount[R];}
    std::vector<char> sbuf(nsend > 0 ? nsend : 1), rbuf(nrecv > 0 ? nrecv : 1);
    for(int R = 0; R < NTask; R++) {if(scount[R] > 0) {memcpy(sbuf.data() + soff[R], out[R].data(), scount[R]);}}
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_BYTE, rbuf.data(), rcount.data(), roff.data(), MPI_BYTE, MPI_COMM_WORLD);
    long long missing = 0;
    for(int r = 0; r < NTask; r++) {
        const int n = rcount[r] / (int) sizeof(struct single_star_light_change);
        for(int m = 0; m < n; m++) {
            struct single_star_light_change c; memcpy(&c, rbuf.data() + roff[r] + (size_t) m * sizeof(c), sizeof(c));
            if(c.kind == 1) {
                const int no = let_light_foreign_node(r, c.target);
                if(no < 0) {missing++; continue;}
                single_star_light_apply(no, &c);
            } else {
                for(int no = c.target; single_star_top_level_node(no); no = Nodes[no].u.d.father) {single_star_light_apply(no, &c);}
            }
        }
    }
    if(missing) {printf("SINGLE_STAR_LIGHT_EXCHANGE: rank %d received %lld changes for imported nodes it does not hold\n", ThisTask, missing); fflush(stdout); endrun(90001101);}
    const double here[3] = {(double) light_pending.size(), (double) (nsend - scount[ThisTask]), MPI_Wtime() - t0};
    light_stats[0] += 1; for(int k = 0; k < 3; k++) {light_stats[k + 1] += here[k];}
    double sum[2] = {0, 0}, tloc[2] = {here[2], light_scan_seconds}, tmax[2] = {0, 0};
    MPI_Reduce(here, sum, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(tloc, tmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(ThisTask == 0) {printf("SINGLE_STAR_LIGHT_EXCHANGE: t=%.10g stars=%.0f bytes=%.0f seconds_max=%.3e scan_seconds_cumulative_max=%.3e\n", All.Time, sum[0], sum[1], tmax[0], tmax[1]);}
    light_pending.clear();
}

/* this rank's cumulative exchange cost: calls, changed stars, bytes sent to other ranks, seconds */
void single_star_light_exchange_stats(double *out4) {for(int k = 0; k < 4; k++) {out4[k] = light_stats[k];}}
#endif

#ifdef GALSF_SINGLE_STARS_LIGHT_REFRESHES_TREE /* opt-in: by default the nodes keep the light of the last build (trunk's bounded staleness) */
/* before a gravity call: refresh the tree node moments (luminosities) if a single star started to shine, started to ionise or went
   dark since the last call, or its band luminosity moved by more than SINGLE_STAR_TREE_LUM_TOL of its own since the nodes last
   saw it; reduced in gravity_tree, so no collective here. A switch or a drift sets 2: on several ranks gravity_tree then
   rebuilds, which re-sends the node light to the ranks holding copies (one rank: a refresh) */
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
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {
            const double l0 = SingleStars.tree_lum[s].L[b], lb = (float) lum[b]; /* as stored */
            if((l0 == 0) != (lb == 0)) {TreeMomentsStaleFlag = 2; return;} /* a band switched on or off */
            if(fabs(lb - l0) > SINGLE_STAR_TREE_LUM_TOL * DMAX(fabs(lb), fabs(l0))) {TreeMomentsStaleFlag = 2; return;}
        }
        if(SingleStars.M_drawn[s] <= All.IMFSampleStellarMassCut) {continue;}
#if defined(GALSF_FB_MECHANICAL)
        if(SingleStars.event_bits[s] & (SINGLE_STAR_EVENT_SN_DONE | SINGLE_STAR_EVENT_AGB_DONE)) {continue;}
#endif
        double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
        double age = single_star_age_yr(i), t_pms = stellar_t_PMS(logM, logZ), t_on = single_star_ionizing_onset_yr(i);
        double t_switch[3] = {t_pms, t_pms + stellar_lifetime(logM, logZ), t_on};
        for(int k = 0; k < 3; k++) {if(t_switch[k] > 0 && age - dt_yr < t_switch[k] && t_switch[k] <= age) {TreeMomentsStaleFlag = 2; return;}}
    }
}

/* after a gravity call: if it built the tree or refreshed the node moments, the node luminosities are the stars' current ones */
void single_star_tree_lum_store(void)
{
    if(!TreeMomentsRenewed) {return;}
    for(int i = 0; i < NumPart; i++) {
        if(P[i].Type != 4 || P[i].StarIdx <= 0) {continue;}
        double lum[N_RT_FREQ_BINS] = {0}; rt_get_source_luminosity(i, 1, lum, P, CellP);
        for(int b = 0; b < N_RT_FREQ_BINS; b++) {SingleStars.tree_lum[single_star_slot(P[i].StarIdx)].L[b] = (float) lum[b];}
    }
}
#endif
