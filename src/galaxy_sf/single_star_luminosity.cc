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

#if defined(GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES) || defined(GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS)

static inline double single_star_band_lum_cgs(double log_l) {double l = pow(10., log_l); return (isfinite(l) && l > 0) ? l : 0;}

/* L[SINGLE_STAR_BAND_*] [erg/s] and the bolometric luminosity; returns 0 (all zero) before the main sequence, after death
   without a record or at or below IMFSampleStellarMassCut. 6-8 eV = L(6-13.6) - L(8-13.6), unclipped as stella_dev */
int single_star_band_luminosities_cgs(int i, double *L, double *L_bol)
{
    for(int b = 0; b < SINGLE_STAR_NBANDS; b++) {L[b] = 0;}
    *L_bol = 0;
    if(P[i].StarIdx <= 0) {return 0;}
    int s = single_star_slot(P[i].StarIdx);
    if(SingleStars.M_drawn[s] <= All.IMFSampleStellarMassCut) {return 0;}
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
    return 1;
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
/* ionising energy [erg/s] and photon rate [1/s] of a shining star from the tables at its current age */
static void single_star_ionizing_rates(int i, double *L_ion, double *Q)
{
    int s = single_star_slot(P[i].StarIdx);
    double logM = log10(SingleStars.M_drawn[s]), logZ = log10(DMAX(SingleStars.Z_birth[s], 1e-10));
    double la = log10(DMAX(evaluate_stellar_age_Gyr(i) * 1.e9 - stellar_t_PMS(logM, logZ), 100.));
    *L_ion = single_star_band_lum_cgs(stellar_log_L_ion_tot(logM, logZ, la)); *Q = single_star_band_lum_cgs(stellar_log_Q_ion(logM, logZ, la));
}
#endif

#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
/* as rt_get_source_luminosity: fills lum[] [code units], returns 1 for a shining star (pp is P here) */
int single_star_rt_band_luminosities(int i, int mode, double *lum, struct particle_data *pp)
{
    double L[SINGLE_STAR_NBANDS], L_bol; (void) pp;
    if(!single_star_band_luminosities_cgs(i, L, &L_bol)) {return 0;}
    if(mode < 0) {return 1;}
    lum[RT_FREQ_BIN_PHOTOELECTRIC] = L[SINGLE_STAR_BAND_PE] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_LYMAN_WERNER] = L[SINGLE_STAR_BAND_LW] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_GENERIC_USER_FREQ] = L[SINGLE_STAR_BAND_68] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_NUV] = L[SINGLE_STAR_BAND_NUV] / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_OPTICAL_NIR] = L[SINGLE_STAR_BAND_OPT] / UNIT_LUM_IN_CGS;
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION
    double L_ion, Q; single_star_ionizing_rates(i, &L_ion, &Q);
    lum[RT_FREQ_BIN_ION_ENERGY] = L_ion / UNIT_LUM_IN_CGS;
    lum[RT_FREQ_BIN_ION_PHOTONS] = Q * 13.6 * ELECTRONVOLT_IN_ERGS / UNIT_LUM_IN_CGS;
#endif
    return 1;
}
#endif

#ifdef GALSF_FB_RADPRESSURE_SINGLE_STAR_BANDS
/* photon momentum a single star hands to the gas this step [g cm/s] (stella_dev estimator): band luminosities absorbed
   over the kernel column at the star, f_b = 1 - exp(-sigma_b DGR N_H) with sigma = 2.0 (PE + 6-8), 1.3 (NUV), 0.5 (OPT)
   x 1e-21 cm^2, LW not counted (inside PE), capped at L_bol; plus the IR boost tau_IR = 5 DGR Sigma [cm^2/g] on the
   absorbed light. GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE adds 6-8 eV to the NUV term as stella_dev production.
   Stars below SINGLE_STAR_RADPRESSURE_MIN_MASS push nothing (stella_dev) */
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
   centres, all deposits scaled by L_emit / sum(L_abs) when above 1 (stella_dev photon cap). The sum over cells is kept running:
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
