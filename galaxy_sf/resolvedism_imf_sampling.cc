#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <gsl/gsl_rng.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "resolvedism_selfcheck.h"
#include "resolvedism_fb_shared.h"
#include "../mesh/kernel.h"

/* LYRA-style single-star IMF sampling (Gutcke+ 2021).
 *
 * Each star formation event draws a stellar mass from the Kroupa IMF,
 * then collects that mass from an accretion region around the parent
 * gas cell.  Every gas cell within the accretion radius loses the same
 * fraction f_acc = M_star / M_enclosed of its mass.  No cell loses more
 * than 50%.  The star inherits CoM velocity and mass-weighted metallicity
 * from all donor cells.
 *
 * The accretion radius and f_acc are pre-computed from the local gas
 * density (known from the density loop), so only a single code_block_xchange
 * pass is needed per iteration.  Iteration is rare — only when the
 * density estimate was too far off (massive stars in patchy ISM).
 *
 * Runtime parameter IMFSampleMinMass controls the IMF lower bound:
 *   0.08 → full IMF  [0.08, 350] Msun
 *   4.0  → truncated  [4.0, 350] Msun  (LYRA-style)
 */

#ifdef GALSF_RESOLVEDISM_SAMPLE_IMF

#if (N_STELLAR_MASS < 5)
#error "GALSF_RESOLVEDISM_SAMPLE_IMF requires N_STELLAR_MASS >= 5 (indices 0-4 used)"
#endif

static const double M_max = 350.;

/* Mass-weighted Kroupa IMF: M * dN/dM.  Slopes are -0.3 (M<0.5) and -1.3 (M>=0.5).
   Used with the LYRA probability P = M_cell/M_star * pfac, where the 1/M in P
   cancels the extra M here, giving the correct number-weighted Kroupa for formed stars. */
static inline double drawMassFromIMF(double x)
{
    double A = 0.126512;
    if(x < 0.5) return 2.0 * A * pow(x, -0.3);
    return A * pow(x, -1.3);
}

/* Envelope function for rejection sampling (upper bound of mass-weighted IMF) */
static inline double envelope_function(double x)
{
    return 2.0 * 0.126512 * pow(x, -0.3);
}

/* Public function: draw a single stellar mass from the mass-weighted Kroupa IMF via rejection sampling.
 * The mass-weighting is cancelled by the LYRA probability P = M_cell/M_star * pfac,
 * giving the correct number-weighted Kroupa distribution for formed stars.
 * Lower bound from All.IMFSampleMinMass (default 0.08), upper bound 350 Msun. */
double draw_one_mass_from_kroupa_IMF(void)
{
    double M_min = All.IMFSampleMinMass;
    if(M_min < 0.08) M_min = 0.08;
    if(M_min > M_max) M_min = M_max * 0.99; /* safety */
    double M_drawn;
    do {
        /* Proposal from CDF inversion of envelope M^{-0.3}: integral gives M^{0.7} */
        M_drawn = pow((pow(M_max, 0.7) - pow(M_min, 0.7)) * gsl_rng_uniform(random_generator)
                      + pow(M_min, 0.7), 1.0 / 0.7);
    } while(gsl_rng_uniform(random_generator) > (drawMassFromIMF(M_drawn) / envelope_function(M_drawn)));
    return M_drawn;
}

/* Public function: finalize a newly sampled star — set up luminosity, winds, Ia fields.
 * Called from assign_stellar_masses() after the accretion walk sets Mass and Vel.
 * Assumes P[i].Mass and P[i].MstarSampleIMF[0] are already set to the correct values. */
void finalize_sampled_star(int i, double M_drawn)
{
    P[i].sampled = 1;

    if(!isfinite(P[i].Mass) || P[i].Mass <= 0)
        printf("NAN_CHECK_IMF: Task=%d ID=%llu Mass=%.6e after IMF sampling\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].Mass);

#ifdef GALSF_RESOLVEDISM_WINDS
    P[i].WindMassAccum = 0;
    P[i].WindMomentumAccum = 0;
    P[i].M_current_old = M_drawn; /* initial mass = ZAMS mass */
    P[i].last_wind_log_age = 0.0; /* log10(1 yr) — cumulative table is ~0 at this age */
#endif
#ifdef GALSF_RESOLVEDISM_TYPE_IA
    P[i].M_drawn_Ia = 0; /* set when star dies as WD */
#endif
#ifdef GALSF_RESOLVEDISM_FB
    P[i].FB_Deposited = 0;
    P[i].FB_NoDepositCount = 0;
    {int k_; for(k_ = 0; k_ < 3; k_++) {P[i].FB_RecoilAccum[k_] = 0; P[i].FB_VecSum[k_] = 0;}}
#endif

    /* Clear temporary storage */
    P[i].MstarSampleIMF[1] = P[i].MstarSampleIMF[2] = P[i].MstarSampleIMF[3] = P[i].MstarSampleIMF[4] = 0;

    /* Compute luminosity from the single drawn star */
#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
    P[i].UV_luminosity = 0;
    P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_PHOTOION
    P[i].Lyman_photons_per_sec = 0;
#endif
    if(M_drawn > All.IMFSampleStellarMassCut) {
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr < get_lifetime(M_drawn, P[i].BirthMetallicity)) {
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
            double logM = log10(M_drawn);
            double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
            double table_age = star_age_yr - stellar_t_PMS(logM, logZ);
            if(table_age > 0) {
                double log_age = log10(DMAX(table_age, 100.0));
#if defined(RADTRANSFER)
                /* M1 RT: FUV = 8-13.6 eV (PE bin), NUV = 3.4-8 eV, OPT = 0.4-3.4 eV */
                P[i].UV_luminosity  = pow(10., stellar_log_L_FUV_M1(logM, logZ, log_age));
#else
                /* Tree-ray: FUV = 6-13.6 eV (PE heating + LW), NUV = 3.4-8 eV, OPT = 0.4-3.4 eV */
                P[i].UV_luminosity  = pow(10., stellar_log_L_FUV_total(logM, logZ, log_age));
#endif
                P[i].LW_luminosity  = pow(10., stellar_log_L_LW(logM, logZ, log_age));
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
                P[i].NUV_luminosity = pow(10., stellar_log_L_NUV(logM, logZ, log_age));
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
                P[i].OPT_luminosity = pow(10., stellar_log_L_OPT_NIR(logM, logZ, log_age));
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
                P[i].Lyman_photons_per_sec = pow(10., stellar_log_Q_ion(logM, logZ, log_age));
#endif
            } /* else PMS: luminosities stay at zero (initialized above) */
#else
            P[i].UV_luminosity = pow(10., get_logL_pe(M_drawn));
            P[i].LW_luminosity = P[i].UV_luminosity;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[i].OPT_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
            P[i].Lyman_photons_per_sec = pow(10., get_logS_ly(M_drawn));
#endif
#endif
        }
    }
    /* Sanity check luminosities */
    if(!isfinite(P[i].UV_luminosity) || P[i].UV_luminosity < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu UV_lum=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].UV_luminosity, M_drawn);
    if(!isfinite(P[i].LW_luminosity) || P[i].LW_luminosity < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu LW_lum=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].LW_luminosity, M_drawn);
#ifdef GALSF_RESOLVEDISM_PHOTOION
    if(!isfinite(P[i].Lyman_photons_per_sec) || P[i].Lyman_photons_per_sec < 0)
        printf("NAN_CHECK_LUM: Task=%d ID=%llu Q_ion=%.6e M=%.2f\n",
            ThisTask, (unsigned long long)P[i].ID, P[i].Lyman_photons_per_sec, M_drawn);
#endif
#endif
}


#ifdef GALSF_RESOLVEDISM_G0_VARIABLE
/* Per-step recompute of FUV/LW (and NUV/OPT) band luminosities for LIVING stars.
 *
 * BUGFIX 2026-06-29: these were set only ONCE, in finalize_sampled_star at formation,
 * where the star age ~0 < t_PMS so the "PMS" branch leaves them at 0. They were never
 * updated as the star reached the main sequence, so every star kept UV_luminosity =
 * LW_luminosity = 0 for its whole life -> tree-walk FUV = 0 -> G0 collapsed to the
 * uniform floor in every cell. Here we re-evaluate them from the stellar tables at the
 * star's CURRENT age, every step. (Lyman_photons_per_sec is recomputed separately each
 * step in resolvedism_photoionize(), so it is intentionally NOT touched here.)
 * Body mirrors the band-luminosity logic in finalize_sampled_star(). */
void recompute_resolvedism_fuv_luminosities(void)
{
    int i;
    for(int ia : ActiveParticleList) {if(P[ia].Type == 4) {TreeMomentsStaleFlag = 1; break;}} /* node luminosities follow per-step stellar lums */
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    /* TestFUVZero=1: silence ALL stellar radiation (UV/LW/NUV/OPT + Lyman) for
     * quiescent-medium experiments (e.g. clean KO15 terminal-momentum ladder:
     * progenitor FUV photoheating drives a ~10 km/s box-wide flow otherwise).
     * Keeps the TREE_RAD columns/chemistry stack fully compiled and valid —
     * stripping TREE_RAD_H2 from a net-17 build is an invalid config (CALC_PHOTO
     * aborts). Test-tag only; no effect in production builds. */
    /* TestFixedQion>0: clean StarBench PI ionization-front test. PIN the ionizing
     * photon rate to a CONSTANT (Q [phot/s], no aging drift) and zero every
     * non-ionizing band, so the source is a pure fixed-Q ioniser directly
     * comparable to the Spitzer/Hosokawa-Inutsuka D-type solution. Checked BEFORE
     * TestFUVZero so it takes precedence. (Tier-2 variable-Q test: leave this =0.) */
    if(All.TestFixedQion > 0) {
        for(int i : ActiveParticleList) {
            if(P[i].Type != 4) continue;
            P[i].UV_luminosity = 0; P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[i].OPT_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
            P[i].Lyman_photons_per_sec = All.TestFixedQion;
#endif
#ifdef TREE_RAY_PI
            P[i].Ion_luminosity = All.TestFixedQion * Resolvedism_Eion_eff_eV * 1.60218e-12; /* table-driven <hnu> (was 27.2) */
#endif
        }
        return;
    }
    /* TestFixedLbol>0: clean RADIATION-PRESSURE test. PIN the bolometric luminosity
     * to a CONSTANT L [erg/s] (no aging drift), split into the UV/NUV/OPT bands with
     * a fixed representative hot-star spectrum (FUV 0.5, NUV 0.2, OPT 0.3; LW = 0.3
     * of FUV), and zero the ionizing rate so radp is isolated from PI. Gives a pure
     * constant-L radiation source directly comparable to the momentum-driven
     * radiation-pressure shell (Draine 2011 / Krumholz-Matzner). Checked before
     * TestFUVZero so it takes precedence. (Variable-L test: leave this =0.) */
    if(All.TestFixedLbol > 0) {
        double L = All.TestFixedLbol;
        for(int i : ActiveParticleList) {
            if(P[i].Type != 4) continue;
            P[i].UV_luminosity = 0.5 * L; P[i].LW_luminosity = 0.15 * L;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[i].NUV_luminosity = 0.2 * L;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[i].OPT_luminosity = 0.3 * L;
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
            P[i].Lyman_photons_per_sec = 0;
#endif
#ifdef TREE_RAY_PI
            P[i].Ion_luminosity = 0;
#endif
        }
        return;
    }
    if(All.TestFUVZero) {
        for(int i : ActiveParticleList) {
            if(P[i].Type != 4) continue;
            P[i].UV_luminosity = 0; P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[i].OPT_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_PHOTOION
            P[i].Lyman_photons_per_sec = 0;
#endif
#ifdef TREE_RAY_PI
            P[i].Ion_luminosity = 0;
#endif
        }
        return;
    }
#endif
#ifdef STELLA_SELFCHECK
    int n_selfcheck_life = 0;
#endif
    for(int i : ActiveParticleList)
    {
        if(P[i].Type != 4) continue;
        if(P[i].sampled != 1) continue;             /* only finalized stars */
        double M_drawn = P[i].MstarSampleIMF[0];     /* ZAMS mass (table index) */

        P[i].UV_luminosity = 0;
        P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
        P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
        P[i].OPT_luminosity = 0;
#endif
#ifdef TREE_RAY_PI
        P[i].Ion_luminosity = 0;
#endif
        if(M_drawn <= All.IMFSampleStellarMassCut) continue;
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
#if defined(STELLA_SELFCHECK) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES) && defined(GALSF_RESOLVEDISM_FB)
        { /* radiation and death must use the same lifetime */
            double lM_ = log10(DMAX(M_drawn, 0.09)), lZ_ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
            double t_rad = get_lifetime(M_drawn, P[i].BirthMetallicity), t_die = get_star_lifetime(M_drawn, lM_, lZ_);
            STELLA_CHECK("lifetime_consistent", fabs(t_rad - t_die) <= 1.0e-9 * t_die, "ID=%llu M=%g t_rad=%g t_die=%g", (unsigned long long)P[i].ID, M_drawn, t_rad, t_die);
            n_selfcheck_life++;
        }
#endif
        if(star_age_yr >= get_lifetime(M_drawn, P[i].BirthMetallicity)) continue; /* dead */
#ifdef GALSF_RESOLVEDISM_STELLAR_TABLES
        double logM = log10(M_drawn);
        double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
        double table_age = star_age_yr - stellar_t_PMS(logM, logZ);
        if(table_age > 0) {
            double log_age = log10(DMAX(table_age, 100.0));
#if defined(RADTRANSFER)
            P[i].UV_luminosity = pow(10., stellar_log_L_FUV_M1(logM, logZ, log_age));
#else
            P[i].UV_luminosity = pow(10., stellar_log_L_FUV_total(logM, logZ, log_age));
#endif
            P[i].LW_luminosity = pow(10., stellar_log_L_LW(logM, logZ, log_age));
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
            P[i].NUV_luminosity = pow(10., stellar_log_L_NUV(logM, logZ, log_age));
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
            P[i].OPT_luminosity = pow(10., stellar_log_L_OPT_NIR(logM, logZ, log_age));
#endif
#ifdef TREE_RAY_PI
            /* ionizing ENERGY luminosity for the TreeRay PI band: Q_ion [photons/s]
             * x the TABLE-DRIVEN population <hnu> (Resolvedism_Eion_eff_eV; was a
             * hard-wired 27.2 eV = 13.6 eV heat/ionization, ~3x the real O-star
             * photoelectron energy). chemcool converts back with the same energy,
             * so the photon count is conserved. */
            P[i].Ion_luminosity = pow(10., stellar_log_Q_ion(logM, logZ, log_age)) * Resolvedism_Eion_eff_eV * 1.60218e-12;
            if(!isfinite(P[i].Ion_luminosity) || P[i].Ion_luminosity < 0) P[i].Ion_luminosity = 0;
#endif
        } /* else PMS: stays 0 */
#else
        P[i].UV_luminosity = pow(10., get_logL_pe(M_drawn));
        P[i].LW_luminosity = P[i].UV_luminosity;
#endif
        /* Defensive guard: never let a non-finite/negative luminosity reach the
         * gravity-tree FUV walk (rare transient edge case ~1e-6 of stars). */
        if(!isfinite(P[i].UV_luminosity) || P[i].UV_luminosity < 0) P[i].UV_luminosity = 0;
        if(!isfinite(P[i].LW_luminosity) || P[i].LW_luminosity < 0) P[i].LW_luminosity = 0;
#ifdef GALSF_RESOLVEDISM_NUV_VARIABLE
        if(!isfinite(P[i].NUV_luminosity) || P[i].NUV_luminosity < 0) P[i].NUV_luminosity = 0;
#endif
#ifdef GALSF_RESOLVEDISM_OPT_VARIABLE
        if(!isfinite(P[i].OPT_luminosity) || P[i].OPT_luminosity < 0) P[i].OPT_luminosity = 0;
#endif
    }
#ifdef STELLA_SELFCHECK
    STELLA_CHECK_RAN("lifetime_consistent", n_selfcheck_life);
#endif
}
#endif


#if defined(GALSF_RESOLVEDISM_SAMPLE_IMF) && defined(GALSF_RESOLVEDISM_STELLAR_TABLES) && (defined(RADTRANSFER) || defined(TREE_RAY_PI))
/* Mean ionizing photon energy of the CURRENT living stellar population, from the
 * stellar tables: E_eff = Sum L_ion_tot / Sum Q_ion (photon-flux weighted), i.e.
 * the spectral hardness the tables actually assign to these stars.
 *
 * BUGFIX 2026-07-29: the M1/TreeRay ionizing band assumed a hard-wired
 * <hnu> = 27.2 eV (upstream rt_chem.cc bookkeeping convention), so the chemcool
 * seam deposited 27.2-13.6 = 13.6 eV of heat PER IONIZATION -- ~3x the real
 * O-star photoelectron energy (~3-5 eV). Every M1/TreeRay-ionized cell
 * equilibrated ~40% hot (14.5 kK measured vs ~10-11 kK expected at 0.1 Zsun).
 * Emission (L_ion from tables) and consumption (Gamma, heating in
 * chem_interface.cc) both read rt_nu_eff_eV[RT_FREQ_BIN_H0], so updating the
 * single global keeps the photon count exactly conserved while fixing the
 * hardness. Multifrequency builds derive per-band energies from the spectrum
 * (rt_chem.cc) and are left untouched. */
double Resolvedism_Eion_eff_eV = 27.2;   /* updated per step; startup fallback = old convention */
void resolvedism_update_ion_photon_energy(void)
{
#ifdef GALSF_RESOLVEDISM_ISOLATED_FB_TEST
    if(All.TestFixedQion > 0) {return;} /* pinned-Q test source: keep current <hnu>, skip table lookups on the synthetic star */
#endif
    int i; double sumL = 0, sumQ = 0, glob[2];
    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 1) continue;
        double M_drawn = P[i].MstarSampleIMF[0];
        if(M_drawn <= DMAX(All.IMFSampleStellarMassCut, 8.0)) continue; /* only ionizing stars matter */
        double star_age_yr = evaluate_stellar_age_Gyr(i) * 1.0e9;
        if(star_age_yr >= get_lifetime(M_drawn, P[i].BirthMetallicity)) continue;
        double logM = log10(M_drawn);
        double logZ = log10(DMAX(P[i].BirthMetallicity, 1e-10));
        double table_age = star_age_yr - stellar_t_PMS(logM, logZ);
        if(table_age <= 0) continue;
        double log_age = log10(DMAX(table_age, 100.0));
        double L = pow(10., stellar_log_L_ion_tot(logM, logZ, log_age));
        double Q = pow(10., stellar_log_Q_ion(logM, logZ, log_age));
        if(isfinite(L) && isfinite(Q) && L > 0 && Q > 0) {sumL += L; sumQ += Q;}
    }
    double loc[2] = {sumL, sumQ};
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(glob[1] > 0)
    {
        double E_eV = glob[0] / glob[1] / ELECTRONVOLT_IN_ERGS;
        if(isfinite(E_eV)) {Resolvedism_Eion_eff_eV = DMIN(DMAX(E_eV, 13.6), 100.0);}
    } /* no living ionizing stars: keep previous value (no photons in flight anyway) */
#if defined(RADTRANSFER) && defined(RT_CHEM_PHOTOION) && !defined(RT_PHOTOION_MULTIFREQUENCY)
    rt_nu_eff_eV[RT_FREQ_BIN_H0] = Resolvedism_Eion_eff_eV;
    rt_ion_G_HI[RT_FREQ_BIN_H0] = (rt_nu_eff_eV[RT_FREQ_BIN_H0] - 13.6) * ELECTRONVOLT_IN_ERGS / UNIT_ENERGY_IN_CGS;
#endif
}
#endif


/* =========================================================================== */
/*  Accretion walk via code_block_xchange framework.                           */
/*  Equal-fraction mass removal from all gas cells within R_acc.               */
/*  f_acc and R_acc pre-computed in assign_stellar_masses(), stored on particle */
/*  MstarSampleIMF[2] = f_acc,  KernelRadius = R_acc.                         */
/* =========================================================================== */

/* Global accumulators — allocated in assign_stellar_masses(), indexed by particle index */
static double *IMF_MassAccreted = NULL;
static std::vector<double> IMF_RmaxDonor; /* per particle: farthest donor distance over all removal passes */
static double *IMF_MomAccreted = NULL;   /* [NumPart * 3] */
#ifdef METALS
static double *IMF_MetalAccreted = NULL;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
static double *IMF_ElemAccreted = NULL;  /* [NumPart * NUM_RESOLVEDISM_ELEMENTS] */
#endif
/* Two-pass mode flag: 0 = count enclosed mass only (no removal), 1 = remove mass */
static int IMF_AccretionMode = 0;
/* Per-star donor tracking: accumulates (star_index, donor_ID, dM) entries during
 * the removal pass. Logged to IMFinfo.txt after the accretion walk completes. */
struct imf_donor_log_entry { int star_index; MyIDType donor_ID; double dM; };
static std::vector<imf_donor_log_entry> IMF_DonorLog;


#define CORE_FUNCTION_NAME resolvedismIMF_evaluate
#define INPUTFUNCTION_NAME particle2in_resolvedismIMF
#define OUTPUTFUNCTION_NAME out2particle_resolvedismIMF
#define CONDITIONFUNCTION_FOR_EVALUATION if(resolvedismIMF_active_check(i))
#include "../system/code_block_xchange_initialize.h"

struct INPUT_STRUCT_NAME
{
    MyDouble Pos[3], KernelRadius;
    MyDouble f_acc;   /* mass fraction to remove from each neighbor */
    int NodeList[NODELISTLENGTH];
}
*DATAIN_NAME, *DATAGET_NAME;

void particle2in_resolvedismIMF(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{
    int k; for(k=0;k<3;k++) {in->Pos[k]=P[i].Pos[k];}
    in->KernelRadius = P[i].KernelRadius;
    if(in->KernelRadius <= 0) in->KernelRadius = All.ForceSoftening[P[i].Type] * 5.0;
    in->f_acc = P[i].MstarSampleIMF[2]; /* set in assign_stellar_masses() pre-processing */
}

#define IMF_MAX_DONORS 128
struct imf_donor_entry { MyIDType ID; MyFloat dM; };

struct OUTPUT_STRUCT_NAME
{
    MyFloat mass_accreted;
    MyFloat r_max_donor;   /* farthest cell that gave mass (comoving code length) */
    MyFloat mom_accreted[3];
#ifdef METALS
    MyFloat metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    MyFloat elem_accreted[NUM_RESOLVEDISM_ELEMENTS];
#endif
    int n_donors;
    struct imf_donor_entry donors[IMF_MAX_DONORS];
}
*DATARESULT_NAME, *DATAOUT_NAME;

void out2particle_resolvedismIMF(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{
    int k;
    /* Always use += : arrays are zeroed before each iteration,
     * mode 0 (local) is always called first, then mode 1 (remote) adds. */
    if(mode == 0) {
        IMF_MassAccreted[i] += out->mass_accreted;
        for(k=0;k<3;k++) IMF_MomAccreted[i*3+k] += out->mom_accreted[k];
#ifdef METALS
        IMF_MetalAccreted[i] += out->metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++)
            IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] += out->elem_accreted[k];
#endif
        /* donor logging — the mode-0 primary loop is threaded too: shared vector needs the critical section */
        if(IMF_AccretionMode == 1) {
            #pragma omp critical (imf_donor_log)
            {
                if(out->r_max_donor > IMF_RmaxDonor[i]) {IMF_RmaxDonor[i] = out->r_max_donor;}
                for(int d = 0; d < out->n_donors; d++)
                    IMF_DonorLog.push_back({i, out->donors[d].ID, (double)out->donors[d].dM});
            }
        }
    } else {
        #pragma omp atomic
        IMF_MassAccreted[i] += out->mass_accreted;
        for(k=0;k<3;k++) {
            #pragma omp atomic
            IMF_MomAccreted[i*3+k] += out->mom_accreted[k];
        }
#ifdef METALS
        #pragma omp atomic
        IMF_MetalAccreted[i] += out->metal_accreted;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        for(k=0;k<NUM_RESOLVEDISM_ELEMENTS;k++) {
            #pragma omp atomic
            IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] += out->elem_accreted[k];
        }
#endif
        /* donor logging — mode 1 may be threaded, use critical section */
        if(IMF_AccretionMode == 1) {
            #pragma omp critical (imf_donor_log)
            {
                if(out->r_max_donor > IMF_RmaxDonor[i]) {IMF_RmaxDonor[i] = out->r_max_donor;}
                for(int d = 0; d < out->n_donors; d++)
                    IMF_DonorLog.push_back({i, out->donors[d].ID, (double)out->donors[d].dM});
            }
        }
    }
}

int resolvedismIMF_active_check(int i);
int resolvedismIMF_active_check(int i)
{
    if(P[i].Type != 4) return 0;
    if(P[i].sampled != 0) return 0;
    if(P[i].MstarSampleIMF[0] <= 0) return 0;
    if(P[i].MstarSampleIMF[2] <= 0) return 0; /* f_acc = 0 means converged or not active */
    return 1;
}


/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below */
int resolvedismIMF_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, k, n;
    double r2, h2;
    struct INPUT_STRUCT_NAME local;
    struct OUTPUT_STRUCT_NAME out;
    memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME));

    if(mode == 0) {particle2in_resolvedismIMF(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];}
    if(local.f_acc <= 0) return 0;
    if(local.KernelRadius <= 0) return 0;
    h2 = local.KernelRadius * local.KernelRadius;

    if(mode == 0) {startnode = All.MaxPart;}
    else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;}

    while(startnode >= 0)
    {
        while(startnode >= 0)
        {
            numngb_inbox = ngb_treefind_variable_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist);
            if(numngb_inbox < 0) return -2;

            for(n = 0; n < numngb_inbox; n++)
            {
                j = ngblist[n];
                if(P[j].Type != 0) continue;
#ifdef GALSF_RESOLVEDISM_PHOTOION
                if(CellP[j].Ionized > 0) continue; /* ionized cells cannot be donor cells for IMF accretion */
#endif

                double Mass_j;
                #pragma omp atomic read
                Mass_j = P[j].Mass;
                if(Mass_j <= 0) continue;

                double dp[3];
                for(k=0;k<3;k++) {dp[k] = local.Pos[k] - P[j].Pos[k];}
                NEAREST_XYZ(dp[0],dp[1],dp[2],1);
                r2=0; for(k=0;k<3;k++) {r2 += dp[k]*dp[k];}
                if(r2 < 0 || r2 >= h2) continue; /* allow r2==0: parent cell at same position */

                double M_floor = 0.01 * All.MinMassForParticleMerger; /* keep donors far from empty (merged later if below threshold) */
                if(Mass_j <= M_floor) continue;
                if(IMF_AccretionMode == 0)
                {
                    /* --- COUNTING PASS: sum available neighbor mass, no removal --- */
                    out.mass_accreted += Mass_j - M_floor;
                }
                else
                {
                    /* --- REMOVAL PASS: use correct f_acc --- */
                    /* threads accreting onto different stars can share a donor: read-compute-subtract is one step */
                    double dM = 0, Vel_j[3] = {0, 0, 0};
#ifdef METALS
                    double Z_j = 0;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    double X_j[NUM_RESOLVEDISM_ELEMENTS];
#endif
                    #pragma omp critical (imf_donor_mass)
                    {
                        double Mj = P[j].Mass;
                        if(Mj > M_floor) {
                            dM = local.f_acc * (Mj - M_floor);
                            if(dM > 0.5 * Mj) dM = 0.5 * Mj;
                            if(dM > Mj - M_floor) dM = Mj - M_floor;
                        }
                        if(dM > 0) {
                            for(k=0;k<3;k++) {Vel_j[k] = P[j].Vel[k];}
#ifdef METALS
                            Z_j = P[j].Metallicity[0];  /* total Z, slot 0 in FIRE-pattern Metallicity[] */
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                            for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {X_j[k] = P[j].Metallicity[MET_OF(k)];}
#endif
                            /* conserved MHD quantities scale with the mass removed (parent side of merge_split.cc:517-525) */
                            double frac = dM / Mj;
#ifdef MAGNETIC
                            for(k=0;k<3;k++) {
                                CellP[j].B[k] -= frac * CellP[j].B[k];
                                CellP[j].BPred[k] -= frac * CellP[j].BPred[k];
                                CellP[j].DtB[k] -= frac * CellP[j].DtB[k];
                            }
                            CellP[j].divB -= frac * CellP[j].divB;
#ifdef DIVBCLEANING_DEDNER
                            CellP[j].Phi -= frac * CellP[j].Phi;
                            CellP[j].PhiPred -= frac * CellP[j].PhiPred;
                            CellP[j].DtPhi -= frac * CellP[j].DtPhi;
                            for(k=0;k<3;k++) {CellP[j].DtB_PhiCorr[k] -= frac * CellP[j].DtB_PhiCorr[k];}
#endif
#endif
                            P[j].Mass -= dM;
                            CellP[j].Mass -= dM;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
                            CellP[j].MassTrue -= dM;
#endif
                        }
                    }
                    if(dM <= 0) continue;
#ifdef METALS
                    out.metal_accreted += dM * Z_j;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
                    /* IMF_ElemAccreted is indexed by StellarElement (0..26 = H..Zn); Met[MET_OF(k)] holds its mass fraction */
                    for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {out.elem_accreted[k] += dM * X_j[k];}
#endif
#ifdef STELLA_SELFCHECK
                    { static long n_don = 0, n_low = 0; n_don++;   /* donor drained below the accretion floor */
                      if(P[j].Mass < (1 - 1.0e-9) * 0.01 * All.MinMassForParticleMerger && ++n_low <= 50)
                          STELLA_CHECK("donor_mass_floor", 0, "donor ID=%llu M_left=%.3e M_merge=%.3e (code) n_low=%ld", (unsigned long long)P[j].ID, (double)P[j].Mass, All.MinMassForParticleMerger, n_low);
                      if(n_don == 1 || n_don % 1000 == 0) STELLA_CHECK_RAN("donor_mass_floor", 1);}
#endif
                    out.mass_accreted += dM;
                    if(r2 > out.r_max_donor * out.r_max_donor) {out.r_max_donor = sqrt(r2);}
                    for(k=0;k<3;k++) out.mom_accreted[k] += dM * Vel_j[k];
                    /* record donor for mass tracing */
                    if(out.n_donors < IMF_MAX_DONORS) {
                        out.donors[out.n_donors].ID = P[j].ID;
                        out.donors[out.n_donors].dM = dM;
                        out.n_donors++;
                    }
                }
            }
        }
        if(mode == 1)
        {
            listindex++;
            if(listindex < NODELISTLENGTH)
            {
                startnode = DATAGET_NAME[target].NodeList[listindex];
                if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode;}
            }
        }
    }

    if(mode == 0) {out2particle_resolvedismIMF(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;}
    return 0;
}


void assign_stellar_masses_xchange(void)
{
    #include "../system/code_block_xchange_perform_ops_malloc.h"
    #include "../system/code_block_xchange_perform_ops.h"
    #include "../system/code_block_xchange_perform_ops_demalloc.h"
}
#include "../system/code_block_xchange_finalize.h"


/* =========================================================================== */
/*  Top-level routine: pre-compute f_acc, run accretion walk, finalize stars    */
/* =========================================================================== */
void assign_stellar_masses(void)
{
    int i, k;

    /* Count unsampled stars (spawned by sfr_eff.cc with sampled=0) */
    int nstars_local = 0;
#ifdef STELLA_SELFCHECK
    int n_selfcheck_tree = 0;
#endif
    for(i = 0; i < NumPart; i++)
        if(P[i].Type == 4 && P[i].sampled == 0 && P[i].MstarSampleIMF[0] > 0) nstars_local++;

    int nstars_total;
    MPI_Allreduce(&nstars_local, &nstars_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(nstars_total == 0) return;

    if(ThisTask == 0) printf("RESOLVEDISM IMF: accreting mass for %d new stars\n", nstars_total);
    PRINT_STATUS(" ..IMF accretion walk for single stars");
    IMF_DonorLog.clear();

    /* Allocate accumulators (persist across iterations, zeroed each pass) */
    IMF_MassAccreted = (double *) mymalloc("IMF_MassAccreted", NumPart * sizeof(double));
    IMF_RmaxDonor.assign(NumPart, 0.0);
    IMF_MomAccreted = (double *) mymalloc("IMF_MomAccreted", NumPart * 3 * sizeof(double));
#ifdef METALS
    IMF_MetalAccreted = (double *) mymalloc("IMF_MetalAccreted", NumPart * sizeof(double));
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    IMF_ElemAccreted = (double *) mymalloc("IMF_ElemAccreted", NumPart * NUM_RESOLVEDISM_ELEMENTS * sizeof(double));
#endif

    /* Zero accumulators before first pass */
    memset(IMF_MassAccreted, 0, NumPart * sizeof(double));
    memset(IMF_MomAccreted, 0, NumPart * 3 * sizeof(double));
#ifdef METALS
    memset(IMF_MetalAccreted, 0, NumPart * sizeof(double));
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    memset(IMF_ElemAccreted, 0, NumPart * NUM_RESOLVEDISM_ELEMENTS * sizeof(double));
#endif

    int iter = 0, max_iter = 10;
    int incomplete_total;

    do {
        /* --- Set R_acc for this iteration --- */
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;

            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double rho = P[i].FormationDensity / All.cf_a3inv; /* convert physical back to comoving for volume estimate */
            if(rho <= 0) rho = 1e-30;

            double h_parent = P[i].KernelRadius;
            if(h_parent <= 0) h_parent = All.ForceSoftening[P[i].Type] * 5.0;

            /* On iterations after the first, check if this star is already converged */
            if(iter > 0) {
                double deficit = M_star_code - IMF_MassAccreted[i];
                if(deficit <= 0.01 * M_star_code) {
                    P[i].MstarSampleIMF[2] = 0; /* converged — skip */
                    continue;
                }
                /* Expand radius for another attempt */
                h_parent *= 1.5;
            }

            /* Estimate R_acc from density: want M_enclosed >= 2 * M_star */
            double M_remaining = (iter > 0) ? (M_star_code - IMF_MassAccreted[i]) : M_star_code;
            double M_kernel = (4.0/3.0) * M_PI * h_parent * h_parent * h_parent * rho;
            double R_acc = h_parent;
            if(2.0 * M_remaining > M_kernel && M_kernel > 0) {
                R_acc = h_parent * pow(2.0 * M_remaining / M_kernel, 1.0/3.0);
            }

            P[i].KernelRadius = R_acc;
            P[i].MstarSampleIMF[2] = 1.0; /* dummy positive value so active_check passes */
        }

        /* --- PASS 1: Count actual enclosed mass --- */
        IMF_AccretionMode = 0;
        /* Use a temporary accumulator for enclosed mass (reuse MassAccreted buffer but save/restore) */
        double *IMF_MassEnclosed = (double *) mymalloc("IMF_MassEnclosed", NumPart * sizeof(double));
        memset(IMF_MassEnclosed, 0, NumPart * sizeof(double));
        /* Temporarily swap the accumulator pointer */
        double *save_MassAccreted = IMF_MassAccreted;
        IMF_MassAccreted = IMF_MassEnclosed;
        assign_stellar_masses_xchange();
        IMF_MassAccreted = save_MassAccreted; /* restore */

        /* --- Compute correct f_acc from actual enclosed mass --- */
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;
            if(P[i].MstarSampleIMF[2] <= 0) continue; /* already converged */

            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double M_remaining = (iter > 0) ? (M_star_code - save_MassAccreted[i]) : M_star_code;
            double M_enclosed = IMF_MassEnclosed[i];

            double f_acc;
            if(M_enclosed > 0) {
                f_acc = M_remaining / M_enclosed;
            } else {
                f_acc = 0; /* no neighbors found */
            }
            /* Cap at 0.5 per cell */
            if(f_acc > 0.5) f_acc = 0.5;

            P[i].MstarSampleIMF[2] = f_acc;

        }
        myfree(IMF_MassEnclosed);

        /* --- PASS 2: Remove mass with correct f_acc --- */
        IMF_AccretionMode = 1;
        /* Zero the per-pass portion of accumulators (MassAccreted keeps running total across iterations) */
        /* We need a temporary to capture THIS pass's accreted mass */
        double *IMF_PassMass = (double *) mymalloc("IMF_PassMass", NumPart * sizeof(double));
        memset(IMF_PassMass, 0, NumPart * sizeof(double));
        /* BUGFIX 2026-06-30: do NOT zero Mom/Metal/Elem accumulators per pass.
         * Accretion is iterative (radius expands over up to 10 passes). These are
         * accreted ONLY in the removal pass (mode 1) and must accumulate across all
         * passes, just like the mass running-total below (IMF_MassAccreted += PassMass).
         * Previously memset here every pass -> multi-pass stars kept only the LAST
         * pass's metals/momentum, divided by the FULL accreted mass -> near-pure-H
         * stars with BirthMetallicity=0 (68% of stars), corrupting every
         * metallicity-dependent lookup (lifetimes, luminosities, yields, remnants).
         * They are zeroed once before the do-loop, so accumulation is correct. */
        double *save2 = IMF_MassAccreted;
        IMF_MassAccreted = IMF_PassMass;
        assign_stellar_masses_xchange();
        IMF_MassAccreted = save2;
        /* Add this pass's accreted mass to running total */
        for(i = 0; i < NumPart; i++) IMF_MassAccreted[i] += IMF_PassMass[i];
        myfree(IMF_PassMass);

        /* Check for incomplete accretion */
        int incomplete_local = 0;
        for(i = 0; i < NumPart; i++) {
            if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;
            double M_star_code = P[i].MstarSampleIMF[0] / UNIT_MASS_IN_SOLAR;
            double deficit = M_star_code - IMF_MassAccreted[i];
            if(deficit > 0.01 * M_star_code) incomplete_local++;
        }
        MPI_Allreduce(&incomplete_local, &incomplete_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        iter++;

        if(ThisTask == 0 && incomplete_total > 0)
            printf("IMF accretion pass %d: %d stars with >5%% mass deficit, iterating\n", iter, incomplete_total);
    } while(incomplete_total > 0 && iter < max_iter);

    if(ThisTask == 0 && iter >= max_iter && incomplete_total > 0)
        printf("WARNING: IMF accretion did not converge after %d passes, %d stars incomplete\n", max_iter, incomplete_total);

    /* === Post-processing: set star mass, CoM velocity, metallicity, then finalize === */
    /* Allocate local IMF logging buffers */
    double *imf_x = (double *)mymalloc("imf_x", DMAX(nstars_local,1) * sizeof(double));
    double *imf_y = (double *)mymalloc("imf_y", DMAX(nstars_local,1) * sizeof(double));
    double *imf_z = (double *)mymalloc("imf_z", DMAX(nstars_local,1) * sizeof(double));
    double *imf_mdrawn = (double *)mymalloc("imf_mdrawn", DMAX(nstars_local,1) * sizeof(double));
    double *imf_macc = (double *)mymalloc("imf_macc", DMAX(nstars_local,1) * sizeof(double));
    double *imf_racc = (double *)mymalloc("imf_racc", DMAX(nstars_local,1) * sizeof(double));
    double *imf_zbirth = (double *)mymalloc("imf_zbirth", DMAX(nstars_local,1) * sizeof(double));
    double *imf_nH = (double *)mymalloc("imf_nH", DMAX(nstars_local,1) * sizeof(double));
    double *imf_T = (double *)mymalloc("imf_T", DMAX(nstars_local,1) * sizeof(double));
    long long *imf_id = (long long *)mymalloc("imf_id", DMAX(nstars_local,1) * sizeof(long long));
    std::vector<double> imf_rdon(DMAX(nstars_local,1), 0.0); /* farthest donor distance [physical kpc] */
    int nimf = 0;

    for(i = 0; i < NumPart; i++)
    {
        if(P[i].Type != 4 || P[i].sampled != 0 || P[i].MstarSampleIMF[0] <= 0) continue;

        double M_drawn = P[i].MstarSampleIMF[0]; /* solar masses */
        double mass_acc = IMF_MassAccreted[i];

        if(mass_acc <= 0) {
            printf("WARNING: IMF accretion failed for star ID=%llu M_star=%.2f Msun, using tiny mass\n",
                (unsigned long long)P[i].ID, M_drawn);
            mass_acc = 1e-10;
        }

        /* Star mass = actual mass accreted (exact conservation) */
        P[i].Mass = mass_acc; TreeMomentsStaleFlag = 1;

        /* CoM velocity from all donor cells */
        if(mass_acc > 0) {
            for(k=0;k<3;k++) P[i].Vel[k] = IMF_MomAccreted[i*3+k] / mass_acc;
        }

        /* Mass-weighted metallicity */
#ifdef METALS
        P[i].Metallicity[0] = (mass_acc > 0) ? IMF_MetalAccreted[i] / mass_acc : 0;
#endif
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
        /* 28-slot layout: Met[0]=total Z, Met[1]=H, Met[2]=He, Met[3..27]=metals.
         * IMF_ElemAccreted[] is in 27-element table order (H=0..Zn=26); write each
         * into Met[MET_OF(k)] and compute Met[0] from metal sum. */
        if(mass_acc > 0) {
            double Z_sum = 0;
            for(k = 0; k < NUM_RESOLVEDISM_ELEMENTS; k++) {
                P[i].Metallicity[MET_OF(k)] = IMF_ElemAccreted[i*NUM_RESOLVEDISM_ELEMENTS+k] / mass_acc;
                if(k >= ELEM_C) Z_sum += P[i].Metallicity[MET_OF(k)];
            }
            P[i].Metallicity[0] = Z_sum;
            /* Force Σ Met[1..27] = 1 bit-exact via H from conservation. */
            double X_H_new = 1.0 - P[i].Metallicity[0] - P[i].Metallicity[MET_OF(ELEM_He)];
            if(X_H_new < 0) X_H_new = 0;
            P[i].Metallicity[MET_OF(ELEM_H)] = (MyFloat)X_H_new;
        } else {
            for(k = 0; k < NUM_METAL_SPECIES; k++) P[i].Metallicity[k] = 0;
        }
#endif
        P[i].BirthMetallicity = P[i].Metallicity[0];  /* total Z, slot 0 in both layouts */

        /* Log SF + IMF accretion info */
        imf_x[nimf] = P[i].Pos[0];
        imf_y[nimf] = P[i].Pos[1];
        imf_z[nimf] = P[i].Pos[2];
        imf_mdrawn[nimf] = M_drawn;
        imf_macc[nimf] = mass_acc * UNIT_MASS_IN_SOLAR;
        imf_racc[nimf] = P[i].KernelRadius;
        imf_rdon[nimf] = IMF_RmaxDonor[i] * All.cf_atime * UNIT_LENGTH_IN_KPC;
        imf_zbirth[nimf] = P[i].BirthMetallicity;
        imf_id[nimf] = (long long)P[i].ID;
        imf_nH[nimf] = P[i].FormationDensity * UNIT_DENSITY_IN_CGS * HYDROGEN_MASSFRAC / PROTONMASS_CGS; /* already physical at formation */
        imf_T[nimf] = P[i].MstarSampleIMF[3]; /* CellP.Temp stored at spawn time */
        nimf++;

        int iparent = (int)P[i].MstarSampleIMF[1]; /* read before finalize clears the temporary slots */
        finalize_sampled_star(i, M_drawn);
        force_add_element_to_tree(iparent, i); /* deferred tree insertion beside the parent cell */
#ifdef STELLA_SELFCHECK
        { /* star must lie inside the tree node it was linked into */
            int no = Father[i]; double dmax = 0;
            if(no >= All.MaxPart) {for(k = 0; k < 3; k++) {dmax = DMAX(dmax, fabs(P[i].Pos[k] - Nodes[no].center[k]) / (0.5 * Nodes[no].len));}}
            STELLA_CHECK("tree_insert", no >= All.MaxPart && dmax <= 1.5, "ID=%llu parent=%d node=%d dist_over_halflen=%g", (unsigned long long)P[i].ID, iparent, no, dmax);
            n_selfcheck_tree++;
        }
#endif
    }

#ifdef STELLA_SELFCHECK
    STELLA_CHECK_RAN("tree_insert", n_selfcheck_tree);
#endif
    /* Gather SF info and write to SFinfo.txt on rank 0 */
    {
        int nimf_total = 0;
        MPI_Allreduce(&nimf, &nimf_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(nimf_total > 0)
        {
            int *recvcounts = NULL, *displs = NULL;
            double *all_x=NULL, *all_y=NULL, *all_z=NULL, *all_mdrawn=NULL, *all_macc=NULL;
            double *all_racc=NULL, *all_zbirth=NULL, *all_nH=NULL, *all_T=NULL;
            long long *all_id = NULL;
            if(ThisTask == 0) {
                recvcounts = (int *)mymalloc("recvcounts_imf", NTask * sizeof(int));
                displs = (int *)mymalloc("displs_imf", NTask * sizeof(int));
            }
            MPI_Gather(&nimf, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                displs[0] = 0;
                for(i = 1; i < NTask; i++) {displs[i] = displs[i-1] + recvcounts[i-1];}
                all_x = (double *)mymalloc("aimf_x", nimf_total * sizeof(double));
                all_y = (double *)mymalloc("aimf_y", nimf_total * sizeof(double));
                all_z = (double *)mymalloc("aimf_z", nimf_total * sizeof(double));
                all_mdrawn = (double *)mymalloc("aimf_md", nimf_total * sizeof(double));
                all_macc = (double *)mymalloc("aimf_ma", nimf_total * sizeof(double));
                all_racc = (double *)mymalloc("aimf_ra", nimf_total * sizeof(double));
                all_zbirth = (double *)mymalloc("aimf_zb", nimf_total * sizeof(double));
                all_nH = (double *)mymalloc("aimf_nH", nimf_total * sizeof(double));
                all_T = (double *)mymalloc("aimf_T", nimf_total * sizeof(double));
                all_id = (long long *)mymalloc("aimf_id", nimf_total * sizeof(long long));
            }
            MPI_Gatherv(imf_x, nimf, MPI_DOUBLE, all_x, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_y, nimf, MPI_DOUBLE, all_y, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_z, nimf, MPI_DOUBLE, all_z, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_mdrawn, nimf, MPI_DOUBLE, all_mdrawn, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_macc, nimf, MPI_DOUBLE, all_macc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_racc, nimf, MPI_DOUBLE, all_racc, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            std::vector<double> all_rdon(ThisTask == 0 ? DMAX(nimf_total,1) : 1, 0.0);
            MPI_Gatherv(imf_rdon.data(), nimf, MPI_DOUBLE, all_rdon.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_zbirth, nimf, MPI_DOUBLE, all_zbirth, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_nH, nimf, MPI_DOUBLE, all_nH, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_T, nimf, MPI_DOUBLE, all_T, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(imf_id, nimf, MPI_LONG_LONG, all_id, recvcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                double worst_err = 0; int worst_idx = 0;
                for(i = 0; i < nimf_total; i++) {
                    fprintf(FdSFinfo, "%12.6f %9.4f %9.4f %9.4f %10lld %8.4f %8.4f %10.6f %10.4e %10.2f %10.2e %10.6f\n",
                        All.Time, all_x[i], all_y[i], all_z[i], all_id[i],
                        all_mdrawn[i], all_macc[i], all_racc[i], all_zbirth[i], all_T[i], all_nH[i], all_rdon[i]);
                    double err = (all_mdrawn[i] > 0) ? fabs(all_macc[i] - all_mdrawn[i]) / all_mdrawn[i] : 0;
                    if(err > worst_err) {worst_err = err; worst_idx = i;}
                }
                fflush(FdSFinfo);
                printf("RESOLVEDISM IMF: drew %d stars, max deviation: ID=%lld M_drawn=%.4f M_acc=%.4f (err=%.2f%%)\n",
                    nimf_total, all_id[worst_idx], all_mdrawn[worst_idx], all_macc[worst_idx], worst_err*100);
                myfree(all_id); myfree(all_T); myfree(all_nH); myfree(all_zbirth); myfree(all_racc);
                myfree(all_macc); myfree(all_mdrawn); myfree(all_z); myfree(all_y); myfree(all_x);
                myfree(displs); myfree(recvcounts);
            }
        }
    }

    /* Write donor log to IMFinfo.txt: one line per (star, donor) pair */
    {
        /* Convert star_index to star_ID and donor dM to solar masses */
        int ndonors_local = (int)IMF_DonorLog.size();
        long long *donor_star_id = (long long *)mymalloc("d_sid", DMAX(ndonors_local,1) * sizeof(long long));
        long long *donor_gas_id = (long long *)mymalloc("d_gid", DMAX(ndonors_local,1) * sizeof(long long));
        double *donor_dM = (double *)mymalloc("d_dM", DMAX(ndonors_local,1) * sizeof(double));
        for(int d = 0; d < ndonors_local; d++) {
            donor_star_id[d] = (long long)P[IMF_DonorLog[d].star_index].ID;
            donor_gas_id[d] = (long long)IMF_DonorLog[d].donor_ID;
            donor_dM[d] = IMF_DonorLog[d].dM * UNIT_MASS_IN_SOLAR;
        }

        int ndonors_total = 0;
        MPI_Allreduce(&ndonors_local, &ndonors_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(ndonors_total > 0) {
            int *rcounts = NULL, *rdispls = NULL;
            long long *all_star_id = NULL, *all_gas_id = NULL;
            double *all_dM = NULL;
            if(ThisTask == 0) {
                rcounts = (int *)mymalloc("d_rc", NTask * sizeof(int));
                rdispls = (int *)mymalloc("d_rd", NTask * sizeof(int));
            }
            MPI_Gather(&ndonors_local, 1, MPI_INT, rcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                rdispls[0] = 0;
                for(i = 1; i < NTask; i++) rdispls[i] = rdispls[i-1] + rcounts[i-1];
                all_star_id = (long long *)mymalloc("d_as", ndonors_total * sizeof(long long));
                all_gas_id = (long long *)mymalloc("d_ag", ndonors_total * sizeof(long long));
                all_dM = (double *)mymalloc("d_am", ndonors_total * sizeof(double));
            }
            MPI_Gatherv(donor_star_id, ndonors_local, MPI_LONG_LONG, all_star_id, rcounts, rdispls, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Gatherv(donor_gas_id, ndonors_local, MPI_LONG_LONG, all_gas_id, rcounts, rdispls, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Gatherv(donor_dM, ndonors_local, MPI_DOUBLE, all_dM, rcounts, rdispls, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if(ThisTask == 0) {
                for(i = 0; i < ndonors_total; i++)
                    fprintf(FdIMFinfo, "%12.6f %12lld %12lld %12.6e\n", All.Time, all_star_id[i], all_gas_id[i], all_dM[i]);
                fflush(FdIMFinfo);
                myfree(all_dM); myfree(all_gas_id); myfree(all_star_id);
                myfree(rdispls); myfree(rcounts);
            }
        }
        myfree(donor_dM); myfree(donor_gas_id); myfree(donor_star_id);
        IMF_DonorLog.clear();
    }

    /* Free IMF logging buffers (LIFO) */
    myfree(imf_id); myfree(imf_T); myfree(imf_nH); myfree(imf_zbirth); myfree(imf_racc);
    myfree(imf_macc); myfree(imf_mdrawn); myfree(imf_z); myfree(imf_y); myfree(imf_x);

    /* Free accumulators (LIFO order for mymalloc) */
#ifdef GALSF_RESOLVEDISM_METALS_INDIVIDUAL
    myfree(IMF_ElemAccreted); IMF_ElemAccreted = NULL;
#endif
#ifdef METALS
    myfree(IMF_MetalAccreted); IMF_MetalAccreted = NULL;
#endif
    myfree(IMF_MomAccreted); IMF_MomAccreted = NULL;
    myfree(IMF_MassAccreted); IMF_MassAccreted = NULL;
    std::vector<double>().swap(IMF_RmaxDonor);

    if(ThisTask == 0 && iter > 1) printf("RESOLVEDISM IMF: accretion required %d passes\n", iter);
}


#endif /* GALSF_RESOLVEDISM_SAMPLE_IMF */
