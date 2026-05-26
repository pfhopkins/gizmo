/* dm_fluid_functions.h — placeholder dark-fluid thermo + cooling hooks for
 * HYDRO_MULTIFLUID_DM (Type=0+FluidType=FLUID_DM particles).
 *
 * Defaults: trivial adiabatic γ=5/3 EOS, strictly local; zero cooling.
 *
 * HYDRO_MULTIFLUID_DM_COOLING (opt-in sub-subflag) activates the consolidated
 * atomic+molecular ADM cooling reference implementation appended below. It
 * was distilled from the legacy gizmo_adm_roy_v2 reference (Roy et al.) and
 * uses four user-supplied parameters on the All struct:
 *
 *    All.ADM_FineStructure       (= alpha_ADM; SM value 7.2973525693e-3)
 *    All.ADM_ProtonMass          (= m_p_ADM in g; SM value PROTONMASS_CGS)
 *    All.ADM_ElectronMass        (= m_e_ADM in g; SM value ELECTRON_MASS_SM_CGS)
 *    All.ADM_MolecularFraction   (= dark-H2 fraction by mass, 0..1)
 *
 * Deltas from the Roy reference baked in here (placeholder machinery —
 * NOT extensively validated; users implementing real ADM science should
 * re-validate):
 *   - GSL dependency replaced by gizmo_gl20_integrate (panel-truncated;
 *     integrands carry exp(-u^2) decay, truncation past u≈12 is below 1e-60).
 *   - find_abundances_and_rates_dm: do/while iteration removed; balance is
 *     closed-form alpha_rec / (alpha_rec + gamma_coll_ion) since there is
 *     no photoionization channel in this placeholder.
 *   - CoolingRate_dm: dead LambdaExcH0/LambdaRecHp zero-assigns dropped.
 *   - cooling_DarkH2: commented-out Glover-2015 experimental branches stripped;
 *     stray printf-on-overflow tightened to a PRINT_WARNING.
 *   - SphP[target].Ne write → cell[target].Ne (fits current branch).
 *
 * Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifdef HYDRO_MULTIFLUID_DM

#include "../declarations/multifluid_helpers.h"

/* Trivial dark-fluid EOS: P = (γ-1) ρ u with γ=5/3. Pure ideal-gas adiabat,
 * no chemistry / cooling / radiation coupling. Mirrors the structure of
 * eos.cc::set_eos_pressure's early-pre-cooling branch.
 *
 * KOKKOS_INLINE_FUNCTION (device-clean): body is pure arithmetic on macro
 * constants (GAMMA_DEFAULT, PROTONMASS_CGS, BOLTZMANN_CGS, UNIT_*) plus
 * device-callable cell_data accessors (density_for_energy, InternalEnergyPred).
 * Lets eos_functions.h::set_eos_pressure_impl dispatch to it unconditionally
 * under HYDRO_MULTIFLUID_DM — no __CUDA_ARCH__ exclusion or
 * POST_COOLING_DEVICE_EOS_SUPPORTED disable needed for safety. */
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif
static KOKKOS_INLINE_FUNCTION void set_dark_eos_pressure(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double gamma_eos_index = GAMMA_DEFAULT;
    double press = (gamma_eos_index - 1) * cell[i].InternalEnergyPred * cell[i].density_for_energy();
    cell[i].Gamma = gamma_eos_index;
    cell[i].Pressure = press;
    double mu_dark = 1.0;
    cell[i].Temperature = (MyFloat)( cell[i].InternalEnergyPred * (gamma_eos_index - 1.0) * mu_dark
                                     * PROTONMASS_CGS / BOLTZMANN_CGS
                                     * UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS );
    cell[i].SoundSpeed = (MyFloat)sqrt(gamma_eos_index * press / cell[i].density_for_energy());
}


/* ============================================================================
 *  Atomic+molecular ADM cooling (HYDRO_MULTIFLUID_DM_COOLING)
 * ============================================================================ */
#ifdef HYDRO_MULTIFLUID_DM_COOLING

#include "../declarations/gizmo_quadrature.h"

/* ---- cooling tables (inline globals: one instance shared across TUs) ---- */
inline int    NCOOLTAB_DM_g     = 2000;     /* table size */
inline double Tmin_dm_g         = -1.0;     /* log10(T_min/K), set in MakeCoolingTable_dm */
inline double Tmax_dm_g         =  9.0;     /* log10(T_max/K) */
inline double deltaT_dm_g       =  0.0;     /* (Tmax-Tmin)/NCOOLTAB */
inline double *BetaH0_dm_g       = nullptr;
inline double *Betaff_dm_g       = nullptr;
inline double *AlphaHp_dm_g      = nullptr;
inline double *AlphaHpRate_dm_g  = nullptr;
inline double *GammaeH0_dm_g     = nullptr;

/* ---- species indices for cooling_DarkH2 (file-scope; was per-function dup) ---- */
static const int QH_DM = 0, QH2_DM = 1, QHj_DM = 2, QE_DM = 3;

/* ---- SM reference constants used to form ADM scaling ratios ---- */
#define FINE_STRUCTURE_SM     (7.2973525693e-3)   /* dimensionless */
#define ELECTRON_MASS_SM_CGS  ELECTRONMASS_CGS    /* m_e in g (from declarations/constants.h) */

/* ============================================================================
 *  Custom semi-infinite integrator: replaces gsl_integration_qagiu.
 *  Integrands carry exp(-u^2) decay, so truncate at upper = max(lower+12, 16);
 *  use 4-panel GL20 for sub-1e-12 accuracy on smooth integrands.
 * ============================================================================ */
static inline double dm_semiinfinite_integrate(gizmo_integrand_fn f, double lower, void *params)
{
    double upper = (lower + 12.0 > 16.0) ? (lower + 12.0) : 16.0;
    const int NPANEL = 4;
    double dx = (upper - lower) / NPANEL;
    double result = 0;
    for(int p = 0; p < NPANEL; p++) {
        result += gizmo_gl20_integrate(f, lower + p*dx, lower + (p+1)*dx, params);
    }
    return result;
}

/* ============================================================================
 *  ATOMIC COOLING — cross-section / rate-coefficient integrands.
 *  Follows Draine-style hydrogenic formulae with ADM mass/coupling ratios.
 * ============================================================================ */

/* Collisional excitation kernel g0 (closed-form rational + log). */
static inline double dm_g0(double x) {
    return 9.0 * (299619.0 + 374220.0*pow(x,2.) + 203040.0*pow(x,4.) + 51840.0*pow(x,6.) + 5120.0*pow(x,8.))
         / (40.0 * pow(9.0 + 4.0*pow(x,2.), 5.))
         + log(x) - 0.5 * log(9.0 + 4.0*pow(x,2.));
}

/* Collisional excitation integrand g(u; y2). */
static inline double dm_g(double u, void *params) {
    double y2 = *(double*)params;
    double xplus  = (u/sqrt(y2)) * (1.0 + sqrt(1.0 - 3.0*y2/(u*u)/4.0));
    double xminus = (u/sqrt(y2)) * (1.0 - sqrt(1.0 - 3.0*y2/(u*u)/4.0));
    if(xminus == 0.0) {
        double x = 0.75 * y2 / (u*u);
        xminus = (u/sqrt(y2)) * (0.5*x + (1.0/8.0)*x*x + (1.0/16.0)*x*x*x);
    }
    return (u * exp(-u*u) / (1.0 + 7.0*y2/(4.0*u*u))) * (dm_g0(xplus) - dm_g0(xminus));
}
static inline double dm_g_integral(double y2) {
    return dm_semiinfinite_integrate(&dm_g, sqrt(3.0*y2)/2.0, &y2);
}

/* Collisional ionization integrand f(u; y2). */
static inline double dm_f(double u, void *params) {
    double y2 = *(double*)params;
    double u2 = u*u;
    double r = y2/u2;
    return (u * exp(-u2) / (1.0 + 2.0*r)) * (1.0 - r - 0.5*(1.0 - r*r)*log(r) + r*log(r)/(1.0 + r));
}
static inline double dm_f_integral(double y2) {
    return dm_semiinfinite_integrate(&dm_f, sqrt(y2), &y2);
}

/* Recombination rate / cooling integrands (N=100 level sum). */
static inline double dm_recomb_rate_f(double u, void *params) {
    double y2 = *(double*)params;
    double sum = 0.0;
    for(int i = 1; i <= 100; i++) {
        sum += u * exp(-u*u) / (u*u * (double)i*i*i + y2 * (double)i);
    }
    return sum;
}
static inline double dm_recomb_cool_f(double u, void *params) {
    double y2 = *(double*)params;
    double sum = 0.0, u3 = u*u*u;
    for(int i = 1; i <= 100; i++) {
        sum += u3 * exp(-u*u) / (u*u * (double)i*i*i + y2 * (double)i);
    }
    return sum;
}
static inline double dm_recomb_rate_integral(double y2) {
    return dm_semiinfinite_integrate(&dm_recomb_rate_f, 0.0, &y2);
}
static inline double dm_recomb_cool_integral(double y2) {
    return dm_semiinfinite_integrate(&dm_recomb_cool_f, 0.0, &y2);
}

/* ============================================================================
 *  MOLECULAR COOLING — dark-H2 rovib (Soliman/Roy adaptation of Glover & Abel
 *  2008 + Glover 2015 SM curves with ADM scalings).
 * ============================================================================ */

static inline double dm_sigmoid(double x, double x0, double s) { return 10.0 / (10.0 + exp(-s * (x - x0))); }

static inline double dm_wCool(double logTgas, double logTmin, double logTmax) {
    double x = (logTgas - logTmin) / (logTmax - logTmin);
    double w = pow(10., 200.0 * (dm_sigmoid(x, -0.2, 50.) * dm_sigmoid(-x, -1.2, 50.) - 1.0));
    if(w < 1e-199) {w = 0.0;}
    if(w > 1.0)    {w = 0.0;}
    return w;
}

/* Glover & Abel 2008 + Glover 2015 SM H2 rovib piece. typeCurve in {0,1,2,3}. */
static inline double dm_computeSMCurve(int typeCurve, double temp, double winScaleLow, double winScaleHigh) {
    double T3 = temp * 1e-3, logt3 = log10(T3), logt = log10(temp);
    double l2 = logt3*logt3, l3 = l2*logt3, l4 = l3*logt3, l5 = l4*logt3;
    if(typeCurve == 0) { /* H2-H */
        if(temp <= 1e2)              return pow(10., -16.818342 + 37.383713*logt3 + 58.145166*l2 + 48.656103*l3 + 20.159831*l4 + 3.8479610*l5);
        else if(temp <= 1e3)         return pow(10., -24.311209 +  3.5692468*logt3 - 11.332860*l2 - 27.850082*l3 - 21.328264*l4 - 4.2519023*l5);
        else if(temp <= 6e3)         return pow(10., -24.311209 +  4.6450521*logt3 -  3.7209846*l2 +  5.9369081*l3 -  5.5108049*l4 + 1.5538288*l5);
        else                          return 1.862314467912518e-22 * dm_wCool(logt, 1.0 + winScaleLow, log10(6e3) + winScaleHigh);
    }
    if(typeCurve == 1) { /* H2-H2 */
        double w24 = dm_wCool(logt, 2.0 + winScaleLow, 4.0 + winScaleHigh);
        double f = pow(10., -23.962112 + 2.09433740*logt3 - 0.77151436*l2 + 0.43693353*l3 - 0.14913216*l4 - 0.033638326*l5);
        return isfinite(f) ? w24 * f : 0.0;
    }
    if(typeCurve == 2) { /* H2-H+ */
        double w14 = dm_wCool(logt, 1.0 + winScaleLow, 4.0 + winScaleHigh);
        if(temp > 1e1 && temp <= 1e4) return pow(10., -22.089523 + 1.5714711*logt3 + 0.015391166*l2 - 0.23619985*l3 - 0.51002221*l4 + 0.32168730*l5);
        return 1.182509139382060e-21 * w14;
    }
    if(typeCurve == 3) { /* H2-e */
        double l6 = l5*logt3, l7 = l6*logt3, l8 = l7*logt3;
        double w24 = dm_wCool(logt, 2.0 + winScaleLow, 4.0 + winScaleHigh);
        double f;
        if(temp <= 5e2) {
            f = pow(10., DMIN(-21.928796 + 16.815730*logt3 + 96.743155*l2 + 343.19180*l3 + 734.71651*l4 + 983.67576*l5 + 801.81247*l6 + 364.14446*l7 + 70.609154*l8, 30.0));
        } else {
            f = pow(10., -22.921189 + 1.6802758*logt3 + 0.93310622*l2 + 4.0406627*l3 - 4.7274036*l4 - 8.8077017*l5 + 8.9167183*l6 + 6.4380698*l7 - 6.3701156*l8);
        }
        return isfinite(f) ? w24 * f : 0.0;
    }
    return 0.0;
}

/* Multi-case piecewise rescaled SM rovib (Soliman et al. arXiv:2106.13245 Appx). */
static inline double dm_computeMultiCase(int typeCurve, double Tgas, double critTemp, double Tscale,
                                          double atomScale, double tempScaleR, double tempScaleV,
                                          double ldlScaleR, double ldlScaleV, double nX) {
    double Tr = Tgas*tempScaleR, Tv = Tgas*tempScaleV;
    double critTempR = critTemp/tempScaleR;
    double critTempT = critTemp/Tscale;
    double critTempV = critTemp/tempScaleV;
    double f = 0.0;
    if(Tgas <= critTempR) {
        f = dm_computeSMCurve(typeCurve, Tr, 1.0, tempScaleR/atomScale) * ldlScaleR;
    } else if(Tgas >= critTempV) {
        f = dm_computeSMCurve(typeCurve, Tv, tempScaleV/tempScaleR, tempScaleV/atomScale) * ldlScaleV;
    } else if(Tgas <= critTempT) {
        double x2 = critTemp, x1 = x2 * 0.98;
        double y2 = dm_computeSMCurve(typeCurve, x2, 1.0, 1.0);
        double y1 = dm_computeSMCurve(typeCurve, x1, 1.0, 1.0);
        double slope = (y2 - y1) / (x2 - x1);
        f = (slope * (Tr - critTemp) + y2) * ldlScaleR;
    } else {
        double x2 = critTemp, x1 = x2 * 1.02;
        double y2 = dm_computeSMCurve(typeCurve, x2, 1.0, 1.0);
        double y1 = dm_computeSMCurve(typeCurve, x1, 1.0, 1.0);
        double slope = (y2 - y1) / (x2 - x1);
        f = (slope * (Tv - critTemp) + y2) * ldlScaleV;
    }
    return f * nX;
}

/* Total dark-H2 cooling rate (low-density + LTE combined). */
static inline double dm_cooling_DarkH2(double *n, double Tgas)
{
    double ra = All.ADM_FineStructure / FINE_STRUCTURE_SM;
    double rP = All.ADM_ProtonMass / PROTONMASS_CGS;
    double rE = All.ADM_ElectronMass / ELECTRON_MASS_SM_CGS;

    double tempScaleR = pow(ra,-2.0) * pow(rE,-2.0) * pow(rP, 1.0);
    double tempScaleV = pow(ra,-2.0) * pow(rE,-1.5) * pow(rP, 0.5);
    double atomScale  = pow(ra,-2.0) * pow(rE,-1.0);
    double ctHt  = pow(ra,-2.0) * pow(rE,-1.5)    * pow(rP, 0.5);
    double ctH2t = pow(ra,-2.0) * pow(rE,-1.5882) * pow(rP, 0.5861);
    double ctpt  = ctHt, ctet = ctHt;
    double hdlScaleR    = pow(ra, 9.0) * pow(rE, 8.0) * pow(rP,-6.0);
    double hdlScaleV    = pow(ra, 9.0) * pow(rE, 5.0) * pow(rP,-3.0);
    double ldlScaleRHH2 = pow(ra, 1.0) * pow(rE, 1.0) * pow(rP,-2.0);
    double ldlScaleVHH2 = pow(ra, 1.0) * pow(rE, 0.25)* pow(rP,-1.25);
    double ldlScaleRp   = pow(ra, 1.0) * pow(rE, 0.5) * pow(rP,-1.5);
    double ldlScaleVp   = pow(ra, 1.0) * pow(rP,-1.0);
    double ldlScaleRe   = pow(ra, 1.0) * pow(rP,-1.0);
    double ldlScaleVe   = pow(ra, 1.0) * pow(rE, 0.5) * pow(rP,-0.5);

    /* SM critical temperatures for H2-X rovib transitions. */
    double critTempH  = 855.833, critTempH2 = 5402.44;
    double critTempHp = 1.0e10,  critTempE  = 1.0e10;

    double cool = 0.0;
    cool += dm_computeMultiCase(0, Tgas, critTempH,  ctHt,  atomScale, tempScaleR, tempScaleV, ldlScaleRHH2, ldlScaleVHH2, n[QH_DM]);
    cool += dm_computeMultiCase(2, Tgas, critTempHp, ctpt,  atomScale, tempScaleR, tempScaleV, ldlScaleRp,   ldlScaleVp,   n[QHj_DM]);
    cool += dm_computeMultiCase(1, Tgas, critTempH2, ctH2t, atomScale, tempScaleR, tempScaleV, ldlScaleRHH2, ldlScaleVHH2, n[QH2_DM]);
    cool += dm_computeMultiCase(3, Tgas, critTempE,  ctet,  atomScale, tempScaleR, tempScaleV, ldlScaleRe,   ldlScaleVe,   n[QE_DM]);

    if(cool > 1.0e30 && n[QH2_DM] > 0.0 && n[QH2_DM] < 1.0e15) {
        PRINT_WARNING("dm_cooling_DarkH2: low-density cool=%g unreasonable; returning 0", cool);
        return 0.0;
    }
    if(cool <= 0.0) {return 0.0;}

    /* LTE combined: 1/(1/HDL + 1/LDL) per H2. HDL from HM79/GP98 with ADM scalings. */
    double t3sr = Tgas * tempScaleR * 1.0e-3;
    double t3sv = Tgas * tempScaleV * 1.0e-3;
    double HDLR = (9.5e-22 * pow(t3sr,3.76)) / (1.0 + 0.12*pow(t3sr,2.1)) * exp(-pow(0.13/t3sr,3)) + 3.0e-24 * exp(-0.51/t3sr);
    HDLR *= hdlScaleR;
    double HDLV = 6.7e-19 * exp(-5.86/t3sv) + 1.6e-18 * exp(-11.7/t3sv);
    HDLV *= hdlScaleV;
    double HDL = HDLR + HDLV;
    if(HDL == 0.0) {return 0.0;}
    double LDL = cool;
    return n[QH2_DM] / (1.0/HDL + 1.0/LDL);
}

/* ============================================================================
 *  COOLING TABLE BUILDER (one-shot at startup)
 * ============================================================================ */
static inline void dm_InitCoolMemory(void)
{
    int N = NCOOLTAB_DM_g + 1;
    BetaH0_dm_g      = (double *) malloc(N * sizeof(double));
    AlphaHp_dm_g     = (double *) malloc(N * sizeof(double));
    AlphaHpRate_dm_g = (double *) malloc(N * sizeof(double));
    GammaeH0_dm_g    = (double *) malloc(N * sizeof(double));
    Betaff_dm_g      = (double *) malloc(N * sizeof(double));
}

static inline void dm_MakeCoolingTable(void)
{
    Tmin_dm_g = (All.MinGasTemp > 0.0) ? log10(All.MinGasTemp) : -1.0;
    deltaT_dm_g = (Tmax_dm_g - Tmin_dm_g) / NCOOLTAB_DM_g;
    /* unit-conversion bookkeeping for recombination rate normalization */
    const double gtoGeV    = 1.0 / 1.79e-24;
    const double TtoGeV    = 1.0 / 1.16e13;
    const double InGeVcm   = 1.9733e-14;
    const double GeVtoergs = 0.001602;
    for(int i = 0; i <= NCOOLTAB_DM_g; i++) {
        double T = pow(10.0, Tmin_dm_g + deltaT_dm_g * i);
        double y2 = All.ADM_ElectronMass * pow(All.ADM_FineStructure, 2.0) * pow(C_LIGHT_CGS, 2.0)
                  / (2.0 * BOLTZMANN_CGS * T);
        BetaH0_dm_g[i]      = 7.4e-18 * pow(All.ADM_FineStructure/0.01, 2.0) * sqrt(ELECTRON_MASS_SM_CGS/All.ADM_ElectronMass) * sqrt(1.0e5/T) * dm_g_integral(y2);
        Betaff_dm_g[i]      = 3.7e-27 * (pow(All.ADM_FineStructure/0.01, 3.0) / pow(All.ADM_ElectronMass/ELECTRON_MASS_SM_CGS, 1.5)) * sqrt(T);
        AlphaHp_dm_g[i]     = (pow(All.ADM_FineStructure, 5.0) / sqrt(pow(T,3.0) * All.ADM_ElectronMass))
                              * sqrt(pow(2.0, 11.0) * M_PI / 27.0)
                              * (C_LIGHT_CGS * pow(InGeVcm, 2.0) / sqrt(gtoGeV * pow(TtoGeV, 3.0)))
                              * dm_recomb_rate_integral(y2);
        AlphaHpRate_dm_g[i] = (pow(All.ADM_FineStructure, 5.0) / sqrt(T * All.ADM_ElectronMass))
                              * sqrt(pow(2.0, 11.0) * M_PI / 27.0)
                              * (C_LIGHT_CGS * pow(InGeVcm, 2.0) * GeVtoergs / sqrt(gtoGeV * TtoGeV))
                              * dm_recomb_cool_integral(y2);
        GammaeH0_dm_g[i]    = 2.2e-7 * pow(ELECTRON_MASS_SM_CGS/All.ADM_ElectronMass, 1.5) * sqrt(1.0e5/T) * dm_f_integral(y2);
    }
}

static inline void InitCool_dm(void) { dm_InitCoolMemory(); dm_MakeCoolingTable(); }

/* ============================================================================
 *  IONIZATION EQUILIBRIUM + ATOMIC LAMBDA (closed-form: no photoionization)
 * ============================================================================ */
static inline double dm_find_abundances_and_rates(double logT, double rho, int target, int return_cooling_mode,
                                                   double *ne_guess, double *nH0_guess, double *nHp_guess,
                                                   double *mu_guess,
                                                   struct gas_cell_data *cell)
{
    if(!isfinite(logT) || !isfinite(rho)) {logT = Tmin_dm_g;}
    if(logT <= Tmin_dm_g) {
        *nH0_guess = 1.0; *nHp_guess = 0.0; *ne_guess = 1.e-22;
        *mu_guess = 1.0/(1.0 + *ne_guess);
        return 0;
    }
    if(logT >= Tmax_dm_g) {
        *nH0_guess = 0.0; *nHp_guess = 1.0; *ne_guess = 1.0;
        *mu_guess = 1.0/(1.0 + *ne_guess);
        return 0;
    }
    double t = (logT - Tmin_dm_g) / deltaT_dm_g;
    int j = (int)t;
    if(j < 0) {j = 0;}
    if(j > NCOOLTAB_DM_g) {j = NCOOLTAB_DM_g;}
    double fhi = t - j, flow = 1.0 - fhi;

    /* Closed-form equilibrium: recombination vs collisional ionization
     * (no photoionization in this placeholder; iteration of Roy's version
     * was structurally pointless since the formula has no n_elec dependence). */
    double aHp     = flow * AlphaHp_dm_g[j]     + fhi * AlphaHp_dm_g[j+1];
    double aHpRate = flow * AlphaHpRate_dm_g[j] + fhi * AlphaHpRate_dm_g[j+1];
    double geH0    = DMAX(flow * GammaeH0_dm_g[j] + fhi * GammaeH0_dm_g[j+1], 1.e-40);
    double nH0     = aHp / (MIN_REAL_NUMBER + aHp + geH0);
    double nHp     = 1.0 - nH0;
    double n_elec  = nHp;

    *nH0_guess = nH0; *nHp_guess = nHp; *ne_guess = n_elec;
    *mu_guess = 1.0 / (1.0 + n_elec);
    if(target >= 0) {cell[target].Ne = n_elec;}

    if(return_cooling_mode == 1) {
        double bH0 = flow * BetaH0_dm_g[j] + fhi * BetaH0_dm_g[j+1];
        double bff = flow * Betaff_dm_g[j] + fhi * Betaff_dm_g[j+1];
        double LambdaExc = bH0 * n_elec * nH0;
        double LambdaIon = 0.5 * All.ADM_ElectronMass * pow(C_LIGHT_CGS, 2.0) * pow(All.ADM_FineStructure, 2.0) * geH0 * n_elec * nH0;
        double LambdaRec = aHpRate * n_elec * nHp;
        double LambdaFF  = bff * nHp * n_elec;
        return LambdaExc + LambdaIon + LambdaRec + LambdaFF;
    }
    return 0;
}

/* Convert specific u to T via Newton iteration on mu. */
static inline double dm_convert_u_to_temp(double u, double rho, int target, double *ne_guess,
                                           double *nH0_guess, double *nHp_guess, double *mu_guess,
                                           struct gas_cell_data *cell)
{
    double T_0 = u * All.ADM_ProtonMass / BOLTZMANN_CGS;
    *mu_guess = 1.0 / (1.0 + *ne_guess);
    double temp = (5./3.-1.) * (*mu_guess) * T_0;
    int iter = 0;
    while(iter < MAXITER) {
        double temp_old = temp;
        dm_find_abundances_and_rates(log10(DMAX(temp, pow(10., Tmin_dm_g))), rho, target, 0,
                                      ne_guess, nH0_guess, nHp_guess, mu_guess, cell);
        temp = (5./3.-1.) * (*mu_guess) * T_0;
        if(fabs(temp - temp_old) / DMAX(temp, 1.0) < 1.e-4) {break;}
        iter++;
    }
    if(temp <= 0) {temp = pow(10., Tmin_dm_g);}
    if(log10(temp) < Tmin_dm_g) {temp = pow(10., Tmin_dm_g);}
    return temp;
}

/* Net cooling rate (heating − cooling) per nH^2 in cgs. */
static inline double dm_CoolingRate(double logT, double rho, double n_elec_guess, int target, struct gas_cell_data *cell)
{
    double n_elec = n_elec_guess, nH0, nHp, mu;
    double nHcgs = rho / All.ADM_ProtonMass;
    if(logT <= Tmin_dm_g) {logT = Tmin_dm_g + 0.5 * deltaT_dm_g;}
    if(!isfinite(rho)) {return 0;}
    double T = pow(10.0, logT);

    double Lambda;
    if(logT < Tmax_dm_g) {
        Lambda = dm_find_abundances_and_rates(logT, rho, target, 1, &n_elec, &nH0, &nHp, &mu, cell);
        /* dark-H2 rovib contribution at low T (cap at the SM-analog 4.4 logT plus ADM scalings) */
        if(logT <= 4.4 + log10(All.ADM_ElectronMass/ELECTRON_MASS_SM_CGS) + 2.0*log10(All.ADM_FineStructure/FINE_STRUCTURE_SM)) {
            double n[4];
            n[QH_DM]  = nH0;
            n[QH2_DM] = All.ADM_MolecularFraction;
            n[QHj_DM] = nHp;
            n[QE_DM]  = n_elec;
            Lambda += dm_cooling_DarkH2(n, T);
        }
    } else {
        /* Fully ionized: only free-free remains. */
        nHp = 1.0; n_elec = nHp;
        Lambda = 3.7e-27 * (pow(All.ADM_FineStructure/0.01, 3.0) / pow(All.ADM_ElectronMass/ELECTRON_MASS_SM_CGS, 1.5)) * sqrt(T) * nHp * n_elec;
    }

    double Q = -Lambda; /* no heating channel in this placeholder */
    if(target >= 0) {Q += cell[target].DtInternalEnergy / nHcgs;}
    return Q;
}

static inline double dm_CoolingRateFromU(double u, double rho, double ne_guess, int target, struct gas_cell_data *cell)
{
    double nH0, nHp, mu; (void)nH0; (void)nHp;
    double temp = dm_convert_u_to_temp(u, rho, target, &ne_guess, &nH0, &nHp, &mu, cell);
    return dm_CoolingRate(log10(temp), rho, ne_guess, target, cell);
}

/* Implicit cooling solver via bisection (mirrors standard DoCooling pattern). */
static inline double dm_DoCooling(double u_old, double rho, double dt, double ne_guess, int target, struct gas_cell_data *cell)
{
    rho   *= UNIT_DENSITY_IN_CGS;
    u_old *= UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS;
    dt    *= UNIT_TIME_IN_CGS;
    double nHcgs = rho / All.ADM_ProtonMass;
    double ratefact = nHcgs * nHcgs / rho;
    double u = u_old, u_lower = u, u_upper = u;
    double LambdaNet = dm_CoolingRateFromU(u, rho, ne_guess, target, cell);
    int iter_upper = 0, iter_lower = 0;
    if(u - u_old - ratefact * LambdaNet * dt < 0) {
        u_upper *= sqrt(1.1); u_lower /= sqrt(1.1);
        while((iter_upper < MAXITER) && (u_upper - u_old - ratefact * dm_CoolingRateFromU(u_upper, rho, ne_guess, target, cell) * dt < 0)) {
            u_upper *= 1.1; u_lower *= 1.1; iter_upper++;
        }
    }
    if(u - u_old - ratefact * LambdaNet * dt > 0) {
        u_lower /= sqrt(1.1); u_upper *= sqrt(1.1);
        while((iter_lower < MAXITER) && (u_lower - u_old - ratefact * dm_CoolingRateFromU(u_lower, rho, ne_guess, target, cell) * dt > 0)) {
            u_upper /= 1.1; u_lower /= 1.1; iter_lower++;
        }
    }
    int iter = 0, iter_condition = 0;
    double du;
    do {
        u = 0.5 * (u_lower + u_upper);
        LambdaNet = dm_CoolingRateFromU(u, rho, ne_guess, target, cell);
        if(u - u_old - ratefact * LambdaNet * dt > 0) {u_upper = u;} else {u_lower = u;}
        du = u_upper - u_lower;
        iter++;
        iter_condition = ((fabs(du/u) > 3.0e-2) || ((fabs(du/u) > 3.0e-4) && (iter < 10))) && (iter < MAXITER);
    } while(iter_condition);
    if(iter >= MAXITER) {
        PRINT_WARNING("dm_DoCooling: failed to converge: u_in=%g rho_in=%g dt=%g ne_in=%g target=%d", u_old, rho, dt, ne_guess, target);
    }
    return u / (UNIT_ENERGY_IN_CGS / UNIT_MASS_IN_CGS);
}

#endif /* HYDRO_MULTIFLUID_DM_COOLING */


/* Placeholder dark cooling entry point. With HYDRO_MULTIFLUID_DM_COOLING off,
 * this is a strict-adiabatic no-op. With it on, dispatches to the consolidated
 * ADM cooling chain above. */
static inline void do_dark_cooling_for_particle(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef HYDRO_MULTIFLUID_DM_COOLING
    double dtime = get_particle_timestep_in_physical(i, pp);
    if((dtime > 0) && (cell[i].Mass > 0) && (pp[i].Type == 0)) {
        double uold = DMAX(All.MinEgySpec, cell[i].InternalEnergy);
        /* same dt-internal-energy limiter pattern as cooling.cc */
        cell[i].DtInternalEnergy = DMAX(cell[i].DtInternalEnergy, -0.99 * cell[i].InternalEnergy / dtime);
        cell[i].DtInternalEnergy = DMIN(cell[i].DtInternalEnergy,  1.e4 * cell[i].InternalEnergy / dtime);
        cell[i].DtInternalEnergy *= (UNIT_ENERGY_IN_CGS/UNIT_MASS_IN_CGS) / UNIT_TIME_IN_CGS * All.ADM_ProtonMass;
        double unew = dm_DoCooling(uold, cell[i].Density * All.cf_a3inv, dtime, cell[i].Ne, i, cell);
        cell[i].InternalEnergy = unew;
        cell[i].InternalEnergyPred = unew;
        cell[i].Pressure = (5./3.-1) * cell[i].Density * unew;
        cell[i].DtInternalEnergy = 0;
    }
#else
    (void)i; (void)pp; (void)cell; /* strict-adiabatic no-op */
#endif
}

#endif /* HYDRO_MULTIFLUID_DM */
