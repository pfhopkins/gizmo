/*
 * jaco.cc -- GIZMO glue for the jaco code-generated microphysics network.
 *
 * Packs a gas cell into the generated SolveVars/Params (gizmo_to_jaco), calls the implicit solver
 * (jaco_solver.cc, no GIZMO dependencies), and writes the answer back (jaco_to_gizmo), with the
 * model's outputs at the answer (microphysics_outputs) for the host to apply over the step. Also
 * builds the CIE table that seeds the ions, and reports per-step solver statistics.
 */

#include "../core/proto.h"
#include "../declarations/allvars.h"
#include "microphysics_func_jac.h"
#include <math.h>
#include <string.h>
#include "jaco_solver.h"
#include "jaco_composition.h"
/* NaN/Inf check — defined in jaco_util.cc to prevent LTO from optimizing it away */
extern "C" int jaco_isfinite(double x);
#define JACO_ABUNDANCE_FLOOR 1e-20

#ifdef JACO

#ifndef COOLING
#error "JACO requires COOLING"
#endif
/* These read the standard cooling module's chemistry or EOS directly, bypassing jaco_cell_eos */
#if defined(OUTPUT_COOLRATE) || defined(GALSF_EFFECTIVE_EQS) || defined(CHIMES) || defined(COOL_GRACKLE)
#error "JACO does not yet support OUTPUT_COOLRATE, GALSF_EFFECTIVE_EQS, CHIMES or COOL_GRACKLE"
#endif
/* do_the_cooling_for_particle returns right after call_jaco, so everything the standard path does afterwards is skipped:
   cosmic-ray losses and the sink thermal-feedback energy injection. Refuse those configurations until the JACO path
   carries them. */
#if defined(COSMIC_RAY_FLUID) || defined(SINK_THERMALFEEDBACK)
#error "JACO does not yet support COSMIC_RAY_FLUID or SINK_THERMALFEEDBACK"
#endif
/* The network takes its cosmic rays from the ISRF scaling, not from the LEBRON field */
#ifdef COSMIC_RAY_SUBGRID_LEBRON
#error "JACO does not yet support COSMIC_RAY_SUBGRID_LEBRON: the network's cosmic-ray ionization and heating ignore the LEBRON cosmic-ray field"
#endif
/* The non-ideal resistivities (calculate_and_assign_nonideal_mhd_coefficients) read Ne but estimate the temperature,
   H2 fraction and cosmic-ray ionization rate themselves, inconsistently with the network's state */
#ifdef MHD_NON_IDEAL
#error "JACO does not yet support MHD_NON_IDEAL: its resistivities use their own temperature, H2 and cosmic-ray estimates, not the network's"
#endif
/* Radiation. Supported: M1 RADTRANSFER whose bands are all solve variables of the model (photon_EUV: RT_CHEM_PHOTOION's
   single H-ionizing band; photon_FUV, photon_NUV, photon_ONIR, photon_IR: RT_PHOTOELECTRIC, RT_NUV, RT_OPTICAL_NIR,
   RT_INFRARED), with the dust temperature solved where the IR band is. The model's processes are every exchange
   between the bands, the gas and the dust; the RT kick and drift transport the bands only (rt_update_driftkick under
   JACO). Every band GIZMO evolves must be one the model solves, and vice versa. Not supported: He photoionization and
   further ionizing bands, a separate Lyman-Werner band, X-ray and free-free bands, ray-based and intensity solvers, the
   dust-only cooling switch, the nuclear-zoom routing. */
#if defined(RADTRANSFER) && !defined(JACO_HAS_VAR_x_photon_EUV)
#error "JACO with RADTRANSFER needs a model whose bands are solve variables (JACO=starforge_legacy_RT or starforge_legacy_RT_EUV)"
#endif
#if defined(JACO_HAS_VAR_x_photon_EUV) && !(defined(RADTRANSFER) && defined(RT_CHEM_PHOTOION))
#error "a jaco model with radiation bands needs M1 RADTRANSFER with RT_CHEM_PHOTOION"
#endif
#if defined(RT_PHOTOELECTRIC) != defined(JACO_HAS_VAR_x_photon_FUV) || defined(RT_NUV) != defined(JACO_HAS_VAR_x_photon_NUV) || \
    defined(RT_OPTICAL_NIR) != defined(JACO_HAS_VAR_x_photon_ONIR) || defined(RT_INFRARED) != defined(JACO_HAS_VAR_x_photon_IR)
#error "the RT bands GIZMO evolves (RT_PHOTOELECTRIC, RT_NUV, RT_OPTICAL_NIR, RT_INFRARED) must be the jaco model's: starforge_legacy_RT has all four, starforge_legacy_RT_EUV none"
#endif
#if defined(RT_INFRARED) && !defined(JACO_HAS_VAR_Td)
#error "JACO with RT_INFRARED needs a model that solves the dust temperature"
#endif
#if defined(RADTRANSFER) && defined(TRANSPORT_SUBCYCLE) && !defined(TRANSPORT_SUBCYCLE_COOLING)
#error "JACO with TRANSPORT_SUBCYCLE needs TRANSPORT_SUBCYCLE_COOLING: the bands would stream through the gas unabsorbed for the whole subcycle"
#endif
#if defined(RT_CHEM_PHOTOION_HE) || defined(RT_PHOTOION_MULTIFREQUENCY) || defined(RT_LYMAN_WERNER) || defined(RT_SOFT_XRAY) || \
    defined(RT_HARD_XRAY) || defined(RT_FREEFREE) || defined(GALSF_FB_FIRE_RT_LONGRANGE) || defined(RT_LEBRON) || \
    defined(RT_EVOLVE_INTENSITIES) || defined(RT_DIFFUSION_CG) || defined(RT_COOLING_DUST_ONLY) || defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM)
#error "JACO does not support He photoionization, multifrequency ionizing bands, RT_LYMAN_WERNER, X-ray or free-free bands, LEBRON, RT intensities, RT_DIFFUSION_CG, RT_COOLING_DUST_ONLY or SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM"
#endif

/* The generated header defines JACO_HAS_VAR_<name> / JACO_HAS_PARAM_<name> for every SolveVars / Params field, so each
   quantity is filled once below under #ifdef. Models that solve the H and He ions get their seeds from the CIE table and
   the cell's Ne. */
#if defined(JACO_HAS_VAR_x_Hplus) && defined(JACO_HAS_VAR_x_Heplus) && defined(JACO_HAS_VAR_x_Heplusplus)
#define JACO_SOLVES_IONS
#ifndef JACO_HAS_PARAM_y
#error "a jaco model that solves the He ions must take the He abundance y"
#endif
#endif

/* Under RT_CHEM_PHOTOION the cell carries HII, which the RT opacity reads: the model's time-dependent H+ starts each step
   there and is written back to it */
#ifdef RT_CHEM_PHOTOION
#if !defined(JACO_VAR_TIME_DEPENDENT_x_Hplus) || !defined(JACO_SOLVES_IONS)
#error "JACO with RT_CHEM_PHOTOION needs a model that solves the ions with H+ time-dependent"
#endif
#define JACO_TRACKS_HII
#endif

#ifdef JACO_DEBUG_PARAMS
#include <stdint.h>
/* Debug builds start every Params field as a NaN, so a field the packer leaves unset is reported (jaco_check_params) or
   poisons the EOS. A quiet-NaN bit pattern, so -ffast-math cannot fold it away. */
static void jaco_poison_params(Params *pr) {
    const uint64_t qnan = 0x7ff8deadbeef0000ULL;
    for (int k = 0; k < N_PARAMS; k++)
        memcpy(&pr->data[k], &qnan, sizeof(qnan));
}
static void jaco_check_params(const Params *pr, const char *where) {
    static const char *names[N_PARAMS] = JACO_PARAM_NAMES;
    for (int k = 0; k < N_PARAMS; k++)
        if (!jaco_isfinite(pr->data[k])) {
            printf("JACO DEBUG: Params.%s is unset or not finite after %s\n", names[k], where);
            endrun(779);
        }
}
#endif

/* ---- CIE lookup table for initial ion abundance guesses ---- */

#define CIE_TABLE_N 200
#define CIE_TABLE_LOG_TMIN 2.5 /* 316 K */
#define CIE_TABLE_LOG_TMAX 6.5 /* 3.16e6 K */
static double cie_log_T[CIE_TABLE_N];
static double cie_xHp[CIE_TABLE_N];
static double cie_xHep[CIE_TABLE_N];
static double cie_xHepp[CIE_TABLE_N];
static int cie_table_initialized = 0;

/* Interpolate from the CIE table in log T */
static double cie_interp(const double *table, double logT) {
    double f = (logT - CIE_TABLE_LOG_TMIN) / (CIE_TABLE_LOG_TMAX - CIE_TABLE_LOG_TMIN) * (CIE_TABLE_N - 1);
    if (f <= 0)
        return table[0];
    if (f >= CIE_TABLE_N - 1)
        return table[CIE_TABLE_N - 1];
    int i = (int)f;
    double t = f - i;
    return (1 - t) * table[i] + t * table[i + 1];
}

/* Free electrons per H in CIE at temperature T (from the table, at its fixed He abundance) */
double jaco_cie_electron_abundance(double T) {
    double logT = log10(DMAX(T, 10.));
    return cie_interp(cie_xHp, logT) + cie_interp(cie_xHep, logT) + 2.0 * cie_interp(cie_xHepp, logT);
}

/* Build the CIE table by sweeping T downward with continuation, solving the chemistry at each
   fixed T with the solver's fixed-T chemistry solve. Called once from InitCool(). */
void jaco_build_cie_table(void) {
    if (cie_table_initialized)
        return;

    /* Default params for CIE (low density, no radiation) */
    struct JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    Params pr = {};
    pr.n_Htot = 1.0;
#ifdef JACO_HAS_PARAM_Delta_t
    pr.Delta_t = 1e15;
#endif
#ifdef JACO_HAS_PARAM_y
    pr.y = 0.0994;
#endif
#ifdef JACO_HAS_PARAM_ISRF
    pr.ISRF = 1.0;
#endif
#ifdef JACO_HAS_PARAM_N_H
    pr.N_H = 1e20;
#endif
#ifdef JACO_HAS_PARAM_G_0
    pr.G_0 = 1.0;
#endif
#ifdef JACO_HAS_PARAM_G_LW
    pr.G_LW = 1.0;
#endif
#ifdef JACO_HAS_PARAM_Td
    pr.Td = 15.0;
#endif
#ifdef JACO_HAS_PARAM_Z_d
    pr.Z_d = 1.0;
#endif
#ifdef JACO_HAS_PARAM_f_d
    pr.f_d = 1.0;
#endif
#ifdef JACO_HAS_PARAM_grad_v
    pr.grad_v = 1e-14;
#endif
#ifdef JACO_HAS_PARAM_Delta_x
    pr.Delta_x = 3e18;
#endif
#ifdef JACO_HAS_PARAM_X
    pr.X = 0.7155;
#endif
#ifdef JACO_HAS_PARAM_x_C_tot
    pr.x_C_tot = 2.1e-4;
#endif
#ifdef JACO_HAS_PARAM_x_N
    pr.x_N = 6.8e-5;
#endif
#ifdef JACO_HAS_PARAM_x_Ne
    pr.x_Ne = 8.5e-5;
#endif
#ifdef JACO_HAS_PARAM_x_Mg
    pr.x_Mg = 3.2e-5;
#endif
#ifdef JACO_HAS_PARAM_x_Si
    pr.x_Si = 3.2e-5;
#endif
#ifdef JACO_HAS_PARAM_x_S
    pr.x_S = 1.3e-5;
#endif
#ifdef JACO_HAS_PARAM_x_Ca
    pr.x_Ca = 2.2e-6;
#endif
#ifdef JACO_HAS_PARAM_x_Fe
    pr.x_Fe = 2.5e-5;
#endif
#ifdef JACO_HAS_PARAM_x_O_tot
    pr.x_O_tot = 4.9e-4;
#endif
#ifdef JACO_HAS_PARAM_rsol
    pr.rsol = 0; /* radiation-free: the bands exchange nothing with the gas and stay empty (over Delta_t = 1e30 the dust's
                    emission would otherwise pile up in them) */
    pr.hnu_EUV = 20.;
#endif
#ifdef JACO_HAS_PARAM_T_rad
    pr.T_rad = pr.T_CMB = 2.73;
#ifdef JACO_HAS_PARAM_Td_initial
    pr.Td_initial = 20.;
#endif
    pr.rho = 2.3e-24; /* n_Htot = 1 */
    pr.Z_metals = 0.014;
    pr.gamma_eos = 5. / 3.;
#endif
#ifdef JACO_VAR_TIME_DEPENDENT_x_Hplus
    pr.Delta_t = 1e30; /* the table is the steady state: no pull towards the continuation's start */
#endif

#ifdef JACO_SOLVES_IONS
    /* Sweep with continuation, from HIGH T to LOW T. The (0.9, 0.01, 0.01)
       guess is in the basin of attraction of the physical solution at high T;
       sweeping downward, the previous (mostly-ionized) solution stays in the
       physical basin as T decreases. Sweeping upward from low T tends to fall
       into the trivial all-zero fixed point. */
    double xHp = 0.99, xHep = 1e-4, xHepp = 0.099;
    for (int i = CIE_TABLE_N - 1; i >= 0; i--) {
        double logT = CIE_TABLE_LOG_TMIN + (CIE_TABLE_LOG_TMAX - CIE_TABLE_LOG_TMIN) * i / (CIE_TABLE_N - 1);
        double T = pow(10.0, logT);
        cie_log_T[i] = logT;

        SolveVars sv = {};
        sv.T = T;
        sv.x_Hplus = xHp;
        sv.x_Heplus = xHep;
        sv.x_Heplusplus = xHepp;
#ifdef JACO_HAS_VAR_x_H_2
        sv.x_H_2 = 1e-20;
#endif
        sv.u = jaco_T_to_u(T, &sv, &pr, NULL);
#ifdef JACO_HAS_PARAM_u_initial
        pr.u_initial = sv.u;
#endif
        jaco_initial_from_state(&sv, &pr);

        int nfeval = 0;
        if (jaco_solve_chemistry(&sv, &pr, &set, &nfeval)) {
            printf("jaco_build_cie_table: chemistry failed at logT=%g (T=%g)\n", logT, T);
            jaco_print_state(stdout, "  state:", &sv, &pr);
            endrun(11);
        }

        cie_xHp[i] = sv.x_Hplus;
        cie_xHep[i] = sv.x_Heplus;
        cie_xHepp[i] = sv.x_Heplusplus;
        /* Continuation */
        xHp = sv.x_Hplus;
        xHep = sv.x_Heplus;
        xHepp = sv.x_Heplusplus;
    }
    if (ThisTask == 0) {
        printf("JACO: built CIE lookup table (%d points, logT %.1f–%.1f)\n", CIE_TABLE_N, CIE_TABLE_LOG_TMIN,
               CIE_TABLE_LOG_TMAX);
        FILE *fd = fopen("jaco_cie_table.dat", "w");
        if (fd) {
            fprintf(fd, "# logT x_Hplus x_Heplus x_Heplusplus\n");
            for (int i = 0; i < CIE_TABLE_N; i++)
                fprintf(fd, "%.6e %.6e %.6e %.6e\n", cie_log_T[i], cie_xHp[i], cie_xHep[i], cie_xHepp[i]);
            fclose(fd);
            printf("JACO: CIE table written to jaco_cie_table.dat\n");
        }
    }
#endif /* JACO_SOLVES_IONS */
    cie_table_initialized = 1;
}

/* ---- GIZMO interface layer ---- */

/* Mass per H nucleus (g) in jaco's EOS for the composition in pr. Species masses conserve every element,
   electrons included, so jaco's mass density is n_Htot times this whatever the ionization and H2 state.
   With no H2 every species carries (3/2)kT, so rho = (3/2) P/u. */
static double jaco_mass_per_H(const Params *pr_in) {
    Params pr = *pr_in;
    pr.n_Htot = 1.0;
    SolveVars s = {};
    s.T = 1e4;
#ifdef JACO_SOLVES_IONS
    s.x_Hplus = 1e-4; /* any value; nonzero keeps the H- expression away from 0/0 */
    s.x_Heplus = s.x_Heplusplus = JACO_ABUNDANCE_FLOOR;
#endif
    return 1.5 * jaco_eos_pressure(&s, &pr) / jaco_T_to_u(s.T, &s, &pr, NULL);
}

/* H mass fraction of cell i and its n_He/n_H (*y), by jaco_mass_fractions */
static inline double jaco_cell_X_H(int i, struct particle_data *pp, double *y) {
    double Z_metals = 0, Y_tracked = 0, Y, Z;
    int he_tracked = 0;
#ifdef METALS
    Z_metals = pp[i].Metallicity[0];
#if (NUM_METAL_SPECIES >= 10)
    Y_tracked = pp[i].Metallicity[1];
    he_tracked = 1;
#endif
#endif
    double X = jaco_mass_fractions(Z_metals, Y_tracked, he_tracked, HYDROGEN_MASSFRAC, &Y, &Z);
    *y = 0.25 * Y / X;
    return X;
}

/* Fill the cell-dependent Params: everything the generated EOS (jaco_eos.cc) reads, including the column that
   shields C+ and so sets the free electrons, plus the other cell properties the rates need, except the radiation
   inputs only the rates read (G_LW, Td: gizmo_to_jaco). The solver and jaco_cell_eos both pack
   through here, so they evaluate the same EOS. rho_cgs is the physical gas density; returns jaco's mass
   density for these Params. */
static double jaco_pack_params(int i, Params *pr, struct particle_data *pp, struct gas_cell_data *cell, double rho_cgs) {
    pr->n_Htot = HYDROGEN_MASSFRAC * rho_cgs / PROTONMASS_CGS;
#ifdef JACO_HAS_PARAM_x_H
    pr->x_H = 1.; /* all H atomic */
#endif

    /* Hydrogen mass fraction and helium abundance by number */
    double y_He, X_H = jaco_cell_X_H(i, pp, &y_He);
    (void)X_H;
    (void)y_He;
#ifdef JACO_HAS_PARAM_y
    pr->y = y_He;
#endif
#ifdef JACO_HAS_PARAM_X
    pr->X = X_H;
#endif

    /* Metal abundances (per H nucleus) from the metallicity array: C, N, O, Ne, Mg, Si, S, Ca, Fe */
    double Z_solar = 0; /* without METALS: no metals, and dust at solar, as gas_dust_heating_coeff assumes */
#ifdef METALS
    Z_solar = All.SolarAbundances[0];
#endif
    double Z_met = 0, x_metal[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    (void)Z_solar;
    (void)Z_met;
    (void)x_metal;
#ifdef METALS
    Z_met = pp[i].Metallicity[0];
    if (NUM_METAL_SPECIES >= 10) {
        /* Metallicity indices: [0]=Z, [1]=He, [2]=C, [3]=N, [4]=O, [5]=Ne, [6]=Mg, [7]=Si, [8]=S, [9]=Ca, [10]=Fe
           Convert mass fractions to number abundances per H: x_s = (X_s / m_s) / (X_H / m_H) = (X_s / m_s_amu) * (1 /
           X_H) */
        static const double mass_number[9] = {12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 56.0};
        double inv_XH = 1.0 / X_H;
        for (int k = 0; k < 9; k++)
            x_metal[k] = pp[i].Metallicity[2 + k] / mass_number[k] * inv_XH;
    } else {
        /* Scale all elemental abundances from total metallicity assuming solar ratios */
        static const double x_solar[9] = {2.1e-4, 6.8e-5, 4.9e-4, 8.5e-5, 3.2e-5, 3.2e-5, 1.3e-5, 2.2e-6, 2.5e-5};
        double Zfac = (Z_solar > 0) ? Z_met / Z_solar : 0;
        for (int k = 0; k < 9; k++)
            x_metal[k] = x_solar[k] * Zfac;
    }
#endif
#ifdef JACO_HAS_PARAM_x_C_tot
    pr->x_C_tot = x_metal[0];
#endif
#ifdef JACO_HAS_PARAM_x_N
    pr->x_N = x_metal[1];
#endif
#ifdef JACO_HAS_PARAM_x_O_tot
    pr->x_O_tot = x_metal[2];
#endif
#ifdef JACO_HAS_PARAM_x_Ne
    pr->x_Ne = x_metal[3];
#endif
#ifdef JACO_HAS_PARAM_x_Mg
    pr->x_Mg = x_metal[4];
#endif
#ifdef JACO_HAS_PARAM_x_Si
    pr->x_Si = x_metal[5];
#endif
#ifdef JACO_HAS_PARAM_x_S
    pr->x_S = x_metal[6];
#endif
#ifdef JACO_HAS_PARAM_x_Ca
    pr->x_Ca = x_metal[7];
#endif
#ifdef JACO_HAS_PARAM_x_Fe
    pr->x_Fe = x_metal[8];
#endif

    /* Dust: solar-normalized dust abundance and sublimation factor */
#ifdef JACO_HAS_PARAM_Z_d
    double Zd_solar = (Z_solar > 0) ? Z_met / Z_solar : 1.0;
    pr->Z_d = DMAX(1e-4, Zd_solar);
#endif
#ifdef JACO_HAS_PARAM_f_d
    pr->f_d = 1.0; /* no sublimation correction for now */
#endif

    /* Kim+23 nebular forbidden-line cooling of photoionized gas: on exactly where the standard module applies it */
#ifdef JACO_HAS_PARAM_f_neb
#if defined(RT_CHEM_PHOTOION) && defined(METALS)
    pr->f_neb = 1.0;
#else
    pr->f_neb = 0.0;
#endif
#endif
#ifdef JACO_HAS_PARAM_f_metal
    pr->f_metal = jaco_metal_line_switch(); /* tabulated metal lines: on where the standard module applies them */
#endif

    /* Radiation: G_0 sets the C+ fraction and so enters the EOS; G_LW and Td only enter the rates (gizmo_to_jaco).
       Cosmic rays scale with sqrt(ISRF), as in Get_CosmicRayEnergyDensity_cgs. */
#ifdef JACO_HAS_PARAM_G_0
    jaco_radiation_inputs(i, cell[i].Temperature, &pr->G_0, NULL, NULL, pp, cell);
#endif
#ifdef JACO_HAS_PARAM_N_H
    pr->N_H = evaluate_NH_from_GradRho(pp[i].GradRho, pp[i].KernelRadius, cell[i].Density, pp[i].NumNgb, 1, i, pp) *
              UNIT_SURFDEN_IN_CGS / PROTONMASS_CGS; /* column for shielding, in nucleons */
#endif
#ifdef JACO_HAS_PARAM_ISRF
    pr->ISRF = 1.0;
#ifdef RT_ISRF_BACKGROUND
    pr->ISRF = All.InterstellarRadiationFieldStrength;
#endif
#endif

    /* Cell size and velocity gradient */
#ifdef JACO_HAS_PARAM_Delta_x
    double dx_code = pp[i].Get_Particle_Size() * All.cf_atime; /* physical cell size in code units */
    pr->Delta_x = dx_code * UNIT_LENGTH_IN_CGS;
#endif
#ifdef JACO_HAS_PARAM_grad_v
    double grad_v = cell[i].velocity_gradient_norm(); /* velocity gradient Frobenius norm in code units */
    if (!jaco_isfinite(grad_v))
        grad_v = 0; /* gradients are not yet computed at the first EOS call after reading the ICs */
    pr->grad_v =
        DMAX(1e-30, grad_v * UNIT_VEL_IN_CGS /
                        UNIT_LENGTH_IN_CGS); /* CGS s^-1, floored to avoid division by zero in LVG expressions */
#endif

    /* Cosmological redshift for inverse Compton cooling */
#ifdef JACO_HAS_PARAM_z
    pr->z = All.ComovingIntegrationOn ? (1.0 / All.Time - 1.0) : 0;
#endif

    double mass_per_H = jaco_mass_per_H(pr);
#ifdef JACO_FAMILY_STARFORGE
    /* the starforge EOS carries the full composition (X, y, metals), so jaco's mass density can match the cell's */
    pr->n_Htot = rho_cgs / mass_per_H;
#endif
    return pr->n_Htot * mass_per_H;
}

#ifdef JACO_SOLVES_IONS
/* CIE ions at temperature T, clamped to the H not in H2 (sv->x_H_2 must be set) and to the cell's He */
static void jaco_cie_species(double T, const Params *pr, SolveVars *sv) {
    double logT = log10(jaco_isfinite(T) ? DMAX(T, 10.) : 1e4), xHp_max = 1.0;
#ifdef JACO_HAS_VAR_x_H_2
    xHp_max -= 2.0 * sv->x_H_2;
#endif
    jaco_cie_ions_for_cell(cie_interp(cie_xHp, logT), cie_interp(cie_xHep, logT), cie_interp(cie_xHepp, logT), xHp_max, pr->y,
                           JACO_ABUNDANCE_FLOOR, &sv->x_Hplus, &sv->x_Heplus, &sv->x_Heplusplus);
}
#endif

#ifdef JACO_SOLVES_IONS
/* Ion electrons ne_ions split onto H+ first (up to xHp_max; or xHp_fixed if >= 0), then He+ (up to y), then He+
   converts to He++ */
static void jaco_split_ion_electrons(double ne_ions, double xHp_max, double y, double xHp_fixed, SolveVars *sv) {
    double xHp = (xHp_fixed >= 0) ? DMIN(xHp_fixed, ne_ions) : DMIN(ne_ions, xHp_max), ne_He = DMAX(ne_ions - xHp, 0);
    double xHepp = DMIN(DMAX(ne_He - y, 0), y), xHep = DMAX(DMIN(ne_He - 2.0 * xHepp, y - xHepp), 0);
    sv->x_Hplus = DMAX(JACO_ABUNDANCE_FLOOR, xHp);
    sv->x_Heplus = DMAX(JACO_ABUNDANCE_FLOOR, xHep);
    sv->x_Heplusplus = DMAX(JACO_ABUNDANCE_FLOOR, xHepp);
}
#endif

#ifdef JACO_TRACKS_HII
/* The cell's HII within the jaco floor and what H2 leaves */
static double jaco_cell_HII(int i, struct gas_cell_data *cell, double xHp_max) {
    double x = cell[i].HII;
    return jaco_isfinite(x) ? DMAX(JACO_ABUNDANCE_FLOOR, DMIN(x, xHp_max)) : JACO_ABUNDANCE_FLOOR;
}
#endif

/* Species from the cell's cached Ne (all free electrons per H) and MolecularMassFraction. The electrons the model
   puts on fixed species (metals), at the cached temperature, are taken out first; they depend only weakly on the
   ions, so two passes settle the split. jaco's u(T) and P(T) depend on the composition only through x_e and x_H2,
   so any split of Ne gives the same EOS. An ion share outside [0, 1 - 2 x_H2 + 2 y] cannot be split; CIE at the
   cached temperature is used instead. */
static void jaco_species_from_cell(int i, struct gas_cell_data *cell, const Params *pr, SolveVars *sv) {
#ifdef JACO_SOLVES_IONS
    double xH2 = 0;
#ifdef JACO_HAS_VAR_x_H_2
    double fmol = cell[i].MolecularMassFraction;
    fmol = jaco_isfinite(fmol) ? DMIN(DMAX(fmol, 0), 1.0) : 0;
    xH2 = 0.5 * fmol;
    sv->x_H_2 = DMAX(JACO_ABUNDANCE_FLOOR, xH2);
#endif
    double ne = cell[i].Ne, y = pr->y, xHp_max = 1.0 - 2.0 * xH2, xHp_fixed = -1;
#ifdef JACO_TRACKS_HII
    xHp_fixed = jaco_cell_HII(i, cell, xHp_max);
#endif
    if (jaco_isfinite(ne) && ne >= 0) {
        SolveVars s = *sv;
        s.T = (jaco_isfinite(cell[i].Temperature) && cell[i].Temperature > 0) ? cell[i].Temperature : 1e4;
        double ne_ions = ne;
        for (int pass = 0; pass < 2; pass++) {
            jaco_split_ion_electrons(ne_ions, xHp_max, y, xHp_fixed, &s);
            ne_ions = DMAX(ne - jaco_fixed_electron_abundance(&s, pr), 0);
        }
        if (ne_ions <= (xHp_max + 2.0 * y) * (1.0 + 1e-10)) {
            jaco_split_ion_electrons(ne_ions, xHp_max, y, xHp_fixed, sv);
            return;
        }
    }
    jaco_cie_species(cell[i].Temperature, pr, sv);
#endif
}

/* T with jaco_T_to_u(T) = u at the composition in sv, by Newton from T_seed (falling back to jaco_u_to_T for an
   unusable seed), to near round-off so the solver's energy row starts there. Returns T_seed itself when it
   already reproduces u, so a solver temperature normally passes through unchanged. *cv is du/dT at the
   returned T. */
static double jaco_T_from_u(double u, double T_seed, const SolveVars *sv, const Params *pr, double *cv) {
    double T = T_seed, T_lo = 1e-3, T_hi = 1e12;
    if (!jaco_isfinite(T) || T <= 0 || T > 1e10)
        T = jaco_u_to_T(u, sv, pr);
    for (int iter = 0; iter < 100; iter++) {
        double du = jaco_T_to_u(T, sv, pr, cv) - u;
        if (fabs(du) <= 1e-13 * fabs(u))
            return T;
        if (du > 0)
            T_hi = fmin(T_hi, T);
        else
            T_lo = fmax(T_lo, T);
        double T_new = T - du / (*cv);
        if (!(T_new > T_lo && T_new < T_hi))
            T_new = sqrt(T_lo * T_hi);
        T = T_new;
    }
    printf("jaco_T_from_u failed to converge: u=%g T_seed=%g T=%g\n", u, T_seed, T);
    T = jaco_u_to_T(u, sv, pr);
    jaco_T_to_u(T, sv, pr, cv);
    return T;
}

#ifdef JACO_SOLVES_IONS
/* T with u = jaco_T_to_u(T) at CIE ions at T (sv->x_H_2 kept), by bisection in log T, since u rises with both T and
   the ionization; leaves those ions in sv and returns T refined at them to round-off */
static double jaco_cie_T_from_u(double u, const Params *pr, SolveVars *sv) {
    double lo = 0, hi = log(1e10), cv; /* 1 K to 1e10 K */
    for (int iter = 0; iter < 60 && hi - lo > 1e-10; iter++) {
        double lnT = 0.5 * (lo + hi);
        jaco_cie_species(exp(lnT), pr, sv);
        if (jaco_T_to_u(exp(lnT), sv, pr, NULL) > u) hi = lnT; else lo = lnT;
    }
    double T = exp(0.5 * (lo + hi));
    jaco_cie_species(T, pr, sv);
    return jaco_T_from_u(u, T, sv, pr, &cv);
}

/* CIE ions in *cie (sv's H2 kept) and the temperature *T_cie to take with them; returns 1 where they show cell i's
   cached free electrons stale (jaco_ions_stale). CIE is taken at the cached temperature, except in a cell whose u was
   raised directly since its last solve (JacoReheated): the cached temperature predates that heating, so CIE is taken
   at the temperature where it reproduces u. */
static int jaco_stale_ions(int i, double u, struct gas_cell_data *cell, const Params *pr, const SolveVars *sv, SolveVars *cie,
                           double *T_cie) {
    *cie = *sv;
    if (cell[i].JacoReheated) {
        *T_cie = jaco_cie_T_from_u(u, pr, cie);
    } else {
        *T_cie = cell[i].Temperature;
        jaco_cie_species(*T_cie, pr, cie);
    }
    return jaco_ions_stale(cell[i].Ne, cie->x_Hplus + cie->x_Heplus + 2.0 * cie->x_Heplusplus);
}
#endif

/* jaco's EOS on the cell's cached state: composition from Ne, MolecularMassFraction and metallicity, at
   specific energy u and physical density rho (both code units), seeded with the cached Temperature. In a cell
   heated directly since its last solve, stale ions are replaced by CIE as the solver's seed replaces them
   (jaco_stale_ions). Uses only (i, pp, cell), so it is safe on the packed copies cooling works on. */
void jaco_cell_eos(int i, struct particle_data *pp, struct gas_cell_data *cell, double u, double rho,
                   struct jaco_eos_state *eos) {
    SolveVars sv = {};
    Params pr = {};
#ifdef JACO_DEBUG_PARAMS
    jaco_poison_params(&pr); /* the EOS reads a subset of Params; an unpacked field it reads makes the state NaN */
#endif
    double rho_jaco = jaco_pack_params(i, &pr, pp, cell, rho * UNIT_DENSITY_IN_CGS), cv, T_seed = cell[i].Temperature;
    jaco_species_from_cell(i, cell, &pr, &sv);
    sv.u = u * UNIT_SPECEGY_IN_CGS;
#ifdef JACO_SOLVES_IONS
    SolveVars cie;
    double T_cie;
    if (cell[i].JacoReheated && jaco_stale_ions(i, sv.u, cell, &pr, &sv, &cie, &T_cie)) {
        sv = cie;
        T_seed = T_cie;
    }
#endif
    sv.T = jaco_T_from_u(sv.u, T_seed, &sv, &pr, &cv);
    eos->T = sv.T;
    eos->P_over_rho = jaco_eos_pressure(&sv, &pr) / rho_jaco;
    eos->gamma = 1.0 + eos->P_over_rho / (cv * sv.T); /* first adiabatic index at frozen composition */
#ifdef JACO_DEBUG_PARAMS
    if (!jaco_isfinite(eos->T) || !jaco_isfinite(eos->P_over_rho) || !jaco_isfinite(eos->gamma)) {
        printf("JACO DEBUG: jaco_cell_eos is not finite (T=%g P/rho=%g gamma=%g); the EOS reads an unpacked Params field\n",
               eos->T, eos->P_over_rho, eos->gamma);
        endrun(779);
    }
#endif
#ifdef JACO_SOLVES_IONS
    eos->x_Hplus = sv.x_Hplus;
    eos->x_Heplus = sv.x_Heplus;
    eos->x_Heplusplus = sv.x_Heplusplus;
    eos->y = pr.y;
#else
    eos->x_Hplus = eos->x_Heplus = eos->x_Heplusplus = eos->y = 0;
#endif
    eos->x_e = jaco_electron_abundance(&sv, &pr);
}

#ifdef JACO_HAS_VAR_x_photon_EUV
static const int jaco_var_initial_param[N_VARS] = JACO_VAR_INITIAL_PARAM_INIT;
#define JACO_PARAM_OF_INITIAL(k) (jaco_var_initial_param[k])
#ifdef JACO_HAS_VAR_Td
static const double jaco_var_floor[N_VARS] = JACO_VAR_FLOOR_INIT, jaco_var_ceiling[N_VARS] = JACO_VAR_CEILING_INIT;
#define JACO_TD_FLOOR (jaco_var_floor[IDX_Td])
#define JACO_TD_CEILING (jaco_var_ceiling[IDX_Td])
#endif
/* The bands the model solves: GIZMO's bin, the solve variable (per H nucleus), and its unit, photons (the ionizing band,
   counted at rt_nu_eff_eV as GIZMO counts them) or eV */
struct jaco_band {
    int bin, var, photons;
};
static const struct jaco_band jaco_bands[] = {
    {RT_FREQ_BIN_H0, IDX_x_photon_EUV, 1},
#ifdef JACO_HAS_VAR_x_photon_FUV
    {RT_FREQ_BIN_PHOTOELECTRIC, IDX_x_photon_FUV, 0},
    {RT_FREQ_BIN_NUV, IDX_x_photon_NUV, 0},
    {RT_FREQ_BIN_OPTICAL_NIR, IDX_x_photon_ONIR, 0},
    {RT_FREQ_BIN_INFRARED, IDX_x_photon_IR, 0},
#endif
};
#define JACO_N_BANDS ((int)(sizeof(jaco_bands) / sizeof(jaco_bands[0])))

/* code energy of band b per unit of its solve variable, in cell i with the model's n_Htot */
static double jaco_band_unit(const struct jaco_band *b, int i, const Params *pr, struct gas_cell_data *cell) {
    double L = UNIT_LENGTH_IN_CGS, volume_cgs = cell[i].Mass / (cell[i].Density * All.cf_a3inv) * L * L * L;
    double eV_per_unit = b->photons ? rt_nu_eff_eV[b->bin] : 1.;
    return volume_cgs * pr->n_Htot * eV_per_unit * ELECTRONVOLT_IN_ERGS / UNIT_ENERGY_IN_CGS;
}

/* The bands' post-transport energies as the step's initial values and seeds, the band constants, and what the
   processes read of the cell: its density, metals, adiabatic index, IR radiation temperature and dust temperature
   seed, and the UV background's share in the photoionization heating and the Lyman-Werner field */
static void jaco_pack_bands(int i, SolveVars *sv, Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    for (int k = 0; k < JACO_N_BANDS; k++) {
        double x = DMAX(cell[i].Rad_E_gamma[jaco_bands[k].bin], 0) / jaco_band_unit(&jaco_bands[k], i, pr, cell);
        sv->data[jaco_bands[k].var] = pr->data[JACO_PARAM_OF_INITIAL(jaco_bands[k].var)] = jaco_isfinite(x) ? x : 0;
    }
    const double L = UNIT_LENGTH_IN_CGS;
    pr->rsol = C_LIGHT_CODE_REDUCED / C_LIGHT_CODE;
    pr->sigma_HI = rt_ion_sigma_HI[RT_FREQ_BIN_H0] * L * L;
    pr->eps_HI = rt_ion_G_HI[RT_FREQ_BIN_H0] * UNIT_ENERGY_IN_CGS;
    pr->hnu_EUV = rt_nu_eff_eV[RT_FREQ_BIN_H0];
#ifdef JACO_HAS_VAR_Td
    double T_floor = get_min_allowed_dustIRrad_temperature(), T_cmb = 2.73 / All.cf_atime;
#ifdef RT_ISRF_BACKGROUND
    if (!All.ComovingIntegrationOn) T_cmb *= 1. + All.RadiationBackgroundRedshift;
#endif
    pr->T_CMB = T_cmb;
    pr->T_rad = DMAX(cell[i].Radiation_Temperature, T_floor);
    pr->rho = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS;
    pr->Z_metals = 0.014;
#ifdef METALS
    pr->Z_metals = pp[i].Metallicity[0];
#endif
    pr->gamma_eos = cell[i].gamma_eos_value();
    double Td = cell[i].Dust_Temperature;
    sv->Td = (jaco_isfinite(Td) && Td > 0) ? DMIN(DMAX(Td, JACO_TD_FLOOR), JACO_TD_CEILING) : DMAX(cell[i].Temperature, JACO_TD_FLOOR);
#ifdef JACO_HAS_PARAM_Td_initial
    pr->Td_initial = sv->Td; /* the dust-absorbed bands' opacity, as the kick takes it */
#endif
    jaco_uvb_inputs(i, cell[i].Temperature, &pr->G_LW_bg, &pr->gamma_12_UVB, &pr->eps_H0_UVB, cell);
#endif
}

/* The solved bands into the cell: energies (the drifted copies synchronized), and fluxes held to the M1 limit
   |F| <= c_tilde E, as the kick does. The other bands' flux is otherwise left to the kick's relaxation with Rad_Kappa,
   as GIZMO's kick absorption leaves it (scaling it with the absorbed energy as well starves dense gas of the
   photoelectric band); the ionizing band's is scaled with its loss to the photoionizations, which reproduces the
   standard module's I-front (without it the front leads by ~5% early on). Then the emission back-reaction on the gas
   velocity as GIZMO's cooling return applies it, and, with the IR band, the dust temperature and the band's radiation
   temperature after the step. Accounting into acc (see jaco_report_solve_stats). */
static void jaco_write_bands(int i, const SolveVars *sv, const Params *pr, const Outputs *out, struct particle_data *pp,
                             struct gas_cell_data *cell, struct jaco_step_outputs *acc) {
    double dE_total = 0;
    for (int k = 0; k < JACO_N_BANDS; k++) {
        const int bin = jaco_bands[k].bin;
        double E_old = DMAX(cell[i].Rad_E_gamma[bin], 0), E_new = sv->data[jaco_bands[k].var] * jaco_band_unit(&jaco_bands[k], i, pr, cell);
        if (!jaco_isfinite(E_new) || E_new < 0) {
            acc->band_clipped++;
            E_new = jaco_isfinite(E_new) ? 0 : E_old;
        }
        double dE = E_new - E_old;
        if (dE < 0) acc->band_removed -= dE; else acc->band_added += dE;
        if (bin == RT_FREQ_BIN_H0) acc->euv_loss -= dE;
        dE_total += dE;
        cell[i].Rad_E_gamma[bin] = cell[i].Rad_E_gamma_Pred[bin] = E_new;
        if (bin == RT_FREQ_BIN_H0 && E_new < E_old) { /* the photoionizations take the photons with their flux */
            double fac = E_new / E_old;
            cell[i].Rad_Flux[bin] *= fac;
            cell[i].Rad_Flux_Pred[bin] *= fac;
        }
        double fmax = C_LIGHT_CODE_REDUCED * E_new, f = cell[i].Rad_Flux[bin].norm(), fp = cell[i].Rad_Flux_Pred[bin].norm();
        if (f > fmax) cell[i].Rad_Flux[bin] *= fmax / f;
        if (fp > fmax) cell[i].Rad_Flux_Pred[bin] *= fmax / fp;
    }
    /* each photoionization takes one photon of the ionizing band, its only sink: the band's loss must be what the
       photoionizations took, to the solver's tolerance (relative, and 1e-13 photons per H nucleus absolute: ten times
       its absolute abundance tolerance) */
    const double unit_euv = jaco_band_unit(&jaco_bands[0], i, pr, cell);
    double photons = pr->rsol * out->photoionization_rate * pr->Delta_t * unit_euv / pr->n_Htot;
    acc->photoionized += photons;
    double E0_euv = pr->data[JACO_PARAM_OF_INITIAL(IDX_x_photon_EUV)] * unit_euv;
    acc->euv_open = fabs(photons - (E0_euv - DMAX(cell[i].Rad_E_gamma[RT_FREQ_BIN_H0], 0))) > 1e-5 * (E0_euv + photons) + 1e-13 * unit_euv;
    double momfac = 1. - dE_total / (cell[i].Mass * C_LIGHT_CODE * C_LIGHT_CODE_REDUCED);
    pp[i].dp += pp[i].Vel * ((momfac - 1.) * cell[i].Mass);
    pp[i].Vel *= momfac;
    cell[i].VelPred *= momfac;
#ifdef JACO_HAS_VAR_Td
    cell[i].Dust_Temperature = DMAX(sv->Td, get_min_allowed_dustIRrad_temperature());
    cell[i].Radiation_Temperature = DMAX(out->T_rad_new, get_min_allowed_dustIRrad_temperature());
#endif
}
#endif

void gizmo_to_jaco(int i, SolveVars *sv, Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    double dtime = get_particle_timestep_in_physical(i, pp);
#ifdef TRANSPORT_SUBCYCLE_COOLING
    dtime *= All.Transport_Subcycle_dt_fraction; /* cooling runs once per transport sub-step, from the sub-step's state */
#endif
    set_PdV_work_heatingrate(i, dtime, pp, cell);
#ifdef JACO_DEBUG_PARAMS
    jaco_poison_params(pr);
#endif
    jaco_pack_params(i, pr, pp, cell, cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS);
#ifdef JACO_HAS_VAR_x_photon_EUV
    jaco_pack_bands(i, sv, pr, pp, cell); /* first: the metals' free electrons the seeds split off read the bands */
#endif

    sv->u = cell[i].InternalEnergy * UNIT_SPECEGY_IN_CGS;
#ifdef JACO_HAS_PARAM_u_initial
    pr->u_initial = sv->u;
#endif
#ifdef JACO_HAS_PARAM_Delta_t
    pr->Delta_t = dtime * UNIT_TIME_IN_CGS;
#endif
    /* set_PdV_work_heatingrate leaves DtInternalEnergy in erg/s per H at HYDROGEN_MASSFRAC, so this product with
       nHcgs() (not pr->n_Htot) is rho*du/dt in erg/cm^3/s */
#ifndef COOLING_OPERATOR_SPLIT
    pr->pdv_work = (cell[i].CoolingIsOperatorSplitThisTimestep == 0) ? cell[i].DtInternalEnergy * cell[i].nHcgs() : 0;
#else
    pr->pdv_work = 0;
#endif

    /* Seed with the state jaco_cell_eos describes: species from the cached Ne and H2, and T on the EOS for u
       at that composition (normally the cached Temperature itself), so the energy row starts at round-off */
    jaco_species_from_cell(i, cell, pr, sv);
    int stale_ions = 0;
    double T_stale = cell[i].Temperature;
#ifdef JACO_SOLVES_IONS
    /* Except where the CIE table is collisionally ionized (x_e > 0.1, T >~ 1.3e4 K): there it bounds the steady state
       from below, so a cached Ne under it is stale (shock heating, or a solve that settled on the near-neutral fixed
       point of the ionization balance, where the gas stops cooling). Seed CIE ions at the cached T, keeping that T:
       re-deriving T at the ionized composition roughly halves it and strands Newton far from the ionized state. A cell
       heated directly since its last solve takes CIE at the temperature consistent with its u instead
       (jaco_stale_ions). */
    SolveVars cie; /* at the cell's He and H2: the table's own He abundance would flag every ionized cell */
    stale_ions = jaco_stale_ions(i, sv->u, cell, pr, sv, &cie, &T_stale);
    if (stale_ions)
        *sv = cie;
#endif
    double cv;
    sv->T = stale_ions ? T_stale : jaco_T_from_u(sv->u, cell[i].Temperature, sv, pr, &cv);
    jaco_initial_from_state(sv, pr); /* the time-dependent species start the step at their seeds */
#ifdef JACO_TRACKS_HII
    {
        double xHp_max = 1.0;
#ifdef JACO_HAS_PARAM_x_H_2_initial
        xHp_max -= 2.0 * pr->x_H_2_initial;
#endif
        pr->x_Hplus_initial = jaco_cell_HII(i, cell, xHp_max); /* even where the seed is CIE */
    }
#endif

#if defined(JACO_HAS_PARAM_G_LW) || defined(JACO_HAS_PARAM_Td)
    {
        /* LW field and dust temperature as the standard cooling module evaluates them, at the cached temperature */
        double *G_LW = NULL, *Td = NULL;
#ifdef JACO_HAS_PARAM_G_LW
        G_LW = &pr->G_LW;
#endif
#ifdef JACO_HAS_PARAM_Td
        Td = &pr->Td;
#endif
        jaco_radiation_inputs(i, cell[i].Temperature, NULL, G_LW, Td, pp, cell);
    }
#endif
#ifdef JACO_DEBUG_PARAMS
    jaco_check_params(pr, "gizmo_to_jaco");
#endif
}

/* Write the solved state back to cell i, the bands included, and, if out, evaluate the model's outputs there with the
   step's energy accounting. */
void jaco_to_gizmo(int i, const SolveVars *sv, const Params *pr, struct particle_data *pp, struct gas_cell_data *cell,
                   struct jaco_step_outputs *out) {
    struct jaco_step_outputs local;
    if (!out) out = &local;
    memset(out, 0, sizeof(*out));
    microphysics_outputs(sv, pr, &out->rate);
    out->dt = pr->Delta_t;
    out->gas_dE = (sv->u / UNIT_SPECEGY_IN_CGS - cell[i].InternalEnergy) * cell[i].Mass;
#ifdef JACO_HAS_VAR_x_photon_EUV
    jaco_write_bands(i, sv, pr, &out->rate, pp, cell, out);
#endif
    cell[i].InternalEnergy = sv->u / UNIT_SPECEGY_IN_CGS;
    cell[i].InternalEnergyPred = cell[i].InternalEnergy;
    cell[i].Temperature = sv->T;

    /* Write solved species back BEFORE set_eos_pressure, whose EOS composition comes from Ne and MolecularMassFraction */
#ifdef JACO_SOLVES_IONS
    cell[i].Ne = jaco_electron_abundance(sv, pr); /* all free electrons, metals' included, as the rates use them */
#endif
#ifdef JACO_TRACKS_HII
    cell[i].HII = sv->x_Hplus;
    cell[i].HI = DMAX(1.0 - sv->x_Hplus, 0); /* neutral H including H2, as DoCooling stores it */
#endif
#ifdef JACO_HAS_VAR_x_H_2
    double xHp = sv->x_Hplus, xH2 = sv->x_H_2;
    cell[i].MolecularMassFraction = 2.0 * xH2;
    double xH0 = DMAX(1.0 - xHp, 0); /* neutral H including H2, as in the standard cooling module */
    cell[i].MolecularMassFraction_perNeutralH = (xH0 > 0) ? DMIN(1, cell[i].MolecularMassFraction / xH0) : 0;
#endif
    cell[i].JacoReheated = 0;

    set_eos_pressure(i, pp, cell); /* P, Gamma and sound speed from jaco's EOS; T moves only by the solver's u(T) residual */
#ifndef COOLING_OPERATOR_SPLIT
    if (cell[i].CoolingIsOperatorSplitThisTimestep == 0) {
        cell[i].DtInternalEnergy = 0;
    }
#endif

    /* Sanity check GIZMO-side variables */
    if (!jaco_isfinite(cell[i].InternalEnergy) || !jaco_isfinite(cell[i].Pressure) ||
        !jaco_isfinite(cell[i].Temperature) || cell[i].InternalEnergy <= 0 || cell[i].Pressure <= 0 ||
        cell[i].Temperature <= 0) {
        printf("JACO FATAL: bad value after jaco_to_gizmo\n");
        printf("  InternalEnergy=%.6e Pressure=%.6e Temperature=%.6e\n", (double)cell[i].InternalEnergy,
               (double)cell[i].Pressure, (double)cell[i].Temperature);
        printf("  Ne=%.6e MolecularMassFraction=%.6e Density=%.6e\n", (double)cell[i].Ne,
               (double)cell[i].MolecularMassFraction, (double)cell[i].Density);
        printf("  SolveVars:");
        for (int k = 0; k < N_VARS; k++)
            printf(" [%d]=%.6e", k, sv->data[k]);
        printf("\n  Params:");
        for (int k = 0; k < N_PARAMS; k++)
            printf(" [%d]=%.6e", k, pr->data[k]);
        printf("\n");
        endrun(778);
    }
}

/* A spawned cell (jet, wind or SN ejecta) carries the composition of the cell it was cloned from. Start it H2-free with
   CIE ions, either at temperature T with u from jaco's EOS, or, for T <= 0, at the temperature where CIE reproduces its
   u; then cache its EOS. Call once its density and metallicity are set. */
void jaco_seed_spawned_cell(int i, double T, struct particle_data *pp, struct gas_cell_data *cell) {
    SolveVars sv = {};
    Params pr = {};
    if (T > 0) cell[i].Temperature = T; /* the radiation inputs to the C+ fraction read it */
    jaco_pack_params(i, &pr, pp, cell, cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS);
#ifdef JACO_HAS_VAR_x_H_2
    sv.x_H_2 = JACO_ABUNDANCE_FLOOR;
#endif
    cell[i].MolecularMassFraction = cell[i].MolecularMassFraction_perNeutralH = 0;
    if (T > 0) {
#ifdef JACO_SOLVES_IONS
        jaco_cie_species(T, &pr, &sv);
#endif
        cell[i].InternalEnergy = cell[i].InternalEnergyPred = jaco_T_to_u(T, &sv, &pr, NULL) / UNIT_SPECEGY_IN_CGS;
    } else {
        double u = cell[i].InternalEnergy * UNIT_SPECEGY_IN_CGS, cv;
#ifdef JACO_SOLVES_IONS
        T = jaco_cie_T_from_u(u, &pr, &sv);
        (void)cv;
#else
        T = jaco_T_from_u(u, -1, &sv, &pr, &cv);
#endif
    }
    sv.T = cell[i].Temperature = T;
#ifdef JACO_SOLVES_IONS
    cell[i].Ne = jaco_electron_abundance(&sv, &pr);
#endif
#ifdef JACO_TRACKS_HII
    cell[i].HII = sv.x_Hplus;
    cell[i].HI = DMAX(1.0 - sv.x_Hplus, 0);
#endif
    cell[i].JacoReheated = 0;
    set_eos_pressure(i, pp, cell);
}

/* ---- Per-step solver statistics ----
   Accumulated over the cells call_jaco() solves in one cooling pass; reported and reset by
   jaco_report_solve_stats(), which every rank must call (MPI collective) once per pass. */
enum { JS_CELLS, JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED, JS_FEVALS, JS_RESYNC, JS_PINNED, JS_NANJ, JS_NANF,
       JS_T1_1, JS_T1_2, JS_T1_3, JS_T1_4, JS_T1_MORE, JS_NANOUT, JS_NONCONS,
       JS_T1F_NONFINITE, JS_T1F_SINGULAR, JS_T1F_LINESEARCH, JS_T1F_BUDGET, JS_T1F_FLOOR, JS_T1F_UNSTABLE, JS_N };
static long jaco_stats[JS_N];
static int jaco_stats_max_nfeval = 0;
/* energy accounting over a cooling pass, code units: band energy removed and added by the model, the gas internal
   energy change, and the ionizing band's photons taken by photoionizations (the model's output) and lost by the band */
enum { JE_REMOVED, JE_ADDED, JE_GAS, JE_PHOTOIONIZED, JE_EUV_LOSS, JE_N };
static double jaco_energy[JE_N];

static void jaco_stats_add(const struct JacoSolveInfo *info, const struct jaco_step_outputs *out) {
    int bin[4] = {JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED};
    int t1 = info->nfeval - info->nfeval_fd; /* evaluations Newton asked for, not finite-difference columns */
    int bad_out = 0;
    for (int k = 0; k < N_OUTPUTS; k++)
        if (!jaco_isfinite(out->rate.data[k])) bad_out = 1;
#ifdef _OPENMP
#pragma omp critical(jaco_stats_update)
#endif
    {
        jaco_stats[JS_CELLS]++;
        jaco_stats[bin[info->tier - 1]]++;
        jaco_stats[JS_FEVALS] += info->nfeval;
        jaco_stats[JS_RESYNC] += info->resynced_T0;
        jaco_stats[JS_PINNED] += info->pinned;
        jaco_stats[JS_NANJ] += info->n_nonfinite_jac > 0;
        jaco_stats[JS_NANF] += info->n_nonfinite_F > 0;
        if (info->tier == JACO_TIER_NEWTON) jaco_stats[t1 <= 4 ? JS_T1_1 + t1 - 1 : JS_T1_MORE]++;
        if (info->tier1_status <= -1 && info->tier1_status >= -6) jaco_stats[JS_T1F_NONFINITE - 1 - info->tier1_status]++;
        jaco_stats[JS_NANOUT] += bad_out;
        /* a subcycled answer's photoionization rate at the end of the step, times the step, is not its integral */
        jaco_stats[JS_NONCONS] += out->band_clipped > 0 || (out->euv_open && info->tier != JACO_TIER_SUBCYCLE);
        if (info->nfeval > jaco_stats_max_nfeval) jaco_stats_max_nfeval = info->nfeval;
        jaco_energy[JE_REMOVED] += out->band_removed;
        jaco_energy[JE_ADDED] += out->band_added;
        jaco_energy[JE_GAS] += out->gas_dE;
        jaco_energy[JE_PHOTOIONIZED] += out->photoionized;
        jaco_energy[JE_EUV_LOSS] += out->euv_loss;
    }
}

/* Called at the end of each cooling pass on every rank: prints the MPI-reduced statistics on
   rank 0 and resets the counters. */
void jaco_report_solve_stats(void) {
    long global[JS_N];
    int global_max = 0;
    double energy[JE_N];
    MPI_Reduce(jaco_stats, global, JS_N, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&jaco_stats_max_nfeval, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(jaco_energy, energy, JE_N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (ThisTask == 0 && global[JS_CELLS] > 0) {
        double n = (double)global[JS_CELLS];
        printf("jaco solve stats: %ld cells | tier1 %ld (%.2f%%) tier2 %ld tier3 %ld FAILED %ld | nfeval mean %.2f max %d | "
               "tier-1 evals 1:%ld 2:%ld 3:%ld 4:%ld >4:%ld | T0 resynced %ld, pinned %ld, non-finite J %ld F %ld outputs %ld\n",
               global[JS_CELLS], global[JS_TIER1], 100.0 * global[JS_TIER1] / n, global[JS_TIER2], global[JS_TIER3],
               global[JS_FAILED], global[JS_FEVALS] / n, global_max, global[JS_T1_1], global[JS_T1_2], global[JS_T1_3],
               global[JS_T1_4], global[JS_T1_MORE], global[JS_RESYNC], global[JS_PINNED], global[JS_NANJ], global[JS_NANF],
               global[JS_NANOUT]);
        printf("jaco tier-1 failures: non-finite %ld singular %ld line search %ld iteration budget %ld below floor %ld "
               "unstable balance root %ld\n", global[JS_T1F_NONFINITE], global[JS_T1F_SINGULAR], global[JS_T1F_LINESEARCH],
               global[JS_T1F_BUDGET], global[JS_T1F_FLOOR], global[JS_T1F_UNSTABLE]);
#ifdef JACO_HAS_VAR_x_photon_EUV
        printf("jaco band energy (code units): removed %.6e added %.6e | gas internal energy change %.6e | ionizing band: "
               "photoionized %.6e band loss %.6e | non-conserving cells %ld\n", energy[JE_REMOVED], energy[JE_ADDED],
               energy[JE_GAS], energy[JE_PHOTOIONIZED], energy[JE_EUV_LOSS], global[JS_NONCONS]);
#endif
        fflush(stdout);
    }
    memset(jaco_stats, 0, sizeof(jaco_stats));
    memset(jaco_energy, 0, sizeof(jaco_energy));
    jaco_stats_max_nfeval = 0;
}

#ifdef OUTPUT_COOLRATE_DETAIL
#if !defined(JACO_HAS_OUTPUT_heating_rate) || !defined(JACO_HAS_OUTPUT_cooling_rate) || \
    !defined(JACO_HAS_OUTPUT_metal_line_cooling_rate) || !defined(JACO_HAS_OUTPUT_photoelectric_heating_rate)
#include <stdint.h>
static const char *jaco_rates_not_output = ""
#ifndef JACO_HAS_OUTPUT_heating_rate
    " HeatingRate"
#endif
#ifndef JACO_HAS_OUTPUT_cooling_rate
    " CoolingRate"
#endif
#ifndef JACO_HAS_OUTPUT_metal_line_cooling_rate
    " MetalCoolingRate"
#endif
#ifndef JACO_HAS_OUTPUT_photoelectric_heating_rate
    " PElecHeatingRate"
#endif
    ;
/* NaN for a rate the model does not output, with a warning the first time */
static double jaco_rate_not_output(void) {
    static int warned = 0;
#ifdef _OPENMP
#pragma omp critical(jaco_coolrate_warn)
#endif
    if (!warned) {
        warned = 1;
        if (ThisTask == 0)
            printf("JACO WARNING: OUTPUT_COOLRATE_DETAIL: the model has no output for%s, written as NaN\n", jaco_rates_not_output);
    }
    const uint64_t qnan = 0x7ff8000000000000ULL; /* a bit pattern, so -ffast-math cannot fold it */
    double x;
    memcpy(&x, &qnan, sizeof(x));
    return x;
}
#endif

/* The standard module's per-cell rate outputs at the solved state, in CoolingRate()'s units (erg cm^3 s^-1, per
   nHcgs()^2): the net radiative rate from the energy row, the separated rates from the model's outputs (NaN where it
   has none). The model's heating and cooling sum each process's heat by its sign, so HeatingRate - CoolingRate =
   NetHeatingRateQ. Call before jaco_to_gizmo, which clears DtInternalEnergy. */
static void jaco_coolrate_detail(const SolveVars *sv, const Params *pr, struct gas_cell_data *cell) {
    Params p = *pr;
    p.pdv_work = 0;
    p.u_initial = sv->u; /* sv->u = u(T, x), so the backward-Euler term vanishes and the energy row is the net heating */
    SolveVars F;
    double J[N_VARS][N_VARS], nH = cell->nHcgs(), nH2 = nH * nH;
    microphysics_func_jac(sv, &p, &F, J);
    cell->NetHeatingRateQ = F.T / nH2;
#ifndef COOLING_OPERATOR_SPLIT
    cell->HydroHeatingRate = cell->DtInternalEnergy / nH;
#endif
    Outputs out;
    microphysics_outputs(sv, pr, &out);
    (void)out;
#ifdef JACO_HAS_OUTPUT_heating_rate
    cell->HeatingRate = out.heating_rate / nH2;
#else
    cell->HeatingRate = jaco_rate_not_output();
#endif
#ifdef JACO_HAS_OUTPUT_cooling_rate
    cell->CoolingRate = out.cooling_rate / nH2;
#else
    cell->CoolingRate = jaco_rate_not_output();
#endif
#ifdef JACO_HAS_OUTPUT_metal_line_cooling_rate
    cell->MetalCoolingRate = out.metal_line_cooling_rate / nH2;
#else
    cell->MetalCoolingRate = jaco_rate_not_output();
#endif
#ifdef JACO_HAS_OUTPUT_photoelectric_heating_rate
    cell->PElecHeatingRate = out.photoelectric_heating_rate / nH2;
#else
    cell->PElecHeatingRate = jaco_rate_not_output();
#endif
}
#endif

void call_jaco(struct particle_data *p, struct gas_cell_data *c) {
    double dtime = get_particle_timestep_in_physical(0, p);
    if (dtime == 0)
        return;
    SolveVars sv = {};
    Params pr = {};
    gizmo_to_jaco(0, &sv, &pr, p, c);

    struct JacoSolverSettings set;
    jaco_solver_default_settings(&set);
    set.T_min = (All.MinGasTemp > 0) ? All.MinGasTemp : 1.0;
    set.u_min = All.MinEgySpec * UNIT_SPECEGY_IN_CGS;
    struct JacoSolveInfo info;
    const SolveVars sv_in = sv;
#ifdef RT_ILIEV_TEST1
    /* Iliev et al. (2006) test 1: the chemistry at a fixed 1e4 K */
    memset(&info, 0, sizeof(info));
    sv.T = 1e4;
    info.tier = jaco_solve_chemistry(&sv, &pr, &set, &info.nfeval) ? JACO_TIER_FAILED : JACO_TIER_NEWTON;
    if (info.tier == JACO_TIER_FAILED) {
        printf("JACO FATAL: the fixed-temperature chemistry failed on task %d\n", ThisTask);
        jaco_print_state(stdout, "  input state:", &sv_in, &pr);
        endrun(10);
    }
#else
    if (jaco_solve(&sv, &pr, &set, &info)) {
        printf("JACO FATAL: every solver tier failed on task %d (nfeval=%d)\n", ThisTask, info.nfeval);
        jaco_print_state(stdout, "  input state:", &sv_in, &pr);
        set.verbose = 2; /* replay with a trace of every iterate */
        sv = sv_in;
        jaco_solve(&sv, &pr, &set, &info);
        fflush(stdout);
        endrun(10);
    }
#endif
#ifdef JACO_DUMP_SLOW
    /* JACO_DUMP_SLOW=N: each rank writes the input state of every Nth solve that tier 1 did not finish (at most 400)
       to jaco_slow_<rank>.txt, for test/jaco_solver's replay */
    if (info.tier > JACO_TIER_NEWTON) {
        static long n_slow = 0;
        static int n_dumped = 0;
#ifdef _OPENMP
#pragma omp critical(jaco_dump_slow)
#endif
        if (n_slow++ % JACO_DUMP_SLOW == 0 && n_dumped < 400) {
            char fname[64];
            snprintf(fname, sizeof(fname), "jaco_slow_%d.txt", ThisTask);
            FILE *fp = fopen(fname, n_dumped ? "a" : "w");
            if (fp) {
                fprintf(fp, "tier %d nfeval %d tier-1 status %d time %g\n", info.tier, info.nfeval, info.tier1_status, All.Time);
                jaco_print_state(fp, "  input state:", &sv_in, &pr);
                fclose(fp);
            }
            n_dumped++;
        }
    }
#endif
#ifdef OUTPUT_COOLRATE_DETAIL
    jaco_coolrate_detail(&sv, &pr, c);
#endif
    struct jaco_step_outputs out;
    jaco_to_gizmo(0, &sv, &pr, p, c, &out);
    jaco_stats_add(&info, &out);
}
#endif
