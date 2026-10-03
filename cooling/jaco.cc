/*
 * jaco.cc -- GIZMO glue for the jaco code-generated microphysics network.
 *
 * Packs a gas cell into the generated SolveVars/Params (gizmo_to_jaco), calls the implicit solver
 * (jaco_solver.cc, no GIZMO dependencies), and writes the answer back (jaco_to_gizmo). Also
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
#if defined(OUTPUT_COOLRATE) || defined(GALSF_EFFECTIVE_EQS) || defined(CHIMES) || defined(COOL_GRACKLE) || defined(RT_CHEM_PHOTOION)
#error "JACO does not yet support OUTPUT_COOLRATE, GALSF_EFFECTIVE_EQS, CHIMES, COOL_GRACKLE or RT_CHEM_PHOTOION"
#endif
/* do_the_cooling_for_particle returns right after call_jaco, so everything the standard path does afterwards is skipped:
   the cooling-radiation return to the RT bands, cosmic-ray losses, the sink thermal-feedback energy injection and the
   subcycle dt scaling. Refuse those configurations until the JACO path carries them. */
#if defined(RADTRANSFER) || defined(COSMIC_RAY_FLUID) || defined(SINK_THERMALFEEDBACK) || defined(TRANSPORT_SUBCYCLE_COOLING)
#error "JACO does not yet support RADTRANSFER, COSMIC_RAY_FLUID, SINK_THERMALFEEDBACK or TRANSPORT_SUBCYCLE_COOLING"
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
#ifdef JACO_HAS_PARAM_x_H_2_initial
        pr.x_H_2_initial = sv.x_H_2;
#endif

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
    double Z_solar = All.SolarAbundances[0];
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
/* Ion electrons ne_ions split onto H+ first (up to xHp_max), then He+ (up to y), then He+ converts to He++ */
static void jaco_split_ion_electrons(double ne_ions, double xHp_max, double y, SolveVars *sv) {
    double xHp = DMIN(ne_ions, xHp_max), ne_He = DMAX(ne_ions - xHp, 0);
    double xHepp = DMIN(DMAX(ne_He - y, 0), y), xHep = DMAX(DMIN(ne_He - 2.0 * xHepp, y - xHepp), 0);
    sv->x_Hplus = DMAX(JACO_ABUNDANCE_FLOOR, xHp);
    sv->x_Heplus = DMAX(JACO_ABUNDANCE_FLOOR, xHep);
    sv->x_Heplusplus = DMAX(JACO_ABUNDANCE_FLOOR, xHepp);
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
    double ne = cell[i].Ne, y = pr->y, xHp_max = 1.0 - 2.0 * xH2;
    if (jaco_isfinite(ne) && ne >= 0) {
        SolveVars s = *sv;
        s.T = (jaco_isfinite(cell[i].Temperature) && cell[i].Temperature > 0) ? cell[i].Temperature : 1e4;
        double ne_ions = ne;
        for (int pass = 0; pass < 2; pass++) {
            jaco_split_ion_electrons(ne_ions, xHp_max, y, &s);
            ne_ions = DMAX(ne - jaco_fixed_electron_abundance(&s, pr), 0);
        }
        if (ne_ions <= (xHp_max + 2.0 * y) * (1.0 + 1e-10)) {
            jaco_split_ion_electrons(ne_ions, xHp_max, y, sv);
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

/* jaco's EOS on the cell's cached state: composition from Ne, MolecularMassFraction and metallicity, at
   specific energy u and physical density rho (both code units), seeded with the cached Temperature. Uses
   only (i, pp, cell), so it is safe on the packed copies cooling works on. */
void jaco_cell_eos(int i, struct particle_data *pp, struct gas_cell_data *cell, double u, double rho,
                   struct jaco_eos_state *eos) {
    SolveVars sv = {};
    Params pr = {};
#ifdef JACO_DEBUG_PARAMS
    jaco_poison_params(&pr); /* the EOS reads a subset of Params; an unpacked field it reads makes the state NaN */
#endif
    double rho_jaco = jaco_pack_params(i, &pr, pp, cell, rho * UNIT_DENSITY_IN_CGS), cv;
    jaco_species_from_cell(i, cell, &pr, &sv);
    sv.u = u * UNIT_SPECEGY_IN_CGS;
    sv.T = jaco_T_from_u(sv.u, cell[i].Temperature, &sv, &pr, &cv);
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

void gizmo_to_jaco(int i, SolveVars *sv, Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    double dtime = get_particle_timestep_in_physical(i, pp);
    set_PdV_work_heatingrate(i, dtime, pp, cell);
#ifdef JACO_DEBUG_PARAMS
    jaco_poison_params(pr);
#endif
    jaco_pack_params(i, pr, pp, cell, cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS);

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
#ifdef JACO_SOLVES_IONS
    /* Except where the CIE table is collisionally ionized (x_e > 0.1, T >~ 1.3e4 K): there it bounds the steady state
       from below, so a cached Ne under it is stale (shock heating, or a solve that settled on the near-neutral fixed
       point of the ionization balance, where the gas stops cooling). Seed CIE ions at the cached T, keeping that T:
       re-deriving T at the ionized composition roughly halves it and strands Newton far from the ionized state. */
    SolveVars cie = *sv; /* at the cell's He and H2: the table's own He abundance would flag every ionized cell */
    jaco_cie_species(cell[i].Temperature, pr, &cie);
    stale_ions = jaco_ions_stale(cell[i].Ne, cie.x_Hplus + cie.x_Heplus + 2.0 * cie.x_Heplusplus);
    if (stale_ions)
        *sv = cie;
#endif
    double cv;
    sv->T = stale_ions ? cell[i].Temperature : jaco_T_from_u(sv->u, cell[i].Temperature, sv, pr, &cv);
#ifdef JACO_HAS_PARAM_x_H_2_initial
    pr->x_H_2_initial = sv->x_H_2;
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

void jaco_to_gizmo(int i, const SolveVars *sv, const Params *pr, struct particle_data *pp, struct gas_cell_data *cell) {
    cell[i].InternalEnergy = sv->u / UNIT_SPECEGY_IN_CGS;
    cell[i].InternalEnergyPred = cell[i].InternalEnergy;
    cell[i].Temperature = sv->T;

    /* Write solved species back BEFORE set_eos_pressure, whose EOS composition comes from Ne and MolecularMassFraction */
#ifdef JACO_SOLVES_IONS
    cell[i].Ne = jaco_electron_abundance(sv, pr); /* all free electrons, metals' included, as the rates use them */
#endif
#ifdef JACO_HAS_VAR_x_H_2
    double xHp = sv->x_Hplus, xH2 = sv->x_H_2;
    cell[i].MolecularMassFraction = 2.0 * xH2;
    double xH0 = DMAX(1.0 - xHp, 0); /* neutral H including H2, as in the standard cooling module */
    cell[i].MolecularMassFraction_perNeutralH = (xH0 > 0) ? DMIN(1, cell[i].MolecularMassFraction / xH0) : 0;
#endif

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

/* ---- Per-step solver statistics ----
   Accumulated over the cells call_jaco() solves in one cooling pass; reported and reset by
   jaco_report_solve_stats(), which every rank must call (MPI collective) once per pass. */
enum { JS_CELLS, JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED, JS_FEVALS, JS_RESYNC, JS_PINNED, JS_NANJ, JS_NANF,
       JS_T1_1, JS_T1_2, JS_T1_3, JS_T1_4, JS_T1_MORE, JS_N };
static long jaco_stats[JS_N];
static int jaco_stats_max_nfeval = 0;

static void jaco_stats_add(const struct JacoSolveInfo *info) {
    int bin[4] = {JS_TIER1, JS_TIER2, JS_TIER3, JS_FAILED};
    int t1 = info->nfeval - info->nfeval_fd; /* evaluations Newton asked for, not finite-difference columns */
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
        if (info->nfeval > jaco_stats_max_nfeval) jaco_stats_max_nfeval = info->nfeval;
    }
}

/* Called at the end of each cooling pass on every rank: prints the MPI-reduced statistics on
   rank 0 and resets the counters. */
void jaco_report_solve_stats(void) {
    long global[JS_N];
    int global_max = 0;
    MPI_Reduce(jaco_stats, global, JS_N, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&jaco_stats_max_nfeval, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (ThisTask == 0 && global[JS_CELLS] > 0) {
        double n = (double)global[JS_CELLS];
        printf("jaco solve stats: %ld cells | tier1 %ld (%.2f%%) tier2 %ld tier3 %ld FAILED %ld | nfeval mean %.2f max %d | "
               "tier-1 evals 1:%ld 2:%ld 3:%ld 4:%ld >4:%ld | T0 resynced %ld, pinned %ld, non-finite J %ld F %ld\n",
               global[JS_CELLS], global[JS_TIER1], 100.0 * global[JS_TIER1] / n, global[JS_TIER2], global[JS_TIER3],
               global[JS_FAILED], global[JS_FEVALS] / n, global_max, global[JS_T1_1], global[JS_T1_2], global[JS_T1_3],
               global[JS_T1_4], global[JS_T1_MORE], global[JS_RESYNC], global[JS_PINNED], global[JS_NANJ], global[JS_NANF]);
        fflush(stdout);
    }
    memset(jaco_stats, 0, sizeof(jaco_stats));
    jaco_stats_max_nfeval = 0;
}

#ifdef OUTPUT_COOLRATE_DETAIL
/* The standard module's per-cell rate outputs at the solved state, in CoolingRate()'s units (erg cm^3 s^-1, per nHcgs()^2).
   jaco's energy row gives only the net radiative rate, so the separated ones (CoolingRate, HeatingRate, MetalCoolingRate,
   PElecHeatingRate) are written as zero. Call before jaco_to_gizmo, which clears DtInternalEnergy. */
static void jaco_coolrate_detail(const SolveVars *sv, const Params *pr, struct gas_cell_data *cell) {
    Params p = *pr;
    p.pdv_work = 0;
    p.u_initial = sv->u; /* sv->u = u(T, x), so the backward-Euler term vanishes and the energy row is the net heating */
    SolveVars F;
    double J[N_VARS][N_VARS], nH = cell->nHcgs();
    microphysics_func_jac(sv, &p, &F, J);
    cell->NetHeatingRateQ = F.T / (nH * nH);
#ifndef COOLING_OPERATOR_SPLIT
    cell->HydroHeatingRate = cell->DtInternalEnergy / nH;
#endif
    cell->CoolingRate = cell->HeatingRate = cell->MetalCoolingRate = cell->PElecHeatingRate = 0;
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
    if (jaco_solve(&sv, &pr, &set, &info)) {
        printf("JACO FATAL: every solver tier failed on task %d (nfeval=%d)\n", ThisTask, info.nfeval);
        jaco_print_state(stdout, "  input state:", &sv_in, &pr);
        set.verbose = 2; /* replay with a trace of every iterate */
        sv = sv_in;
        jaco_solve(&sv, &pr, &set, &info);
        fflush(stdout);
        endrun(10);
    }
#ifdef OUTPUT_COOLRATE_DETAIL
    jaco_coolrate_detail(&sv, &pr, c);
#endif
    jaco_to_gizmo(0, &sv, &pr, p, c);
    jaco_stats_add(&info);
}
#endif
