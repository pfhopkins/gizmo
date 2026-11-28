/* Energy step trace (OUTPUT_ENERGY_STEP_TRACE; stella_dev core/energy_step_trace.cc with the fix/trace-imf-2026-11-17 changes):
 * per-stage energy closure, the energy twin of the mass trace (OUTPUT_MASS_BALANCE_STEP_CHECKPOINTS). Writes
 * <OutputDir>/energy_step_trace.txt, checked by gizmo.conservation_trace.
 *
 * E = kinetic (all types) + thermal (gas) + magnetic + CR (+ raw Rad_E_gamma) [+ potential at full syncs].
 * Stage brackets (ESTEP_BEGIN/END) are rank-local O(N) sums, no communication; every booked source or sink is added on the rank
 * where it happens (ESTEP_BOOK, or a ledger row through estep_ledger_hook). One MPI_Reduce at the end of every Nth step
 * (N = flag value) and of every full-sync step closes the window:
 *   STAGE line: an exact stage whose measured dE minus its booked terms exceeds the round-off bar.
 *   SYNC  line: at full syncs, E_tot incl. potential, cumulative stage changes, bookings and the integration residual
 *               dE_tot - sum(stage dE) - integration bookings. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Kokkos_Core.hpp>
#include "../declarations/gpu_all_mirror.h"  /* before allvars.h */
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../declarations/gpu_dispatch_templates.h"
#include "loop_routes.h"
#include "../gravity/analytic_gravity_functions.h" /* Hu+21 potential */
#if defined(PMGRID) && defined(PM_PLACEHIGHRESREGION)
#include "../gravity/pm_highres_region.h"
#endif
#ifdef COSMIC_RAY_FLUID
#include "../eos/cosmic_ray_fluid/cosmic_ray_functions.h" /* GAMMA_COSMICRAY with a CR spectrum */
#endif

#ifdef OUTPUT_ENERGY_STEP_TRACE

#define ESTEP_EVERY ((OUTPUT_ENERGY_STEP_TRACE + 0) > 0 ? (OUTPUT_ENERGY_STEP_TRACE + 0) : 1) /* flag value N: close a window every N steps */
#define ESTEP_BAR_EXACT 1.0e-12 /* exact stages: relative to E_abs + |booked| */

enum {EC_KIN, EC_TH, EC_MAG, EC_CR, EC_RAD, EC_N};
static const char *estep_comp_name[EC_N] = {"kin", "th", "mag", "cr", "rad"};

static const char *estep_stage_name[] = {"merge_split", "fb_mech", "fb_radp", "photoion", "cooling", "sf", "rt_inject",
    "int_grav", "int_hydro", "int_kick1", "int_kick2", "int_drift", "int_nbody"};
/* 1 = exact (booked terms must close it to round-off), 0 = measured, no energy booking yet (a source in tot_res),
 * 2 = part of the time integration, measured for attribution only (contains nested stages; in no closure) */
static const int estep_stage_exact[] = {
    1, /* merge_split, with the SN ejecta cell's energy (sn_cell_th) */
    1, 0, 1, /* fb_mech, fb_radp (measured: the radiation's work), photoion (marks cells; the heat is booked in cooling) */
#if defined(SGCHEM) && !defined(TRANSPORT_SUBCYCLE_COOLING) /* subcycled cooling runs outside the cooling bracket */
    1,
#else
    0,
#endif
#if defined(GALSF_SINGLE_STARS_FROM_IMF) /* the single-star spawn and gas collection book their changes; other SF paths do not yet */
    1,
#else
    0,
#endif
    0, /* rt_inject: measured, the sources' photons */
    2, 2, 2, 2, 2, 2};

static const char *estep_book_name[] = {"fb_sn", "fb_ia", "fb_failed", "fb_wind", "fb_agb", "fb_remnant_kin", "cool_solve", "cool_pi", "cool_floor", "cool_cr_loss", "cool_rad_exchange",
    "sf_th_locked", "sf_mag_locked", "sf_donor", "sf_donor_mag", "sf_star_kin", "sf_cr", "merge_heat_not_returned", "merge_mag_lost", "sn_cell_th", "sn_cell_mag",
    "ketju_chain_pot", "ketju_collision", "ketju_expand", "kick_floor", "cr_clip", "outflow", "rt_emit", "rt_abs", "rt_toir",
#ifdef MERGE_INACTIVE_PARENT_COMPLETES_STEP_WITH_OWN_RATES
    "merge_parent_kick", "merge_parent_kick_to_int",
#endif
    "info_mhd_num", "info_cr_shock",
    "info_cool_id_diff", "info_cool_id_gross", "info_cool_id_evals"};
/* stage each booking belongs to; -1 = time integration (kicks/drift), -2 = information only (in no closure) */
static const int estep_book_stage[] = {ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_FB_MECH, ESTEP_COOLING, ESTEP_COOLING, ESTEP_COOLING,
    ESTEP_COOLING, ESTEP_COOLING, ESTEP_SF, ESTEP_SF, ESTEP_SF, ESTEP_SF, ESTEP_SF, ESTEP_SF, ESTEP_MERGE_SPLIT, ESTEP_MERGE_SPLIT, ESTEP_MERGE_SPLIT, ESTEP_MERGE_SPLIT,
    -1, -1, -1, -1, -1, -1, -1, -1, -1,
#ifdef MERGE_INACTIVE_PARENT_COMPLETES_STEP_WITH_OWN_RATES
    ESTEP_MERGE_SPLIT, -1,
#endif
    -2, -2, -2, -2, -2};
/* magnetic bookings stay out of the stage closure: E_B = 1/2 |BV|^2 / V with V = m/rho moves whenever a tree walk drifts a cell; they count at full syncs */
static int estep_book_is_mag(int b) {return b == EBK_SF_MAG_LOCKED || b == EBK_SF_DONOR_MAG || b == EBK_MERGE_MAG_LOST || b == EBK_SN_CELL_MAG;}

static_assert(sizeof(estep_stage_name) / sizeof(estep_stage_name[0]) == ESTEP_NSTAGE, "stage names");
static_assert(sizeof(estep_stage_exact) / sizeof(estep_stage_exact[0]) == ESTEP_NSTAGE, "stage kinds");
static_assert(sizeof(estep_book_name) / sizeof(estep_book_name[0]) == EBK_N, "booking names");
static_assert(sizeof(estep_book_stage) / sizeof(estep_book_stage[0]) == EBK_N, "booking stages");

double EstepBooked[EBK_N];                    /* rank-local bookings in this window [code energy] */
static double estep_pre[ESTEP_NSTAGE][EC_N];  /* rank-local E at the open bracket */
static double estep_dE[ESTEP_NSTAGE][EC_N];   /* rank-local measured stage change in this window */
static int estep_open[ESTEP_NSTAGE];
static double estep_wall; /* seconds this rank spent in the trace */

/* magnetic energy of a gas cell from the conserved B*V: 1/2 |B|^2 V, V = m/rho (as energy_statistics) */
KOKKOS_INLINE_FUNCTION double estep_cell_mag_energy_of(const struct particle_data *pp, const struct gas_cell_data *cell, int i, double a2inv, double a3inv)
{
#ifdef MAGNETIC
    if(pp[i].Type != 0 || pp[i].Mass <= 0 || cell[i].Density <= 0) {return 0;}
    Vec3<double> b = cell[i].B * (cell[i].Density / pp[i].Mass) * a2inv;
    return 0.5 * b.norm_sq() * pp[i].Mass / (cell[i].Density * a3inv);
#else
    (void)pp; (void)cell; (void)i; (void)a2inv; (void)a3inv; return 0;
#endif
}
double estep_cell_mag_energy(int i) {return estep_cell_mag_energy_of(P, CellP, i, All.cf_a2inv, All.cf_a3inv);}

/* all energy of mass m of a particle, every component [code energy]; with m = P[i].Mass what it carries */
double estep_particle_energy_for_mass(int i, double m)
{
    if(m <= 0) {return 0;}
    double e = 0.5 * m * P[i].Vel.norm_sq() * All.cf_a2inv;
    if(P[i].Type == 0) {
        e += m * CellP[i].InternalEnergy;
#ifdef MAGNETIC
        if(CellP[i].Density > 0) {e += 0.5 * (CellP[i].B * (CellP[i].Density / m) * All.cf_a2inv).norm_sq() * m / (CellP[i].Density * All.cf_a3inv);}
#endif
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {e += CellP[i].CosmicRayEnergy[k];}
#endif
#ifdef RADTRANSFER
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {e += CellP[i].Rad_E_gamma[k];}
#endif
    }
    return e;
}

/* ledger rows that are trace bookings: the row's own sign convention turned into the energy the stage adds */
#ifdef OUTPUT_MASS_METAL_ENERGY_LEDGER
void estep_ledger_hook(int row, double x)
{
    switch(row) {
#ifdef SGCHEM
        case LEDGER_CHEM_HEATING: ESTEP_BOOK(EBK_COOL_SOLVE, x); break;   /* the solve only, |dE| */
        case LEDGER_CHEM_COOLING: ESTEP_BOOK(EBK_COOL_SOLVE, -x); break;
        case LEDGER_E_KICK_FLOOR: ESTEP_BOOK(EBK_KICK_FLOOR, x); break;   /* what the kick limiter and floor add */
#ifdef MAGNETIC
        case LEDGER_E_MHD_NUMERICAL: ESTEP_BOOK(EBK_INFO_MHD_NUM, x); break;
#endif
#endif
#if defined(GALSF_FB_PHOTOIONIZATION_PER_DIRECTION) && defined(SGCHEM)
        case LEDGER_PI_HEATING: ESTEP_BOOK(EBK_COOL_PI, x); break;       /* the raise to 1e4 K in the chemistry call */
#endif
#if defined(SGCHEM) && defined(COSMIC_RAY_FLUID)
        case LEDGER_E_CR_LOSSES: ESTEP_BOOK(EBK_COOL_CR_LOSS, -x); break; /* CR energy the losses remove */
#endif
#ifdef COSMIC_RAY_FLUID
        case LEDGER_E_CR_CLIP: ESTEP_BOOK(EBK_CR_CLIP, x); break;
        case LEDGER_E_CR_SHOCK: ESTEP_BOOK(EBK_INFO_CR_SHOCK, x); break; /* thermal -> CR at shocks: a transfer */
        case LEDGER_E_CR_COLLECTED: ESTEP_BOOK(EBK_SF_CR, -x); break;     /* CR energy that leaves with the donated gas */
#endif
        default: break;
    }
}
#endif

#ifdef PMGRID
/* a particle's own long-range mesh potential in P.Potential (with G): -G m / (sqrt(pi) r_s), the r -> 0 limit of the Gaussian split's
   -G m erf(r / 2 r_s) / r; energy sums remove it from 1/2 sum m Phi (stella_dev 5e3384335). rs0, rs1: All.Asmth[0], [1] */
KOKKOS_INLINE_FUNCTION double estep_pm_self_potential(const struct particle_data *pp, int i, double G, double rs0, double rs1)
{
    double rs = rs0;
#ifdef PM_PLACEHIGHRESREGION
    Vec3<double> pos = pp[i].Pos; if(pmforce_is_particle_high_res(pp[i].Type, pos)) {rs = rs1;}
#endif
    (void) rs1; return -G * pp[i].Mass / (sqrt(M_PI) * rs);
}
#endif

/* Neumaier-compensated sums of the components and of 1/2 m Phi (slot EC_N), and their join for a Kokkos reduction */
struct EstepSums {double s[EC_N + 1], c[EC_N + 1];};
KOKKOS_INLINE_FUNCTION void estep_neumaier_add(EstepSums &v, int k, double x) {double t = v.s[k] + x; v.c[k] += (fabs(v.s[k]) >= fabs(x)) ? (v.s[k] - t) + x : (x - t) + v.s[k]; v.s[k] = t;}
struct EstepNeumaierJoin {
    using reducer = EstepNeumaierJoin;
    using value_type = EstepSums;
    using result_view_type = Kokkos::View<value_type, Kokkos::AnonymousSpace, Kokkos::MemoryUnmanaged>;
    static_assert(sizeof(value_type) % sizeof(int) == 0, "EstepSums must be a whole number of int-sized pieces");
    value_type &value;
    KOKKOS_INLINE_FUNCTION EstepNeumaierJoin(value_type &v) : value(v) {}
    KOKKOS_INLINE_FUNCTION void join(value_type &dst, const value_type &src) const {for(int k = 0; k <= EC_N; k++) {estep_neumaier_add(dst, k, src.s[k]); dst.c[k] += src.c[k];}}
    KOKKOS_INLINE_FUNCTION void init(value_type &v) const {for(int k = 0; k <= EC_N; k++) {v.s[k] = v.c[k] = 0;}}
    KOKKOS_INLINE_FUNCTION value_type &reference() const {return value;}
    KOKKOS_INLINE_FUNCTION result_view_type view() const {return result_view_type(&value);}
    KOKKOS_INLINE_FUNCTION bool references_scalar() const {return true;}
};

/* local sums of the components (and of 1/2 m Phi when pot != NULL): one Kokkos kernel over the rank's particles, on the device
   or on the host execution space (gizmo_stage_on_device). A Serial host space runs the particles in order with one accumulator,
   as a plain loop; an OpenMP or Threads host space and the device join partial sums with the same compensation, so the sums
   differ from a plain loop at round-off (see gizmo_stage_reduce in gpu_dispatch_templates.h) */
static void estep_local_sums(double e[EC_N], double *pot)
{
    const int n = NumPart, on_device = gizmo_stage_on_device(n), with_pot = (pot != NULL);
    if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH();} /* the high-resolution region test can read All */
    const struct particle_data *kp = P; const struct gas_cell_data *kc = CellP;
    const double a2inv = All.cf_a2inv, a3inv = All.cf_a3inv;
#ifdef PMGRID
    const double G = All.G, rs0 = All.Asmth[0], rs1 = All.Asmth[1];
#endif
#ifdef GRAVITY_ANALYTIC_TALLBOX_HU21
    const struct hu_tallbox_units hu = hu_tallbox_units_now();
    const double z_mid = boxHalf_Z;
#endif
    EstepSums v;
    auto body = KOKKOS_LAMBDA(int i, EstepSums &acc) {
        if(kp[i].Mass <= 0) {return;}
        estep_neumaier_add(acc, EC_KIN, 0.5 * kp[i].Mass * kp[i].Vel.norm_sq() * a2inv);
#ifdef EVALPOTENTIAL
        if(with_pot) {estep_neumaier_add(acc, EC_N, 0.5 * kp[i].Mass * kp[i].Potential);}
#ifdef PMGRID
        if(with_pot) {estep_neumaier_add(acc, EC_N, -0.5 * kp[i].Mass * estep_pm_self_potential(kp, i, G, rs0, rs1));} /* no self-energy */
#endif
#endif
#ifdef GRAVITY_ANALYTIC_TALLBOX_HU21
        if(with_pot) {estep_neumaier_add(acc, EC_N, kp[i].Mass * hu_tallbox_potential(kp[i].Pos[2] - z_mid, hu));} /* external field: full weight */
#endif
        if(kp[i].Type != 0) {return;}
        estep_neumaier_add(acc, EC_TH, kp[i].Mass * kc[i].InternalEnergy);
        estep_neumaier_add(acc, EC_MAG, estep_cell_mag_energy_of(kp, kc, i, a2inv, a3inv));
#ifdef COSMIC_RAY_FLUID
        for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {estep_neumaier_add(acc, EC_CR, kc[i].CosmicRayEnergy[k]);}
#endif
#ifdef RADTRANSFER
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {estep_neumaier_add(acc, EC_RAD, kc[i].Rad_E_gamma[k]);}
#endif
        (void) with_pot;
    };
    EstepNeumaierJoin(v).init(v); /* nothing to sum on an empty rank */
    gizmo_stage_reduce("estep_local_sums", on_device, n, body, EstepNeumaierJoin(v));
    loop_route_count("estep_local_sums", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    for(int k = 0; k < EC_N; k++) {e[k] = v.s[k] + v.c[k];}
    if(pot) {*pot = v.s[EC_N] + v.c[EC_N];}
}

void estep_begin(int stage)
{
    double t0 = MPI_Wtime();
    estep_local_sums(estep_pre[stage], NULL);
    estep_open[stage] = 1;
    estep_wall += MPI_Wtime() - t0;
}

void estep_end(int stage)
{
    if(!estep_open[stage]) {return;}
    double t0 = MPI_Wtime(), e[EC_N];
    estep_local_sums(e, NULL);
    for(int k = 0; k < EC_N; k++) {estep_dE[stage][k] += e[k] - estep_pre[stage][k];}
    estep_open[stage] = 0;
    estep_wall += MPI_Wtime() - t0;
}

void energy_step_trace_report(void)
{
    static int first = 1;
    static long long n_calls = 0;
    int is_sync = (All.HighestActiveTimeBin == All.HighestOccupiedTimeBin);
    n_calls++;
    if(!first && !is_sync && (n_calls % ESTEP_EVERY) != 0) {return;} /* rank-uniform: step count and timebins are global */

    double t0 = MPI_Wtime();
    const int NS = ESTEP_NSTAGE * EC_N, NV = NS + EBK_N + EC_N + 2;
    double loc[NV + 2], glob[NV + 2];
    for(int s = 0; s < ESTEP_NSTAGE; s++) {for(int k = 0; k < EC_N; k++) {loc[s * EC_N + k] = estep_dE[s][k]; estep_dE[s][k] = 0;}}
    for(int b = 0; b < EBK_N; b++) {loc[NS + b] = EstepBooked[b]; EstepBooked[b] = 0;}
    double e[EC_N], pot = 0;
    estep_local_sums(e, &pot);
    double eabs = 0;
    for(int k = 0; k < EC_N; k++) {loc[NS + EBK_N + k] = e[k]; eabs += fabs(e[k]);}
    loc[NV - 2] = eabs; loc[NV - 1] = pot;
    loc[NV + 1] = 0; /* Layzer-Irvine weights: sum 3(gamma-1) m u + 3(gamma_CR-1) E_CR, the expansion loss rate over H */
    if(All.ComovingIntegrationOn && is_sync) {
        const int n = NumPart, on_device = gizmo_stage_on_device(n);
        if(on_device) {GIZMO_GPU_ENSURE_ALL_FRESH();} /* the CR adiabatic index of a spectrum can read All */
        const struct particle_data *kp = P; const struct gas_cell_data *kc = CellP;
        loc[NV + 1] = gizmo_stage_sum<double>("estep_layzer_irvine", on_device, n, KOKKOS_LAMBDA(int i, double &w) {
            if(kp[i].Type != 0 || kp[i].Mass <= 0) {return;}
            w += 3. * (kc[i].gamma_eos_value() - 1.) * kp[i].Mass * kc[i].InternalEnergy;
#ifdef COSMIC_RAY_FLUID
            for(int k = 0; k < N_CR_PARTICLE_BINS; k++) {w += 3. * (GAMMA_COSMICRAY(k) - 1.) * kc[i].CosmicRayEnergy[k];}
#endif
        });
        loop_route_count("estep_layzer_irvine", on_device ? LOOP_ROUTE_DEVICE_KERNEL : LOOP_ROUTE_HOST_KERNEL, n);
    }
    loc[NV] = estep_wall + (MPI_Wtime() - t0); /* rank sum of trace time up to this reduce */
    MPI_Reduce(loc, glob, NV + 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    estep_wall += MPI_Wtime() - t0;
    if(ThisTask != 0) {first = 0; return;}

    int pot_ok = 0;
#if defined(EVALPOTENTIAL) && !defined(ADAPTIVE_TREEFORCE_UPDATE) /* skipped tree walks keep an old Potential */
    pot_ok = 1;
#endif
    static FILE *fd = NULL;
    static double cum_stage[ESTEP_NSTAGE], cum_book[EBK_N], E0 = 0, E0_abs = 0;
    static int have_E0 = 0;
    if(!fd) {
        char path[DEFAULT_PATH_BUFFERSIZE_TOUSE];
        snprintf(path, DEFAULT_PATH_BUFFERSIZE_TOUSE, "%s/energy_step_trace.txt", All.OutputDir);
        fd = fopen(path, "a");
        if(!fd) {endrun(88731);}
        if(ftell(fd) == 0) {
            fprintf(fd, "# Energy step trace [code energy units; erg = x %.10e]. key=value lines.\n", UNIT_ENERGY_IN_CGS);
            fprintf(fd, "# E = kin(all types, peculiar) + th + mag + cr + rad(raw Rad_E_gamma); pot = 1/2 sum m Phi / a at full syncs.\n");
            fprintf(fd, "# Comoving: LI = int (2 E_kin + E_pot + sum 3(gamma-1) m u + E_mag + 3(gamma_CR-1) E_cr + E_rad) dln a (Layzer-Irvine), trapezoid over syncs; 0 if not comoving.\n");
            fprintf(fd, "# STAGE: exact stage whose window residual = dE(kin+th+cr+rad) - booked exceeds bar = %.1e x (E_abs + |booked|).\n", ESTEP_BAR_EXACT);
            fprintf(fd, "#        d_mag and b_*mag* are shown but not closed per stage: tree walks drift Density, and E_B = 1/2|BV|^2 m/rho moves with it.\n");
            fprintf(fd, "# SYNC: full sync. int_res = (E_tot - E_tot0) + LI - sum(stage dE) - integration bookings (ketju_chain_pot, ketju_collision, ketju_expand, kick_floor, cr_clip, outflow, rt_emit, rt_abs, rt_toir).\n");
            fprintf(fd, "#       b_info_*: information only, in no closure. tot_res = (E_tot - E_tot0) - all bookings - measured stages.\n");
            fprintf(fd, "#       E_scale: |E| + |E_pot| at the first sync, fixed for the run (the checker adds the booked SN energy from the SN on).\n");
            fprintf(fd, "# cooling stage: the solve (ledger), PI raise, input and output floors, CR losses booked from their formulas at the site;\n");
            fprintf(fd, "#        unbooked, so a STAGE finding: the NaN revert, the hydro du around the solve without COOLING_OPERATOR_SPLIT.\n");
            fprintf(fd, "#        the solve, PI raise and CR loss rates are booked by construction. info_cool_id_*: at every RHS evaluation of the\n");
            fprintf(fd, "#        chemistry solve (explicit hot path, accepted substeps, final state) sum |energy RHS + reported net cooling| x dt,\n");
            fprintf(fd, "#        the gross rates x dt (per evaluation, times the cell step: only their ratio is meaningful), the evaluations;\n");
            fprintf(fd, "#        equal when every term of the energy equation is reported (not net 18).\n");
            fprintf(fd, "# exact stages:");
            for(int s = 0; s < ESTEP_NSTAGE; s++) {if(estep_stage_exact[s] == 1) {fprintf(fd, " %s", estep_stage_name[s]);}}
            fprintf(fd, "\n# pot_valid=%d every=%d\n", pot_ok, ESTEP_EVERY);
#if defined(SGCHEM) && (CHEMISTRYNETWORK != 18)
            fprintf(fd, "# cooling_identity=1\n"); /* the checker requires covered evaluations */
#endif
        }
        if(RestartFlag == 1) {fprintf(fd, "RESTART t=%.16e\n", All.Time);}
    }

    double Eabs = glob[NV - 2];
    for(int s = 0; s < ESTEP_NSTAGE; s++) {
        double dE = 0, dE_nomag = 0, book = 0, babs = 0;
        for(int k = 0; k < EC_N; k++) {dE += glob[s * EC_N + k]; if(k != EC_MAG) {dE_nomag += glob[s * EC_N + k];}}
        for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == s && !estep_book_is_mag(b)) {book += glob[NS + b]; babs += fabs(glob[NS + b]);}}
        double res = dE_nomag - book, bar = ESTEP_BAR_EXACT * (Eabs + babs);
        if(estep_stage_exact[s] == 1 && fabs(res) > bar) {
            fprintf(fd, "STAGE step=%lld t=%.16e stage=%s res=%+.16e bar=%.6e booked=%+.16e", (long long)All.NumCurrentTiStep, All.Time, estep_stage_name[s], res, bar, book);
            for(int k = 0; k < EC_N; k++) {fprintf(fd, " d_%s=%+.16e", estep_comp_name[k], glob[s * EC_N + k]);}
            for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == s && glob[NS + b] != 0) {fprintf(fd, " b_%s=%+.16e", estep_book_name[b], glob[NS + b]);}}
            fprintf(fd, " E_abs=%.16e\n", Eabs);
        }
        if(!first) {cum_stage[s] += dE;}
    }
    double Etot = 0; for(int k = 0; k < EC_N; k++) {Etot += glob[NS + EBK_N + k];}
    double Epot = pot_ok ? glob[NV - 1] / All.cf_atime : 0; /* physical: 1/2 sum m Phi / a */
    Etot += Epot;
    if(!first) {for(int b = 0; b < EBK_N; b++) {cum_book[b] += glob[NS + b];}}
    if(is_sync) {
        static double li = 0, li_lna = 0, li_T = 0;
        double lna = All.ComovingIntegrationOn ? log(All.Time) : 0.;
        double T = 2. * glob[NS + EBK_N + EC_KIN] + Epot + glob[NV + 1] + glob[NS + EBK_N + EC_MAG] + glob[NS + EBK_N + EC_RAD];
        if(!have_E0) {E0 = Etot; E0_abs = Eabs + fabs(Epot); have_E0 = 1; for(int s = 0; s < ESTEP_NSTAGE; s++) {cum_stage[s] = 0;} for(int b = 0; b < EBK_N; b++) {cum_book[b] = 0;} li = 0;}
        else if(All.ComovingIntegrationOn) {li += 0.5 * (T + li_T) * (lna - li_lna);}
        li_lna = lna; li_T = T;
        double sum_stage = 0, int_book = 0, sum_book = 0, sources = 0;
        for(int s = 0; s < ESTEP_NSTAGE; s++) {if(estep_stage_exact[s] == 2) {continue;} sum_stage += cum_stage[s]; if(estep_stage_exact[s] == 0) {sources += cum_stage[s];}}
        for(int b = 0; b < EBK_N; b++) {if(estep_book_stage[b] == -2) {continue;} sum_book += cum_book[b]; if(estep_book_stage[b] == -1) {int_book += cum_book[b];}}
        double dEtot = Etot - E0 + li; /* comoving: the expansion work, so dEtot is again the source/sink budget */
        fprintf(fd, "SYNC step=%lld t=%.16e", (long long)All.NumCurrentTiStep, All.Time);
        for(int k = 0; k < EC_N; k++) {fprintf(fd, " E_%s=%.16e", estep_comp_name[k], glob[NS + EBK_N + k]);}
        if(pot_ok) {fprintf(fd, " E_pot=%.16e E_tot=%.16e LI=%+.16e dE_tot=%+.16e int_res=%+.16e tot_res=%+.16e", Epot, Etot, li, dEtot, dEtot - sum_stage - int_book, dEtot - sum_book - sources);}
        else {fprintf(fd, " E_pot=nan E_tot=nan dE_tot=nan int_res=nan tot_res=nan");}
        fprintf(fd, " E_scale=%.16e trace_cpu_s=%.4e", E0_abs, glob[NV]); /* fixed: |E| + |E_pot| at the first sync */
        for(int s = 0; s < ESTEP_NSTAGE; s++) {fprintf(fd, " s_%s=%+.16e", estep_stage_name[s], cum_stage[s]);}
        for(int b = 0; b < EBK_N; b++) {fprintf(fd, " b_%s=%+.16e", estep_book_name[b], cum_book[b]);}
        fprintf(fd, "\n");
    }
    fflush(fd);
    first = 0;
}
#endif /* OUTPUT_ENERGY_STEP_TRACE */
