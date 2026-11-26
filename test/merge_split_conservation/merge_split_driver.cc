/* Driver for test_merge_split_conservation.py: runs the production split and merge code (mesh/merge_split.cc) on hand-built
   cells and prints PASS|FAIL per conserved quantity. It replaces core/main.cc in the link of the build under test (CPU builds).
   CellP has a canary region past MaxPartGas to catch writes for non-gas indices. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "declarations/allvars.h"
#include "core/proto.h"

/* core/main.cc, which this driver replaces in the link, owns the executable fingerprint the snapshot header writes */
unsigned long long executable_fnv1a64(void) {return 0;}
long long executable_size_in_bytes(void) {return -1;}

static FILE *out;
static int n_fail = 0, n_check = 0;
static const double RTOL = 1e-13;

static void check(const char *name, bool ok, double got, double want)
{
    n_check++; if(!ok) {n_fail++;}
    fprintf(out, "%s %s got=%.17g want=%.17g\n", ok ? "PASS" : "FAIL", name, got, want);
}
static void check_rel(const char *name, double got, double want, double scale)
{
    check(name, fabs(got - want) <= RTOL * fabs(scale), got, want);
}

/* ---- particle arrays with a canary region past MaxPartGas ---- */
static const int N_CANARY = 64;
static const unsigned char CANARY = 0xA5;

/* aligned like mymalloc: -march=native builds use aligned vector stores on these structs */
static void *zalloc_aligned(size_t bytes)
{
    void *p = NULL;
    if(posix_memalign(&p, 64, bytes) != 0) {fprintf(stderr, "allocation failed\n"); exit(1);}
    memset(p, 0, bytes);
    return p;
}

static void setup(int max_part, int max_part_gas)
{
    free(P); free(CellP);
    All.MaxPart = max_part; All.MaxPartGas = max_part_gas;
    P = (struct particle_data *) zalloc_aligned(max_part * sizeof(struct particle_data));
    CellP = (struct gas_cell_data *) zalloc_aligned((max_part_gas + N_CANARY) * sizeof(struct gas_cell_data));
    memset((void *)(CellP + max_part_gas), CANARY, N_CANARY * sizeof(struct gas_cell_data));
    NextInTimeBin.assign(max_part, -1); PrevInTimeBin.assign(max_part, -1);
    NumPart = N_gas = 0;
    for(int b = 0; b < TIMEBINS; b++) {TimeBinActive[b] = 0;} /* nothing active: merge_and_split_particles then only runs the SN spawn */
}
static bool canary_intact()
{
    const unsigned char *c = (const unsigned char *)(CellP + All.MaxPartGas);
    for(size_t b = 0; b < N_CANARY * sizeof(struct gas_cell_data); b++) {if(c[b] != CANARY) {return false;}}
    return true;
}

/* a gas cell with every extensive field set */
static void make_gas(int i, double x, double y, double z, double mass, double X_H, double Y_He, double Z_met, double seed)
{
    P[i].Type = 0; P[i].ID = i + 1; P[i].Mass = CellP[i].Mass = mass;
    P[i].Pos[0] = x; P[i].Pos[1] = y; P[i].Pos[2] = z;
    P[i].Vel[0] = 1.0 + seed; P[i].Vel[1] = -2.0 * seed; P[i].Vel[2] = 0.5;
    CellP[i].VelPred = P[i].Vel;
    CellP[i].Density = 1.0 + seed; P[i].NumNgb = 32;
    P[i].KernelRadius = 2. * cbrt(mass / CellP[i].Density);
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    P[i].AGS_KernelRadius = P[i].KernelRadius;
#endif
    P[i].TimeBin = 10; P[i].dt_step = ((integertime)1) << 10;
    CellP[i].InternalEnergy = CellP[i].InternalEnergyPred = 100. * (1. + seed);
#ifdef METALS
    P[i].Metallicity[0] = Z_met;
    for(int k = 1; k < NUM_METAL_SPECIES; k++) {P[i].Metallicity[k] = Z_met * 0.01 * k;}
#if (NUM_LIVE_SPECIES_FOR_COOLTABLES >= 10)
    P[i].Metallicity[1] = Y_He; /* helium is tracked: the hydrogen mass fraction is 1 - Z - Y */
#endif
#endif
    (void) X_H; (void) Y_He; (void) Z_met;
#if defined(GALSF_ISMDUSTCHEM_MODEL) /* dust as mass fractions of the cell */
    for(int k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {CellP[i].ISMDustChem_Dust_Metal[k] = 1e-3 * (k + 1) * (1. + seed);}
    for(int k = 0; k < NUM_ISMDUSTCHEM_SPECIES; k++) {CellP[i].ISMDustChem_Dust_Species[k] = 2e-3 * (k + 1) * (1. + seed);}
#endif
#if defined(SGCHEM)
    for(int k = 0; k < SGCHEM_NUM_TRACKED_SPECIES; k++) {CellP[i].TracAbund[k] = 1e-2 * (k + 1) * (1. + 2. * seed);}
#endif
#ifdef MAGNETIC
    CellP[i].B[0] = 1e-3 * (1. + seed); CellP[i].B[1] = -2e-3 * seed; CellP[i].B[2] = 5e-4; CellP[i].BPred = CellP[i].B;
#endif
#if defined(SINK_THERMALFEEDBACK)
    CellP[i].Injected_Sink_Energy = 7.0 * mass * (1. + seed);
#endif
}

static void make_other(int i, int type, double mass, double x)
{
    P[i].Type = type; P[i].ID = 100000 + i; P[i].Mass = mass;
    P[i].Pos[0] = x; P[i].Pos[1] = 0.5; P[i].Pos[2] = 0.5;
    P[i].Vel[0] = 0.3; P[i].KernelRadius = 0.1; P[i].NumNgb = 32; P[i].TimeBin = 10; P[i].dt_step = ((integertime)1) << 10;
#ifdef AGS_KERNELRADIUS_CALCULATION_IS_ACTIVE
    P[i].AGS_KernelRadius = 0.1;
#endif
}

/* ---- conserved totals over live cells ---- */
#ifdef METALS
static const int NZ = NUM_METAL_SPECIES;
#else
static const int NZ = 0;
#endif
struct Totals {double M = 0, p[3] = {0, 0, 0}, B[3] = {0, 0, 0}, E_sink = 0, H2 = 0, CO = 0; std::vector<double> MZ, dust;};

static Totals totals(int n)
{
    Totals t; t.MZ.assign(NZ, 0.);
#if defined(GALSF_ISMDUSTCHEM_MODEL)
    t.dust.assign(NUM_ISMDUSTCHEM_ELEMENTS + NUM_ISMDUSTCHEM_SPECIES, 0.);
#endif
    for(int i = 0; i < n; i++)
    {
        if(P[i].Mass <= 0) {continue;} /* merged away */
        t.M += P[i].Mass; for(int k = 0; k < 3; k++) {t.p[k] += P[i].Mass * P[i].Vel[k];}
#ifdef METALS
        for(int k = 0; k < NZ; k++) {t.MZ[k] += P[i].Mass * P[i].Metallicity[k];}
#endif
        if(P[i].Type != 0) {continue;}
#if defined(GALSF_ISMDUSTCHEM_MODEL)
        for(int k = 0; k < NUM_ISMDUSTCHEM_ELEMENTS; k++) {t.dust[k] += P[i].Mass * CellP[i].ISMDustChem_Dust_Metal[k];}
        for(int k = 0; k < NUM_ISMDUSTCHEM_SPECIES; k++) {t.dust[NUM_ISMDUSTCHEM_ELEMENTS + k] += P[i].Mass * CellP[i].ISMDustChem_Dust_Species[k];}
#endif
#ifdef MAGNETIC
        for(int k = 0; k < 3; k++) {t.B[k] += CellP[i].B[k];}
#endif
#if defined(SINK_THERMALFEEDBACK)
        t.E_sink += CellP[i].Injected_Sink_Energy;
#endif
#if defined(SGCHEM)
        double X_H = 1; /* TracAbund are n_X/n_H: the mass in a species goes with the cell's hydrogen mass */
#if (NUM_LIVE_SPECIES_FOR_COOLTABLES >= 10)
        X_H = 1. - P[i].Metallicity[0] - P[i].Metallicity[1];
#endif
        t.H2 += CellP[i].TracAbund[0] * 2. * X_H * P[i].Mass;
        if(SGCHEM_NUM_TRACKED_SPECIES > 2) {t.CO += CellP[i].TracAbund[2] * 28. * X_H * P[i].Mass;}
#endif
    }
    return t;
}

static void check_totals(const char *what, const Totals &a, const Totals &b)
{
    char name[256];
    snprintf(name, sizeof(name), "%s:mass", what); check_rel(name, b.M, a.M, a.M);
    double pscale = fabs(a.p[0]) + fabs(a.p[1]) + fabs(a.p[2]);
    for(int k = 0; k < 3; k++) {snprintf(name, sizeof(name), "%s:momentum[%d]", what, k); check_rel(name, b.p[k], a.p[k], pscale);}
    for(int k = 0; k < NZ; k++) {snprintf(name, sizeof(name), "%s:metal_mass[%d]", what, k); check_rel(name, b.MZ[k], a.MZ[k], a.MZ[k]);}
    for(size_t k = 0; k < a.dust.size(); k++) {snprintf(name, sizeof(name), "%s:dust_mass[%zu]", what, k); check_rel(name, b.dust[k], a.dust[k], a.dust[k]);}
#ifdef MAGNETIC
    double bscale = fabs(a.B[0]) + fabs(a.B[1]) + fabs(a.B[2]);
    for(int k = 0; k < 3; k++) {snprintf(name, sizeof(name), "%s:B_conserved[%d]", what, k); check_rel(name, b.B[k], a.B[k], bscale);}
#endif
#if defined(SINK_THERMALFEEDBACK)
    snprintf(name, sizeof(name), "%s:injected_sink_energy", what); check_rel(name, b.E_sink, a.E_sink, a.E_sink);
#endif
#if defined(SGCHEM)
    snprintf(name, sizeof(name), "%s:H2_mass", what); check_rel(name, b.H2, a.H2, a.H2);
    if(SGCHEM_NUM_TRACKED_SPECIES > 2) {snprintf(name, sizeof(name), "%s:CO_mass", what); check_rel(name, b.CO, a.CO, a.CO);}
#endif
}

/* ---- the cases ---- */
static void test_split()
{
    setup(64, 64);
    make_gas(0, 0.5, 0.5, 0.5, 2.0, 0.70, 0.28, 0.02, 0.0);
    make_gas(1, 0.6, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.3);
    make_gas(2, 0.5, 0.7, 0.5, 1.0, 0.72, 0.27, 0.01, 0.6);
    NumPart = N_gas = 3;
    Totals before = totals(3);
    Vec3<double> x0 = P[0].Pos;
    int ok = split_particle_i(0, 0, 0.1); /* distance to cell 1, the nearest */
    check("split:succeeded", ok == 1, ok, 1);
    check_totals("split", before, totals(4));
    /* daughters straddle the parent, each within 0.4 of the nearest-neighbour distance (0.1) */
    Vec3<double> d0 = P[0].Pos - x0, d1 = P[3].Pos - x0;
    check("split:daughters_symmetric_about_parent", (d0 + d1).norm() < 1e-12, (d0 + d1).norm(), 0);
    check("split:offset_within_0.4_r_nearest", d0.norm() > 0 && d0.norm() <= 0.4 * 0.1 * (1 + 1e-12), d0.norm(), 0.04);
    check("split:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);
}

static void test_merge()
{
    setup(64, 64);
    /* ejecta-like cell into ISM-like cell: different X_H */
    make_gas(0, 0.5, 0.5, 0.5, 0.8, 0.30, 0.40, 0.30, 0.1);
    make_gas(1, 0.55, 0.5, 0.5, 1.5, 0.74, 0.25, 0.01, 0.7);
    NumPart = N_gas = 2;
    Totals before = totals(2);
    int ok = merge_particles_ij(0, 1);
    check("merge:succeeded", ok == 1, ok, 1);
    check_totals("merge", before, totals(2));
    check("merge:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);
}

/* thermal + kinetic energy of live gas; pred=1 uses the predicted state */
static double gas_energy(int n, int pred)
{
    double e = 0;
    for(int i = 0; i < n; i++)
    {
        if(P[i].Mass <= 0 || P[i].Type != 0) {continue;}
        Vec3<double> v = pred ? Vec3<double>(CellP[i].VelPred) : Vec3<double>(P[i].Vel);
        e += P[i].Mass * ((pred ? CellP[i].InternalEnergyPred : CellP[i].InternalEnergy) + 0.5 * v.norm_sq() * All.cf_a2inv);
    }
    return e;
}

static void test_merge_energy()
{
    /* light cell into heavier one at ~Mach 0.8 relative speed: the merge must not lose (1/2) mu |dv|^2 */
    setup(64, 64);
    make_gas(0, 0.5, 0.5, 0.5, 0.4, 0.72, 0.27, 0.01, 0.);
    make_gas(1, 0.55, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.);
    P[0].Vel = Vec3<double>{5., 0., 0.}; P[1].Vel = Vec3<double>{-3., 0., 0.};
    CellP[0].VelPred = Vec3<double>{5.5, 0., 0.}; CellP[1].VelPred = Vec3<double>{-3.2, 0., 0.};
    NumPart = N_gas = 2;
    double e0 = gas_energy(2, 0), e0p = gas_energy(2, 1);
    double mu = 0.4 * 1.0 / 1.4, ke_com = 0.5 * mu * 64. * All.cf_a2inv;
#ifdef MAGNETIC
    double V0 = P[0].Mass / CellP[0].Density, V1 = P[1].Mass / CellP[1].Density;
    double eb0 = 0.5 * (CellP[0].B.norm_sq() / V0 + CellP[1].B.norm_sq() / V1);
#endif
    int ok = merge_particles_ij(0, 1);
    check("merge_energy:succeeded", ok == 1, ok, 1);
    double e1 = gas_energy(2, 0), e1p = gas_energy(2, 1);
    fprintf(out, "INFO merge_energy: E_before=%.10g E_after=%.10g dE/E=%.4e KE_com/E=%.4e\n", e0, e1, (e1 - e0) / e0, ke_com / e0);
#ifdef MAGNETIC
    double eb1 = 0.5 * CellP[1].B.norm_sq() / (V0 + V1); /* summed VB over summed volume until the next density pass */
    fprintf(out, "INFO merge_energy: E_B_before=%.6g E_B_after=%.6g (relative to E: %.3e)\n", eb0, eb1, (eb1 - eb0) / e0);
#endif
#ifndef GENERAL_RELATIVITY /* the merge returns the pair's centre-of-mass kinetic energy as heat */
    check_rel("merge_energy:thermal_plus_kinetic_conserved", e1, e0, e0);
    check_rel("merge_energy:predicted_state_conserved", e1p, e0p, e0p);
#endif
}

static void test_merge_rates()
{
    /* the merged cell's rates are per unit mass and carry both parents' total momentum and energy rates (MFM: no DtMass) */
    setup(64, 64);
    make_gas(0, 0.5, 0.5, 0.5, 0.4, 0.72, 0.27, 0.01, 0.);
    make_gas(1, 0.55, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.3);
    CellP[0].HydroAccel = Vec3<double>{2., -1., 0.5}; CellP[1].HydroAccel = Vec3<double>{-0.7, 0.4, 1.1};
    CellP[0].DtInternalEnergy = 30.; CellP[1].DtInternalEnergy = -12.;
    NumPart = N_gas = 2;
    double a = All.cf_atime, m0 = P[0].Mass, m1 = P[1].Mass, mtot = m0 + m1;
    Vec3<double> p_rate = CellP[0].HydroAccel * m0 + CellP[1].HydroAccel * m1;
#ifdef HYDRO_KICK_TOTAL_ENERGY_EXACT /* the velocity the kick turns the total-energy rate into du/dt with */
    CellP[0].HydroVelRef = CellP[0].VelPred / a; CellP[1].HydroVelRef = CellP[1].VelPred / a;
#define RATE_VEL(i) (CellP[i].HydroVelRef)
#else
#define RATE_VEL(i) (CellP[i].VelPred / a)
#endif
    double e_rate = m0 * (CellP[0].DtInternalEnergy + dot(RATE_VEL(0), CellP[0].HydroAccel)) + m1 * (CellP[1].DtInternalEnergy + dot(RATE_VEL(1), CellP[1].HydroAccel));
    int ok = merge_particles_ij(0, 1);
    check("merge_rates:succeeded", ok == 1, ok, 1);
    Vec3<double> p_rate1 = CellP[1].HydroAccel * mtot;
    double e_rate1 = mtot * (CellP[1].DtInternalEnergy + dot(RATE_VEL(1), CellP[1].HydroAccel));
    for(int k = 0; k < 3; k++) {check_rel("merge_rates:momentum_rate_conserved", p_rate1[k], p_rate[k], p_rate.norm());}
    check_rel("merge_rates:energy_rate_conserved_per_unit_mass", e_rate1, e_rate, fabs(e_rate));
}

static void test_nongas_never_touches_CellP()
{
    /* gas 0..3, collisionless filler, then two stars at indices past MaxPartGas */
    setup(48, 8);
    for(int i = 0; i < 4; i++) {make_gas(i, 0.2 + 0.1 * i, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.1 * i);}
    for(int i = 4; i < 30; i++) {make_other(i, 1, 1.0, 0.01 * i);}
    make_other(30, 4, 2.0, 0.40); make_other(31, 4, 3.0, 0.41); make_other(32, 4, 0.0, 0.42);
    NumPart = 33; N_gas = 4;
    double m_before = P[30].Mass + P[31].Mass;
    merge_particles_ij(30, 31); /* star-star merge */
    check_rel("star_merge:mass", P[30].Mass + P[31].Mass, m_before, m_before);
    merge_particles_ij(32, 31); /* early return on a zero-mass star (i) */
    merge_particles_ij(31, 32); /* early return on a zero-mass star (j) */
    check("star_merge:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);

#ifndef MAINTAIN_TREE_IN_REARRANGE /* that path edits the tree, which the driver does not build */
    /* rearrange_particle_sequence eliminates a zero-mass star past MaxPartGas */
    setup(48, 8);
    for(int i = 0; i < 4; i++) {make_gas(i, 0.2 + 0.1 * i, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.1 * i);}
    for(int i = 4; i < 30; i++) {make_other(i, 1, 1.0, 0.01 * i);}
    make_other(30, 4, 0.0, 0.42);
    NumPart = 31; N_gas = 4;
    rearrange_particle_sequence();
    check("rearrange:star_eliminated", NumPart == 30, NumPart, 30);
    check("rearrange:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);
#endif
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    gizmo_kokkos_initialize(argc, argv);
    if(argc < 2) {fprintf(stderr, "usage: %s params.txt\n", argv[0]); return 1;}

    /* the subset of begrun() merge/split depends on */
    RestartFlag = 0;
    strncpy(ParameterFile, argv[1], sizeof(ParameterFile)-1);
    read_parameter_file(ParameterFile);
    mymalloc_init();
    set_units();
    set_cosmo_factors_for_current_time();
    All.Time = All.TimeBegin;
    All.Timebase_interval = (All.TimeMax - All.TimeBegin) / TIMEBASE;
    set_softenings();
#ifdef METALS /* as init() sets them: the cooling chain divides by these */
    for(int k=0;k<NUM_METAL_SPECIES;k++) {All.SolarAbundances[k]=0;}
    All.SolarAbundances[0]=0.02;
#ifdef COOL_METAL_LINES_BY_SPECIES
    {const double sa[11]={0.02,0.28,3.26e-3,1.32e-3,8.65e-3,2.22e-3,9.31e-4,1.08e-3,6.44e-4,1.01e-4,1.73e-3}; for(int k=1;k<11 && k<NUM_METAL_SPECIES;k++) {All.SolarAbundances[k]=sa[k];}}
#endif
#endif
#if defined(COOLING) && !defined(SGCHEM)
    InitCool(); /* the merge calls set_eos_pressure, which reads the cooling tables */
#endif

    fflush(stdout);
    out = fdopen(dup(fileno(stdout)), "w");
    {FILE *w = tmpfile(); if(!w || dup2(fileno(w), fileno(stdout)) < 0) {fprintf(stderr, "could not divert stdout\n"); return 1;}}

    test_split();
    test_merge();
    test_merge_energy();
    test_merge_rates();
    test_nongas_never_touches_CellP();
    fprintf(out, "DONE checks=%d failed=%d\n", n_check, n_fail);
    fflush(out);
    gizmo_kokkos_finalize();
    MPI_Finalize();
    return n_fail;
}
