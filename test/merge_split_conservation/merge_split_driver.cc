/* Driver for test_merge_split_conservation.py: runs the production split/merge/SN-spawn code on hand-built cells and prints
   PASS|FAIL per conserved quantity. CellP has a canary region past MaxPartGas to catch writes for non-gas indices. */
#include <mpi.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "declarations/allvars.h"
#include "core/proto.h"

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
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL)
    /* [0]=Z, [1]=H, [2]=He, [3..]=metals, sum([1..]) = 1 */
    int n_met = NUM_METAL_SPECIES - 3; double zsum = 0;
    for(int k = 3; k < NUM_METAL_SPECIES; k++) {P[i].Metallicity[k] = Z_met * (k - 2) / (0.5 * n_met * (n_met + 1)); zsum += P[i].Metallicity[k];}
    P[i].Metallicity[0] = zsum; P[i].Metallicity[MET_OF(ELEM_H)] = X_H; P[i].Metallicity[MET_OF(ELEM_He)] = Y_He;
#else
    P[i].Metallicity[0] = Z_met;
    for(int k = 1; k < NUM_METAL_SPECIES; k++) {P[i].Metallicity[k] = Z_met * 0.01 * k;}
#if (NUM_LIVE_SPECIES_FOR_COOLTABLES >= 10)
    P[i].Metallicity[1] = Y_He;
#endif
#endif
#endif
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
    for(int k = 0; k < NUM_METAL_SPECIES + NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION; k++) {CellP[i].Dyield_pending[k] = 1e-3 * mass * (k % 2 ? -1. : 1.) * (1. + seed + 0.1 * k);}
#endif
#if defined(GALSF_RESOLVEDISM_DUST)
    for(int k = 0; k < NUM_RESOLVEDISM_DUST; k++) {CellP[i].Dust[k] = 1e-3 * (k + 1) * (1. + seed);}
#endif
#if defined(CHEMCOOL)
    for(int k = 0; k < TRAC_NUM; k++) {CellP[i].TracAbund[k] = 1e-2 * (k + 1) * (1. + 2. * seed);}
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
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
static const int NPEND = NUM_METAL_SPECIES + NUM_ADDITIONAL_PASSIVESCALAR_SPECIES_FOR_YIELDS_AND_DIFFUSION;
#else
static const int NPEND = 0;
#endif

struct Totals {double M = 0, p[3] = {0, 0, 0}, B[3] = {0, 0, 0}, E_sink = 0, H2 = 0, CO = 0; std::vector<double> MZ, pend, dust;};

static Totals totals(int n)
{
    Totals t; t.MZ.assign(NZ, 0.); t.pend.assign(NPEND, 0.);
#if defined(GALSF_RESOLVEDISM_DUST)
    t.dust.assign(NUM_RESOLVEDISM_DUST, 0.);
#endif
    for(int i = 0; i < n; i++)
    {
        if(P[i].Mass <= 0) {continue;} /* merged away */
        t.M += P[i].Mass; for(int k = 0; k < 3; k++) {t.p[k] += P[i].Mass * P[i].Vel[k];}
#ifdef METALS
        for(int k = 0; k < NZ; k++) {t.MZ[k] += P[i].Mass * P[i].Metallicity[k];}
#endif
        if(P[i].Type != 0) {continue;}
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
        for(int k = 0; k < NPEND; k++) {t.pend[k] += CellP[i].Dyield_pending[k];}
#endif
#if defined(GALSF_RESOLVEDISM_DUST)
        for(int k = 0; k < NUM_RESOLVEDISM_DUST; k++) {t.dust[k] += P[i].Mass * CellP[i].Dust[k];}
#endif
#ifdef MAGNETIC
        for(int k = 0; k < 3; k++) {t.B[k] += CellP[i].B[k];}
#endif
#if defined(SINK_THERMALFEEDBACK)
        t.E_sink += CellP[i].Injected_Sink_Energy;
#endif
#if defined(CHEMCOOL) && defined(HAVE_PARTICLE_HYDROGEN_MASSFRAC)
        double X_H = particle_hydrogen_massfrac(i); /* TracAbund are n_X/n_H */
        t.H2 += CellP[i].TracAbund[0] * 2. * X_H * P[i].Mass;
        if(TRAC_NUM > 2) {t.CO += CellP[i].TracAbund[2] * 28. * X_H * P[i].Mass;}
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
    double pend_scale = 0; for(int k = 0; k < NPEND; k++) {pend_scale += fabs(a.pend[k]);}
    for(int k = 0; k < NPEND; k++) {snprintf(name, sizeof(name), "%s:pending_metal_exchange[%d]", what, k); check_rel(name, b.pend[k], a.pend[k], pend_scale);}
    for(size_t k = 0; k < a.dust.size(); k++) {snprintf(name, sizeof(name), "%s:dust_mass[%zu]", what, k); check_rel(name, b.dust[k], a.dust[k], a.dust[k]);}
#ifdef MAGNETIC
    double bscale = fabs(a.B[0]) + fabs(a.B[1]) + fabs(a.B[2]);
    for(int k = 0; k < 3; k++) {snprintf(name, sizeof(name), "%s:B_conserved[%d]", what, k); check_rel(name, b.B[k], a.B[k], bscale);}
#endif
#if defined(SINK_THERMALFEEDBACK)
    snprintf(name, sizeof(name), "%s:injected_sink_energy", what); check_rel(name, b.E_sink, a.E_sink, a.E_sink);
#endif
#if defined(CHEMCOOL) && defined(HAVE_PARTICLE_HYDROGEN_MASSFRAC)
    snprintf(name, sizeof(name), "%s:H2_mass", what); check_rel(name, b.H2, a.H2, a.H2);
    if(TRAC_NUM > 2) {snprintf(name, sizeof(name), "%s:CO_mass", what); check_rel(name, b.CO, a.CO, a.CO);}
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
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
    check("split:daughter_starts_with_no_pending_exchange", CellP[3].Dyield_pending[0] == 0, CellP[3].Dyield_pending[0], 0);
#endif
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
#if defined(HAVE_PARTICLE_HYDROGEN_MASSFRAC) && defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL)
    double X_H = particle_hydrogen_massfrac(1), want = (0.8 * 0.30 + 1.5 * 0.74) / 2.3;
    check_rel("merge:merged_X_H_is_mass_weighted", X_H, want, want);
#endif
    check("merge:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);
}

static void test_hydrogen_massfrac()
{
#ifdef HAVE_PARTICLE_HYDROGEN_MASSFRAC
    setup(8, 8);
    make_gas(0, 0.5, 0.5, 0.5, 1.0, 0.71, 0.27, 0.02, 0.);
    double X_H = particle_hydrogen_massfrac(0), want;
#if defined(GALSF_RESOLVEDISM_METALS_INDIVIDUAL) || (NUM_LIVE_SPECIES_FOR_COOLTABLES >= 10)
    want = 1. - P[0].Metallicity[0] - 0.27; /* helium is tracked: X = 1 - Z - Y */
#else
    want = HYDROGEN_MASSFRAC;
#endif
    check_rel("hydrogen_massfrac:matches_layout", X_H, want, want);
    check("hydrogen_massfrac:is_not_total_Z", fabs(X_H - P[0].Metallicity[0]) > 0.1, X_H, P[0].Metallicity[0]);
#endif
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

#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
void hydro_final_operations_and_cleanup(void);
static void test_metal_floor()
{
    /* an exchange taking 99.5% of a cell's metal is applied as is */
    setup(8, 8);
    make_gas(0, 0.5, 0.5, 0.5, 1.0, 0.72, 0.27, 0.02, 0.);
    NumPart = N_gas = 1;
    for(int k = 0; k < NPEND; k++) {CellP[0].Dyield_pending[k] = CellP[0].Dyield[k] = 0;}
    double MZ0 = P[0].Mass * P[0].Metallicity[0], dMZ = -0.995 * MZ0;
    CellP[0].Dyield[0] = dMZ;
    ActiveParticleList.assign(1, 0);
    hydro_final_operations_and_cleanup();
    check_rel("metal_floor:exchange_applied_exactly", P[0].Mass * P[0].Metallicity[0], MZ0 + dMZ, MZ0);
}
#endif

#if defined(GALSF_RESOLVEDISM_SN_SPAWN) && defined(GALSF_RESOLVEDISM_FB)
static void make_sn_star(int i, double mass, double ej_mass, double ej_energy, double x)
{
    make_other(i, 4, mass, x);
    P[i].SN_SpawnPending = 1; P[i].SpawnEjMass = ej_mass; P[i].SpawnEjEnergy = ej_energy;
    double w = 0; for(int e = 0; e < NUM_RESOLVEDISM_ELEMENTS; e++) {w += 1. / (1 + e);}
    for(int e = 0; e < NUM_RESOLVEDISM_ELEMENTS; e++) {P[i].SpawnEjZ[e] = ej_mass * (1. / (1 + e)) / w;}
}

static void test_sn_spawn()
{
    /* (a) ordinary spawn: mass moves star -> new cell, which holds only ejecta */
    setup(64, 64);
    for(int i = 0; i < 6; i++) {make_gas(i, 0.3 + 0.05 * i, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.1 * i);}
    make_sn_star(6, 20.0, 5.0, 40.0, 0.31);
    NumPart = 7; N_gas = 6;
    double m_before = 0; for(int i = 0; i < NumPart; i++) {m_before += P[i].Mass;}
    merge_and_split_particles();
    int j = 7; /* new cell, before rearrange_particle_sequence */
    double m_after = 0; for(int i = 0; i <= j; i++) {m_after += P[i].Mass;}
    check("sn_spawn:cell_created", P[j].Type == 0 && P[j].Mass > 0, P[j].Mass, 5.0);
    check_rel("sn_spawn:mass_conserved", m_after, m_before, m_before);
    check_rel("sn_spawn:specific_energy", CellP[j].InternalEnergy, 40.0 / 5.0, 8.0);
#ifdef MAGNETIC
    double bt = CellP[0].B.norm(), b = CellP[j].B.norm(); /* template is cell 0 */
    check_rel("sn_spawn:B_scaled_to_ejecta_mass", b, bt * 5.0 / P[0].Mass, bt);
#endif
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
    double pend = 0; for(int k = 0; k < NPEND; k++) {pend += fabs(CellP[j].Dyield_pending[k]);}
    check("sn_spawn:no_template_pending_exchange", pend == 0, pend, 0);
#endif
#if defined(GALSF_RESOLVEDISM_DUST)
    double dust = 0; for(int k = 0; k < NUM_RESOLVEDISM_DUST; k++) {dust += CellP[j].Dust[k];}
    check("sn_spawn:no_template_dust", dust == 0, dust, 0);
#endif
#ifdef WAKEUP
    check("sn_spawn:woken", P[j].wakeup != 0, P[j].wakeup, -2);
#endif
    check("sn_spawn:payload_cleared", P[6].SN_SpawnPending == 0, P[6].SN_SpawnPending, 0);

    /* (b) a star holding less than its ejecta: the payload scales down, mass is conserved */
    setup(64, 64);
    for(int i = 0; i < 6; i++) {make_gas(i, 0.3 + 0.05 * i, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.1 * i);}
    make_sn_star(6, 3.0, 5.0, 40.0, 0.31);
    NumPart = 7; N_gas = 6;
    m_before = 0; for(int i = 0; i < NumPart; i++) {m_before += P[i].Mass;}
    merge_and_split_particles();
    m_after = 0; for(int i = 0; i <= 7; i++) {m_after += P[i].Mass;}
    check_rel("sn_spawn_overdrawn:mass_conserved", m_after, m_before, m_before);
    check_rel("sn_spawn_overdrawn:specific_energy_kept", CellP[7].InternalEnergy, 40.0 / 5.0, 8.0);

    /* (c) no room left for gas: defer, and never write CellP past MaxPartGas */
    setup(64, 10);
    for(int i = 0; i < 6; i++) {make_gas(i, 0.3 + 0.05 * i, 0.5, 0.5, 1.0, 0.72, 0.27, 0.01, 0.1 * i);}
    for(int i = 6; i < 12; i++) {make_other(i, 1, 1.0, 0.01 * i);}
    make_sn_star(12, 20.0, 5.0, 40.0, 0.31);
    NumPart = 13; N_gas = 6;
    merge_and_split_particles();
    check("sn_spawn_full:deferred", P[12].SN_SpawnPending == 1, P[12].SN_SpawnPending, 1);
    check("sn_spawn_full:no_CellP_write_past_MaxPartGas", canary_intact(), canary_intact(), 1);
}
#endif

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    if(argc < 2) {fprintf(stderr, "usage: %s params.txt\n", argv[0]); return 1;}

    /* the subset of begrun() merge/split depends on */
    memset(&All, 0, sizeof(All));
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
#if defined(COOLING) && !defined(CHEMCOOL)
    InitCool(); /* the merge calls set_eos_pressure, which reads the cooling tables */
#endif

    fflush(stdout);
    out = fdopen(dup(fileno(stdout)), "w");
    {FILE *w = tmpfile(); if(!w || dup2(fileno(w), fileno(stdout)) < 0) {fprintf(stderr, "could not divert stdout\n"); return 1;}}

    test_split();
    test_merge();
    test_hydrogen_massfrac();
    test_nongas_never_touches_CellP();
#if defined(TURB_DIFF_METALS_EXACT_CONSERVATION)
    test_metal_floor();
#endif
#if defined(GALSF_RESOLVEDISM_SN_SPAWN) && defined(GALSF_RESOLVEDISM_FB)
    test_sn_spawn();
#endif
    fprintf(out, "DONE checks=%d failed=%d\n", n_check, n_fail);
    fflush(out);
    MPI_Finalize();
    return n_fail;
}
