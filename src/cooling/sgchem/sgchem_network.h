/* C++ implementation: Ulrich Steinwandel (uli@mpa-garching.mpg.de).
   Based on the SGChem Fortran networks by Simon Glover (GNU GPL, as distributed in Arepo). */
/* Per-network species facts used on the C side of SGCHEM: molecular weights of the tracked species (TracAbund, n_X/n_H),
   and, with chem_interface.h included first, the electrons per H, the photoionized (HII) state and the element-pool clamps.
   Networks 5 and 17 reproduce the code these replace; 3 and 15 follow their rate_eq files (rate_eq_mol.F, rate_eq_nl99.F). */
#ifndef SGCHEM_NETWORK_H
#define SGCHEM_NETWORK_H

#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 20   /* H2, H+, CO (net 20: CO is the equilibrium value) */
static const double sgchem_trac_molwt[] = {2.0, 1.0, 28.0};
#elif CHEMISTRYNETWORK == 18     /* H2, H+, CO, He0-He2, then every stage of C N O Ne Mg Si S Fe */
#include "cxx/chem_net18_layout.h"
static const double sgchem_trac_molwt[] = {2.0, 1.0, 28.0, 4.0, 4.0, 4.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0, 56.0};
#elif CHEMISTRYNETWORK == 17     /* H2, H+, CO, He+, He++, D+, HD */
static const double sgchem_trac_molwt[] = {2.0, 1.0, 28.0, 4.0, 4.0, 2.0, 3.0};
#elif CHEMISTRYNETWORK == 15     /* H2, H+, C+, CHx, OHx, CO, HCO+, He+, M+ (M: generic low-IP metal, taken as Mg) */
static const double sgchem_trac_molwt[] = {2.0, 1.0, 12.0, 13.0, 17.0, 28.0, 29.0, 4.0, 24.0};
#elif CHEMISTRYNETWORK == 16     /* Gong+17: as 15, with M = Si */
static const double sgchem_trac_molwt[] = {2.0, 1.0, 12.0, 13.0, 17.0, 28.0, 29.0, 4.0, 28.0};
#elif CHEMISTRYNETWORK == 3      /* H2, H+, C+, Si+, O+, D+, HD, He+, He++, CO, C2, OH, H2O, O2, HCO+, CH, CH2, Si++, CH3+ */
static const double sgchem_trac_molwt[] = {2.0, 1.0, 12.0, 28.0, 16.0, 2.0, 3.0, 4.0, 4.0, 28.0, 24.0, 17.0, 18.0, 32.0, 29.0, 13.0, 14.0, 28.0, 15.0};
#else
#error "SGCHEM: no species table for this CHEMISTRYNETWORK in cooling/sgchem/sgchem_network.h"
#endif
static_assert(sizeof(sgchem_trac_molwt)/sizeof(sgchem_trac_molwt[0]) == TRAC_NUM, "sgchem_trac_molwt must list TRAC_NUM species");

#ifdef CHEM_INTERFACE_H

/* electrons per H; species not tracked by the network count as singly ionized at their full pool */
static inline double sgchem_electrons(const double *ab, const struct ChemInputs *in)
{
#if CHEMISTRYNETWORK == 18
    (void)in;
    return n18_electrons_trac(ab);
#elif CHEMISTRYNETWORK == 20
    (void)ab; (void)in;
    return chem_cxx_net20_electrons();   /* equilibrium x_e of the cell just finished */
#elif CHEMISTRYNETWORK == 5
    return ab[IHP] + in->abundc + in->abundsi;
#elif CHEMISTRYNETWORK == 17
    return ab[IHP] + ab[IHEP] + 2.0*ab[IHEPP] + in->abundc + in->abundsi;
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    return ab[IHP] + ab[IC] + ab[IHCOP] + ab[IHEP] + ab[IMP];
#elif CHEMISTRYNETWORK == 3
    return ab[IHP] + ab[IDP] + ab[IHEP] + ab[IC] + ab[IO] + ab[ISi] + ab[IHCOP] + 2.0*ab[ISIPP] + ab[ICH3P] + 2.0*ab[IHEPP];
#endif
}

/* uniform ISRF-shaped (Draine 1978) field of strength G0 (6-13.6 eV) and G0_LW (11.2-13.6 eV), Habing units: band energies
   through the shared chem_fuv_field_from_band_energy, floors 0.324e-2 G0_bg (LW: a tenth in 6-13.6 eV energy) as stella_dev */
static inline struct ChemFUVField sgchem_uniform_field(double G0, double G0_LW, double G0_bg)
{
    double g0_floor = 0.324e-2 * G0_bg;
    double u_uv = DMAX(G0, g0_floor) * HABING_EGY_DENSITY_6_13P6EV_CGS;
    double u_lw = DMAX(G0_LW * HABING_EGY_DENSITY_11P2_13P6EV_CGS, 0.1 * g0_floor * HABING_EGY_DENSITY_6_13P6EV_CGS);
    return chem_fuv_field_from_band_energy(u_uv, HABING_EGY_DENSITY_6_13P6EV_CGS, u_lw, G0 * DUST_EGY_DENSITY_3P4_6EV_CGS,
                                           G0 * DUST_EGY_DENSITY_0P4_3P4EV_CGS, 1, 1, g0_floor);
}

/* photoionized cell: molecules destroyed, H, D, He and the low-IP metals singly ionized */
static inline void sgchem_set_ionized(double *ab, const struct ChemInputs *in)
{
    ab[IH2] = 0.0;
    ab[IHP] = 0.9998;
    ab[ICO] = 0.0;
#if CHEMISTRYNETWORK == 17
    ab[IHEP]  = in->abhe;     /* He -> He+ (13.6-54.4 eV photons) */
    ab[IHEPP] = 0.0;          /* He++ negligible for most O stars */
    ab[IDP]   = in->abundD;   /* D -> D+ (same IP as H) */
    ab[IHD]   = 0.0;          /* HD dissociated */
#elif CHEMISTRYNETWORK == 18
    {   /* He -> He+, metals singly ionized */
        const double pool[N18_NEL] = {1.0, in->abhe, in->abundc, in->abundN, in->abundo, in->abundNe, in->abundMg, in->abundsi, in->abundS, in->abundFe};
        int e, j;
        ab[N18_T_HE] = 0.0; ab[N18_T_HE + 1] = in->abhe; ab[N18_T_HE + 2] = 0.0;
        for(e = 2; e < N18_NEL; e++) for(j = 0; j <= n18_Z[e]; j++) ab[n18_toff[e] + j] = (j == 1) ? pool[e] : 0.0;
    }
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    ab[IC] = in->abundc; ab[ICHX] = 0.0; ab[IOHX] = 0.0; ab[IHCOP] = 0.0;
    ab[IHEP] = in->abhe; ab[IMP] = in->abundM;
#elif CHEMISTRYNETWORK == 3
    ab[IC] = in->abundc; ab[IO] = in->abundo; ab[ISi] = in->abundsi; ab[ISIPP] = 0.0;
    ab[IDP] = in->abundD; ab[IHD] = 0.0; ab[IHEP] = in->abhe; ab[IHEPP] = 0.0;
    ab[IC2] = ab[IOH] = ab[IH2O] = ab[IO2] = ab[IHCOP] = ab[ICH] = ab[ICH2] = ab[ICH3P] = 0.0;
#endif
}

/* scale species k[0..n-1] (n_k atoms of the element each) down so they hold at most `pool` of it */
static inline void sgchem_clamp_group(double *ab, const int *k, const double *n, int nk, double pool)
{
    double sum = 0; int i;
    if(pool < 0) {pool = 0;}
    for(i = 0; i < nk; i++) {sum += n[i] * ab[k[i]];}
    if(sum > pool && sum > 0) {double f = pool / sum; for(i = 0; i < nk; i++) {ab[k[i]] *= f;}}
}

/* species cannot hold more of an element than its gas-phase pool (upstream injection/diffusion can leave them inconsistent) */
static inline void sgchem_clamp_pools(double *ab, const struct ChemInputs *in)
{
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 17 || CHEMISTRYNETWORK == 20
    double co_pool = (in->abundc < in->abundo) ? in->abundc : in->abundo;
    if(co_pool < 0) co_pool = 0;
    if(ab[ICO] > co_pool) ab[ICO] = co_pool;  /* CO <= min(C_gas, O_gas) */
#if CHEMISTRYNETWORK == 17
    {   /* He+ + He++ <= abhe; D+ + HD <= abundD: rescale each pair proportionally */
        double he_sum = ab[IHEP] + ab[IHEPP];
        if(he_sum > in->abhe && he_sum > 0) {double f = in->abhe / he_sum; ab[IHEP] *= f; ab[IHEPP] *= f;}
        double d_sum = ab[IDP] + ab[IHD];
        if(d_sum > in->abundD && d_sum > 0) {double f = in->abundD / d_sum; ab[IDP] *= f; ab[IHD] *= f;}
    }
#endif
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    {const int k[] = {IC, ICHX, ICO, IHCOP}; const double n[] = {1, 1, 1, 1}; sgchem_clamp_group(ab, k, n, 4, in->abundc);}
    {const int k[] = {IOHX, ICO, IHCOP}; const double n[] = {1, 1, 1}; sgchem_clamp_group(ab, k, n, 3, in->abundo);}
    {const int k[] = {IHEP}; const double n[] = {1}; sgchem_clamp_group(ab, k, n, 1, in->abhe);}
    {const int k[] = {IMP}; const double n[] = {1}; sgchem_clamp_group(ab, k, n, 1, in->abundM);}
#elif CHEMISTRYNETWORK == 3
    {const int k[] = {IC, ICO, IC2, IHCOP, ICH, ICH2, ICH3P}; const double n[] = {1, 1, 2, 1, 1, 1, 1}; sgchem_clamp_group(ab, k, n, 7, in->abundc);}
    {const int k[] = {IO, ICO, IOH, IH2O, IO2, IHCOP}; const double n[] = {1, 1, 1, 1, 2, 1}; sgchem_clamp_group(ab, k, n, 6, in->abundo);}
    {const int k[] = {ISi, ISIPP}; const double n[] = {1, 1}; sgchem_clamp_group(ab, k, n, 2, in->abundsi);}
    {const int k[] = {IHEP, IHEPP}; const double n[] = {1, 1}; sgchem_clamp_group(ab, k, n, 2, in->abhe);}
    {const int k[] = {IDP, IHD}; const double n[] = {1, 1}; sgchem_clamp_group(ab, k, n, 2, in->abundD);}
#endif
    /* hydrogen pool, after the metal clamps (which only lower these species) */
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 20
    {const int k[] = {IH2, IHP}; const double n[] = {2, 1}; sgchem_clamp_group(ab, k, n, 2, 1.0);}
#elif CHEMISTRYNETWORK == 17
    {const int k[] = {IH2, IHP, IHD}; const double n[] = {2, 1, 1}; sgchem_clamp_group(ab, k, n, 3, 1.0);}
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    {const int k[] = {IH2, IHP, ICHX, IOHX, IHCOP}; const double n[] = {2, 1, 1, 1, 1}; sgchem_clamp_group(ab, k, n, 5, 1.0);}
#elif CHEMISTRYNETWORK == 3
    {const int k[] = {IH2, IHP, IHD, IOH, IH2O, IHCOP, ICH, ICH2, ICH3P}; const double n[] = {2, 1, 1, 1, 2, 1, 1, 2, 3}; sgchem_clamp_group(ab, k, n, 9, 1.0);}
#endif
}

#endif /* CHEM_INTERFACE_H */
#endif /* SGCHEM_NETWORK_H */
