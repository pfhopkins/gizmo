/* Per-network species facts used on the C side of CHEMCOOL: molecular weights of the tracked species (TracAbund, n_X/n_H),
   and, with chem_interface.h included first, the electrons per H, the photoionized (HII) state and the element-pool clamps.
   Networks 5 and 17 reproduce the code these replace; 3 and 15 follow their rate_eq files (rate_eq_mol.F, rate_eq_nl99.F). */
#ifndef CHEMCOOL_NETWORK_H
#define CHEMCOOL_NETWORK_H

#if CHEMISTRYNETWORK == 5        /* H2, H+, CO */
static const double chemcool_trac_molwt[] = {2.0, 1.0, 28.0};
#elif CHEMISTRYNETWORK == 17     /* H2, H+, CO, He+, He++, D+, HD */
static const double chemcool_trac_molwt[] = {2.0, 1.0, 28.0, 4.0, 4.0, 2.0, 3.0};
#elif CHEMISTRYNETWORK == 15     /* H2, H+, C+, CHx, OHx, CO, HCO+, He+, M+ (M: generic low-IP metal, taken as Mg) */
static const double chemcool_trac_molwt[] = {2.0, 1.0, 12.0, 13.0, 17.0, 28.0, 29.0, 4.0, 24.0};
#elif CHEMISTRYNETWORK == 16     /* Gong+17: as 15, with M = Si */
static const double chemcool_trac_molwt[] = {2.0, 1.0, 12.0, 13.0, 17.0, 28.0, 29.0, 4.0, 28.0};
#elif CHEMISTRYNETWORK == 3      /* H2, H+, C+, Si+, O+, D+, HD, He+, He++, CO, C2, OH, H2O, O2, HCO+, CH, CH2, Si++, CH3+ */
static const double chemcool_trac_molwt[] = {2.0, 1.0, 12.0, 28.0, 16.0, 2.0, 3.0, 4.0, 4.0, 28.0, 24.0, 17.0, 18.0, 32.0, 29.0, 13.0, 14.0, 28.0, 15.0};
#else
#error "CHEMCOOL: no species table for this CHEMISTRYNETWORK in cooling/chemcool/chemcool_network.h"
#endif
static_assert(sizeof(chemcool_trac_molwt)/sizeof(chemcool_trac_molwt[0]) == TRAC_NUM, "chemcool_trac_molwt must list TRAC_NUM species");

#ifdef CHEM_INTERFACE_H

/* electrons per H; species not tracked by the network count as singly ionized at their full pool */
static inline double chemcool_electrons(const double *ab, const struct ChemInputs *in)
{
#if CHEMISTRYNETWORK == 5
    return ab[IHP] + in->abundc + in->abundsi;
#elif CHEMISTRYNETWORK == 17
    return ab[IHP] + ab[IHEP] + 2.0*ab[IHEPP] + in->abundc + in->abundsi;
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    return ab[IHP] + ab[IC] + ab[IHCOP] + ab[IHEP] + ab[IMP];
#elif CHEMISTRYNETWORK == 3
    return ab[IHP] + ab[IDP] + ab[IHEP] + ab[IC] + ab[IO] + ab[ISi] + ab[IHCOP] + 2.0*ab[ISIPP] + ab[ICH3P] + 2.0*ab[IHEPP];
#endif
}

/* photoionized cell: molecules destroyed, H, D, He and the low-IP metals singly ionized */
static inline void chemcool_set_ionized(double *ab, const struct ChemInputs *in)
{
    ab[IH2] = 0.0;
    ab[IHP] = 0.9998;
    ab[ICO] = 0.0;
#if CHEMISTRYNETWORK == 17
    ab[IHEP]  = in->abhe;     /* He -> He+ (13.6-54.4 eV photons) */
    ab[IHEPP] = 0.0;          /* He++ negligible for most O stars */
    ab[IDP]   = in->abundD;   /* D -> D+ (same IP as H) */
    ab[IHD]   = 0.0;          /* HD dissociated */
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
static inline void chemcool_clamp_group(double *ab, const int *k, const double *n, int nk, double pool)
{
    double sum = 0; int i;
    if(pool < 0) {pool = 0;}
    for(i = 0; i < nk; i++) {sum += n[i] * ab[k[i]];}
    if(sum > pool && sum > 0) {double f = pool / sum; for(i = 0; i < nk; i++) {ab[k[i]] *= f;}}
}

/* species cannot hold more of an element than its gas-phase pool (upstream injection/diffusion can leave them inconsistent) */
static inline void chemcool_clamp_pools(double *ab, const struct ChemInputs *in)
{
#if CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 17
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
    {const int k[] = {IC, ICHX, ICO, IHCOP}; const double n[] = {1, 1, 1, 1}; chemcool_clamp_group(ab, k, n, 4, in->abundc);}
    {const int k[] = {IOHX, ICO, IHCOP}; const double n[] = {1, 1, 1}; chemcool_clamp_group(ab, k, n, 3, in->abundo);}
    {const int k[] = {IHEP}; const double n[] = {1}; chemcool_clamp_group(ab, k, n, 1, in->abhe);}
    {const int k[] = {IMP}; const double n[] = {1}; chemcool_clamp_group(ab, k, n, 1, in->abundM);}
#elif CHEMISTRYNETWORK == 3
    {const int k[] = {IC, ICO, IC2, IHCOP, ICH, ICH2, ICH3P}; const double n[] = {1, 1, 2, 1, 1, 1, 1}; chemcool_clamp_group(ab, k, n, 7, in->abundc);}
    {const int k[] = {IO, ICO, IOH, IH2O, IO2, IHCOP}; const double n[] = {1, 1, 1, 1, 2, 1}; chemcool_clamp_group(ab, k, n, 6, in->abundo);}
    {const int k[] = {ISi, ISIPP}; const double n[] = {1, 1}; chemcool_clamp_group(ab, k, n, 2, in->abundsi);}
    {const int k[] = {IHEP, IHEPP}; const double n[] = {1, 1}; chemcool_clamp_group(ab, k, n, 2, in->abhe);}
    {const int k[] = {IDP, IHD}; const double n[] = {1, 1}; chemcool_clamp_group(ab, k, n, 2, in->abundD);}
#endif
    /* hydrogen pool, after the metal clamps (which only lower these species) */
#if CHEMISTRYNETWORK == 5
    {const int k[] = {IH2, IHP}; const double n[] = {2, 1}; chemcool_clamp_group(ab, k, n, 2, 1.0);}
#elif CHEMISTRYNETWORK == 17
    {const int k[] = {IH2, IHP, IHD}; const double n[] = {2, 1, 1}; chemcool_clamp_group(ab, k, n, 3, 1.0);}
#elif CHEMISTRYNETWORK == 15 || CHEMISTRYNETWORK == 16
    {const int k[] = {IH2, IHP, ICHX, IOHX, IHCOP}; const double n[] = {2, 1, 1, 1, 1}; chemcool_clamp_group(ab, k, n, 5, 1.0);}
#elif CHEMISTRYNETWORK == 3
    {const int k[] = {IH2, IHP, IHD, IOH, IH2O, IHCOP, ICH, ICH2, ICH3P}; const double n[] = {2, 1, 1, 1, 2, 1, 1, 2, 3}; chemcool_clamp_group(ab, k, n, 9, 1.0);}
#endif
}

#endif /* CHEM_INTERFACE_H */
#endif /* CHEMCOOL_NETWORK_H */
