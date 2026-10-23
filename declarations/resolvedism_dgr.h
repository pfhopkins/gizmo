#ifndef RESOLVEDISM_DGR_H
#define RESOLVEDISM_DGR_H
/* Per-cell dust-to-gas ratio in Milky-Way units (1 = RESOLVEDISM_DGR_MW_ABS), shared by chemistry,
 * the TREECOL dust column, radiation pressure and the IR opacity. Live dust: sum(Dust[])/MW anchor;
 * no-dust proxy: DGRnormalized x Z/Zsun. Capped at 10 (certified one-zone domain). */
#define RESOLVEDISM_DGR_MW_C_ABS   1.24e-3 /* carbonaceous part: RESOLVEDISM_DEPL_C x Asplund09 C (Dwek 2005 1.9e-3 exceeds that C budget) */
#define RESOLVEDISM_DGR_MW_SIL_ABS 3.5e-3  /* silicate part (Dwek 2005, as Hu+23/26) */
#define RESOLVEDISM_DGR_MW_ABS     (RESOLVEDISM_DGR_MW_C_ABS + RESOLVEDISM_DGR_MW_SIL_ABS) /* = the solar seed */

static inline double resolvedism_cell_DGR_solar(int p)
{
    double dgr;
#ifdef GALSF_RESOLVEDISM_DUST
    {
        double DGR_actual = 0; int kd;
        for(kd = 0; kd < NUM_RESOLVEDISM_DUST; kd++) {DGR_actual += CellP[p].Dust[kd];}
        dgr = DGR_actual / RESOLVEDISM_DGR_MW_ABS;
    }
#elif defined(CHEMCOOL)
#ifdef METALS
    dgr = All.DGRnormalized * P[p].Metallicity[0] / All.SolarAbundances[0];
#else
    dgr = All.DGRnormalized;
#endif
#else
    dgr = 1.0; /* no chemistry: DGRnormalized not defined (pure N-body / non-Stella builds) */
#endif
    if(dgr > 10.0) {dgr = 10.0;}
    if(dgr < 0) {dgr = 0;}
#ifdef CHEMCOOL
    if(CellP[p].Temp > 1.0e5) {dgr = 0;} /* sputtering regime: same gate as the chemistry input (chem_interface.cc) */
#endif
    return dgr;
}
#endif
