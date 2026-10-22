#ifndef RESOLVEDISM_DGR_H
#define RESOLVEDISM_DGR_H
/* Single source of truth for the per-cell dust-to-gas ratio in SOLAR units
 * (1.0 = MW-like DGR_solar ~ 0.01 absolute), shared by chemistry (chemcool.cc),
 * the dust-weighted tree column (forcetree.cc) and radiation pressure
 * (resolvedism_fb_momentum.cc), so attenuation, absorption and heating all see
 * the SAME dust field. Mirrors the chemcool.cc convention exactly (2026-07-28):
 * evolved-dust branch sums Dust[]/0.01; no-dust branch scales the dust-to-metal
 * efficiency DGRnormalized with the LOCAL metallicity (constant DGRnormalized with
 * CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY); capped at 10x solar (the
 * certified one-zone domain — fresh unmixed ejecta would otherwise reach ~50x). */
#if defined(CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY) && defined(GALSF_RESOLVEDISM_DUST)
#error "CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY has no effect with GALSF_RESOLVEDISM_DUST (live dust sets the DGR)"
#endif
#define RESOLVEDISM_DGR_SOLAR_ABS 0.01  /* absolute MW dust-to-gas mass ratio at DGR=1 (solar units) */

/* raw DGR formula, no cap/gate (chemistry applies its own cap and solve-T sputter gate) */
static inline double resolvedism_cell_DGR_solar_raw(int p)
{
    double dgr;
#ifdef GALSF_RESOLVEDISM_DUST
    {
        double DGR_actual = 0; int kd;
        for(kd = 0; kd < NUM_RESOLVEDISM_DUST; kd++) {DGR_actual += CellP[p].Dust[kd];}
        dgr = DGR_actual / RESOLVEDISM_DGR_SOLAR_ABS;
    }
#elif defined(CHEMCOOL)
#if defined(METALS) && !defined(CHEMCOOL_DGR_INDEPENDENT_OF_METALLICITY)
    dgr = All.DGRnormalized * P[p].Metallicity[0] / All.SolarAbundances[0];
#else
    dgr = All.DGRnormalized;
#endif
#else
    dgr = 1.0; /* no chemistry: DGRnormalized not defined (pure N-body / non-Stella builds) */
#endif
    return dgr;
}

static inline double resolvedism_cell_DGR_solar(int p)
{
    double dgr = resolvedism_cell_DGR_solar_raw(p);
    if(dgr > 10.0) {dgr = 10.0;}
    if(dgr < 0) {dgr = 0;}
#ifdef CHEMCOOL
    if(CellP[p].Temp > 1.0e5) {dgr = 0;} /* sputtering regime: same gate as the chemistry input (chem_interface.cc) */
#endif
    return dgr;
}
#endif
