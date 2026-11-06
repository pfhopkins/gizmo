/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* GALSF_RESOLVEDISM_TREECOL_PIXELS=N: per-direction columns (N HEALPix pixels) and stellar band fluxes accumulated in the
   gravity walk for the SGChem chemistry. Shared by the node-moment paths and the host and device walks. */
#ifndef GRAVTREE_TREECOL_PIXELS_H
#define GRAVTREE_TREECOL_PIXELS_H
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS

#include "../cooling/sgchem/sgchem_consts.h"
#include "../cooling/sgchem/sgchem_dust_to_gas.h"
#include "../cooling/sgchem/chem_interface.h" /* band energy densities of the Habing field */

/* node species masses: dust-weighted gas (dust-to-gas ratio x mass), H2 and CO */
#define TREECOL_PIXELS_DUST 0
#define TREECOL_PIXELS_H2   1
#define TREECOL_PIXELS_CO   2

/* a particle's species masses [code units]; zero for non-gas. H nuclei per mass as the chemistry, 1/((1+4 ABHE) m_p) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_species_masses(const struct particle_data *pp, const struct gas_cell_data *cell, int i, double *m)
{
    m[TREECOL_PIXELS_DUST] = m[TREECOL_PIXELS_H2] = m[TREECOL_PIXELS_CO] = 0;
    if(pp[i].Type != 0) {return;}
    double dgr = (cell[i].Temp > 1.0e5) ? 0 : fmax(sgchem_cell_dust_to_gas_ratio(pp, cell, i), 0.); /* hot gas: grains sputtered */
    m[TREECOL_PIXELS_DUST] = dgr * pp[i].Mass;
    double m_H = pp[i].Mass / (1.0 + 4.0 * ABHE);
    m[TREECOL_PIXELS_H2] = 2.0 * cell[i].TracAbund[IH2] * m_H;
    m[TREECOL_PIXELS_CO] = 28.0 * cell[i].TracAbund[ICO] * m_H;
}


#define TREECOL_PIXELS_N GALSF_RESOLVEDISM_TREECOL_PIXELS

/* HEALPix RING pixel from z = cos(theta) and tt = phi/(pi/2) in [0,4] (Gorski+05) */
KOKKOS_INLINE_FUNCTION int treecol_hpx_ring_pix(int nside, double z, double tt)
{
    double za = fabs(z);
    if(za <= 2.0/3.0)
    {
        double t1 = nside * (0.5 + tt), t2 = nside * z * 0.75;
        long jp = (long)(t1 - t2), jm = (long)(t1 + t2), ir = nside + 1 + jp - jm, kshift = 1 - (ir & 1);
        long ip = ((jp + jm - nside + kshift + 1) / 2) % (4 * nside);
        return (int)(nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip);
    }
    double tp = tt - (long)tt, tmp = nside * sqrt(3.0 * (1.0 - za));
    long jp = (long)(tp * tmp), jm = (long)((1.0 - tp) * tmp), ir = jp + jm + 1, ip = ((long)(tt * ir)) % (4 * ir);
    return (int)((z > 0) ? 2 * ir * (ir - 1) + ip : 12 * nside * nside - 2 * ir * (ir + 1) + ip);
}

/* atan2-free tt = phi/(pi/2) in [0,4): polynomial atan on one octant, |error| < TREECOL_HPX_TT_ERR */
#define TREECOL_HPX_TT_ERR 1e-8
KOKKOS_INLINE_FUNCTION double treecol_hpx_tt_approx(double x, double y)
{
    double ax = fabs(x), ay = fabs(y), t = fmin(ax, ay) / fmax(ax, ay), q = t * t, a;
    a = t * (0.9999999975460201 + q * (-0.3333328229551207 + q * (0.1999823064037613 + q * (-0.14261573680246378 + q * (0.10940198964830647
        + q * (-0.0837206394697871 + q * (0.05746355782881671 + q * (-0.030717508884063723 + q * (0.010680719435017704 + q * (-0.0017437011413061324))))))))));
    a += (ay > ax ? 1.0 : 0.0) * (0.5 * M_PI - 2.0 * a);
    a += (x < 0 ? 1.0 : 0.0) * (M_PI - 2.0 * a);
    a *= (y < 0 ? -1.0 : 1.0);
    a += (a < 0 ? 2.0 * M_PI : 0.0);
    return a * (2.0 / M_PI);
}

/* the ring pixel when no edge decision can change within the tt error, else -1 */
KOKKOS_INLINE_FUNCTION int treecol_hpx_ring_pix_checked(int nside, double z, double tt)
{
    double M = (8.0 * nside + 8.0) * (TREECOL_HPX_TT_ERR * (2.0 / M_PI)) * 4.0, M1 = 1.0 - M, za = fabs(z);
    if(za <= 2.0/3.0)
    {
        double t1 = nside * (0.5 + tt), t2 = nside * z * 0.75, a = t1 - t2, b = t1 + t2, fa = floor(a), fb = floor(b);
        if(a - fa < M || a - fa > M1 || b - fb < M || b - fb > M1) {return -1;}
        long jp = (long)fa, jm = (long)fb, ir = nside + 1 + jp - jm, kshift = 1 - (ir & 1);
        long ip = ((jp + jm - nside + kshift + 1) / 2) % (4 * nside);
        return (int)(nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip);
    }
    double ft = floor(tt), tp = tt - ft;
    if(tp < M || tp > M1) {return -1;}
    double tmp = nside * sqrt(3.0 * (1.0 - za)), u = tp * tmp, v = (1.0 - tp) * tmp, fu = floor(u), fv = floor(v);
    if(u - fu < M || u - fu > M1 || v - fv < M || v - fv > M1) {return -1;}
    long ir = (long)fu + (long)fv + 1;
    double w = tt * ir, fw = floor(w);
    if(w - fw < M || w - fw > M1) {return -1;}
    long ip = ((long)fw) % (4 * ir);
    return (int)((z > 0) ? 2 * ir * (ir - 1) + ip : 12 * nside * nside - 2 * ir * (ir + 1) + ip);
}

/* HEALPix RING pixel of the direction d (= vec2pix_ring); atan2 only near pixel edges */
KOKKOS_INLINE_FUNCTION int treecol_hpx_pixel(const Vec3<double> &d)
{
    double x = d[0], y = d[1], r = sqrt(x * x + y * y + d[2] * d[2]);
    if(r == 0) {return 0;}
    double z = d[2] / r, ax = fabs(x), ay = fabs(y);
    if(ax > 1e-100 && ay > 1e-100 && ax < 1e100 && ay < 1e100 && fabs(z) <= 1.0)
    {
        int ip = treecol_hpx_ring_pix_checked(TREECOL_PIXELS_NSIDE, z, treecol_hpx_tt_approx(x, y));
        if(ip >= 0) {return ip;}
    }
    double phi = atan2(y, x); if(phi < 0) {phi += 2.0 * M_PI;}
    return treecol_hpx_ring_pix(TREECOL_PIXELS_NSIDE, z, phi / (0.5 * M_PI));
}

#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
/* band luminosity times its dust cross-section [1e-21 cm^2 per H at solar DGR] (stella_dev). Counted twice: LW and 6-8 eV, as production */
KOKKOS_INLINE_FUNCTION double treecol_radp_sigma_lum(const double *lum)
{
    double s = 2.0 * (lum[RT_FREQ_BIN_PHOTOELECTRIC] + lum[RT_FREQ_BIN_GENERIC_USER_FREQ]) + 1.3 * lum[RT_FREQ_BIN_NUV] + 0.5 * lum[RT_FREQ_BIN_OPTICAL_NIR];
#ifdef GALSF_FB_RADPRESSURE_COUNT_6_8EV_TWICE
    s += 2.0 * lum[RT_FREQ_BIN_LYMAN_WERNER] + 1.3 * lum[RT_FREQ_BIN_GENERIC_USER_FREQ];
#endif
    return s;
}

/* unit vector of HEALPix RING pixel ip, NSIDE 1 (pix2vec_ring) */
KOKKOS_INLINE_FUNCTION Vec3<double> treecol_radp_pixel_centre(int ip)
{
    const double q = 0.7453559924999299; /* sqrt(1 - (2/3)^2) / 1 */
    const double c = 0.7071067811865476; /* cos(pi/4) */
    Vec3<double> n = {};
    if(ip < 4 || ip >= 8)
    {
        const int k = (ip < 4) ? ip : ip - 8; /* phi = pi/4 + k pi/2 */
        n[0] = q * ((k == 0 || k == 3) ? c : -c); n[1] = q * ((k < 2) ? c : -c); n[2] = (ip < 4) ? 2.0 / 3.0 : -2.0 / 3.0;
        return n;
    }
    const int k = ip - 4; /* phi = k pi/2 */
    n[0] = (k == 0) ? 1.0 : ((k == 2) ? -1.0 : 0.0); n[1] = (k == 1) ? 1.0 : ((k == 3) ? -1.0 : 0.0);
    return n;
}
#endif

/* per-target sums during the walk: species columns and band fluxes per pixel, three band totals */
struct treecol_pixels_sums_t {
    double col[TREECOL_PIXELS_NSPECIES][TREECOL_PIXELS_N]; /* [code mass / code length^2] */
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    double uv[TREECOL_PIXELS_N], lw[TREECOL_PIXELS_N];    /* 6-13.6 (PE + 6-8) and 11.2-13.6 eV [erg/s / code length^2] */
    double nuv, opt, f68;                                  /* 3.444-6, 0.4133-3.444, 6-8 eV, totals */
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    double radp[4];                                        /* sigma-weighted flux: vector over pixel centres, scalar */
#endif
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    double gas[TREECOL_PIXELS_N];                          /* gas column [code mass / code length^2] */
#endif
};

KOKKOS_INLINE_FUNCTION void treecol_pixels_zero(treecol_pixels_sums_t &a)
{
    for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.col[s][k] = 0;}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.uv[k] = a.lw[k] = 0;}
    a.nuv = a.opt = a.f68 = 0;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {a.radp[k] = 0;}
#endif
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.gas[k] = 0;}
#endif
}

/* combine two partial sums (lane partials, in the engine's fixed fold order) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_add(treecol_pixels_sums_t &a, const treecol_pixels_sums_t &b)
{
    for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.col[s][k] += b.col[s][k];}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.uv[k] += b.uv[k]; a.lw[k] += b.lw[k];}
    a.nuv += b.nuv; a.opt += b.opt; a.f68 += b.f68;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {a.radp[k] += b.radp[k];}
#endif
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.gas[k] += b.gas[k];}
#endif
}

/* one accepted source (leaf or node) for a gas target: column in the pixel of its mass centre dr, inside
   TreeColShieldingLength (0 = no limit), unsoftened; band fluxes L/r_L^2 in the pixel of the light
   centroid d_lum (a second lookup only for luminous nodes, whose light and mass centres differ), r_L floored at r_floor
   (the target's Plummer-equivalent softening) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_accumulate(treecol_pixels_sums_t &a, const Vec3<double> &dr, double r, double fac_accel,
                                                      double mass, double gasmass, const double *species, const Vec3<double> &d_lum, const double *lum, double r_floor)
{
    if(r <= 0 || mass <= 0) {return;}
    const int ip = treecol_hpx_pixel(dr);
    if(All.TreeColShieldingLength <= 0 || r < All.TreeColShieldingLength / All.cf_atime)
    {
        const double w = 1.0 / ((4.0 * M_PI / TREECOL_PIXELS_N) * r * r); /* unsoftened, as stella_dev */
        for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {a.col[s][ip] += w * species[s];}
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
        a.gas[ip] += w * gasmass;
#endif
    }
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    const double l_pe = lum[RT_FREQ_BIN_PHOTOELECTRIC], l_68 = lum[RT_FREQ_BIN_GENERIC_USER_FREQ], l_lw = lum[RT_FREQ_BIN_LYMAN_WERNER];
    const double l_nuv = lum[RT_FREQ_BIN_NUV], l_opt = lum[RT_FREQ_BIN_OPTICAL_NIR];
    if(l_pe + l_68 > 0 || l_lw > 0 || l_nuv > 0 || l_opt > 0)
    {
        const double f = UNIT_LUM_IN_CGS / fmax(d_lum.norm_sq(), r_floor * r_floor);
        const int il = (d_lum[0] == dr[0] && d_lum[1] == dr[1] && d_lum[2] == dr[2]) ? ip : treecol_hpx_pixel(d_lum);
        a.uv[il] += (l_pe + l_68) * f; a.lw[il] += l_lw * f;
        a.nuv += l_nuv * f; a.opt += l_opt * f; a.f68 += l_68 * f;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
        const double fw = treecol_radp_sigma_lum(lum) / fmax(d_lum.norm_sq(), r_floor * r_floor);
        const Vec3<double> nh = treecol_radp_pixel_centre(il);
        a.radp[0] += fw * nh[0]; a.radp[1] += fw * nh[1]; a.radp[2] += fw * nh[2]; a.radp[3] += fw;
#endif
    }
#else
    (void) d_lum; (void) lum; (void) r_floor;
#endif
    (void) gasmass;
}

/* per-pixel cell storage: 16-bit log10 fixed point over 20 dex above log10_min (step 3.05e-4 dex, rounded to nearest),
   0 = zero, saturating at the top, x scaled to the range unit; float with GALSF_RESOLVEDISM_TREECOL_PIXELS_STORED_AS_FLOAT */
#define TREECOL_PIXELS_LOG16_SPAN 20.0
#define TREECOL_PIXELS_LOG16_MIN_COLUMN (-16.0) /* [g/cm^2 at the code normalisation] */
#define TREECOL_PIXELS_LOG16_MIN_EGY (-24.0)    /* [erg/cm^3] */
KOKKOS_INLINE_FUNCTION TREECOL_PIXEL_STORAGE_T treecol_pixel_store(double x, double scale, double log10_min)
{
#ifdef TREECOL_PIXELS_LOG16
    const double q = (x > 0) ? (log10(x * scale) - log10_min) * (65535.0 / TREECOL_PIXELS_LOG16_SPAN) : 0;
    return (TREECOL_PIXEL_STORAGE_T) ((q < 0.5) ? 0 : ((q >= 65534.5) ? 65535 : (int) (q + 0.5)));
#else
    (void) scale; (void) log10_min; return (TREECOL_PIXEL_STORAGE_T) x;
#endif
}
KOKKOS_INLINE_FUNCTION double treecol_pixel_load(TREECOL_PIXEL_STORAGE_T v, double scale, double log10_min)
{
#ifdef TREECOL_PIXELS_LOG16
    return v ? pow(10., log10_min + v * (TREECOL_PIXELS_LOG16_SPAN / 65535.0)) / scale : 0;
#else
    (void) scale; (void) log10_min; return v;
#endif
}
#define TREECOL_PIXELS_COLUMN(c, s, k) treecol_pixel_load((c).TreeColColumn[s][k], UNIT_SURFDEN_IN_CGS, TREECOL_PIXELS_LOG16_MIN_COLUMN) /* [code] */
#define TREECOL_PIXELS_EGY_UV(c, k) treecol_pixel_load((c).TreeColEgyUV[k], 1.0, TREECOL_PIXELS_LOG16_MIN_EGY) /* [erg/cm^3] */

/* walk end: write a gas cell's columns and band energy densities [erg/cm^3]; Lyman-Werner floored per pixel at
   0.1 x the 6-13.6 eV background floor 0.324e-2 SGChemBackgroundG0 / N (stella_dev), then summed */
KOKKOS_INLINE_FUNCTION void treecol_pixels_write_cell(struct gas_cell_data &c, const treecol_pixels_sums_t &a)
{
    for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {c.TreeColColumn[s][k] = treecol_pixel_store(a.col[s][k], UNIT_SURFDEN_IN_CGS, TREECOL_PIXELS_LOG16_MIN_COLUMN);}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    const double u = All.cf_a2inv / (4.0 * M_PI * C_LIGHT_CGS * UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS); /* flux [erg/s / code length^2] to energy density */
    const double lw_floor = 0.1 * 0.324e-2 * All.SGChemBackgroundG0 / TREECOL_PIXELS_N * HABING_EGY_DENSITY_6_13P6EV_CGS;
    double lw = 0;
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {c.TreeColEgyUV[k] = treecol_pixel_store(a.uv[k] * u, 1.0, TREECOL_PIXELS_LOG16_MIN_EGY); lw += fmax(a.lw[k] * u, lw_floor);}
    c.TreeColEgyLW = (float) lw; c.TreeColEgyNUV = (float) (a.nuv * u); c.TreeColEgyOPT = (float) (a.opt * u); c.TreeColEgy68 = (float) (a.f68 * u);
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {c.TreeColRadpFlux[k] = (float) a.radp[k];}
#endif
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    { /* STARFORGE, as stella_dev: 1 below N_H = 1e21 cm^-2 (2.23e-3 g cm^-2), then exp(-Sigma / 100 g cm^-2) (1e21 cm^-2 / N_H), pixel mean */
        double f = 0, sigma_0 = 2.23e-3;
        for(int k = 0; k < TREECOL_PIXELS_N; k++) {
            double sigma = a.gas[k] * UNIT_SURFDEN_IN_CGS * All.cf_a2inv;
            f += (sigma < sigma_0) ? 1. : exp(fmax(-sigma / 100., -90.)) * sigma_0 / sigma;
        }
        c.TreeColCRAttenuation = (float) (f / TREECOL_PIXELS_N);
    }
#endif
}

/* device walk: the cell outputs from the device particle mirror back to the host cell */
static inline void treecol_pixels_copy_cell(struct gas_cell_data &d, const struct gas_cell_data &c)
{
    for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {d.TreeColColumn[s][k] = c.TreeColColumn[s][k];}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {d.TreeColEgyUV[k] = c.TreeColEgyUV[k];}
    d.TreeColEgyLW = c.TreeColEgyLW; d.TreeColEgyNUV = c.TreeColEgyNUV; d.TreeColEgyOPT = c.TreeColEgyOPT; d.TreeColEgy68 = c.TreeColEgy68;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {d.TreeColRadpFlux[k] = c.TreeColRadpFlux[k];}
#endif
#endif
#ifdef CR_BACKGROUND_IONRATE_ATTENUATED
    d.TreeColCRAttenuation = c.TreeColCRAttenuation;
#endif
}

#endif
#endif
