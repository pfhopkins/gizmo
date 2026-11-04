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
#define TREECOL_PIXELS_HI   3 /* ionising-photon absorbers: H atoms + 1.34 per H2, the mean sigma_H2/sigma_HI per molecule (at <hnu> 18.6 eV) */
#define TREECOL_SHELLS_H2_PER_HI 1.34

/* a particle's species masses [code units]; zero for non-gas. H nuclei per mass as the chemistry, 1/((1+4 ABHE) m_p) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_species_masses(const struct particle_data *pp, const struct gas_cell_data *cell, int i, double *m)
{
    for(int s = 0; s < TREECOL_PIXELS_NSPECIES; s++) {m[s] = 0;}
    if(pp[i].Type != 0) {return;}
    double dgr = (cell[i].Temp > 1.0e5) ? 0 : fmax(sgchem_dust_to_gas_ratio(pp[i].Metallicity[0]), 0.); /* hot gas: grains sputtered */
    m[TREECOL_PIXELS_DUST] = dgr * pp[i].Mass;
    double m_H = pp[i].Mass / (1.0 + 4.0 * ABHE);
    m[TREECOL_PIXELS_H2] = 2.0 * cell[i].TracAbund[IH2] * m_H;
    m[TREECOL_PIXELS_CO] = 28.0 * cell[i].TracAbund[ICO] * m_H;
#ifdef TREECOL_SHELLS_ION_N
    m[TREECOL_PIXELS_HI] = (fmax(0., 1.0 - cell[i].TracAbund[IHP] - 2.0 * cell[i].TracAbund[IH2]) + TREECOL_SHELLS_H2_PER_HI * cell[i].TracAbund[IH2]) * HYDROGEN_MASSFRAC * pp[i].Mass;
#endif
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
KOKKOS_INLINE_FUNCTION int treecol_hpx_pixel_nside(const Vec3<double> &d, int nside)
{
    double x = d[0], y = d[1], r = sqrt(x * x + y * y + d[2] * d[2]);
    if(r == 0) {return 0;}
    double z = d[2] / r, ax = fabs(x), ay = fabs(y);
    if(ax > 1e-100 && ay > 1e-100 && ax < 1e100 && ay < 1e100 && fabs(z) <= 1.0)
    {
        int ip = treecol_hpx_ring_pix_checked(nside, z, treecol_hpx_tt_approx(x, y));
        if(ip >= 0) {return ip;}
    }
    double phi = atan2(y, x); if(phi < 0) {phi += 2.0 * M_PI;}
    return treecol_hpx_ring_pix(nside, z, phi / (0.5 * M_PI));
}
KOKKOS_INLINE_FUNCTION int treecol_hpx_pixel(const Vec3<double> &d) {return treecol_hpx_pixel_nside(d, TREECOL_PIXELS_NSIDE);}

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
    double col[TREECOL_PIXELS_NCOLUMNS][TREECOL_PIXELS_N]; /* [code mass / code length^2] */
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    double uv[TREECOL_PIXELS_N], lw[TREECOL_PIXELS_N];    /* 6-13.6 (PE + 6-8) and 11.2-13.6 eV [erg/s / code length^2] */
    double nuv, opt, f68;                                  /* 3.444-6, 0.4133-3.444, 6-8 eV, totals */
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    double radp[4];                                        /* sigma-weighted flux: vector over pixel centres, scalar */
#endif
#endif
#ifdef TREECOL_SHELLS_N
    float sh[TREECOL_SHELLS_NFIELD][TREECOL_SHELLS_N];     /* per (pixel, shell): band L/r^2 [code units], dust column */
#endif
#ifdef TREECOL_SHELLS_ION_N
    float sh_ion[TREECOL_SHELLS_ION_NFIELD][TREECOL_SHELLS_ION_N]; /* per (ionising pixel, shell): photons, energy, HI column */
#endif
};

KOKKOS_INLINE_FUNCTION void treecol_pixels_zero(treecol_pixels_sums_t &a)
{
    for(int s = 0; s < TREECOL_PIXELS_NCOLUMNS; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.col[s][k] = 0;}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.uv[k] = a.lw[k] = 0;}
    a.nuv = a.opt = a.f68 = 0;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {a.radp[k] = 0;}
#endif
#endif
#ifdef TREECOL_SHELLS_N
    for(int f = 0; f < TREECOL_SHELLS_NFIELD; f++) {for(int k = 0; k < TREECOL_SHELLS_N; k++) {a.sh[f][k] = 0;}}
#endif
#ifdef TREECOL_SHELLS_ION_N
    for(int f = 0; f < TREECOL_SHELLS_ION_NFIELD; f++) {for(int k = 0; k < TREECOL_SHELLS_ION_N; k++) {a.sh_ion[f][k] = 0;}}
#endif
}

/* combine two partial sums (lane partials, in the engine's fixed fold order) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_add(treecol_pixels_sums_t &a, const treecol_pixels_sums_t &b)
{
    for(int s = 0; s < TREECOL_PIXELS_NCOLUMNS; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.col[s][k] += b.col[s][k];}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {a.uv[k] += b.uv[k]; a.lw[k] += b.lw[k];}
    a.nuv += b.nuv; a.opt += b.opt; a.f68 += b.f68;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {a.radp[k] += b.radp[k];}
#endif
#endif
#ifdef TREECOL_SHELLS_N
    for(int f = 0; f < TREECOL_SHELLS_NFIELD; f++) {for(int k = 0; k < TREECOL_SHELLS_N; k++) {a.sh[f][k] += b.sh[f][k];}}
#endif
#ifdef TREECOL_SHELLS_ION_N
    for(int f = 0; f < TREECOL_SHELLS_ION_NFIELD; f++) {for(int k = 0; k < TREECOL_SHELLS_ION_N; k++) {a.sh_ion[f][k] += b.sh_ion[f][k];}}
#endif
}

#ifdef TREECOL_SHELLS_N
/* log-radial shell of r2: TREECOL_SHELLS_NSHELL shells over 3 dex below TreeColShieldingLength, clamped at both ends */
KOKKOS_INLINE_FUNCTION int treecol_shell_of(double r2)
{
    const double sl = All.TreeColShieldingLength / All.cf_atime;
    const int s = TREECOL_SHELLS_NSHELL - 1 + (int) floor(0.5 * log10(r2 / (sl * sl)) * (TREECOL_SHELLS_NSHELL / 3.0));
    return (s < 0) ? 0 : ((s >= TREECOL_SHELLS_NSHELL) ? TREECOL_SHELLS_NSHELL - 1 : s);
}
#endif

/* one accepted source (leaf or node) for a gas target: column in the pixel of its mass centre dr, inside
   TreeColShieldingLength (0 = no limit), unsoftened; band fluxes L/r_L^2 in the pixel of the light
   centroid d_lum (a second lookup only for luminous nodes, whose light and mass centres differ), r_L floored at r_floor
   (the target's Plummer-equivalent softening) */
KOKKOS_INLINE_FUNCTION void treecol_pixels_accumulate(treecol_pixels_sums_t &a, const Vec3<double> &dr, double r, double fac_accel,
                                                      double mass, const double *species, const Vec3<double> &d_lum, const double *lum, double r_floor)
{
    if(r <= 0 || mass <= 0) {return;}
    const int ip = treecol_hpx_pixel(dr);
    if(All.TreeColShieldingLength <= 0 || r < All.TreeColShieldingLength / All.cf_atime)
    {
        const double w = 1.0 / ((4.0 * M_PI / TREECOL_PIXELS_N) * r * r); /* unsoftened, as stella_dev */
        for(int s = 0; s < TREECOL_PIXELS_NCOLUMNS; s++) {a.col[s][ip] += w * species[s];}
#ifdef TREECOL_SHELLS_N
        a.sh[TREECOL_SHELL_DUST][ip * TREECOL_SHELLS_NSHELL + treecol_shell_of(r * r)] += w * species[TREECOL_PIXELS_DUST];
#endif
#ifdef TREECOL_SHELLS_ION_N
        if(species[TREECOL_PIXELS_HI] > 0) {a.sh_ion[TREECOL_SHELL_ION_HI][treecol_hpx_pixel_nside(dr, GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION) * TREECOL_SHELLS_NSHELL + treecol_shell_of(r * r)] += w * species[TREECOL_PIXELS_HI] * ((double) TREECOL_SHELLS_ION_NPIX / TREECOL_PIXELS_N);}
#endif
    }
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    const double l_pe = lum[RT_FREQ_BIN_PHOTOELECTRIC], l_68 = lum[RT_FREQ_BIN_GENERIC_USER_FREQ], l_lw = lum[RT_FREQ_BIN_LYMAN_WERNER];
    const double l_nuv = lum[RT_FREQ_BIN_NUV], l_opt = lum[RT_FREQ_BIN_OPTICAL_NIR];
#ifdef TREECOL_SHELLS_ION_N
    if(lum[RT_FREQ_BIN_ION_PHOTONS] > 0)
    {
        const double r2_i = fmax(d_lum.norm_sq(), r_floor * r_floor);
        const int ii = treecol_hpx_pixel_nside(d_lum, GALSF_RESOLVEDISM_TREECOL_PIXELS_RADIAL_SHELLS_PHOTOION) * TREECOL_SHELLS_NSHELL + treecol_shell_of(r2_i);
        a.sh_ion[TREECOL_SHELL_ION_PHOTONS][ii] += lum[RT_FREQ_BIN_ION_PHOTONS] / r2_i; a.sh_ion[TREECOL_SHELL_ION_ENERGY][ii] += lum[RT_FREQ_BIN_ION_ENERGY] / r2_i;
    }
#endif
    if(l_pe + l_68 > 0 || l_lw > 0 || l_nuv > 0 || l_opt > 0)
    {
        const double r2_l = fmax(d_lum.norm_sq(), r_floor * r_floor), f = UNIT_LUM_IN_CGS / r2_l;
        const int il = (d_lum[0] == dr[0] && d_lum[1] == dr[1] && d_lum[2] == dr[2]) ? ip : treecol_hpx_pixel(d_lum);
#ifdef TREECOL_SHELLS_N
        const int is = il * TREECOL_SHELLS_NSHELL + treecol_shell_of(r2_l);
        a.sh[TREECOL_SHELL_UV][is] += (l_pe + l_68) / r2_l; a.sh[TREECOL_SHELL_LW][is] += l_lw / r2_l;
        a.sh[TREECOL_SHELL_NUV][is] += l_nuv / r2_l; a.sh[TREECOL_SHELL_OPT][is] += l_opt / r2_l;
        a.f68 += l_68 * f;
#else
        a.uv[il] += (l_pe + l_68) * f; a.lw[il] += l_lw * f;
        a.nuv += l_nuv * f; a.opt += l_opt * f; a.f68 += l_68 * f;
#endif
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
        const double fw = treecol_radp_sigma_lum(lum) / fmax(d_lum.norm_sq(), r_floor * r_floor);
        const Vec3<double> nh = treecol_radp_pixel_centre(il);
        a.radp[0] += fw * nh[0]; a.radp[1] += fw * nh[1]; a.radp[2] += fw * nh[2]; a.radp[3] += fw;
#endif
    }
#else
    (void) d_lum; (void) lum; (void) r_floor;
#endif
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

#ifdef TREECOL_SHELLS_N
/* each pixel outward: shell sources behind the dust of the interior shells plus half their own; sigma per H at the solar
   dust-to-gas ratio: 2.5 A_V (chemistry df25) for 6-13.6 eV, 2, 1.3, 0.5e-21 cm^2 for LW, NUV, optical. Fluxes [erg/s / code length^2] */
#define TREECOL_SHELLS_SIGMA_UV (2.5 * 5.348e-22)
#define TREECOL_SHELLS_SIGMA_LW 2.0e-21
KOKKOS_INLINE_FUNCTION void treecol_shells_sweep(const treecol_pixels_sums_t &a, double *uv, double *lw, double &nuv, double &opt)
{
    const double to_nh = UNIT_SURFDEN_IN_CGS * All.cf_a2inv * HYDROGEN_MASSFRAC / PROTONMASS_CGS;
    nuv = opt = 0;
    for(int p = 0; p < TREECOL_PIXELS_N; p++)
    {
        double tau_uv = 0, tau_d = 0; uv[p] = lw[p] = 0;
        for(int s = 0; s < TREECOL_SHELLS_NSHELL; s++)
        {
            const int k = p * TREECOL_SHELLS_NSHELL + s;
            const double nh = a.sh[TREECOL_SHELL_DUST][k] * to_nh, dtau_uv = TREECOL_SHELLS_SIGMA_UV * nh, dtau_d = TREECOL_SHELLS_SIGMA_LW * nh;
            const double t_d = tau_d + 0.5 * dtau_d;
            uv[p] += a.sh[TREECOL_SHELL_UV][k] * UNIT_LUM_IN_CGS * exp(-(tau_uv + 0.5 * dtau_uv));
            lw[p] += a.sh[TREECOL_SHELL_LW][k] * UNIT_LUM_IN_CGS * exp(-t_d);
            nuv += a.sh[TREECOL_SHELL_NUV][k] * UNIT_LUM_IN_CGS * exp(-t_d * (1.3 / 2.0));
            opt += a.sh[TREECOL_SHELL_OPT][k] * UNIT_LUM_IN_CGS * exp(-t_d * (0.5 / 2.0));
            tau_uv += dtau_uv; tau_d += dtau_d;
        }
    }
}
#endif

#ifdef TREECOL_SHELLS_ION_N
/* ionising light outward through the HI column of each pixel, one band at sigma_HI; attenuated photon and energy flux summed over pixels */
#define TREECOL_SHELLS_SIGMA_HI 6.3e-18
KOKKOS_INLINE_FUNCTION void treecol_shells_ion_sweep(const treecol_pixels_sums_t &a, float &F_ph, float &F_en)
{
    const double to_n = UNIT_SURFDEN_IN_CGS * All.cf_a2inv / PROTONMASS_CGS, f = UNIT_LUM_IN_CGS * All.cf_a2inv / (4.0 * M_PI * UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS);
    double ph = 0, en = 0;
    for(int p = 0; p < TREECOL_SHELLS_ION_NPIX; p++)
    {
        double tau = 0;
        for(int s = 0; s < TREECOL_SHELLS_NSHELL; s++)
        {
            const int k = p * TREECOL_SHELLS_NSHELL + s;
            const double dtau = TREECOL_SHELLS_SIGMA_HI * a.sh_ion[TREECOL_SHELL_ION_HI][k] * to_n, att = exp(-(tau + 0.5 * dtau));
            ph += a.sh_ion[TREECOL_SHELL_ION_PHOTONS][k] * att; en += a.sh_ion[TREECOL_SHELL_ION_ENERGY][k] * att;
            tau += dtau;
        }
    }
    F_ph = (float) (ph * f / (13.6 * ELECTRONVOLT_IN_ERGS)); F_en = (float) (en * f);
}
#endif

/* walk end: write a gas cell's columns and band energy densities [erg/cm^3]; Lyman-Werner floored per pixel at
   0.1 x the 6-13.6 eV background floor 0.324e-2 SGChemBackgroundG0 / N (stella_dev), then summed */
KOKKOS_INLINE_FUNCTION void treecol_pixels_write_cell(struct gas_cell_data &c, const treecol_pixels_sums_t &a)
{
    for(int s = 0; s < TREECOL_PIXELS_NCOLUMNS; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {c.TreeColColumn[s][k] = treecol_pixel_store(a.col[s][k], UNIT_SURFDEN_IN_CGS, TREECOL_PIXELS_LOG16_MIN_COLUMN);}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    const double u = All.cf_a2inv / (4.0 * M_PI * C_LIGHT_CGS * UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS); /* flux [erg/s / code length^2] to energy density */
    const double lw_floor = 0.1 * 0.324e-2 * All.SGChemBackgroundG0 / TREECOL_PIXELS_N * HABING_EGY_DENSITY_6_13P6EV_CGS;
#ifdef TREECOL_SHELLS_N
    double a_uv[TREECOL_PIXELS_N], a_lw[TREECOL_PIXELS_N], a_nuv, a_opt;
    treecol_shells_sweep(a, a_uv, a_lw, a_nuv, a_opt);
#else
    const double *a_uv = a.uv, *a_lw = a.lw, a_nuv = a.nuv, a_opt = a.opt;
#endif
    double lw = 0;
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {c.TreeColEgyUV[k] = treecol_pixel_store(a_uv[k] * u, 1.0, TREECOL_PIXELS_LOG16_MIN_EGY); lw += fmax(a_lw[k] * u, lw_floor);}
    c.TreeColEgyLW = (float) lw; c.TreeColEgyNUV = (float) (a_nuv * u); c.TreeColEgyOPT = (float) (a_opt * u); c.TreeColEgy68 = (float) (a.f68 * u);
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {c.TreeColRadpFlux[k] = (float) a.radp[k];}
#endif
#endif
#ifdef TREECOL_SHELLS_ION_N
    treecol_shells_ion_sweep(a, c.TreeRayIonPhotonFlux, c.TreeRayIonEnergyFlux);
#endif
}

/* device walk: the cell outputs from the device particle mirror back to the host cell */
static inline void treecol_pixels_copy_cell(struct gas_cell_data &d, const struct gas_cell_data &c)
{
    for(int s = 0; s < TREECOL_PIXELS_NCOLUMNS; s++) {for(int k = 0; k < TREECOL_PIXELS_N; k++) {d.TreeColColumn[s][k] = c.TreeColColumn[s][k];}}
#ifdef GALSF_RESOLVEDISM_TREECOL_PIXELS_STELLAR_BAND_FLUXES
    for(int k = 0; k < TREECOL_PIXELS_N; k++) {d.TreeColEgyUV[k] = c.TreeColEgyUV[k];}
    d.TreeColEgyLW = c.TreeColEgyLW; d.TreeColEgyNUV = c.TreeColEgyNUV; d.TreeColEgyOPT = c.TreeColEgyOPT; d.TreeColEgy68 = c.TreeColEgy68;
#ifdef GALSF_FB_RADPRESSURE_ABSORPTION_FROM_TREECOL
    for(int k = 0; k < 4; k++) {d.TreeColRadpFlux[k] = c.TreeColRadpFlux[k];}
#endif
#endif
#ifdef TREECOL_SHELLS_ION_N
    d.TreeRayIonPhotonFlux = c.TreeRayIonPhotonFlux; d.TreeRayIonEnergyFlux = c.TreeRayIonEnergyFlux;
#endif
}

#endif
#endif
