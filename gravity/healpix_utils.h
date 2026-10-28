/* Minimal HEALPix utility for tree-column angular binning.
 * Provides vec2pix_ring() - maps a 3D direction vector to a HEALPix pixel index (ring ordering).
 * This avoids requiring the external chealpix library for this single function.
 */
#ifndef HEALPIX_UTILS_H
#define HEALPIX_UTILS_H

#include <math.h>
#include <stdlib.h>

/* pixel from z and tt = phi/(pi/2) in [0,4]; the exact HEALPix ring formulas */
static inline long hpx_ring_pix(long nside, double z, double tt)
{
    long npix = 12 * nside * nside;
    double za = fabs(z);

    if(za <= 2.0/3.0) {
        /* Equatorial region */
        double temp1 = nside * (0.5 + tt);
        double temp2 = nside * z * 0.75;
        long jp = (long)(temp1 - temp2); /* ascending edge line index */
        long jm = (long)(temp1 + temp2); /* descending edge line index */
        long ir = nside + 1 + jp - jm;   /* ring number counted from z=2/3 */
        long kshift = 1 - (ir & 1);      /* kshift=1 if ir even, 0 if odd */
        long ip = (jp + jm - nside + kshift + 1) / 2;
        ip = ip % (4 * nside);
        return nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip;
    } else {
        /* Polar caps */
        double tp = tt - (long)(tt);
        double tmp = nside * sqrt(3.0 * (1.0 - za));
        long jp = (long)(tp * tmp);       /* increasing edge line index */
        long jm = (long)((1.0 - tp) * tmp); /* decreasing edge line index */
        long ir = jp + jm + 1;            /* ring number counted from closest pole */
        long ip = (long)(tt * ir);
        ip = ip % (4 * ir);
        if(z > 0) {return 2 * ir * (ir - 1) + ip;}
        return npix - 2 * ir * (ir + 1) + ip;
    }
}

/* atan2-free tt = phi/(pi/2) in [0,4), |error| < HPX_TT_ERR; polynomial atan on [0,1], needs x,y != 0 */
#define HPX_TT_ERR 1e-8
static inline double hpx_tt_approx(double x, double y)
{
    double ax = fabs(x), ay = fabs(y), t = fmin(ax, ay) / fmax(ax, ay), s = t * t, a;
    a = t * (0.9999999975460201 + s * (-0.3333328229551207 + s * (0.1999823064037613 + s * (-0.14261573680246378 + s * (0.10940198964830647
        + s * (-0.0837206394697871 + s * (0.05746355782881671 + s * (-0.030717508884063723 + s * (0.010680719435017704 + s * (-0.0017437011413061324))))))))));
    a += (ay > ax ? 1.0 : 0.0) * (0.5 * M_PI - 2.0 * a); /* branch-free octant unfolding */
    a += (x < 0 ? 1.0 : 0.0) * (M_PI - 2.0 * a);
    a *= (y < 0 ? -1.0 : 1.0);
    a += (a < 0 ? 2.0 * M_PI : 0.0);
    return a * (2.0 / M_PI);
}

/* hpx_ring_pix(nside, z, tt) when no truncation/floor decision can change for tt within HPX_TT_ERR, else -1 */
static inline long hpx_ring_pix_checked(long nside, double z, double tt)
{
    double M = (8.0 * nside + 8.0) * (HPX_TT_ERR * (2.0 / M_PI)) * 4.0, M1 = 1.0 - M;
    double za = fabs(z);
    if(za <= 2.0/3.0) {
        double temp1 = nside * (0.5 + tt), temp2 = nside * z * 0.75, a = temp1 - temp2, b = temp1 + temp2;
        double fa = floor(a), fb = floor(b), ga = a - fa, gb = b - fb;
        if(ga < M || ga > M1 || gb < M || gb > M1) {return -1;}
        long jp = (long)fa, jm = (long)fb, ir = nside + 1 + jp - jm, kshift = 1 - (ir & 1);
        long ip = ((jp + jm - nside + kshift + 1) / 2) % (4 * nside);
        return nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip;
    } else {
        double ft = floor(tt), tp = tt - ft;
        if(tp < M || tp > M1) {return -1;}
        double tmp = nside * sqrt(3.0 * (1.0 - za)), u = tp * tmp, v = (1.0 - tp) * tmp, fu = floor(u), fv = floor(v);
        double gu = u - fu, gv = v - fv;
        if(gu < M || gu > M1 || gv < M || gv > M1) {return -1;}
        long ir = (long)fu + (long)fv + 1;
        double w = tt * ir, fw = floor(w), gw = w - fw;
        if(gw < M || gw > M1) {return -1;}
        long ip = ((long)fw) % (4 * ir);
        if(z > 0) {return 2 * ir * (ir - 1) + ip;}
        return 12 * nside * nside - 2 * ir * (ir + 1) + ip;
    }
}

static inline void vec2pix_ring(long nside, const double *vec, long *ipix)
{
    double x = vec[0], y = vec[1], z = vec[2];
    double r = sqrt(x*x + y*y + z*z);
    if(r == 0) { *ipix = 0; return; }
    z /= r; /* cos(theta) */
    double ax = fabs(x), ay = fabs(y);
    if(ax > 1e-100 && ay > 1e-100 && ax < 1e100 && ay < 1e100 && fabs(z) <= 1.0) /* generic direction: skip atan2 unless the pixel could be ambiguous */
    {
        long ip = hpx_ring_pix_checked(nside, z, hpx_tt_approx(x, y));
        if(ip >= 0) {*ipix = ip; return;}
    }
    double phi = atan2(y, x);
    if(phi < 0) phi += 2.0 * M_PI;
    *ipix = hpx_ring_pix(nside, z, phi / (0.5 * M_PI)); /* tt: phi in units of pi/2, range [0,4] */
}

/* pix2vec_ring(): return the CENTER unit vector of RING-ordered pixel ipix.
 * Consistency requirement (Uli 2026-07-08): the per-cell tree radiation-
 * pressure force multiplies the flux binned into each pixel by that pixel's
 * direction, so the returned centroid MUST match how vec2pix_ring() binned the
 * flux — NOT an independent analytic HEALPix center that could differ by the
 * quirks of the minimal vec2pix_ring above. So we build the centroids by
 * INVERTING vec2pix_ring: sample a dense, area-uniform grid of directions over
 * the sphere (cos(theta) uniform, phi uniform), accumulate each sample into the
 * pixel vec2pix_ring assigns it to, and normalize the mean direction per pixel.
 * Computed once and cached (per nside). For nside=1 (NPIX=12) this yields the
 * expected 12 face centers: 4 at z=+2/3, 4 at z=0, 4 at z=-2/3. */
static inline void pix2vec_ring(long nside, long ipix, double *vec)
{
    static double *cache = NULL;
    static long cache_nside = -1;
    if(nside != cache_nside)
    {
        long npix = 12 * nside * nside;
        if(cache) {free(cache);}
        cache = (double *)malloc(3 * npix * sizeof(double));
        long p; for(p = 0; p < 3*npix; p++) {cache[p] = 0.0;}
        long Nt = 400 * nside, Np = 800 * nside, it, iq;
        for(it = 0; it < Nt; it++)
        {
            double ct = 1.0 - 2.0 * ((double)it + 0.5) / (double)Nt; /* cos(theta) uniform in (-1,1) */
            double st = sqrt(fmax(0.0, 1.0 - ct*ct));
            for(iq = 0; iq < Np; iq++)
            {
                double phi = 2.0 * M_PI * ((double)iq + 0.5) / (double)Np;
                double v[3]; v[0] = st*cos(phi); v[1] = st*sin(phi); v[2] = ct;
                long ix; vec2pix_ring(nside, v, &ix);
                if(ix < 0 || ix >= npix) {continue;}
                cache[3*ix+0] += v[0]; cache[3*ix+1] += v[1]; cache[3*ix+2] += v[2];
            }
        }
        for(p = 0; p < npix; p++)
        {
            double x = cache[3*p+0], y = cache[3*p+1], z = cache[3*p+2];
            double r = sqrt(x*x + y*y + z*z);
            if(r > 0) {cache[3*p+0] = x/r; cache[3*p+1] = y/r; cache[3*p+2] = z/r;}
        }
        cache_nside = nside;
    }
    vec[0] = cache[3*ipix+0]; vec[1] = cache[3*ipix+1]; vec[2] = cache[3*ipix+2];
}

#endif /* HEALPIX_UTILS_H */
