/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* HEALPix RING scheme (Gorski et al. 2005): pixel of a direction and pixel centre; plain doubles, no GIZMO state */
#ifndef HEALPIX_RING_H
#define HEALPIX_RING_H
#include <math.h>
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

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

/* unit vector of HEALPix RING pixel ip, NSIDE 1 (pix2vec_ring) */
KOKKOS_INLINE_FUNCTION void treecol_hpx_pixel_centre(int nside, int ip, double *n)
{
    (void) nside;
    const double q = 0.7453559924999299; /* sqrt(1 - (2/3)^2) / 1 */
    const double c = 0.7071067811865476; /* cos(pi/4) */
    n[0] = n[1] = n[2] = 0;
    if(ip < 4 || ip >= 8)
    {
        const int k = (ip < 4) ? ip : ip - 8; /* phi = pi/4 + k pi/2 */
        n[0] = q * ((k == 0 || k == 3) ? c : -c); n[1] = q * ((k < 2) ? c : -c); n[2] = (ip < 4) ? 2.0 / 3.0 : -2.0 / 3.0;
        return;
    }
    const int k = ip - 4; /* phi = k pi/2 */
    n[0] = (k == 0) ? 1.0 : ((k == 2) ? -1.0 : 0.0); n[1] = (k == 1) ? 1.0 : ((k == 3) ? -1.0 : 0.0);
}

#endif
