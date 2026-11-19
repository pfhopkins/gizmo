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

/* unit vector of the centre of RING pixel ip at nside (pix2ang_ring; Gorski+05 eqs. 4-9) */
KOKKOS_INLINE_FUNCTION void treecol_hpx_pixel_centre(int nside, int ip, double *n)
{
    const long ns = nside, npix = 12 * ns * ns, ncap = 2 * ns * (ns - 1);
    long iring, iphi; double z, phi;
    if(ip < ncap) { /* north polar cap: ring iring holds 4 iring pixels */
        iring = (long)((1 + sqrt(1.0 + 2.0 * ip)) / 2); while(2 * iring * (iring - 1) > ip) {iring--;} while(2 * (iring + 1) * iring <= ip) {iring++;}
        iphi = ip + 1 - 2 * iring * (iring - 1); z = 1.0 - (double)(iring * iring) / (3.0 * ns * ns); phi = (iphi - 0.5) * M_PI / (2.0 * iring);
    } else if(ip < npix - ncap) { /* equatorial belt: 4 nside pixels per ring, every other ring shifted by half a pixel */
        const long ip2 = ip - ncap; iring = ip2 / (4 * ns) + ns; iphi = ip2 % (4 * ns) + 1;
        z = (2.0 * ns - iring) * 2.0 / (3.0 * ns); phi = (iphi - (((iring + ns) % 2) ? 1.0 : 0.5)) * M_PI / (2.0 * ns);
    } else { /* south polar cap, counted from the pole */
        const long ip2 = npix - ip; iring = (long)((1 + sqrt(2.0 * ip2 - 1.0)) / 2); while(2 * iring * (iring - 1) >= ip2) {iring--;} while(2 * (iring + 1) * iring < ip2) {iring++;}
        iphi = 4 * iring + 1 - (ip2 - 2 * iring * (iring - 1)); z = -1.0 + (double)(iring * iring) / (3.0 * ns * ns); phi = (iphi - 0.5) * M_PI / (2.0 * iring);
    }
    const double st = sqrt((1.0 - z) * (1.0 + z));
    n[0] = st * cos(phi); n[1] = st * sin(phi); n[2] = z;
}

#endif
