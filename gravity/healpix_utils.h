/* Minimal HEALPix utility for TREE_RAD angular binning.
 * Provides vec2pix_ring() - maps a 3D direction vector to a HEALPix pixel index (ring ordering).
 * This avoids requiring the external chealpix library for this single function.
 */
#ifndef HEALPIX_UTILS_H
#define HEALPIX_UTILS_H

#include <math.h>

static inline void vec2pix_ring(long nside, const double *vec, long *ipix)
{
    double x = vec[0], y = vec[1], z = vec[2];
    double r = sqrt(x*x + y*y + z*z);
    if(r == 0) { *ipix = 0; return; }
    z /= r; /* cos(theta) */
    double phi = atan2(y, x);
    if(phi < 0) phi += 2.0 * M_PI;

    long npix = 12 * nside * nside;
    double za = fabs(z);
    double tt = phi / (0.5 * M_PI); /* phi in units of pi/2, range [0,4) */

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
        *ipix = nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip;
    } else {
        /* Polar caps */
        double tp = tt - (long)(tt);
        double tmp = nside * sqrt(3.0 * (1.0 - za));
        long jp = (long)(tp * tmp);       /* increasing edge line index */
        long jm = (long)((1.0 - tp) * tmp); /* decreasing edge line index */
        long ir = jp + jm + 1;            /* ring number counted from closest pole */
        long ip = (long)(tt * ir);
        ip = ip % (4 * ir);
        if(z > 0) {
            *ipix = 2 * ir * (ir - 1) + ip;
        } else {
            *ipix = npix - 2 * ir * (ir + 1) + ip;
        }
    }
}

#endif /* HEALPIX_UTILS_H */
