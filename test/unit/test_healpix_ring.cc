// Unit test: HEALPix RING pixel centres (src/gravity/healpix_ring.h), used as the radiation-pressure push direction of each
// TreeCol pixel. For NSIDE 1, 2, 4 (12, 48, 192 pixels) every centre is a unit vector that the pixel finder of the walk
// (treecol_hpx_ring_pix) puts back in its own pixel, the centres are distinct, and NSIDE 1 keeps the former table
// (Gorski+05 eqs. 4-9: z = 2/3, 0, -2/3; phi = pi/4 + k pi/2 on the cap rings, k pi/2 on the equator).
#include "test_harness.h"
#include <cmath>
#include "src/gravity/healpix_ring.h"

static int pixel_of(int nside, const double *n)
{
    double phi = atan2(n[1], n[0]); if(phi < 0) {phi += 2.0 * M_PI;}
    return treecol_hpx_ring_pix(nside, n[2], phi / (0.5 * M_PI));
}

TEST_CASE("pixel centres round-trip for NSIDE 1, 2, 4") {
    for(int nside : {1, 2, 4}) {
        const int npix = 12 * nside * nside;
        int bad = 0;
        for(int ip = 0; ip < npix; ip++) {
            double n[3]; treecol_hpx_pixel_centre(nside, ip, n);
            if(fabs(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0) > 1e-12 || pixel_of(nside, n) != ip) {bad++;}
            for(int jp = 0; jp < ip; jp++) {
                double m[3]; treecol_hpx_pixel_centre(nside, jp, m);
                if(fabs(n[0] - m[0]) + fabs(n[1] - m[1]) + fabs(n[2] - m[2]) < 1e-9) {bad++;}
            }
        }
        printf("NSIDE %d: %d of %d pixel centres wrong\n", nside, bad, npix);
        CHECK(bad == 0);
    }
}

TEST_CASE("NSIDE 1 keeps the former table") {
    const double q = sqrt(1.0 - 4.0 / 9.0), c = sqrt(0.5);
    const double ref[12][3] = {{q * c, q * c, 2. / 3}, {-q * c, q * c, 2. / 3}, {-q * c, -q * c, 2. / 3}, {q * c, -q * c, 2. / 3},
                               {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0},
                               {q * c, q * c, -2. / 3}, {-q * c, q * c, -2. / 3}, {-q * c, -q * c, -2. / 3}, {q * c, -q * c, -2. / 3}};
    for(int ip = 0; ip < 12; ip++) {
        double n[3]; treecol_hpx_pixel_centre(1, ip, n);
        for(int k = 0; k < 3; k++) {CHECK_CLOSE(n[k], ref[ip][k], 1e-14);}
    }
}

TEST_MAIN()
