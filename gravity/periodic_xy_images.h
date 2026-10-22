#ifndef PERIODIC_XY_IMAGES_H
#define PERIODIC_XY_IMAGES_H
/* GRAVITY_PERIODIC_XY_IMAGES: self-gravity periodic in x/y by explicit image sums, open in z (needs mesh/kernel.h first) */
#ifdef GRAVITY_PERIODIC_XY_IMAGES
/* calls add(d, fac_accel, fac2_tidal, fac_pot) for each image (a,b) != (0,0), |a|,|b| <= All.GravityPeriodicXYImages,
   of a source of mass m at nearest-image separation dr; h = softening of the pair */
template<typename F> static inline void gravity_periodic_xy_images(double m, const Vec3<double>& dr, double h, F&& add)
{
    const int n = All.GravityPeriodicXYImages;
    for(int a = -n; a <= n; a++)
    {
        for(int b = -n; b <= n; b++)
        {
            if(a == 0 && b == 0) {continue;}
            Vec3<double> d = {dr[0] + a * boxSize_X, dr[1] + b * boxSize_Y, dr[2]};
            double r2 = d.norm_sq(), r = sqrt(r2), fac_accel, fac2_tidal, fac_pot;
            if(r >= h) {fac_accel = m / (r2 * r); fac2_tidal = 3. * fac_accel / r2; fac_pot = -m / r;}
            else {double h_inv = 1. / h, h3_inv = h_inv * h_inv * h_inv, u = r * h_inv;
                fac_accel = m * kernel_gravity(u, h_inv, h3_inv, 1); fac2_tidal = m * kernel_gravity(u, h_inv, h3_inv, 2); fac_pot = m * kernel_gravity(u, h_inv, h3_inv, -1);}
            add(d, fac_accel, fac2_tidal, fac_pot);
        }
    }
}
#endif
#endif
