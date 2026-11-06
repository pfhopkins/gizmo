/* Copyright (c) 2026 Ulrich Steinwandel
   Author: Ulrich Steinwandel (uli@mpa-garching.mpg.de) */
/* IMF draws for single stars: one initial mass from the mass-weighted Kroupa IMF (M dN/dM, slopes -0.3 below 0.5 Msun and
   -1.3 above, on [max(m_min, 0.08), 350] Msun), by the exact inverse of its cumulative distribution (one uniform), on the
   keyed counter-based generator of gpu_rng.h. With the star-formation acceptance probability M_cell/M_star the formed
   stars follow the number-weighted Kroupa IMF (LYRA, Gutcke et al. 2021). Host and device. */
#ifndef SINGLE_STAR_IMF_H
#define SINGLE_STAR_IMF_H
#include <math.h>
#include <stdint.h>
#include "../declarations/gpu_rng.h"

static constexpr double SINGLE_STAR_IMF_MASS_MAX = 350.;  /* upper end of the sampled IMF [Msun] */
static constexpr double SINGLE_STAR_IMF_MASS_BREAK = 0.5; /* Kroupa break [Msun] */
static constexpr uint64_t SINGLE_STAR_IMF_RNG_SALT = gizmo_loop_rng_salt("single_star_imf_mass");

/* cumulative mass-weighted IMF from 0, unnormalised (continuous at the break: 2 m^-0.3 below, m^-1.3 above) */
GIZMO_RNG_INLINE double single_star_imf_cumulative(double m)
{
    const double b = SINGLE_STAR_IMF_MASS_BREAK, G_b = 2.0 * pow(b, 0.7) / 0.7;
    return (m <= b) ? 2.0 * pow(m, 0.7) / 0.7 : G_b + (pow(b, -0.3) - pow(m, -0.3)) / 0.3;
}

/* its inverse */
GIZMO_RNG_INLINE double single_star_imf_inverse_cumulative(double G)
{
    const double b = SINGLE_STAR_IMF_MASS_BREAK, G_b = 2.0 * pow(b, 0.7) / 0.7;
    return (G <= G_b) ? pow(0.35 * G, 1.0 / 0.7) : pow(pow(b, -0.3) - 0.3 * (G - G_b), -1.0 / 0.3);
}

/* one mass [Msun] in [max(m_min, 0.08), 350]; key = the particle's event key, event = the integer time of the draw */
GIZMO_RNG_INLINE double single_star_imf_draw_massweighted(double m_min, uint64_t key, uint64_t event)
{
    double M_max = SINGLE_STAR_IMF_MASS_MAX, M_min = (m_min < 0.08) ? 0.08 : m_min;
    if(M_min > M_max) {M_min = 0.99 * M_max;}
    double G0 = single_star_imf_cumulative(M_min), G1 = single_star_imf_cumulative(M_max);
    double u = gizmo_gpu_rand_double(key, gizmo_rng_event_counter(SINGLE_STAR_IMF_RNG_SALT, event, 0));
    double M = single_star_imf_inverse_cumulative(G0 + u * (G1 - G0));
    return (M < M_min) ? M_min : ((M > M_max) ? M_max : M);
}

/* one mass [Msun] from the number-weighted IMF: the mass-weighted draw accepted with M_min/M (draw indices 1, 2, ... of the event) */
GIZMO_RNG_INLINE double single_star_imf_draw_numberweighted(double m_min, uint64_t key, uint64_t event)
{
    double M_max = SINGLE_STAR_IMF_MASS_MAX, M_min = (m_min < 0.08) ? 0.08 : m_min;
    if(M_min > M_max) {M_min = 0.99 * M_max;}
    double G0 = single_star_imf_cumulative(M_min), G1 = single_star_imf_cumulative(M_max);
    for(uint64_t k = 0; k < 10000; k++)
    {
        double u = gizmo_gpu_rand_double(key, gizmo_rng_event_counter(SINGLE_STAR_IMF_RNG_SALT, event, 1 + 2 * k));
        double M = single_star_imf_inverse_cumulative(G0 + u * (G1 - G0));
        M = (M < M_min) ? M_min : ((M > M_max) ? M_max : M);
        if(gizmo_gpu_rand_double(key, gizmo_rng_event_counter(SINGLE_STAR_IMF_RNG_SALT, event, 2 + 2 * k)) <= M_min / M) {return M;}
    }
    return M_min;
}
#endif
