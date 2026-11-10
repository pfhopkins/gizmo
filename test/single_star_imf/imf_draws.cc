/* writes N single-star IMF draws (doubles): imf_draws N M_MIN EVENT OUT [reference]
   default: GIZMO's draw for keys 1..N at one event; "reference": the rejection sampler of the reference implementation
   (resolvedism_imf_sampling.cc: M^-0.3 envelope, accept with ratio), on std::mt19937_64 seeded with EVENT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include "galaxy_sf/single_star_imf.h"

static double reference_draw(double m_min, std::mt19937_64 &g)
{
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double M_max = 350., M_min = (m_min < 0.08) ? 0.08 : m_min, A = 0.126512;
    double M;
    do {M = pow((pow(M_max, 0.7) - pow(M_min, 0.7)) * U(g) + pow(M_min, 0.7), 1.0 / 0.7);}
    while(U(g) > ((M < 0.5) ? 2.0 * A * pow(M, -0.3) : A * pow(M, -1.3)) / (2.0 * A * pow(M, -0.3)));
    return M;
}

int main(int argc, char **argv)
{
    long N = atol(argv[1]); double m_min = atof(argv[2]); uint64_t ev = strtoull(argv[3], NULL, 10);
    int ref = (argc > 5 && strcmp(argv[5], "reference") == 0);
    std::mt19937_64 g(ev);
    FILE *fd = fopen(argv[4], "wb");
    for(long n = 1; n <= N; n++) {
        double m = ref ? reference_draw(m_min, g) : single_star_imf_draw_massweighted(m_min, gizmo_rng_combine(0, (uint64_t)n), ev);
        fwrite(&m, sizeof(double), 1, fd);
    }
    fclose(fd);
    return 0;
}
