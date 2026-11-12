// Prints the UV-background shielding factors of src/cooling/sgchem/sgchem_uvb_shield.h on a grid of HI, HeI columns.
#include <cstdio>
#include "src/cooling/sgchem/sgchem_uvb_shield.h"

int main()
{
    const double NHI[] = {0, 1e16, 1e17, 3e17, 1e18, 3e18, 1e19, 1e20}, fHe[] = {0, 0.079};
    for(double f : fHe) for(double n : NHI) {
        double N[3] = {n, f * n, 0}, g[3], q[3];
        sgchem_uvb_shield_factors(N, g, q);
        std::printf("FAC NHI=%.6e NHeI=%.6e fgHI=%.17g fgHeI=%.17g fgHeII=%.17g fqHI=%.17g fqHeI=%.17g fqHeII=%.17g\n", N[0], N[1], g[0], g[1], g[2], q[0], q[1], q[2]);
    }
    return 0;
}
