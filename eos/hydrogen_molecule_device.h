/* hydrogen_molecule_device.h — KOKKOS_INLINE_FUNCTION versions of the H2
 * partition-function routines needed by convert_temp_to_u on the GPU.
 *
 * cooling.cc includes this header so the GPU cooling kernel can call these
 * without -rdc=true (they are inlined at the call site).  hydrogen_molecule.cc
 * retains its own non-inline host definitions for non-GPU builds and for any
 * host callers from other TUs.
 *
 * Include order: must come after Kokkos headers (for KOKKOS_INLINE_FUNCTION)
 * and after allvars.h (for BOLTZMANN_CGS).
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef EOS_SUBSTELLAR_ISM

KOKKOS_INLINE_FUNCTION
void hydrogen_molecule_zrot_mixture(double temp, double result[3]) {
    const double EPSILON = 2.220446049250313e-16;
    const double THETA_ROT = 85.4;  // in K
    const double ortho_frac = 0.75; // 3:1 mixture
    const double para_frac = 1 - ortho_frac;
    const double x = THETA_ROT / temp;
    const double expmx = exp(-x);
    const double expmx4 = pow(expmx, 4);

    double error = 1e100;
    double z[2] = {0}; // index 0 for para, 1 for ortho
    double dz_dtemp[2] = {0};
    double d2z_dtemp2[2] = {0};
    double zterm[2] = {0};

    z[0] = zterm[0] = 1.0;
    z[1] = zterm[1] = 9.0;
    double expterm = expmx4 * expmx * expmx;

    int j = 2;
    double dzterm, d2zterm;
    while (error > EPSILON) {
        int s = j % 2;
        zterm[s] *= (2 * j + 1) * expterm / (2 * j - 3);
        int jjplusone = j * (j + 1);
        if (s == 1) {
            dzterm = (jjplusone - 2) * x * zterm[1];
            d2zterm = ((jjplusone - 2) * x - 2) * dzterm;
        } else {
            dzterm = jjplusone * x * zterm[0];
            d2zterm = (jjplusone * x - 2) * dzterm;
        }
        z[s] += zterm[s];
        dz_dtemp[s] += dzterm;
        d2z_dtemp2[s] += d2zterm;
        double err0 = zterm[0] / z[0];
        double err1 = zterm[1] / z[1];
        error = (err1 > err0) ? err1 : err0;
        expterm *= expmx4;
        j++;
    }

    result[0] = exp(para_frac * log(z[0]) + ortho_frac * log(z[1]));
    result[1] = BOLTZMANN_CGS * temp * (para_frac * dz_dtemp[0] / z[0] + ortho_frac * dz_dtemp[1] / z[1]);
    result[2] = BOLTZMANN_CGS * (ortho_frac * (2 * dz_dtemp[1] + d2z_dtemp2[1] - dz_dtemp[1] * dz_dtemp[1] / z[1]) / z[1] +
                                 para_frac * (2 * dz_dtemp[0] + d2z_dtemp2[0] - dz_dtemp[0] * dz_dtemp[0] / z[0]) / z[0]);
}

KOKKOS_INLINE_FUNCTION
void hydrogen_molecule_zvib(double temp, double result[3]) {
    const double THETA_VIB = 6140;
    const double x = THETA_VIB / temp;
    result[0] = -1.0 / expm1(-x);
    result[1] = BOLTZMANN_CGS * THETA_VIB / expm1(x);
    result[2] = THETA_VIB * result[0] * result[1] / (temp * temp);
}

KOKKOS_INLINE_FUNCTION
void hydrogen_molecule_partitionfunc(double temp, double result[3]) {
    double zrot[3], zvib[3];
    hydrogen_molecule_zrot_mixture(temp, zrot);
    hydrogen_molecule_zvib(temp, zvib);
    double etot = 1.5 * BOLTZMANN_CGS * temp;
    double cv = 1.5 * BOLTZMANN_CGS;
    etot += zrot[1];
    cv += zrot[2];
    etot += zvib[1];
    cv += zvib[2];
    double gamma = (cv / BOLTZMANN_CGS + 1) / (cv / BOLTZMANN_CGS);
    result[0] = etot;
    result[1] = cv;
    result[2] = gamma;
}

#endif /* EOS_SUBSTELLAR_ISM */
