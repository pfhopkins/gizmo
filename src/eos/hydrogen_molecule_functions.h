/* hydrogen_molecule_functions.h — Canonical KOKKOS_INLINE_FUNCTION implementations
 * of H2 partition function routines.  Single source of truth for both CPU and GPU.
 *
 * Include order: after allvars.h (for BOLTZMANN_CGS). */
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

    /* Para (even j) and ortho (odd j) are kept in separate named scalars rather than z[s]/zterm[s]
       indexed by the loop-variant s = j % 2. Intel icpx at -O2 and above miscompiles the indexed form
       (the ortho update is dropped, its error term stays pinned at 1 and the loop never exits). */
    double error = 1e100;
    double z_para = 1.0, z_ortho = 9.0;          // partition function sums
    double zterm_para = 1.0, zterm_ortho = 9.0;  // current term
    double dz_para = 0, dz_ortho = 0, d2z_para = 0, d2z_ortho = 0;
    double expterm = expmx4 * expmx * expmx;

    int j = 2;
    /* Bounded. EPSILON is DBL_EPSILON, so the exit test asks the series to converge to machine
       precision; if the terms stagnate at FP granularity the loop never terminates. This body is
       device-callable and reached from the offloaded cooling chain, where that hangs the kernel
       rather than one rank. The series needs about 15 terms below 3000 K, above which H2 is
       dissociated, so this cap cannot bind in normal operation. */
    const int max_partition_terms = 200;
    for (int iter = 0; error > EPSILON && iter < max_partition_terms; ++iter) {
        int jjplusone = j * (j + 1);
        if (j & 1) { // ortho (odd j)
            zterm_ortho *= (2 * j + 1) * expterm / (2 * j - 3);
            double dzterm = (jjplusone - 2) * x * zterm_ortho;
            z_ortho += zterm_ortho; dz_ortho += dzterm; d2z_ortho += ((jjplusone - 2) * x - 2) * dzterm;
        } else { // para (even j)
            zterm_para *= (2 * j + 1) * expterm / (2 * j - 3);
            double dzterm = jjplusone * x * zterm_para;
            z_para += zterm_para; dz_para += dzterm; d2z_para += (jjplusone * x - 2) * dzterm;
        }
        double err_para = zterm_para / z_para;
        double err_ortho = zterm_ortho / z_ortho;
        error = (err_ortho > err_para) ? err_ortho : err_para;
        expterm *= expmx4;
        j++;
    }

    result[0] = exp(para_frac * log(z_para) + ortho_frac * log(z_ortho));
    result[1] = BOLTZMANN_CGS * temp * (para_frac * dz_para / z_para + ortho_frac * dz_ortho / z_ortho);
    result[2] = BOLTZMANN_CGS * (ortho_frac * (2 * dz_ortho + d2z_ortho - dz_ortho * dz_ortho / z_ortho) / z_ortho +
                                 para_frac * (2 * dz_para + d2z_para - dz_para * dz_para / z_para) / z_para);
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

KOKKOS_INLINE_FUNCTION
double hydrogen_molecule_energy(double temp) {
    if (temp < 12.5) {
        return 1.5 * BOLTZMANN_CGS * temp; // only translation
    } else if (temp > 1e5) {
        return 3.5 * BOLTZMANN_CGS * temp; // all DOF excited
    }
    double zrot[3], zvib[3];
    hydrogen_molecule_zrot_mixture(temp, zrot);
    hydrogen_molecule_zvib(temp, zvib);
    double etot = 1.5 * BOLTZMANN_CGS * temp; // translation
    etot += zrot[1];                          // rotation
    etot += zvib[1];                          // vibration
    return etot;
}

KOKKOS_INLINE_FUNCTION
double hydrogen_molecule_gamma(double temp) {
    if (temp < 12.5) { return 5. / 3; }
    else if (temp > 1e5) { return 9. / 7; }
    double zrot[3], zvib[3];
    hydrogen_molecule_zrot_mixture(temp, zrot);
    hydrogen_molecule_zvib(temp, zvib);
    double cv = 1.5 + zrot[2] / BOLTZMANN_CGS + zvib[2] / BOLTZMANN_CGS;
    return (cv + 1) / cv;
}

#endif /* EOS_SUBSTELLAR_ISM */
