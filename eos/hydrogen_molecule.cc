/*
Special routines for computing thermodynamic properties of the hydrogen
molecule, assuming a standard 3:1 ortho:para mixture and accounting for
rotational, vibration, and translational degrees of freedom.

Equations follow Boley 2007, ApJ, 656, L89
*/

#include "../declarations/allvars.h"
#include <math.h>

#ifdef EOS_SUBSTELLAR_ISM

void hydrogen_molecule_zrot_mixture(double temp, double result[3]) {
    /*
    Rotational partition function of hydrogen molecule and derived quantities,
    considering a 3:1 mixture of ortho- and parahydrogen that cannot efficiently
    come into equilibrium.

    Parameters
    ----------
    temp: double
        Temperature in K
    ortho_frac: double
        Fraction of ortho-H2 (default is 3:1 ortho:para mixture)
    result: double[3]
        Stores the partition function value, the average rotational energy per
    molecule, and the heat capacity per molecule at constant volume.
    */

    const double EPSILON = 2.220446049250313e-16;
    const double THETA_ROT = 85.4;  // in K
    const double ortho_frac = 0.75; // 3:1 mixture
    const double para_frac = 1 - ortho_frac;
    const double x = THETA_ROT / temp;
    const double expmx = exp(-x);
    const double expmx4 = pow(expmx, 4);

    /* NOTE: para (even j) and ortho (odd j) are kept in separate named scalars rather than
       zterm[s]/z[s] indexed by the loop-variant s=j%2. The indexed form is miscompiled by
       Intel oneAPI icpx at -O2 and above (the ortho update is dropped, so zterm_ortho stays
       at its initial 9.0, err_ortho stays pinned at 1, and the loop never terminates -- an
       infinite hang, not a wrong answer). MAX_ITER is a second, compiler-independent guard:
       an unbounded convergence loop is a latent hang for any input. The series needs <=~15
       terms below 3000 K (H2 is dissociated above that), so 200 is very generous. */
    const int MAX_ITER = 200;
    double error = 1e100;
    double z_para = 1.0, z_ortho = 9.0;          // partition function sums
    double zterm_para = 1.0, zterm_ortho = 9.0;  // current term
    double dz_para = 0, dz_ortho = 0, d2z_para = 0, d2z_ortho = 0;
    double expterm = expmx4 * expmx * expmx;

    // Summing over rotational levels
    int j = 2, iter = 0;
    while (error > EPSILON && iter < MAX_ITER) {
        iter++;
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

    result[0] = exp(para_frac * log(z_para) + ortho_frac * log(z_ortho));                    // partition function
    result[1] = BOLTZMANN_CGS * temp * (para_frac * dz_para / z_para + ortho_frac * dz_ortho / z_ortho); // mean energy per molecule
    result[2] = BOLTZMANN_CGS * (ortho_frac * (2 * dz_ortho + d2z_ortho - dz_ortho * dz_ortho / z_ortho) / z_ortho +
                                 para_frac * (2 * dz_para + d2z_para - dz_para * dz_para / z_para) / z_para); // heat capacity
}

void hydrogen_molecule_zvib(double temp, double result[3]) {
    /*
    Vibrational partition function of hydrogen molecule and derived quantities.

    Parameters
    ----------
    temp: double
        Temperature in K
    result: double[3]
        Stores the partition function value, the average rotational energy per
    molecule, and the heat capacity per molecule at constant volume.
    */
    const double THETA_VIB = 6140;
    const double x = THETA_VIB / temp;
    result[0] = -1.0 / expm1(-x);
    result[1] = BOLTZMANN_CGS * THETA_VIB / expm1(x);
    result[2] = THETA_VIB * result[0] * result[1] / (temp * temp);
}

void hydrogen_molecule_partitionfunc(double temp, double result[3]) {
    /*
    Thermodynamic quantities derived from the partition function of the
    hydrogen molecule.

    Parameters
    ----------
    temp: double
        Temperature in K
    result: double[3]
        Stores the the average rotational energy per molecule in erg,
        the heat capacity per molecule at constan volume in erg/K,
        and the adiabatic index
    */

    double zrot[3], zvib[3];
    hydrogen_molecule_zrot_mixture(temp, zrot);
    hydrogen_molecule_zvib(temp, zvib);
    double etot = 1.5 * BOLTZMANN_CGS * temp; // translation
    double cv = 1.5 * BOLTZMANN_CGS;
    etot += zrot[1]; // rotation
    cv += zrot[2];
    etot += zvib[1]; // vibration
    cv += zvib[2];
    double gamma = (cv / BOLTZMANN_CGS + 1) / (cv / BOLTZMANN_CGS);
    result[0] = etot;
    result[1] = cv;
    result[2] = gamma;
}

double hydrogen_molecule_energy(double temp) {
    /*
    Average energy of a H2 molecule in thermodynamic equilibrium

    Parameters
    ----------
    temp: double
        Temperature in K

    Returns
    -------
    etot: double
        Average energy of a H2 molecule of temperture T in erg
    */

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

double hydrogen_molecule_gamma(double temp) {
    /*
    First adiabatic index of hydrogen molecule assuming
    a 3:1 ortho:para mixture

    Parameters
    ----------
    temp: double
        Temperature in K

    Returns
    -------
    gamma: double
        Adiabatic index
    */

    if (temp < 12.5) {
        return 5. / 3; // only translation
    } else if (temp > 1e5) {
        return 9. / 7; // all DOF excited
    }

    double zrot[3], zvib[3];
    hydrogen_molecule_zrot_mixture(temp, zrot);
    hydrogen_molecule_zvib(temp, zvib);
    double cv = 1.5;               // translation
    cv += zrot[2] / BOLTZMANN_CGS; // rotation
    cv += zvib[2] / BOLTZMANN_CGS; // vibration
    return (cv + 1) / cv;
}

#endif // EOS_SUBSTELLAR_ISM
