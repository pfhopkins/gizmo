#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"

/* In this file we implement functions for computing ISM chemical rates and abundances under
various simplifying steady-state assumptions. Most prescriptions follow Kim et al. 2023ApJS..264...10K
and references therein.
 */

#ifdef SIMPLE_STEADYSTATE_CHEMISTRY
/* C photoionization rate based on local FUV flux and accounting for CR-generated LW-band photons, e.g. Gredel 1987 */
MyFloat photoionization_rate_C(int i, MyFloat shieldfac)
{
    MyFloat G0 = get_FUV_G0(i, shieldfac, 1); // mode 1 accounts for dust-, self-, and H2 cross-shielding
    return 3.43e-10 * G0 + 520 * CellP[i].MolecularMassFraction * Get_CosmicRayIonizationRate_cgs(i);
}

/* direct cosmic ray ionization rate of C */
MyFloat cosmic_ray_ionization_rate_C(int i)
{
    return 3.85 * Get_CosmicRayIonizationRate_cgs(i);
}

/* Total ionization rate*/
MyFloat total_ionization_rate_C(int i, MyFloat shieldfac)
{
    return photoionization_rate_C(i, shieldfac) + cosmic_ray_ionization_rate_C(i);
}

/* Grain charging parameter psi = G0 sqrt(T) / ne in cgs units */
MyFloat grain_charge_psi(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac)
{
    MyFloat ne = CellP[i].Density * All.cf_a3inv * HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS * x_elec;
    MyFloat G0 = get_FUV_G0(i, shieldfac, 0);
    return G0 * sqrt(temp) / ne + 50; // add 50 to prevent from becoming too small in strongly shielded gas, following Kim 2023
}

#define NUM_RECOMB_TABLE_IONS 12
char *ion_names[NUM_RECOMB_TABLE_IONS] = {"H+", "He+", "C+", "Na+", "Mg+", "Si+", "S+", "K+", "Ca+", "Mn+", "Fe+", "Ca++"};

int ion_name_to_index(char *ion_name)
{
    for (int i = 0; i < NUM_RECOMB_TABLE_IONS; i++)
    {
        if (strcmp(ion_name, ion_names[i]) == 0)
        {
            return i;
        }
    }
    return 0;
}

/* Grain-assisted recomibination rate coefficient in cm^3 s^-1, from Weingartner & Draine 2001

Parameters
----------
i: int
    Index of particle
ion: string
    Name of the ion to output the recombination coefficient for. Options: H+, He+, C+, Na+, Mg+, Si+, S+, K+, Ca+, Mn+, Fe+, Ca++

Returns
-------
alpha_gr: float
    Grain-assisted recombination coefficient in cm^3 s^-1, such that the volumetric recombination rate is alpha_gr n_ion n_e
*/
MyFloat alpha_recomb_grain(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, char *ion_name)
{
    MyFloat psi = grain_charge_psi(i, temp, x_elec, shieldfac);
    // MyFloat temp = get_temperature(i);
    int j = ion_name_to_index(ion_name);

    MyFloat C[NUM_RECOMB_TABLE_IONS][7] = {
        {12.25, 8.074E-6, 1.378, 5.087E2, 1.586E-2, 0.4723, 1.102E-5}, // H+
        {5.572, 3.185E-7, 1.512, 5.115E3, 3.903E-7, 0.4956, 5.494E-7}, // He+
        {45.58, 6.089E-3, 1.128, 4.331E2, 4.845E-2, 0.8120, 1.333E-4}, // C+
        {2.178, 1.732E-7, 2.133, 1.029E4, 1.859E-6, 1.0341, 3.223E-5}, // Na+
        {2.510, 8.116E-8, 1.864, 6.170E4, 2.169E-6, 0.9605, 7.232E-5}, // Mg+
        {2.166, 5.678E-8, 1.874, 4.375E4, 1.635E-6, 0.8964, 7.538E-5}, // Si+
        {3.064, 7.769E-5, 1.319, 1.087E2, 3.475E-1, 0.4790, 4.689E-2}, // S+
        {1.596, 1.907E-7, 2.123, 8.138E3, 1.530E-5, 1.0380, 4.550E-5}, // K+
        {1.636, 8.208E-9, 2.289, 1.254E5, 1.349E-9, 1.1506, 7.204E-4}, // Ca+
        {2.029, 1.433E-6, 1.673, 1.403E4, 1.865E-6, 0.9358, 4.339E-9}, // Mn+
        {1.701, 9.554E-8, 1.851, 5.763E4, 4.116E-8, 0.9456, 2.198E-5}, // Fe+
        {8.270, 2.051E-4, 1.252, 1.590E2, 6.072E-2, 0.5980, 4.497E-7}  // Ca++
    };

    MyFloat Z = P[i].Metallicity[0] / All.SolarAbundances[0];
    return Z * 1e-14 * C[j][0] / (1 + C[j][1] * pow(psi, C[j][2]) * (1 + C[j][3] * pow(temp, C[j][4]) * pow(psi, -C[j][5] - C[j][6] * log(temp))));
}

/* Fraction of C atoms in C+ */
MyFloat f_Cplus(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac)
{
    MyFloat ionization_rate = total_ionization_rate_C(i, shieldfac);
    MyFloat alpha = sqrt(temp / 6.67e-3), beta = sqrt(temp / 1.943e6), gamma = 0.7849 + 0.1597 * exp(-49550 / temp);
    MyFloat k_rr = 2.995e-9 / (alpha * pow(1 + alpha, 1. - gamma) * pow(1 + beta, 1 + gamma));                                       // radiative recombination coefficient  - Gong 2017 Table 1 Eq 17
    MyFloat k_dr = pow(temp, -1.5) * (6.346e-9 * exp(-12.17 / temp) + 9.793e-9 * exp(-73.8 / temp) + 1.634e-6 * exp(-15230 / temp)); // dielectronic recombination - Gong 2017 Table 1 Eq 17
    MyFloat k_gr = alpha_recomb_grain(i, temp, x_elec, shieldfac, "C+");
    MyFloat k_cplus_H2 = 2.31e-13 * pow(temp, -1.3) * exp(-23 / temp);
    MyFloat nHcgs = nH_CGS(i);
    MyFloat ne = nHcgs * x_elec;
    MyFloat nH2 = 0.5 * nHcgs * CellP[i].MolecularMassFraction;
    MyFloat result = ionization_rate / (ionization_rate + k_gr * nHcgs + (k_rr + k_dr) * ne + k_cplus_H2 * nH2);
    return result;
}

/* Fraction of O in O+ from neutral hydrogen fraction */
MyFloat f_Oplus(MyFloat nHp)
{
    return nHp; // assume ionization fraction of O is the same as that of H due to ~equal ionization energy and efficient charge exchange
}

/* Fraction of C atoms in CO: Kim 2023 Eq 25 */
MyFloat f_CO(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, MyFloat nHp)
{
    MyFloat xi_cr16 = Get_CosmicRayIonizationRate_cgs(i) / 1e-16, Zd = P[i].Metallicity[0] / All.SolarAbundances[0];
    MyFloat G0 = get_FUV_G0(i, shieldfac, 0);
    MyFloat n_COcrit = pow(4e3 * Zd / (xi_cr16 * xi_cr16), cbrt(G0)) * (50 * xi_cr16 / pow(Zd, 1.4));
    MyFloat nHcgs = nH_CGS(i);
    MyFloat f_CO = 0.5 * CellP[i].MolecularMassFraction * (1 - DMAX(f_Cplus(i, temp, x_elec, shieldfac), f_Oplus(nHp))) / (1 + pow(n_COcrit / nHcgs, 2));
    return f_CO;
}

/* Contribution of C+ to electron abundance */
MyFloat return_electron_fraction_from_Cplus(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac){
    MyFloat x_Cplus = P[i].Metallicity[2]/All.SolarAbundances[2] * 1.6e-4 * f_Cplus(i, temp, x_elec, shieldfac); // Assumes gas-phase C abundance 1.6e-4 (Sofia 2004)
    return x_Cplus;
}

/* Contribution of O+ to electrons - essentially always negligible in ISM conditions but included for completeness */
MyFloat return_electron_fraction_from_Oplus(int i, MyFloat nHp){
    MyFloat x_Oplus = P[i].Metallicity[4]/All.SolarAbundances[4] * 3.2e-4 * f_Oplus(nHp); // Assumes gas-phase O abundance 3.2e-4 (Savage & Sembach 1996)
    return x_Oplus;
}

/* Contribution of molecular ions to electron abundance */
MyFloat return_electron_fraction_from_molecular_ions(int i, MyFloat temp){
    MyFloat zeta_cr = Get_CosmicRayIonizationRate_cgs(i);
    MyFloat beta_recomb = 3e-6 / sqrt(DMAX(All.MinGasTemp, temp)); // Fromang, Terquem & Balbus 2002 eq. 9
    MyFloat xe= sqrt(zeta_cr / (beta_recomb * DMAX(1e2, nH_CGS(i))));
    return sqrt(zeta_cr / (beta_recomb * DMAX(1e2, nH_CGS(i)))); // Armitage 2010 eq. 24
}


/* 
Contribution of collisionally-dissociated alkali ions to electron abundance

Uses fit to Saha equation solution from Balbus & Hawley 2000, capped at abundance of K.

This is only accurate for low ionizations, with an attempt to smoothly interpolate to the fully-ionized limit.
To do this properly, solve the Saha equation as in e.g. Wurster arxiv:1608.00983 section 2.2
 */
MyFloat return_electron_fraction_from_alkali(int i, MyFloat temp){
    if(temp<100){
        return 0.; // negligible below critical temperature
    }
    MyFloat x_K =  1e-7 * P[i].Metallicity[0]/All.SolarAbundances[0];
    MyFloat xe = 6.47e-13 * sqrt(x_K/1e-7) * sqrt(sqrt(temp*temp*temp/1e9))  * sqrt(2.4e15 / nH_CGS(i)) * exp(-25188/temp)/1.15e-11; // low-ionization approximation
    xe = 1./(1/x_K + 1/xe); // smooth interpolant to limit to x_K
    return xe;
}

#endif
