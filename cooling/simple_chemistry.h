/* simple_chemistry.h — Canonical KOKKOS_INLINE_FUNCTION implementations of
 * ISM steady-state chemistry functions.  Single source of truth for both
 * CPU and GPU.  simple_chemistry.cc retains host-only helpers (f_CO,
 * ion_name_to_index, string-based alpha_recomb_grain wrapper).
 *
 * alpha_recomb_grain uses integer ion index (0=H+, 1=He+, 2=C+, etc.)
 * instead of strcmp-based string lookup for GPU compatibility.
 *
 * Include order: after allvars.h, proto.h. */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef SIMPLE_STEADYSTATE_CHEMISTRY

/* Grain charging parameter psi = G0 sqrt(T) / ne in cgs units */
KOKKOS_INLINE_FUNCTION
MyFloat grain_charge_psi(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat ne = cell[i].Density * All.cf_a3inv * HYDROGEN_MASSFRAC * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS * x_elec;
    MyFloat G0 = get_FUV_G0(i, shieldfac, 0, pp, cell);
    return G0 * sqrt(temp) / ne + 50;
}

/* Grain-assisted recombination rate using integer ion index instead of strcmp.
 * Ion index: 0=H+, 1=He+, 2=C+, 3=Na+, 4=Mg+, 5=Si+, 6=S+, 7=K+, 8=Ca+, 9=Mn+, 10=Fe+, 11=Ca++ */
KOKKOS_INLINE_FUNCTION
MyFloat alpha_recomb_grain(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, int j, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat psi = grain_charge_psi(i, temp, x_elec, shieldfac, pp, cell);
    /* Weingartner & Draine 2001 Table 3 coefficients: C[ion][7] */
    const double C_table[12][7] = {
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
    if(j < 0 || j >= 12) {j = 0;}
    MyFloat Z = pp[i].Metallicity[0] / All.SolarAbundances[0];
    return Z * 1e-14 * C_table[j][0] / (1 + C_table[j][1] * pow(psi, C_table[j][2]) * (1 + C_table[j][3] * pow(temp, C_table[j][4]) * pow(psi, -C_table[j][5] - C_table[j][6] * log(temp))));
}

/* C photoionization rate */
KOKKOS_INLINE_FUNCTION
MyFloat photoionization_rate_C(int i, MyFloat shieldfac, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat G0 = get_FUV_G0(i, shieldfac, 1, pp, cell);
    return 3.43e-10 * G0 + 520 * cell[i].MolecularMassFraction * Get_CosmicRayIonizationRate_cgs(i, pp, cell);
}

/* direct cosmic ray ionization rate of C */
KOKKOS_INLINE_FUNCTION
MyFloat cosmic_ray_ionization_rate_C(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    return 3.85 * Get_CosmicRayIonizationRate_cgs(i, pp, cell);
}

/* Total ionization rate of C */
KOKKOS_INLINE_FUNCTION
MyFloat total_ionization_rate_C(int i, MyFloat shieldfac, struct particle_data *pp, struct gas_cell_data *cell)
{
    return photoionization_rate_C(i, shieldfac, pp, cell) + cosmic_ray_ionization_rate_C(i, pp, cell);
}

/* Fraction of O in O+ */
KOKKOS_INLINE_FUNCTION
MyFloat f_Oplus(MyFloat nHp) { return nHp; }

/* Fraction of C atoms in C+ */
KOKKOS_INLINE_FUNCTION
MyFloat f_Cplus(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat ionization_rate = total_ionization_rate_C(i, shieldfac, pp, cell);
    MyFloat alpha = sqrt(temp / 6.67e-3), beta = sqrt(temp / 1.943e6), gamma = 0.7849 + 0.1597 * exp(-49550 / temp);
    MyFloat k_rr = 2.995e-9 / (alpha * pow(1 + alpha, 1. - gamma) * pow(1 + beta, 1 + gamma));
    MyFloat k_dr = pow(temp, -1.5) * (6.346e-9 * exp(-12.17 / temp) + 9.793e-9 * exp(-73.8 / temp) + 1.634e-6 * exp(-15230 / temp));
    MyFloat k_gr = alpha_recomb_grain(i, temp, x_elec, shieldfac, 2 /* C+ index */, pp, cell);
    MyFloat k_cplus_H2 = 2.31e-13 * pow(temp, -1.3) * exp(-23 / temp);
    MyFloat nHcgs = HYDROGEN_MASSFRAC * cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
    MyFloat ne = nHcgs * x_elec;
    MyFloat nH2 = 0.5 * nHcgs * cell[i].MolecularMassFraction;
    MyFloat result = ionization_rate / (ionization_rate + k_gr * nHcgs + (k_rr + k_dr) * ne + k_cplus_H2 * nH2);
    return result;
}

/* Contribution of C+ to electron abundance */
KOKKOS_INLINE_FUNCTION
MyFloat return_electron_fraction_from_Cplus(int i, MyFloat temp, MyFloat x_elec, MyFloat shieldfac, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat x_Cplus = pp[i].Metallicity[2]/All.SolarAbundances[2] * 1.6e-4 * f_Cplus(i, temp, x_elec, shieldfac, pp, cell);
    return x_Cplus;
}

/* Contribution of O+ to electrons */
KOKKOS_INLINE_FUNCTION
MyFloat return_electron_fraction_from_Oplus(int i, MyFloat nHp, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat x_Oplus = pp[i].Metallicity[4]/All.SolarAbundances[4] * 3.2e-4 * f_Oplus(nHp);
    return x_Oplus;
}

/* Contribution of molecular ions to electron abundance */
KOKKOS_INLINE_FUNCTION
MyFloat return_electron_fraction_from_molecular_ions(int i, MyFloat temp, struct particle_data *pp, struct gas_cell_data *cell)
{
    MyFloat zeta_cr = Get_CosmicRayIonizationRate_cgs(i, pp, cell);
    MyFloat beta_recomb = 3e-6 / sqrt(DMAX(All.MinGasTemp, temp));
    MyFloat nHcgs_local = HYDROGEN_MASSFRAC * cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
    return sqrt(zeta_cr / (beta_recomb * DMAX(1e2, nHcgs_local)));
}

/* Contribution of alkali ions to electron abundance */
KOKKOS_INLINE_FUNCTION
MyFloat return_electron_fraction_from_alkali(int i, MyFloat temp, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(temp < 100) { return 0.; }
    MyFloat nHcgs_local = HYDROGEN_MASSFRAC * cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_CGS / PROTONMASS_CGS;
    MyFloat x_K = 1e-7 * pp[i].Metallicity[0]/All.SolarAbundances[0];
    MyFloat xe = 6.47e-13 * sqrt(x_K/1e-7) * sqrt(sqrt(temp*temp*temp/1e9)) * sqrt(2.4e15 / nHcgs_local) * exp(-25188/temp)/1.15e-11;
    xe = 1./(1/x_K + 1/xe);
    return xe;
}

#endif /* SIMPLE_STEADYSTATE_CHEMISTRY */
