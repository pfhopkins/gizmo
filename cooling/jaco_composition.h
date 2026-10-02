#pragma once
/* Composition of a gas cell for the jaco network, in the standard cooling module's convention. No GIZMO dependencies, so
   test/jaco_composition can check it standalone. */

/* Mass fractions as Get_Gas_Mean_Molecular_Weight_mu counts them: metals Z_metals capped at 0.25; helium the tracked
   abundance Y_tracked capped at 0.35, or its default mass fraction 1 - X_default when the metal array does not carry it
   (he_tracked = 0). Returns X = 1 - Y - Z and stores Y and Z; n_He/n_H is then Y/(4X) and n_k/n_H = Z_k/(A_k X), so
   the species masses add up to the cell's mass density. */
static inline double jaco_mass_fractions(double Z_metals, double Y_tracked, int he_tracked, double X_default, double *Y,
                                         double *Z) {
    *Z = (Z_metals < 0.25) ? Z_metals : 0.25;
    *Y = he_tracked ? ((Y_tracked < 0.35) ? Y_tracked : 0.35) : 1.0 - X_default;
    return 1.0 - *Y - *Z;
}
