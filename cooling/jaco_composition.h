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

/* CIE ions for a cell from the CIE table's x_H+, x_He+ and x_He++ (tabulated at the table's own He abundance): He++ capped
   at the cell's y, x_H+ at the H not in H2 (xHp_max), He+ at the He left, each floored at x_floor. Returns the ions'
   free electrons. */
static inline double jaco_cie_ions_for_cell(double xHp_tab, double xHep_tab, double xHepp_tab, double xHp_max, double y,
                                            double x_floor, double *xHp, double *xHep, double *xHepp) {
    double hepp = (xHepp_tab < y) ? xHepp_tab : y, hp = (xHp_tab < xHp_max) ? xHp_tab : xHp_max;
    double hep = (xHep_tab < y - hepp) ? xHep_tab : y - hepp;
    *xHp = (hp > x_floor) ? hp : x_floor;
    *xHep = (hep > x_floor) ? hep : x_floor;
    *xHepp = (hepp > x_floor) ? hepp : x_floor;
    return *xHp + *xHep + 2.0 * *xHepp;
}

/* A cell's cached free electrons are stale where its CIE ions (ne_cie, at its own composition) are collisionally ionized
   and it has fewer than 99% of them: the table is CIE at n_H = 1, which bounds the steady state from below only roughly. */
static inline int jaco_ions_stale(double ne_cell, double ne_cie) {
    return ne_cie > 0.1 && !(ne_cell >= 0.99 * ne_cie);
}
