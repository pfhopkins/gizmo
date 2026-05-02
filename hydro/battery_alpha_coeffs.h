/* battery_alpha_coeffs.h -- generalized-Ohm coefficients (alpha_O, alpha_H,
 * alpha_A) for an electron + ion + dust + neutral plasma. These are the same
 * coefficients that drive the existing GIZMO non-ideal MHD module (Wardle 2007,
 * Pandey & Wardle 2008, Keith & Wardle 2014); see Soliman, Hopkins & Squire
 * 2025 (SHS25) §2.4 for the explicit confirmation that the alpha's in
 * SHS25 Eqs. 10-12 reduce to the standard Wardle/Pandey-Wardle expressions
 * in the J_d -> 0 limit.
 *
 * The dust battery uses these alphas applied to the dust current J_d
 * (or its TVA-derived equivalent), yielding the EMF
 *   E'_bat,d = -alpha_O J_d - alpha_H (J_d x b_hat) - alpha_A ((J_d x b_hat) x b_hat)
 *
 * The non-ideal MHD module uses the SAME alphas applied to (J - J_d), yielding
 * the resistive EMF E'_J. SHS25 Eqs. 10-12:
 *
 *   alpha_O = (omega_en * omega_+n) / (mu_+ omega_en + mu_e omega_+n)
 *
 *   alpha_H = [ mu_+ Omega_+ (omega_en^2 + Omega_e^2)
 *             + mu_e Omega_e (omega_+n^2 + Omega_+^2) ]
 *           / { [mu_+ (omega_en - Omega_e) + mu_e (omega_+n - Omega_+)]
 *             * [mu_+ (omega_en + Omega_e) + mu_e (omega_+n + Omega_+)] }
 *
 *   alpha_A = mu_e * mu_+ * (Omega_e * omega_+n + Omega_+ * omega_en)^2
 *           / [psi_H * (mu_+ omega_en + mu_e omega_+n)]
 *           + 2 * Omega_e * Omega_+ * (mu_+^2 omega_en^2 + mu_e^2 omega_+n^2)
 *           / [psi_H * (mu_+ omega_en + mu_e omega_+n)]
 *
 * where:
 *   mu_j     = n_j * q_j^2 / m_j   [s^-2]    (mass-density-times-(q/m)^2)
 *   Omega_j  = q_j * B / (m_j * c) [s^-1]    (signed cyclotron frequency)
 *   omega_jn = momentum-transfer collision frequency of species j with
 *              neutrals, rho_n <sigma v>_jn / (m_j + m_n)         [s^-1]
 *
 * Reference: Soliman, Hopkins & Squire 2025, ApJ 985, 55 (arXiv:2410.21461).
 *
 * Header-only KOKKOS_INLINE_FUNCTION, no rdc.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifndef BATTERY_ALPHA_COEFFS_H
#define BATTERY_ALPHA_COEFFS_H

#if defined(MHD_BATTERY_MECHANISMS) && (MHD_BATTERY_MECHANISMS & (4|8))

/* Plasma state inputs to the SHS25 alpha coefficients. All inputs are physical
 * cgs except where noted. Caller fills the struct, helper returns the alphas. */
struct sh25_plasma_state
{
    double mu_e;          /* n_e e^2 / m_e   [s^-2]                            */
    double mu_p;          /* n_+ q_+^2 / m_+ [s^-2]                            */
    double Omega_e;       /* signed e cyclotron freq, q_e B / (m_e c) [s^-1]   */
    double Omega_p;       /* signed +ion cyclotron freq                  [s^-1]*/
    double omega_en;      /* electron-neutral momentum-transfer rate     [s^-1]*/
    double omega_pn;      /* ion-neutral momentum-transfer rate          [s^-1]*/
    double omega_dn;      /* dust-neutral momentum-transfer rate         [s^-1]*/
    double mu_d;          /* n_d q_d^2 / m_d [s^-2]                            */
    double Omega_d;       /* signed grain cyclotron freq                 [s^-1]*/
};

struct sh25_alpha
{
    double alpha_O;       /* SHS25 Eq. 10                                      */
    double alpha_H;       /* SHS25 Eq. 11                                      */
    double alpha_A;       /* SHS25 Eq. 12                                      */
};

/* SHS25 Eqs. 10-12. Closed-form algebra; no iteration. */
KOKKOS_INLINE_FUNCTION
struct sh25_alpha sh25_alpha_coeffs(const struct sh25_plasma_state &s)
{
    struct sh25_alpha a = {0,0,0};

    const double denom_O = s.mu_p * s.omega_en + s.mu_e * s.omega_pn;
    if(denom_O > 0) {
        a.alpha_O = (s.omega_en * s.omega_pn) / denom_O;
    }

    /* Hall denominator psi_H = product of two factors symmetric in +/- Omega */
    const double f_minus = s.mu_p * (s.omega_en - s.Omega_e) + s.mu_e * (s.omega_pn - s.Omega_p);
    const double f_plus  = s.mu_p * (s.omega_en + s.Omega_e) + s.mu_e * (s.omega_pn + s.Omega_p);
    const double psi_H   = f_minus * f_plus;

    if(fabs(psi_H) > 0) {
        const double num_H = s.mu_p * s.Omega_p * (s.omega_en * s.omega_en + s.Omega_e * s.Omega_e)
                           + s.mu_e * s.Omega_e * (s.omega_pn * s.omega_pn + s.Omega_p * s.Omega_p);
        a.alpha_H = num_H / psi_H;
    }

    if((fabs(psi_H) > 0) && (denom_O > 0)) {
        const double common_denom = psi_H * denom_O;
        const double term1 = s.mu_e * s.mu_p
            * pow(s.Omega_e * s.omega_pn + s.Omega_p * s.omega_en, 2);
        const double term2 = 2.0 * s.Omega_e * s.Omega_p
            * (s.mu_p * s.mu_p * s.omega_en * s.omega_en + s.mu_e * s.mu_e * s.omega_pn * s.omega_pn);
        a.alpha_A = (term1 + term2) / common_denom;
    }

    return a;
}

#endif /* MHD_BATTERY_MECHANISMS & (4|8) */

#endif /* BATTERY_ALPHA_COEFFS_H */
