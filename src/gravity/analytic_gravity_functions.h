/* analytic_gravity_functions.h -- device-callable analytic gravitational fields (KOKKOS_INLINE_FUNCTION), shared by the host
 * routines in analytic_gravity.h and by kernels in other files. Include after allvars.h. */
#ifndef ANALYTIC_GRAVITY_FUNCTIONS_H
#define ANALYTIC_GRAVITY_FUNCTIONS_H
#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifdef GRAVITY_ANALYTIC_TALLBOX_HU21
/* Hu, Sternberg & van Dishoeck 2021 (arXiv:2103.03889) fixed vertical potential at R0, added to self-gravity; z from the box midplane, non-comoving.
 *   disk: Phi = 2 pi G Sigma_* H_* ln cosh(z/H_*),  a_z = -2 pi G Sigma_* tanh(z/H_*),  T_zz = -2 pi G Sigma_* / (H_* cosh^2(z/H_*))
 *   NFW:  Phi = -4 pi G rho_s r_s^3 [ln(1+x)/r - ln(1+R0/r_s)/R0], r = sqrt(z^2+R0^2), x = r/r_s, a_z = -G M(<r) z/r^3
 * Device-callable: the units come in as values (hu_tallbox_units_now on the host). */
struct hu_tallbox_units {double G, pc, msun;}; /* G, one parsec and one solar mass in code units */
KOKKOS_INLINE_FUNCTION void hu_tallbox_field(double zc, const struct hu_tallbox_units &u, double *az, double *tzz, double *phi)
{
    const double pc = u.pc, msun = u.msun;
    const double Sigma_star = 40. * msun / (pc * pc), z_star = 250. * pc; /* 40 Msun/pc^2, H_* = 250 pc */
    const double rho_s = 9.5e-3 * msun / (pc * pc * pc), r_s = 17.e3 * pc, R0 = 8.e3 * pc; /* NFW rho_s, r_s; solar radius */
    const double two_piG_Sig = 2. * M_PI * u.G * Sigma_star, fourpiG_rhos_rs3 = 4. * M_PI * u.G * rho_s * r_s * r_s * r_s;
    double ch = cosh(zc / z_star);
    double a_disk = -two_piG_Sig * tanh(zc / z_star), t_disk = -two_piG_Sig / (z_star * ch * ch), p_disk = two_piG_Sig * z_star * log(ch);
    double r = sqrt(zc*zc + R0*R0), x = r / r_s, r3 = r*r*r;
    double GM = fourpiG_rhos_rs3 * (log(1. + x) - x / (1. + x)), dGM = fourpiG_rhos_rs3 * x / ((1. + x) * (1. + x) * r_s);
    double a_dm = -GM * zc / r3, t_dm = -(dGM * zc * zc / (r*r3) + GM / r3 - 3. * GM * zc * zc / (r3*r*r));
    double p_dm = -fourpiG_rhos_rs3 * (log(1. + x) / r - log(1. + R0 / r_s) / R0);
    *az = a_disk + a_dm; *tzz = t_disk + t_dm; *phi = p_disk + p_dm;
}
KOKKOS_INLINE_FUNCTION double hu_tallbox_potential(double zc, const struct hu_tallbox_units &u) {double az, tzz, phi; hu_tallbox_field(zc, u, &az, &tzz, &phi); return phi;} /* Phi(midplane) = 0 */
static inline struct hu_tallbox_units hu_tallbox_units_now(void) {struct hu_tallbox_units u = {All.G, 1. / UNIT_LENGTH_IN_PC, 1. / UNIT_MASS_IN_SOLAR}; return u;}
#endif

#endif /* ANALYTIC_GRAVITY_FUNCTIONS_H */
