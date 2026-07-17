/* cosmology_functions.h — Canonical implementations of basic cosmology
 * helpers (Hubble rate, time-since formation) shared by host and device.
 * The host externals in cosmology.cc are one-line wrappers around these
 * cores, so there is a single copy of each formula.
 *
 * Body visibility: a device translation unit that will evaluate the source
 * payload on-device defines GRAVTREE_SOURCE_DEVICE_TU before its includes and,
 * when GRAVTREE_SOURCE_LAZY_SUPPORTED holds (every gravity-tree source-payload
 * helper enabled in this build is device-callable), sees the bodies as device
 * inlines; translation units that own the host wrapper externals define
 * GRAVTREE_SOURCE_HOST_OWNER_TU before their includes and always see the bodies
 * (host-inline), whatever the compiler. TUs that define neither marker never
 * parse the bodies (e.g. cooling.cc, which includes this header before proto.h).
 *
 * GR_TABULATED_COSMOLOGY[_H/_G/_W] un-defines GRAVTREE_SOURCE_LAZY_SUPPORTED: its branches
 * below interpolate file-loaded tables through host-only helpers
 * (DarkEnergy_a, dHfak, hubble_function_external), so all its tabular
 * work runs exclusively on host and such builds use the eager host
 * source precompute — expect some slowdown, especially on small-N active
 * timesteps.  Note the tables enter the age evaluation only for comoving
 * integrations with |1-OmegaMatter-OmegaLambda|>0.01; the flat branch
 * uses the closed-form solution and never calls hubble_function.
 *
 * Caller must include allvars.h and proto.h first (standard convention for
 * these headers; proto.h has no include guard).
 */
#ifndef COSMOLOGY_FUNCTIONS_H
#define COSMOLOGY_FUNCTIONS_H

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#if defined(GRAVTREE_SOURCE_HOST_OWNER_TU) || (defined(GRAVTREE_SOURCE_DEVICE_TU) && defined(GRAVTREE_SOURCE_LAZY_SUPPORTED))

/* Hubble rate H(a) in code units */
KOKKOS_INLINE_FUNCTION double hubble_function_core(double a)
{
    double hubble_a;

#ifdef GR_TABULATED_COSMOLOGY_H
    hubble_a = All.Hubble_H0_CodeUnits * hubble_function_external(a);
#else
    hubble_a = All.OmegaRadiation / (a*a*a*a) + All.OmegaMatter / (a*a*a) + (1 - All.OmegaMatter - All.OmegaLambda - All.OmegaRadiation) / (a*a)
#ifdef GR_TABULATED_COSMOLOGY
    + DarkEnergy_a(a);
#else
    + All.OmegaLambda;
#endif
    hubble_a = All.Hubble_H0_CodeUnits * sqrt(hubble_a);
#endif
#ifdef GR_TABULATED_COSMOLOGY_G
    hubble_a *= dHfak(a);
#endif
    return (hubble_a);
}

/* time (in Gyr) between the code time/scale-factor t_initial and the current All.Time */
KOKKOS_INLINE_FUNCTION double evaluate_time_since_t_initial_in_Gyr_core(double t_initial)
{
    double age,a0,a1,a2,x0,x1,x2;
    if(All.ComovingIntegrationOn)
    {
        a0 = t_initial;
        a2 = All.Time;
        if(fabs(1-(All.OmegaMatter+All.OmegaLambda))<=0.01)
        {
            /* use exact solution for flat universe, ignoring the radiation-dominated epoch [no stars forming then] */
            x0 = (All.OmegaMatter/(1-All.OmegaMatter))/(a0*a0*a0);
            x2 = (All.OmegaMatter/(1-All.OmegaMatter))/(a2*a2*a2);
            age = (2./(3.*sqrt(1-All.OmegaMatter)))*log(sqrt(x0*x2)/((sqrt(1+x2)-1)*(sqrt(1+x0)+1)));
            age *= 1./All.Hubble_H0_CodeUnits;
        } else {
            /* use simple trap rule integration */
            a1 = 0.5*(a0+a2);
            x0 = 1./(a0*hubble_function_core(a0));
            x1 = 1./(a1*hubble_function_core(a1));
            x2 = 1./(a2*hubble_function_core(a2));
            age = (a2-a0)*(x0+4.*x1+x2)/6.;
        }
    } else {
        /* time variable is simple time, when not in comoving coordinates */
        age=All.Time-t_initial;
    }
    age *= UNIT_TIME_IN_GYR; // convert to absolute Gyr
    if(isnan(age) || age<=0) {age = 0;}
    return age;
}

#endif /* GRAVTREE_SOURCE_HOST_OWNER_TU || (GRAVTREE_SOURCE_DEVICE_TU && GRAVTREE_SOURCE_LAZY_SUPPORTED) */

#endif /* COSMOLOGY_FUNCTIONS_H */
