#ifndef TREE_PAIR_KERNEL_H
#define TREE_PAIR_KERNEL_H
/* Softened pair gravity of one tree interaction (target: h, zeta, ptype, pmass; secondary: mass, h_p, zeta_sec, ptype_sec).
   Single source for force_treeevaluate and the KETJU pair subtraction, which must remove exactly what the tree added. */
/* requires mesh/kernel.h (no include guard there): include it before this header */

static inline void tree_pair_gravity_factors(double r, double r2, double mass, double h, double h_p, int ptype, int ptype_sec,
                                             double zeta, double zeta_sec, double pmass, int AGS_kernel_shared_BITFLAG,
                                             double &fac_accel, double &fac_pot, double &fac_tidal, double &fac2_tidal)
{
    double h_inv, h3_inv, u, h_p_inv, h_p3_inv, u_p;
    fac_pot = 0; fac_tidal = 0; fac2_tidal = 0;
    (void)r2; (void)zeta; (void)zeta_sec; (void)pmass; (void)ptype; (void)ptype_sec; (void)AGS_kernel_shared_BITFLAG; (void)h_p_inv; (void)h_p3_inv; (void)u_p; (void)fac_pot; (void)fac_tidal; (void)fac2_tidal;
                if((r >= h) && (r >= h_p)) // can safely do a purely-Newtonian force (can be done by kernel-gravity as well, but no need to enter all the conditional statements below so just do it here for simplicity //
                {
                    fac_accel = mass / (r2 * r);
#ifdef EVALPOTENTIAL
                    fac_pot = -mass / r;
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE
                    fac_tidal = fac_accel; fac2_tidal = 3.0 * mass / (r2 * r2 * r); /* second derivative of potential needs this factor */
#endif
                }
                else
                {
                    double h_grav = h;
#if !defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
                    if(h_p > h_grav) {h_grav = h_p;} // in this case, symmetrize by taking the maximum here always
#endif
                    h_inv=1./h_grav; h3_inv=h_inv*h_inv*h_inv; u=r*h_inv; // set here to ensure this is using the correct values //
                    fac_accel = mass * kernel_gravity(u, h_inv, h3_inv, 1);
#ifdef EVALPOTENTIAL
                    fac_pot = mass * kernel_gravity(u, h_inv, h3_inv, -1);
#endif
#ifdef COMPUTE_TIDAL_TENSOR_IN_GRAVTREE /* second derivatives needed -> calculate them from softened potential */
                    fac_tidal = fac_accel; fac2_tidal = mass * kernel_gravity(u, h_inv, h3_inv, 2);  /* save original fac_accel without shortrange_table factor or zeta terms (needed for tidal field calculation) */
#endif
                    
#if defined(ADAPTIVE_GRAVSOFT_SYMMETRIZE_FORCE_BY_AVERAGING)
                    if(h_p > 0) // first, appropriately symmetrize the forces between particles. only do this is secondary is a particle, so has a type and softening! //
                    {
                        int symmetrize_by_averaging = 0; // default here to symmetrize by taking the maximum, but this will vary below //
                        // the 'zeta' terms for conservation with adaptive softening assume kernel-scale forces are averaged to symmetrize, to make them continuous
                        if(ptype_sec>=0) {if((1 << ptype_sec) & (AGS_kernel_shared_BITFLAG)) {symmetrize_by_averaging=1;}} // symmetrize by averaging only for particles which have a shared AGS structure since this is how our correction terms are derived //
#ifdef SINGLE_STAR_SINK_DYNAMICS
                        if((ptype!=0) || (ptype_sec!=0)) {symmetrize_by_averaging=0;} // we don't want to do the symmetrization below for sink interactions because it can create very noisy interactions between tiny sink particles and diffuse gas. However we do want it for gas-gas interactions so we keep the below
#endif
                        double prefac_corr_p=1., prefac_corr_orig=1.; // this will give a symmetrized pair by linear averaging
                        if(symmetrize_by_averaging==0) {prefac_corr_p=2; prefac_corr_orig=0.;} // symmetrize instead with the old method of simply taking the larger of the pair. here only act if the softening of the particle whose force is being summed is greater than the target //
                        if((symmetrize_by_averaging==1) || (h_p>h)) // condition to need to evaluate the alternate particle ('p' side)
                        {
                            h_p_inv=1./h_p; h_p3_inv=h_p_inv*h_p_inv*h_p_inv; u_p=r*h_p_inv;
                            fac_accel = 0.5 * (prefac_corr_orig * fac_accel + prefac_corr_p * mass * kernel_gravity(u_p, h_p_inv, h_p3_inv, 1)); // average with neighbor
#ifdef EVALPOTENTIAL
                            fac_pot = 0.5 * (prefac_corr_orig * fac_pot + prefac_corr_p * mass * kernel_gravity(u_p, h_p_inv, h_p3_inv, -1)); // average with neighbor
#endif
#if defined(COMPUTE_TIDAL_TENSOR_IN_GRAVTREE)
                            fac_tidal = fac_accel; fac2_tidal = 0.5 * (prefac_corr_orig * fac2_tidal + prefac_corr_p * mass * kernel_gravity(u_p, h_p_inv, h_p3_inv, 2)); // average forces -> average in tidal tensor as well. also save updated fac_tidal
#endif
                        }
                    } // closes if((h_p > 0)) clause
#endif // closes clause to symmetrize by averaging instead of taking the larger softening
                    
#if defined(ADAPTIVE_GRAVSOFT_FORGAS) || defined(ADAPTIVE_GRAVSOFT_FORALL)
                    double fac_corr = 0; int add_ags_zeta_terms_primary=1, add_ags_zeta_terms_secondary=1; u_p=r/h_p; // define the correction factor but also a clause to see if we should apply any 'correction' term at all
                    if((r<=0) || (pmass<=0) || (mass<=0) || (ptype_sec<0)) {add_ags_zeta_terms_primary=0; add_ags_zeta_terms_secondary=0;} // define conditions to add these terms at all (don't go forward if any of these conditions are met)
                    if((zeta == 0) || (u >= 1) || (h <= 0)) {add_ags_zeta_terms_primary=0;} // other conditions that mean -dont- use the term for the ab side
                    if((zeta_sec == 0) || (u_p >= 1) || (h_p <= 0)) {add_ags_zeta_terms_secondary=0;} // other conditions that mean -dont- use the term for the ba side
                    // correction only applies to 'shared-kernel' particles: so this needs to check if these are the same particles for which the 'shared' kernel lengths are computed
#if defined(ADAPTIVE_GRAVSOFT_FORGAS)
                    if(ptype != 0 || ptype_sec != 0) {add_ags_zeta_terms_primary=0; add_ags_zeta_terms_secondary=0;} // primary and secondary must be gas for ab side or ba side
#else
                    if((ptype_sec < 0) || !((1 << ptype) & (ADAPTIVE_GRAVSOFT_FORALL)) || !((1 << ptype_sec) & (ags_gravity_kernel_shared_BITFLAG(ptype)))) {add_ags_zeta_terms_primary=0;} // primary must be a valid ags particle and 'see' secondary for ab side
                    if((ptype_sec < 0) || !((1 << ptype_sec) & (ADAPTIVE_GRAVSOFT_FORALL)) || !((1 << ptype) & (ags_gravity_kernel_shared_BITFLAG(ptype_sec)))) {add_ags_zeta_terms_secondary=0;} // secondary must be a valid ags particle and 'see' primary for ba side
#endif
                    if(add_ags_zeta_terms_primary) // ab side
                    {
                        double dWdr, wp; kernel_main(u, h3_inv, h3_inv*h_inv, &wp, &dWdr, 1);
                        fac_corr += -(zeta/pmass) * dWdr / r; // go ahead and add the term
                    }
                    if(add_ags_zeta_terms_secondary) // ba side
                    {
                        double dWdr, wp; h_p_inv=1./h_p; h_p3_inv=h_p_inv*h_p_inv*h_p_inv; kernel_main(u_p, h_p3_inv, h_p3_inv*h_p_inv, &wp, &dWdr, 1);
                        fac_corr += -(zeta_sec/pmass) * dWdr / r; // go ahead and add the term
                    }
                    if(!isnan(fac_corr)) {fac_accel += fac_corr;}
#endif
                } // closes r < h (else) clause [where we need to deal with inside-the-softening factors]
}
#endif
