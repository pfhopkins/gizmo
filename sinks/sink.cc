#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"


/*! \file sink.c
 *  \brief routines for gas accretion onto sink particles, and sink particle mergers
 */
/*!
 * This file is largely written by Phil Hopkins and Mike Grudic for GIZMO.
 *   Some parts evolved from the 'blackhole' routines in GADGET3 by Volker Springel,
 *   but the physical modules for sink particle accretion and feedback have been
 *   replaced, and the algorithms for their coupling are new to GIZMO.  This file was modified
 *   on 1/9/15 by Paul Torrey (ptorrey@mit.edu) for clarity by parsing the existing code into
 *   smaller files and routines.  Some communication and sink particle structures were modified
 *   to reduce memory usage. Cleanup, de-bugging, and consolidation of routines by Xiangcheng Ma
 *   (xchma@caltech.edu) followed on 05/15/15; re-integrated by PFH. Massive updates in 2019
 *   from David Guszejnov and Mike Grudic to incorporate and develop single star modules, and
 *   from PFH to integrate those and re-write the parallelism entirely to conform to the newer
 *   code standards and be properly multi-threaded.
 */

#ifdef SINK_PARTICLES // top-level flag [needs to be here to prevent compiler breaking when this is not active] //


/*  This is the parent routine for the sink (black hole, star, planet, etc) physics modules.
 *  It is called in calculate_non_standard_physics in run.c */
void sink_accretion(void)
{
    if(All.TimeStep == 0.) return; /* no evolution */
    PRINT_STATUS("Sink operations begin...");
#if defined(SINK_GRAVCAPTURE_NONGAS)
    long i; for(i = 0; i < NumPart; i++) {P[i].SwallowID = 0;} /* zero out accretion */ // This zero-out loop is effectively performed in density.c now, only on -gas- particles that are actually going to be looked at this timestep, to reduce overhead when only a few particles are active. But it still needs to be done for non-gas particles.
#endif
    sink_start();              /* allocates and cleans SinkTempInfo struct */

    /* this is the PRE-PASS loop.*/
    sink_environment_loop();    /* populates SinkTempInfo based on surrounding gas (sink_environment.c).
                                      If using gravcap the desired mass accretion rate is calculated and set to SinkTempInfo.mass_to_swallow_edd */
#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
    sink_environment_second_loop();    /* Here we compute quantities that require knowledge of previous environment variables --> Bulge-Disk kinematic decomposition for gravitational torque accretion  */
#endif
    /*----------------------------------------------------------------------
     Now do a first set of local operations based on sink environment calculation:
     calculate mdot, dynamical friction, and other 'sink-centric' operations.
     No MPI comm necessary.
     ----------------------------------------------------------------------*/
    sink_properties_loop();       /* do 'sink-centric' operations such as dyn-fric, mdot, etc. This loop is at the end of this file.  */
    /*----------------------------------------------------------------------
     Now we perform a second pass over the sink particle environment.
     Re-evaluate the decision to stochastically swallow gas if we exceed eddington.
     Use the above info to determine the weight functions for feedback
     ----------------------------------------------------------------------*/
    sink_feed_loop();       /* sink mergers and gas/star/dm accretion events are evaluated - P[j].SwallowID's are set */
    /*----------------------------------------------------------------------
     Now we do a THIRD pass over the particles, and
     this is where we can do the actual 'swallowing' operations
     (sink_evaluate_swallow), and 'kicking' operations
     ----------------------------------------------------------------------*/
    sink_swallow_and_kick_loop();
    /*----------------------------------------------------------------------
     Now do final operations on the results from the last pass
     ----------------------------------------------------------------------*/
    sink_final_operations(); /* final operations on the sink with tabulated quantities (not a neighbor loop) */
    sink_end();            /* frees SinkTempInfo; cleans up */
    PRINT_STATUS(" ..closing sink operations");
#if defined(SINK_GRAVCAPTURE_NONGAS)
    for(i = 0; i < NumPart; i++) {P[i].SwallowID = 0;} /* re-zero accretion */
#endif
}





/* calculate escape velocity to use for bounded-ness calculations relative to the sink */
double sink_vesc(int j, double mass, double r_code, double sink_softening)
{
    double cs_to_add = 10. / UNIT_VEL_IN_KMS; /* we can optionally add a 'fudge factor' to v_esc to set a minimum value; useful for -some- galaxy applications */
#if defined(SINK_SEED_GROWTH_TESTS) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    cs_to_add = 0;
#endif
    double m_eff = mass + P[j].Mass; // acount for 2-body mass
#if defined(SINK_SEED_GROWTH_TESTS) || defined(SINGLE_STAR_SINK_DYNAMICS) || defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
    double gas_density = -1; if(P[j].Type == 0) {gas_density = CellP[j].Density;}
#ifdef GRAIN_FLUID
    if((1<<P[j].Type) & GRAIN_PTYPES) {gas_density = P[j].Gas_Density;}
#endif
    if(gas_density > 0) {m_eff += 4.*M_PI * r_code*r_code*r_code * gas_density;} // assume an isothermal sphere interior, for Shu-type solution
#endif
    double hinv = 1./sink_softening, fac=2.*All.G*m_eff/All.cf_atime;
#if defined(SINK_REPOSITION_ON_POTMIN)
    return sqrt(fac/r_code + cs_to_add*cs_to_add); // in this case BH dynamics are intentionally inexact and gravitational BH velocities not well-resolved, so use the larger Keplerian term here
#endif
    return sqrt(fac*fabs(kernel_gravity(r_code*hinv,hinv,hinv*hinv*hinv,-1)) + cs_to_add*cs_to_add); // accounts for softening [non-Keplerian inside softening]
}

/* check whether a particle is sufficiently bound to the BH to qualify for 'gravitational capture' */
int sink_check_boundedness(int j, double vrel, double vesc, double dr_code, double sink_radius) // need to know the sink radius, which can be distinct from both the softening and search radii
{
#if defined(SINK_REPOSITION_ON_POTMIN)
    if(P[j].Type == 5) {return 1;} // if this module is active, we aren't doing exact BH-BH dynamics, so default is to be unrestrictive for BH-BH mergers //
#endif

    /* if pair is a gas particle make sure to account for its pressure and internal energy */
    double cs=0; if(P[j].Type==0) {
        double vA = Get_Gas_Alfven_speed_i(j);
        if(fabs(GAMMA(j)-1) < 0.1) {cs = sqrt(vA*vA + 3.*CellP[j].Pressure/CellP[j].Density);} // assume you're running gamma ~ 1 to hack an isothermal EOS, so we assume gamma=5/3 for boundedness calculation
        else {cs = sqrt(vA*vA + 2.*CellP[j].InternalEnergy);} // effective speed [since what we really want is internal energy] to add to relative velocity to compare with escape speed for boundedness check
    } // use the fast MHD wavespeed to account for magnetic+thermal energy (but not e.g. cosmic ray), in allowing accretion //

#ifdef SINGLE_STAR_SINK_DYNAMICS
    double gas_density = -1; if(P[j].Type == 0) {gas_density = CellP[j].Density;}
#ifdef GRAIN_FLUID
    if((1<<P[j].Type) & GRAIN_PTYPES) {gas_density = P[j].Gas_Density;}
#endif
    if(gas_density > 0)
    {
        if((Get_Particle_Size(j) > sink_radius*1.396263) && (P[j].Type == 0)) {return 0;} // particle volume should be less than sink volume, enforcing a minimum spatial resolution around the sink
#if defined(COOLING)  // check if we're probably sitting at the bottom of a quasi-hydrostatic Larson core
        double nHcgs = HYDROGEN_MASSFRAC * (gas_density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS);
        if(nHcgs > 1e13 && (cs > 0.1 * vrel || P[j].Type != 0)) {double m_eff = 4. * M_PI * dr_code * dr_code * dr_code * gas_density; vesc = DMAX(sqrt(2*All.G * m_eff / dr_code), vesc);} // assume an isothermal sphere interior, for Shu-type solution, and re-estimate vesc using self-gravity of the gas
#endif
    }
#endif

    double v2 = (vrel*vrel+cs*cs)/(vesc*vesc); int bound = 0;
    if(v2 < 1)
    {
        double apocenter = dr_code / (1.-v2); // furthest distance the cell -could- get from the sink, on a purely radial orbit [ignoring internal energy effects, in e.g. a Keplerian potential, this is approximately twice the equivalent circular orbit, while for a highly-eccentric orbit, this is exactly the apocentric radius]
        double apocenter_max = 2.*SinkParticle_GravityKernelRadius; //  = few x epsilon (softening length); check that this is within 2x epsilon is statement that circular orbit with equivalent energy is entirely inside epsilon //
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS // Bate 1995-style criterion, with a fixed sink/accretion radius that is distinct from both the force softening and the search radius
#ifdef GRAIN_FLUID
        if((1 << P[j].Type) & (GRAIN_PTYPES)) {if(dr_code>1.4*sink_radius) {return 0;} else {return 1;}} // simply yes-no, if bound and within sink radius, gets accreted
#endif
        if(dr_code>sink_radius) {return 0;} else {return 1;} // simply yes-no, if bound and within sink radius, gets accreted
#endif
#if !defined(SINGLE_STAR_SINK_DYNAMICS) && !defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS) && defined(SINK_SEED_GROWTH_TESTS)
        double r_j = ForceSoftening_KernelRadius(j);
        if(P[j].Type == 0) {r_j = DMAX(r_j , P[j].KernelRadius);}
        apocenter_max = DMAX(10.0*SinkParticle_GravityKernelRadius,DMIN(50.0*SinkParticle_GravityKernelRadius,r_j));
        if(P[j].Type == 5) {apocenter_max = DMIN(apocenter_max , 1.*SinkParticle_GravityKernelRadius);}
#endif
        if(apocenter < apocenter_max) {bound = 1;}
    }
    return bound;
}



/* weight function for local (short-range) coupling terms from the sink particle, including the single-scattering radiation pressure and the bal winds */
double sink_fb_angleweight_localcoupling(int j, double cos_theta, double r, double H_kernel)
{
    // this follows what we do with SNe, and applies a normalized weight based on the fraction of the solid angle subtended by the particle //
    if(r <= 0 || H_kernel <= 0) return 0;
    double H_j = P[j].KernelRadius;
    if((r >= H_j && r >= H_kernel) || (H_j <= 0) || (P[j].Mass <= 0) || (CellP[j].Density <= 0)) return 0;
    /* now calculate all the kernel quantities we need */
    double hinv=1./H_kernel,hinv_j=1./H_j,hinv3_j=hinv_j*hinv_j*hinv_j,wk_j=0,u=r/H_kernel; /* note these lines and many below assume 3D sims! */
    double dwk_j=0,u_j=r*hinv_j,hinv4_j=hinv_j*hinv3_j,V_j=P[j].Mass/CellP[j].Density;
    double hinv3=hinv*hinv*hinv,hinv4=hinv*hinv3,wk=0,dwk=0;
    if(u<1) {kernel_main(u,hinv3,hinv4,&wk,&dwk,0);} else {wk=dwk=0;}
    if(u_j<1) {kernel_main(u_j,hinv3_j,hinv4_j,&wk_j,&dwk_j,0);} else {wk_j=dwk_j=0;}
    double V_i = 4.*M_PI/3. * H_kernel*H_kernel*H_kernel / (All.DesNumNgb * All.SinkNgbFactor); // this is approximate, will be wrong (but ok b/c just increases weight to neighbors) when not enough neighbors found //
    if(V_i<0 || isnan(V_i)) {V_i=0;}
    if(V_j<0 || isnan(V_j)) {V_j=0;}
    double face_area = fabs(V_i*V_i*dwk + V_j*V_j*dwk_j); // effective face area //
    wk = 0.5 * (1. - 1./sqrt(1. + face_area / (M_PI*r*r))); // corresponding geometric weight //
#if defined(SINK_FB_VOLUMEWEIGHTED)
    wk = 0.5 * (V_j/V_i) * (V_i*wk + V_j*wk_j); // weight in the limit N_particles >> 1 for equal-mass particles (accounts for self-shielding if some in dense disk)
#endif
#if defined(SINK_FB_COLLIMATED)
    double costheta2=cos_theta*cos_theta, eps_width_jet=0.35; eps_width_jet*=eps_width_jet; /* eps_width^2 is approximate with in radians of 'core' of jet */
    wk *= eps_width_jet * (eps_width_jet + costheta2) / ((eps_width_jet + 1) * (eps_width_jet + (1-costheta2)));
#endif
    if((wk <= 0)||(isnan(wk))) return 0; // no point in going further, there's no physical weight here
    return wk;
}



/* function below is used for long-range sink particle radiation fields -- used only in the forcetree routines (where they
    rely this for things like the long-range radiation pressure and compton heating) */
double sink_fb_angleweight(double sink_lum_input, MyFloat sink_angle[3], double dx, double dy, double dz)
{
#ifdef SINGLE_STAR_SINK_DYNAMICS
    return sink_lum_input;
#endif
    if(sink_lum_input <= 0) return 0;
    double r2 = dx*dx+dy*dy+dz*dz; if(r2 <= 0) return 0;
    if(r2*UNIT_LENGTH_IN_PC*UNIT_LENGTH_IN_PC*All.cf_atime*All.cf_atime < 1) return 0; /* no force at < 1pc */
#if defined(SINK_FB_COLLIMATED)
    double cos_theta = fabs((dx*sink_angle[0] + dy*sink_angle[1] + dz*sink_angle[2])/sqrt(r2*(sink_angle[0]*sink_angle[0]+sink_angle[1]*sink_angle[1]+sink_angle[2]*sink_angle[2]))); if(!isfinite(cos_theta)) {cos_theta=1;}
    double wt_normalized = 0.0847655*exp(4.5*cos_theta*cos_theta); // ~exp(-x^2/2*hR^2), normalized appropriately to give the correct total flux, for hR~0.3
    return sink_lum_input * wt_normalized; // ~exp(-x^2/2*hR^2), normalized appropriately to give the correct total flux, for hR~0.3
#endif
    return sink_lum_input;
}







void set_sink_mdot(int i, int n, double dt)
{
    double mdot=0; int k; k=0;
    double soundspeed2; soundspeed2 = convert_internalenergy_soundspeed2(n,SinkTempInfo[i].Sink_SurroudingGasInternalEnergy);
#ifdef SINK_GRAVACCRETION
    double m_tmp_for_accrate, mdisk_for_accrate, sink_mass, fac;
    double rmax_for_accrate,fgas_for_accrate,f_disk_for_accrate, f0_for_accrate;
#endif
#ifdef SINK_SUBGRIDBHVARIABILITY
    long nsubgridvar; int jsub; double varsg1,varsg2,omega_ri,n0_sgrid_elements,norm_subgrid,time_var_subgridvar;
    gsl_rng *random_generator_forbh;
#endif
#ifdef SINK_ENFORCE_EDDINGTON_LIMIT
    double meddington = sink_eddington_mdot(P[n].Sink_Mass);
#endif



#ifdef SINK_GRAVACCRETION
    /* calculate mdot: gravitational instability accretion rate from Hopkins & Quataert 2011 */
    if(SinkTempInfo[i].Mgas_in_Kernel > 0)
    {
        sink_mass = P[n].Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
        sink_mass += P[n].Sink_Mass_Reservoir;
#endif
        rmax_for_accrate = P[n].KernelRadius * All.cf_atime; /* convert to physical units */

#if defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0)
        /* DAA: torque rate based on kinematic B/D decomposition as in Angles-Alcazar et al. [here in brackets because it requires an extra pass] */
        m_tmp_for_accrate = SinkTempInfo[i].Mgas_in_Kernel + SinkTempInfo[i].Mstar_in_Kernel;
        double mbulge_for_accrate = SinkTempInfo[i].MstarBulge_in_Kernel;
        if(mbulge_for_accrate>SinkTempInfo[i].Mstar_in_Kernel) {mbulge_for_accrate=SinkTempInfo[i].Mstar_in_Kernel;}
        mdisk_for_accrate = m_tmp_for_accrate - mbulge_for_accrate;
        f_disk_for_accrate = mdisk_for_accrate / m_tmp_for_accrate;
#else
        /* DAA: Jalt_in_Kernel is now the TOTAL angular momentum (need to subtract Jgas here) */
        m_tmp_for_accrate = SinkTempInfo[i].Malt_in_Kernel;
        double j_tmp_for_accrate = (SinkTempInfo[i].Jalt_in_Kernel - SinkTempInfo[i].Jgas_in_Kernel).norm(); /* jx,y,z, is independent of 'a_scale' b/c ~ m*r*v, vphys=v/a, rphys=r*a */
        fgas_for_accrate = SinkTempInfo[i].Mgas_in_Kernel / m_tmp_for_accrate;

        fac = m_tmp_for_accrate * rmax_for_accrate * sqrt(All.G*(m_tmp_for_accrate+sink_mass)/rmax_for_accrate); /* All.G is G in code (physical) units */

        f_disk_for_accrate = fgas_for_accrate + (1.75*j_tmp_for_accrate/fac);
        if(f_disk_for_accrate>1) {f_disk_for_accrate=1;}
        mdisk_for_accrate = m_tmp_for_accrate * f_disk_for_accrate;
#endif  // if(SINK_GRAVACCRETION == 1)
        if(mdisk_for_accrate>0) {fgas_for_accrate=SinkTempInfo[i].Mgas_in_Kernel/mdisk_for_accrate;} else {fgas_for_accrate=0;}

        if((sink_mass <=0)||(fgas_for_accrate<=0)||(m_tmp_for_accrate<=0))
        {
            mdot = 0;
        } else {
            double menc_all, omega_dyn, mgas_in_racc, mdisk_for_accrate_units, sink_mass_units, rmax_for_accrate_units, r0_accretion;
            menc_all = m_tmp_for_accrate + P[n].Mass; // total enclosed mass in kernel (note P[n].Mass can be large if SINK_INCREASE_DYNAMIC_MASS is set large)
            omega_dyn = sqrt(All.G * menc_all / (rmax_for_accrate*rmax_for_accrate*rmax_for_accrate)); // 1/t_dyn for all mass inside maximum kernel radius
            mdisk_for_accrate_units = mdisk_for_accrate * UNIT_MASS_IN_SOLAR / 1.e9; /* mdisk/1e9msun */
            sink_mass_units = sink_mass * UNIT_MASS_IN_SOLAR / 1.e8; /* mbh/1e8msun */
            rmax_for_accrate_units = rmax_for_accrate * UNIT_LENGTH_IN_PC / 100.; /* r0/100pc */
            f0_for_accrate = 0.31*f_disk_for_accrate*f_disk_for_accrate*pow(mdisk_for_accrate_units,-1./3.); /* dimensionless factor for equations */
            mgas_in_racc = SinkTempInfo[i].Mgas_in_Kernel; r0_accretion = rmax_for_accrate; // -total- gas mass inside of search radius [rmax_for_accrate]
            mgas_in_racc = (4.*M_PI/3.) * (P[n].DensityAroundParticle*All.cf_a3inv) * r0_accretion*r0_accretion*r0_accretion; // use -local- estimator of gas mass in accretion radius //

            fac = 5.0 / (UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR); // basic normalization (use alpha=5, midpoint of values alpha=[1,10] from Hopkins and Quataert 2011 //
            fac *= pow(f_disk_for_accrate, 3./2.) * pow(sink_mass_units,1./6.) / (1. + f0_for_accrate/fgas_for_accrate); // dimensionless dependence on f_disk and m_sink (latter is weak)
            mdot = fac * mdisk_for_accrate_units * pow(rmax_for_accrate_units,-3./2.); // these are the quantities which scale strongly with the scale where we evaluate the BHAR
#if (SINK_GRAVACCRETION >= 1)
            r0_accretion = DMAX(rmax_for_accrate , 3.*SinkParticle_GravityKernelRadius*All.cf_atime); // set to some fixed multiple of the force softening, which is effectively not changing below low redshifts for the BH, and gives a constant physical anchor point
#endif
            mdot *= pow(r0_accretion/rmax_for_accrate , 3./2.); // scales to estimator value at specified 'accretion radius' r0_accretion, assuming f_disk, etc constant and M_enclosed is a constant-density profile //

#if (SINK_GRAVACCRETION == 2) // gas accreted at a constant rate per free-fall time from the specified 'accretion radius'
            mdot = mgas_in_racc * omega_dyn;
#endif
#if (SINK_GRAVACCRETION == 3) // gravito-turbulent estimator: accrete at constant fraction per free-fall from 'accretion radius' with that fraction being f_disk^2
            mdot = (mdisk_for_accrate / menc_all) * (mdisk_for_accrate / menc_all) * mgas_in_racc * omega_dyn;
#endif
#if (SINK_GRAVACCRETION == 4) || (SINK_GRAVACCRETION == 6) || (SINK_GRAVACCRETION == 7) // accrete constant fraction per free-fall time from accretion radius set to minimum of BH radius of gravitational dominance over Vc or cs (basically where gas more tightly bound to BH) - has Bondi-like form
            double Vc2_rmax = All.G * menc_all / rmax_for_accrate; // this is in physical units now
            mdot = 4.*M_PI * All.G*All.G * P[n].Sink_Mass*menc_all * (P[n].DensityAroundParticle*All.cf_a3inv) / pow(soundspeed2 + Vc2_rmax, 1.5);
#if (SINK_GRAVACCRETION == 6) || (SINK_GRAVACCRETION == 7)
            double bhvel2 = SinkTempInfo[i].Sink_SurroundingGasVel.norm_sq();
            double veldisp2_eff = bhvel2/3. + soundspeed2, masscorrfac = pow( menc_all/(1.e-10*menc_all + P[n].Sink_Mass) , 0.25 );
            mdot = masscorrfac * 4.*M_PI * All.G*All.G * P[n].Sink_Mass*menc_all * (P[n].DensityAroundParticle*All.cf_a3inv) / pow(1.e-5*soundspeed2 + Vc2_rmax, 1.5);
            mdot /= 1 + sqrt(veldisp2_eff/Vc2_rmax) * DMIN( veldisp2_eff/Vc2_rmax , masscorrfac );
#if (SINK_GRAVACCRETION == 7)
            mdot = 4.*M_PI * All.G*All.G * menc_all*menc_all * (P[n].DensityAroundParticle*All.cf_a3inv) / pow(veldisp2_eff + Vc2_rmax, 1.5);
#endif
#endif
#endif
#if (SINK_GRAVACCRETION == 5) // use default torques estimator, but then allow gas to accrete as Bondi-Hoyle when its circularization radius is inside the BH radius of influence
            double j_tmp_for_accrate=SinkTempInfo[i].Jgas_in_Kernel.norm_sq(),jcirc_crit=0;
            j_tmp_for_accrate=sqrt(j_tmp_for_accrate); jcirc_crit = SinkTempInfo[i].Mgas_in_Kernel * rmax_for_accrate*rmax_for_accrate*omega_dyn;
            jcirc_crit *= pow(sink_mass/m_tmp_for_accrate,2./3.);
            if(j_tmp_for_accrate < jcirc_crit) /* circularization within BH-dominated region, Bondi accretion valid */
            {
                double bhvel2 = SinkTempInfo[i].Sink_SurroundingGasVel.norm_sq();
                double rho = P[n].DensityAroundParticle*All.cf_a3inv; /* we want all quantities in physical units */
                double vcs_fac = pow(soundspeed2+bhvel2, 1.5);
                mdot = 4.*M_PI * All.G*All.G * P[n].Sink_Mass*P[n].Sink_Mass * rho / vcs_fac;
            } /* otherwise, circularization outside BH-dominated region, efficiency according to usual [above] */
#endif
#if (SINK_GRAVACCRETION == 8)
            double hubber_mdot_from_vr_estimator=MIN_REAL_NUMBER, hubber_mdot_disk_estimator=MIN_REAL_NUMBER; /* our computed 'hubber_mdot_vr_estimator' is their estimate of the radial inflow time from a Bondi flow: but care is needed, for any non-Bondi flow this can give unphysical or negative answers, so we need to limit it and be very cautious using it */
            if(SinkTempInfo[i].hubber_mdot_vr_estimator > 0) { hubber_mdot_from_vr_estimator = SinkTempInfo[i].hubber_mdot_vr_estimator; }
            if(SinkTempInfo[i].hubber_mdot_disk_estimator > 0) { hubber_mdot_disk_estimator = 0.01 * SinkTempInfo[i].Mgas_in_Kernel / (sqrt(All.G * P[n].Mass) * SinkTempInfo[i].hubber_mdot_disk_estimator);}
            double j_eff=SinkTempInfo[i].Jalt_in_Kernel.norm_sq(),m_eff=SinkTempInfo[i].Malt_in_Kernel;
            double facc_which_hubber_mdot = DMIN(1, 1.75*sqrt(j_eff)/(m_eff*sqrt(All.G*(m_eff+P[n].Mass)*rmax_for_accrate))); /* disk fraction estimator */
            mdot = DMAX( SinkTempInfo[i].hubber_mdot_bondi_limiter , pow(hubber_mdot_from_vr_estimator,1-facc_which_hubber_mdot)*pow(hubber_mdot_disk_estimator,facc_which_hubber_mdot));
#endif
#ifdef SINK_GRAVACCRETION_STELLARFBCORR
            double sigma_crit = 3000. * (2.09e-4 / UNIT_SURFDEN_IN_CGS); // from MG's fit to isolated cloud sims [converts from Msun/pc^2 to code units]
            double sigma_enc = (SinkTempInfo[i].Malt_in_Kernel + P[n].Mass) / (M_PI*rmax_for_accrate*rmax_for_accrate); // effective surface density [total gravitating mass / area]
            mdot *= sigma_enc / (sigma_enc + sigma_crit);
#endif
#ifdef OUTPUT_ADDITIONAL_RUNINFO
            printf(" ..Sink accretion kernel :: mdot %g Norm %g fdisk %g mass_1e8msun %g fgas %g f0 %g mdisk_9 %g rmax_100 %g \n",
                   mdot,fac,f_disk_for_accrate,sink_mass_units,fgas_for_accrate,f0_for_accrate,mdisk_for_accrate_units,rmax_for_accrate_units);
#endif
        } // if(fgas_for_accrate<=0)
    } // if(SinkTempInfo[i].Mgas_in_Kernel > 0)
    mdot *= All.SinkAccretionFactor; // this is a pure normalization multiplier here

#if (SINK_GRAVACCRETION >= 9) && (SINK_GRAVACCRETION <= 11) /* heres where we calculate the Bondi accretion rate, if that's going to be used */
    double bhvel2 = 0, rho = P[n].DensityAroundParticle * All.cf_a3inv; /* we want all quantities in physical units */
    bhvel2 = SinkTempInfo[i].Sink_SurroundingGasVel.norm_sq();
#if (SINK_GRAVACCRETION == 10)
    bhvel2 = 0;
#endif
    double fac = pow(soundspeed2+bhvel2, 1.5);
    if(fac > 0) {
        double AccretionFactor = All.SinkAccretionFactor;
#if (SINK_GRAVACCRETION == 11) /* variable-alpha model (Booth&Schaye 2009): now All.SinkAccretionFactor is the slope of the density dependence */
        AccretionFactor = 1.0; if(rho > All.PhysDensThresh) {AccretionFactor = pow(rho/All.PhysDensThresh, All.SinkAccretionFactor);}
#endif
        mdot = 4. * M_PI * AccretionFactor * All.G * All.G * P[n].Sink_Mass * P[n].Sink_Mass * rho / fac;
    } else {mdot=0;}
#endif

#endif // ifdef SINK_GRAVACCRETION


/* DAA: note that we should have mdot=0 here , otherwise the mass accreted is counted twice [mdot*dt in set_sink_new_mass, accreted mass in sink_swallow_and_kick] */
#ifdef SINK_GRAVCAPTURE_GAS
    mdot = 0; /* force mdot=0 despite any earlier settings here.  If this is set, we have to wait to swallow step to eval mdot. */
#endif


#ifdef SINK_ALPHADISK_ACCRETION
    /* use the mass in the accretion disk from the previous timestep to determine the BH accretion rate */
    double x_MdiskSelfGravLimiter = P[n].Sink_Mass_Reservoir / (SINK_ALPHADISK_ACCRETION * P[n].Sink_Mass);
    if(x_MdiskSelfGravLimiter > 20.) {mdot=0;} else {mdot *= exp(-0.5*x_MdiskSelfGravLimiter*x_MdiskSelfGravLimiter);}
    SinkTempInfo[i].mdot_reservoir = mdot;  mdot = 0;  /* if SINK_GRAVCAPTURE_GAS is off, this gets the accretion rate */
    if(P[n].Sink_Mass_Reservoir > 0)
    {
        /* this below is a more complicated expression using the outer-disk expression from Shakura & Sunyaev. Simpler expression
            below captures the same physics with considerably less potential to extrapolate to rather odd scalings in extreme regimes :
           mdot = (2.45/(UNIT_MASS_IN_SOLAR/UNIT_TIME_IN_YR)) * pow( 0.1 , 8./7.) * // normalization, then viscous disk 'alpha'
             pow( P[n].Sink_Mass*UNIT_MASS_IN_SOLAR/1.e8 , -5./14. ) * pow( P[n].Sink_Mass_Reservoir*UNIT_MASS_IN_SOLAR/1.e8 , 10./7. ) * // mbh , m_disk dependence
             pow( DMIN(0.2,DMIN(P[n].KernelRadius,SinkParticle_GravityKernelRadius)*All.cf_atime*UNIT_LENGTH_IN_PC) , -25./14. ); // r_disk dependence */
        double t_acc_disk = (4.2e7/UNIT_TIME_IN_YR) * pow((P[n].Sink_Mass_Reservoir+P[n].Sink_Mass) / P[n].Sink_Mass_Reservoir, 0.4); /* shakura-sunyaev disk, integrated out to Q~1 radius, approximately */
#ifdef SINK_TIMESCALE_SET
        double f_disk_sink = P[n].Sink_Mass_Reservoir / P[n].Sink_Mass; t_acc_disk = (SINK_TIMESCALE_SET/UNIT_TIME_IN_YR) / sqrt( f_disk_sink * (1 + f_disk_sink) );
#endif

#ifdef SINGLE_STAR_SINK_DYNAMICS
        double reff = SinkParticle_GravityKernelRadius, Gm_i = 1./(All.G*P[n].Mass);
#if defined(SINK_GRAVCAPTURE_FIXEDSINKRADIUS)
        double cs_min = 0.2 / UNIT_VEL_IN_KMS;
        reff = All.G*P[n].Sink_Formation_Mass/(cs_min*cs_min), Gm_i = 1./(All.G*P[n].Sink_Formation_Mass); // effectively setting the value to the freefall time the particle has when it forms, for both 'size' of disk and 'effective mass' (for dynamical time below)
#endif
        t_acc_disk = sqrt(reff*reff*reff * Gm_i); // dynamical time at radius "reff", essentially fastest-possible accretion time
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM) && defined(SINK_MDOT_FROM_ALPHAMODEL)
        reff = SinkParticle_GravityKernelRadius; Gm_i = 1./(All.G*P[n].Mass); // these need to be reset in case they are re-defined above
        double j = P[n].Sink_Specific_AngMom.norm_sq();  // calculate magnitude of specific ang mom
        j = sqrt(j) * (1. + 1./SINK_ALPHADISK_ACCRETION); // correction assuming a ratio of accretion disk to sink mass ~SINK_ALPHADISK_ACCRETION [max allowed], with the material in the sink having given its angular momentum to the sink [which is what should happen]
        t_acc_disk = 2.*M_PI*j*j*j*Gm_i*Gm_i / fabs(SINK_MDOT_FROM_ALPHAMODEL); // orbital time at circularization radius of the alpha-disk: SINK_MDOT_FROM_ALPHAMODEL is approximately equivalent to the 'alpha' parameter, setting how rapidly accretion occurs (=0.01 -> 100 orbits)
        if(SINK_MDOT_FROM_ALPHAMODEL>0) {t_acc_disk = 100. * t_acc_disk * (1 / (Gm_i * DMIN(reff, j*j*Gm_i))) / soundspeed2;} // Shakura-Sunyaev prescription with alpha=0.01, using minimum of sink and circularization radius
#endif
#if defined(SINGLE_STAR_AND_SSP_NUCLEAR_ZOOM_SPECIALBOUNDARIES)
        t_acc_disk = DMIN(t_acc_disk , 50. / UNIT_TIME_IN_YR); /* custom hack for now for 'fast' disk accretion - will be replaced by physical model later */
#endif
#if defined(TIDAL_TIMESTEP_CRITERION) /* limit the accretion timescale to not be more than some multiple of the maximum disk dynamical time allowed by tidal truncation */
        double tidal_tensor_mag2 = P[n].tidal_tensorps.frobenius_norm_sq();
        if(tidal_tensor_mag2 > 0) {
            double t_dyn_extgrav = sqrt(1. / (All.cf_a3inv * sqrt(tidal_tensor_mag2 / 6.))); /* recovers tdyn=1/Omega=sqrt[r^3/GM] for a Keplerian potential */
            t_acc_disk = DMIN(t_acc_disk, 100. * t_dyn_extgrav); /* coefficient here is the arbitrary number of dynamical times to limit accretion timescale to */
        }
#endif
#endif

#if defined(SINK_RIAF_SUBEDDINGTON_MODEL)
        /* subgrid model for convective vs disk regimes, with accretion rates from the high-res simulations */
        double psi_magdisk = 0.1;
        double r_kernel = P[n].KernelRadius * All.cf_atime;
        if(r_kernel < ForceSoftening_KernelRadius(n)) {r_kernel = ForceSoftening_KernelRadius(n);}
        double m_sink = P[n].Sink_Mass;
        double m_kernel = SinkTempInfo[i].Malt_in_Kernel + P[n].Mass;
        double m_kernel_nonBH = m_kernel - m_sink;
        double mdot_ROI = SinkTempInfo[i].mdot_reservoir;
        double sigma2_eff = All.G * m_kernel_nonBH / r_kernel;
        double sigma_min = (10.*1.e5/UNIT_VEL_IN_CGS);
        if(sigma2_eff < sigma_min*sigma_min) {sigma2_eff = sigma_min*sigma_min;}
        double r_grav = 2. * All.G * m_sink / (C_LIGHT_CODE*C_LIGHT_CODE);
        double r_roi = All.G * m_sink / sigma2_eff;
        if(r_roi < r_grav) {r_roi = r_grav;}
        int ROI_resolved = 0;
        if(r_roi > r_kernel) {ROI_resolved = 1;}
        if(ROI_resolved)
        {
            double mgas_enc = SinkTempInfo[i].Mgas_in_Kernel + P[n].Sink_Mass_Reservoir;
            double omega_enc = sqrt(All.G * m_kernel / (r_kernel*r_kernel*r_kernel));
            mdot_ROI = psi_magdisk * mgas_enc * omega_enc;
            SinkTempInfo[i].mdot_reservoir = mdot_ROI;
        } else {
            double omega_enc = sqrt(All.G * P[n].Mass / (r_roi*r_roi*r_roi));
            double mdot_ROI_alt = psi_magdisk * P[n].Sink_Mass_Reservoir * omega_enc;
            if(mdot_ROI_alt > mdot_ROI) {mdot_ROI = mdot_ROI_alt;}
        }
        double mdot_Edd = m_sink / (5.e7 / UNIT_TIME_IN_YR);
        double mdot_crit_ROI = (2.*psi_magdisk) * mdot_Edd;
        mdot = 0;
        if(mdot_ROI > mdot_crit_ROI)
        {
            mdot = pow(r_grav/r_roi, 0.15) * mdot_ROI;
        } else {
            mdot = pow(r_grav/r_roi, 0.5) * mdot_ROI;
        }
        if(dt > 0) {
            double mdot_ROI_max = P[n].Sink_Mass_Reservoir / dt;
            if(mdot_ROI_max < mdot_ROI) {
                double mdot_max = (mdot/mdot_ROI) * mdot_ROI_max;
                mdot_ROI = mdot_ROI_max;
                mdot = mdot_max;
            }
        }
        if(mdot <= 1.e-30 || !isfinite(mdot)) {mdot = 1.e-30;}
        t_acc_disk = P[n].Sink_Mass_Reservoir / mdot;
        P[n].Sink_Mdot_ROI = mdot_ROI;
        P[n].Sink_ROI = r_roi;
#endif
        
#if defined(SINK_GRAVCAPTURE_GAS)
        t_acc_disk /= All.SinkAccretionFactor; // when using GRAVCAPTURE, this won't multiply the continuous mdot, but rather mdot from disk to BH
#endif
        if(dt > 0) {t_acc_disk = DMAX(t_acc_disk , 3.*dt);} /* make sure accretion timescale is at least a few timesteps to avoid over-shoot, etc */
        mdot = P[n].Sink_Mass_Reservoir / t_acc_disk;
    }
#endif //ifdef SINK_ALPHADISK_ACCRETION



#ifdef SINK_SUBGRIDBHVARIABILITY /* account for sub-grid accretion rate variability */
    if((mdot>0)&&(dt>0)&&(P[n].DensityAroundParticle>0))
    {
        omega_ri=sqrt(All.G*P[n].DensityAroundParticle*All.cf_a3inv); /* dynamical frequency in physical units */
        n0_sgrid_elements=10.0; norm_subgrid=0.55*3.256/sqrt(n0_sgrid_elements);
        nsubgridvar=(long)P[n].ID + (long)(All.Time/((All.TimeMax-All.TimeBegin)/1000.));
        /* this line just allows 'resetting' the time constants every so often, while generally keeping them steady */
        double fac;
        if(All.ComovingIntegrationOn) {fac=omega_ri * (evaluate_time_since_t_initial_in_Gyr(0.001)/(UNIT_TIME_IN_GYR));} else {fac=omega_ri * All.Time;} /* All.Time is physical time, this is good */
        random_generator_forbh=gsl_rng_alloc(gsl_rng_ranlxd1);
        gsl_rng_set(random_generator_forbh, nsubgridvar);
        if(n0_sgrid_elements >= 1) {
            for(jsub=1;jsub<=n0_sgrid_elements;jsub++) {
                varsg1=gsl_rng_uniform(random_generator_forbh);
                varsg2=gsl_ran_ugaussian(random_generator_forbh);
                time_var_subgridvar=fac*pow(omega_ri*dt,-((float)jsub)/n0_sgrid_elements) + 2.*M_PI*varsg1;
                mdot *= exp( norm_subgrid*cos(time_var_subgridvar)*varsg2 );
            }}
        gsl_rng_free(random_generator_forbh);
    } // if(mdot > 0)
#endif



#ifdef SINK_ALPHADISK_ACCRETION
    /* if there -is- an alpha-disk, protect the alpha-disk not to be over-depleted (i.e. overshooting into negative alpha-disk masses) */
    if(dt>0)
    {
#if defined(SINK_WIND_SPAWN)
        if(dt>0 && mdot > P[n].Sink_Mass_Reservoir/dt*All.Sink_accreted_fraction) mdot = P[n].Sink_Mass_Reservoir/dt*All.Sink_accreted_fraction;
#else
        if(dt>0 && mdot > P[n].Sink_Mass_Reservoir/dt) mdot = P[n].Sink_Mass_Reservoir/dt;
#endif
    }
#ifdef SINK_WIND_KICK /* DAA: correct the mdot into the accretion disk for the mass loss in "kick" winds */
    SinkTempInfo[i].mdot_reservoir *= All.Sink_accreted_fraction;
#endif

#else // SINK_ALPHADISK_ACCRETION

#if defined(SINK_WIND_KICK) || defined(SINK_WIND_SPAWN)
    mdot *= All.Sink_accreted_fraction; /* if there is no alpha-disk, the BHAR defined above is really an mdot into the accretion disk. the rate -to the hole- should be corrected for winds */
#endif
#endif // SINK_ALPHADISK_ACCRETION


#ifdef SINK_ENFORCE_EDDINGTON_LIMIT /* cap the maximum at the Eddington limit */
    if(mdot > All.SinkEddingtonFactor * meddington) {mdot = All.SinkEddingtonFactor * meddington;}
#endif

#if defined(SINK_RETURN_ANGMOM_TO_GAS) /* pre-calculate some quantities for 'angular momentum feedback' here, these have to be based on the mdot estimator above */
    double jmag=0,lmag=0,mdot_eff=mdot,return_timescale,angmom_toreturn; SinkTempInfo[i].angmom_norm_topass_in_swallowloop=0;
    jmag = SinkTempInfo[i].angmom_prepass_sum_for_passback.norm_sq();
    lmag = P[n].Sink_Specific_AngMom.norm_sq();
    lmag = P[n].Mass * sqrt(lmag); // this stores the magnitude of the _total_ angular momentum (mass * length * vel) internal to the sink
    return_timescale = P[n].Mass / mdot_eff; // rate of angular momentum return is (total angular momentum) / (return timescale) - here set to (total mass / mdot). Set this for your desired prescription.
    angmom_toreturn = lmag * DMIN(0.5, dt/return_timescale); // the actual angular momentum we will return this timestep (with a mild limiter so we don't dump it all at once)
    angmom_toreturn = DMIN(angmom_toreturn, 0.1 * All.G * (P[n].Mass+SinkTempInfo[i].Mgas_in_Kernel) * SinkTempInfo[i].Mgas_in_Kernel / DMAX(P[n].KernelRadius, SinkParticle_GravityKernelRadius) * dt); // this limiter explicitly ensures that we never apply a torquing force that is comparable in magnitude to the gravitational force: L < (small fraction) * G Mtot Mgas / R * dt (ie. torque is less than torque needed to change the orbit within an orbital time)
    if(jmag>0 && lmag>0) {SinkTempInfo[i].angmom_norm_topass_in_swallowloop = angmom_toreturn / sqrt(jmag);} /* this should be in units such that, times CODE radius and (code=physical) ang-mom, gives CODE velocity: looks ok at present */
#endif

    /* alright, now we can FINALLY set the BH accretion rate */
    if(isnan(mdot)) {mdot=0;}
    P[n].Sink_Mdot = DMAX(mdot,0);
}



/* Update the Sink_Mass and the Sink_Mass_Reservoir */
void set_sink_new_mass(int i, int n, double dt)
{
    int k; k=0; if(P[n].Sink_Mdot <= 0) {P[n].Sink_Mdot=0;} /* check unphysical values */
    if(dt <= 0 || !isfinite(dt)) {dt = MIN_REAL_NUMBER;}

    /* before mass update, track angular momentum in disk for 'smoothed' accretion case [using continuous accretion rate and specific AM of all material in kernel around BH] */
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM) && (SINK_FOLLOW_ACCRETED_ANGMOM == 1)
    double dm_acc_for_j = P[n].Sink_Mdot * dt, m_tot_for_j = P[n].Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
    dm_acc_for_j = SinkTempInfo[i].mdot_reservoir * dt; m_tot_for_j = P[n].Sink_Mass + P[n].Sink_Mass_Reservoir;
#endif
    P[n].Sink_Specific_AngMom = (P[n].Sink_Specific_AngMom*m_tot_for_j + SinkTempInfo[i].Jgas_in_Kernel*(dm_acc_for_j/(MIN_REAL_NUMBER + SinkTempInfo[i].Mgas_in_Kernel))) / (m_tot_for_j + dm_acc_for_j);
#endif

/*  for SINK_WIND_SPAWN
        - we accrete the winds first, either explicitly to the BH or implicitly into the disk -
        - then we remove the wind mass in the final loop
    for SINK_WIND_KICK
        - the BH grows according to the mdot set above (including the mass loss in winds)
        - if there is an alpha-disk, the mass going out in winds has been subtracted from mdot_reservoir
    for SINK_WIND_KICK + SINK_GRAVCAPTURE_GAS
        - the ratio of BH/disk growth-to-outflow rate is enforced explicitly in sink_swallow_and_kick */

    double dMSINK_continuous_accretion; dMSINK_continuous_accretion = P[n].Sink_Mdot * dt;
#ifdef SINK_ALPHADISK_ACCRETION
    P[n].Sink_Mass += dMSINK_continuous_accretion;   // mdot comes from the disk - no mass loss here regardless of BAL model -
    double dm_reservoir = SinkTempInfo[i].mdot_reservoir * dt - dMSINK_continuous_accretion;
    if(dm_reservoir < -P[n].Sink_Mass_Reservoir) {dm_reservoir=-P[n].Sink_Mass_Reservoir; P[n].Sink_Mass_Reservoir=0;} else {P[n].Sink_Mass_Reservoir += dm_reservoir;}
    if(P[n].Sink_Mass_Reservoir<0) {P[n].Sink_Mass_Reservoir=0;}
    if(P[n].Mass<0) {P[n].Mass=0;}
    dMSINK_continuous_accretion += dm_reservoir;
#else // #ifdef SINK_ALPHADISK_ACCRETION
#if defined(SINK_WIND_SPAWN)
    P[n].Sink_Mass += dMSINK_continuous_accretion / All.Sink_accreted_fraction; // accrete the winds first, then remove the wind mass in the final loop
#else
    P[n].Sink_Mass += dMSINK_continuous_accretion;
#endif
#endif // #else SINK_ALPHADISK_ACCRETION
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
    P[n].Sink_AccretionDeficit += dMSINK_continuous_accretion; // this is mass continuously accreted, which needs to be stochastically 'caught up to'
#endif
#ifdef JET_DIRECTION_FROM_KERNEL_AND_SINK
    double mtot = SinkTempInfo[i].Mgas_in_Kernel + P[n].Mass;
    SinkTempInfo[i].Sink_SurroundingGasCOM /= mtot; // this now stores the COM of the sink-gas system, relative to the sink position
    // We need the angular momentum in the COM frame of the sink-gas system, so must apply the correction -r x p. This is negligible when MBH >> Mgas but important when Mgas > MBH, e.g. low-mass stars at modest mass resolution
    double Mgas_over_mtot_sq = (SinkTempInfo[i].Mgas_in_Kernel/mtot) * SinkTempInfo[i].Mgas_in_Kernel;
    SinkTempInfo[i].Jgas_in_Kernel -= cross(SinkTempInfo[i].Sink_SurroundingGasCOM, SinkTempInfo[i].Sink_SurroundingGasVel) * Mgas_over_mtot_sq;
    P[n].Jgas_in_Kernel = SinkTempInfo[i].Jgas_in_Kernel + P[n].Sink_Specific_AngMom * P[n].Mass; // this stores the _total_ angular momentum (sink + gas) in the COM frame of the sink-gas system, including internal to the sink
    P[n].Mgas_in_Kernel=SinkTempInfo[i].Mgas_in_Kernel;
#endif
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    P[n].ZAMS_Mass = DMAX(P[n].ZAMS_Mass, P[n].Sink_Mass); // keep a running maximum of the stellar mass as the ZAMS mass
#endif
}



void set_sink_drag(int i, int n, double dt)
{
#if defined(SINK_REPOSITION_ON_POTMIN)
    int k;
    double sink_mass, x;
    if(SinkTempInfo[i].DF_mmax_particles>0) /* found something in the kernel, we can proceed */
    {
        /* averaged value for colomb logarithm and integral over the distribution function */
        /* fac_friction = log(lambda) * [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]                  */
        /*       lambda = b_max * v^2 / G / (M+m)                                          */
        /*        b_max = Size of system (e.g. Rvir)                                       */
        /*            v = Relative velocity of BH with respect to the environment          */
        /*            M = Mass of BH                                                       */
        /*            m = individual mass elements composing the large system (e.g. m<<M)  */
        /*            x = v/sqrt(2)/sigma                                                  */
        /*        sigma = width of the max. distr. of the host system                      */
        /*                (e.g. sigma = v_disp / 3                                         */
        sink_mass = P[n].Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
        sink_mass += P[n].Sink_Mass_Reservoir;
#endif
        double bhvel2_df = SinkTempInfo[i].DF_mean_vel.norm_sq();
        double fac, fac_friction;
        /* First term is approximation of the error function */
        fac = 8 * (M_PI - 3) / (3 * M_PI * (4. - M_PI));
        x = sqrt(bhvel2_df) / (sqrt(2) * SinkTempInfo[i].DF_rms_vel); x = fabs(x);
        fac_friction =  x / fabs(x) * sqrt(1 - exp(-x * x * (4 / M_PI + fac * x * x) / (1 + fac * x * x))) - 2 * x / sqrt(M_PI) * exp(-x * x);
        if(x < 0.2) {fac_friction = 4.*x*x*x/(3.*sqrt(M_PI)) * (1.-0.598843094706047*x*x);} // series expansion to prevent numerical errors for small-x
        if(x > 3.0) {fac_friction = 1;} // negligible error in this approximation
        /* now the Coulomb logarithm */
        fac = 50. / UNIT_LENGTH_IN_KPC; /* impact parameter */
        fac_friction *= log(1. + fac * bhvel2_df / (All.G * sink_mass));
        /* now we add a correction to only apply this force if M_SINK is not >> <m_particles> */
        fac_friction *= 1 / (1 + sink_mass / (5.*SinkTempInfo[i].DF_mmax_particles));
        /* now the dimensional part of the force */
        double Mass_in_Kernel = SinkTempInfo[i].Malt_in_Kernel;
        Mass_in_Kernel = SinkTempInfo[i].Malt_in_Kernel - SinkTempInfo[i].Mgas_in_Kernel;
        fac = Mass_in_Kernel / ( (4*M_PI/3) * pow(P[n].KernelRadius*All.cf_atime,3) ); /* mean density of all mass inside kernel */
        fac_friction *= 4*M_PI * All.G * All.G * fac * sink_mass / (bhvel2_df*sqrt(bhvel2_df));
        /* now apply this to the actual acceleration */
        if(fac_friction<0) fac_friction=0; if(isnan(fac_friction)) fac_friction=0;
        /* ok, here we have a special catch - the friction isn't standard dynamical friction, but rather we are already moving
            towards a potential mininum and want to be sure that we don't overshoot or retain large velocities that will
            launch us out, so we want the BH to 'relax' towards moving with the local flow */
        if(bhvel2_df > 0 && dt > 0)
        {
            double dv_magnitude=sqrt(bhvel2_df)*All.cf_atime, fac_vel=0, afac_vel=0; // physical velocity difference between 'target' and BH
            afac_vel = All.G * Mass_in_Kernel / pow(P[n].KernelRadius*All.cf_atime,2); // GMenc/r^2 estimate of local acceleration //
            afac_vel = DMIN(dv_magnitude*UNIT_TIME_IN_MYR , DMAX( DMIN(DMAX(-2.*P[n].Sink_PotentialMinimumOfNeighbors/(P[n].KernelRadius*All.cf_atime*All.cf_atime), 0), 10.*dv_magnitude/dt), afac_vel)); // free-fall-acceleration [checked-to-zero], limited to multiple of actual vel difference in timestep
            fac_vel = afac_vel * dt / dv_magnitude; // rate at which de-celeration/damping occurs
            if(fac_vel > 1.e-4) {fac_vel = 1.-exp(-fac_vel);}
            {auto dv_df = SinkTempInfo[i].DF_mean_vel * (All.cf_atime * fac_vel); P[n].dp += dv_df * P[n].Mass; P[n].Vel += dv_df;}
        }
    }
#endif // repositioning algorithm active
}




void set_sink_long_range_rp(int i, int n) /* pre-set quantities needed for long-range radiation pressure terms */
{
#if defined(SINK_CALC_LOCAL_ANGLEWEIGHTS) && !defined(SINK_FOLLOW_ACCRETED_ANGMOM)
    /* use the gradrho vector as a surrogate to hold the orientation of the angular momentum if we aren't evolving it explicitly
     (this is done because the long-range radiation routines for the BH require the angular momentum vector for non-isotropic emission) */
    P[n].GradRho = {0, 0, 1};
    if(SinkTempInfo[i].Mgas_in_Kernel > 0) {
        double fac = SinkTempInfo[i].Jgas_in_Kernel.norm_sq();
        if(fac>0) {double inv_norm = 1./sqrt(fac); for(int k=0;k<3;k++) {P[n].GradRho[k] = SinkTempInfo[i].Jgas_in_Kernel[k]*inv_norm;}}}
        /* now, the P[n].GradRho[k] field for the BH holds the orientation of the UNIT angular momentum vector
         NOTE it is important that HARD-WIRED into the code, this sink calculation comes after the density calculation
         but before the forcetree update and walk; otherwise, this won't be used correctly there */
#endif
    return;
}





/* determine if a sink is active to execute the neighbor loops above */
int sink_isactive(int i)
{
    if(P[i].Type != 5) {return 0;} // only sinks
    if(P[i].KernelRadius <= 0) {return 0;} // only H>0, i.e. found neighbors
    if(P[i].Mass <= 0) {return 0;} // only mass>0, i.e. not already marked for deletion
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
    if(!P[i].do_gas_search_this_timestep) {return 0;} // only do loops if we have actually done the density pass
#endif
    return 1; // otherwise yes, we are active
}





void sink_final_operations(void)
{
    int i, k, n, bin; double dt, mass_disk, mdot_disk, MgasBulge, MstarBulge, r0; k=0;
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    int count_sink_elim=0, tot_sink_elim;
#endif

#ifdef SINK_REPOSITION_ON_POTMIN
    for (int n : ActiveParticleList)
        if(sink_isactive(n))
            if(P[n].Sink_PotentialMinimumOfNeighbors < 0.5 * SINK_MINPOTVALUE_INIT)
            {
                double fac_sink_shift=0;
                dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(n);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
                dt = P[n].dt_since_last_gas_search;
#endif
                double dr_min = (P[n].Sink_PotentialMinimumOfNeighborsPos - P[n].Pos).norm_sq();
                if(dr_min > 0 && dt > 0)
                {
                    dr_min=sqrt(dr_min)*All.cf_atime; // offset to be covered
                    // in general don't let the shift be more than 0.5 of the distance in a single timestep, but let it move at reasonable ~few km/s speeds minimum, and cap at the free-fall velocity //
                    double dv_shift = sqrt(DMAX(-2.*P[n].Sink_PotentialMinimumOfNeighbors/All.cf_atime , 0)); // free-fall velocity, in [physical] code units, capped zero
                    dv_shift = DMAX(DMIN(dv_shift, dr_min/dt), 10./UNIT_VEL_IN_KMS); // set minimum at ~10 km/s, max at speed which 'jumps' full distance
                    fac_sink_shift = dv_shift * dt / dr_min; // dimensionless shift factor
                    if(fac_sink_shift > 1.e-4) {fac_sink_shift = 1.-exp(-fac_sink_shift);} // make sure we can't overshoot by using this smooth interpolation function
                }
                P[n].Pos += (P[n].Sink_PotentialMinimumOfNeighborsPos - P[n].Pos) * fac_sink_shift;
            }
#endif

    for(n = 0; n < TIMEBINS; n++) {if(TimeBinActive[n]) {TimeBin_Sink_mass[n] = 0; TimeBin_Sink_dynamicalmass[n] = 0; TimeBin_Sink_Mdot[n] = 0; TimeBin_Sink_Medd[n] = 0;}}

    for(i=0; i<N_active_loc_Sink; i++)
    {
        n = SinkTempInfo[i].index;
        int update_sink_moments = 0; // flag whether to go into the block below updating conserved quantities like mass, momentum, etc
        if(P[n].Mass > 0)
        {
            if((SinkTempInfo[i].accreted_Mass>0)||(SinkTempInfo[i].accreted_Sink_Mass>0)||(SinkTempInfo[i].accreted_Sink_Mass_reservoir>0)) { update_sink_moments = 1;}
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
            for(k=0; k<3; k++) {if(SinkTempInfo[i].accreted_J[k] != 0) {update_sink_moments = 1;}}
#endif
        }

        if(update_sink_moments)
        {
#ifdef HERMITE_INTEGRATION
            P[n].AccretedThisTimestep = 1;
#endif
            double m_new; m_new = P[n].Mass + SinkTempInfo[i].accreted_Mass;
#if (SINK_FOLLOW_ACCRETED_ANGMOM == 1) /* in this case we are only counting this if its coming from BH particles */
            m_new = P[n].Mass + SinkTempInfo[i].accreted_Sink_Mass + SinkTempInfo[i].accreted_Sink_Mass_reservoir;
#endif
#if defined(SINK_FOLLOW_ACCRETED_MOMENTUM) && !defined(SINK_REPOSITION_ON_POTMIN)
            P[n].dp += P[n].Vel*(m_new - P[n].Mass) + SinkTempInfo[i].accreted_momentum;
            P[n].Vel = (P[n].Vel*m_new + SinkTempInfo[i].accreted_momentum) / m_new;
#else
            P[n].dp += P[n].Vel*(m_new - P[n].Mass); /* track momentum change from mass gain even without SINK_FOLLOW_ACCRETED_MOMENTUM */
#endif
#if defined(SINK_FOLLOW_ACCRETED_COM) && !defined(SINK_REPOSITION_ON_POTMIN)
            P[n].Pos = (P[n].Pos*m_new + SinkTempInfo[i].accreted_centerofmass) / m_new;
#endif
#if defined(SINK_FOLLOW_ACCRETED_ANGMOM)
            P[n].Sink_Specific_AngMom = (P[n].Sink_Specific_AngMom*P[n].Mass + SinkTempInfo[i].accreted_J) / m_new;
            P[n].Sink_Specific_AngMom -= cross(SinkTempInfo[i].accreted_centerofmass, SinkTempInfo[i].accreted_momentum) / (m_new*m_new);
#endif
#if defined(SINK_RETURN_BFLUX)
            P[n].B += SinkTempInfo[i].accreted_B;
#endif
            P[n].Mass += SinkTempInfo[i].accreted_Mass;
#if defined(SINK_SWALLOWGAS) && !defined(SINK_GRAVCAPTURE_GAS)
            P[n].Sink_AccretionDeficit += SinkTempInfo[i].Sink_AccretionDeficit;
#endif
            P[n].Sink_Mass += SinkTempInfo[i].accreted_Sink_Mass;
#ifdef SINK_ALPHADISK_ACCRETION
            P[n].Sink_Mass_Reservoir += SinkTempInfo[i].accreted_Sink_Mass_reservoir;
#endif
#ifdef GRAIN_FLUID
            P[n].Sink_Dust_Mass += SinkTempInfo[i].accreted_dust_Mass;
#endif
#ifdef RT_REINJECT_ACCRETED_PHOTONS
	    P[n].Sink_accreted_photon_energy += SinkTempInfo[i].accreted_photon_energy;
#endif
        } // if(masses > 0) check
#ifdef HERMITE_INTEGRATION
        else { P[n].AccretedThisTimestep = 0; }
#endif
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
        if(All.ComovingIntegrationOn) {P[n].SinkRadius = DMIN(P[n].SinkRadius, SinkParticle_GravityKernelRadius);} // update sink radius if simulation has it dynamically evolving.
#endif

        /* Correct for the mass loss due to radiation and BAL winds */
        /* always substract the radiation energy from P[n].Sink_Mass && P[n].Mass */
        dt = GET_PARTICLE_FEEDBACK_TIMESTEP_IN_PHYSICAL(n);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
        dt = P[n].dt_since_last_gas_search;
#endif
        double dm = P[n].Sink_Mdot * dt;
        double radiation_loss = evaluate_sink_radiative_efficiency(P[n].Sink_Mdot,P[n].Sink_Mass,n) * dm;
        if(radiation_loss > DMIN(P[n].Mass,P[n].Sink_Mass)) radiation_loss = DMIN(P[n].Mass,P[n].Sink_Mass);
#ifdef SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
        if(All.SinkRadiativeEfficiency > 0 && All.SinkRadiativeEfficiency < 1 && P[n].ProtoStellarStage != 7) {radiation_loss = 0;} // negligible radiation loss term unless the object is actually a compact relic
#endif
        P[n].dp -= P[n].Vel * radiation_loss; P[n].Mass -= radiation_loss; P[n].Sink_Mass -= radiation_loss;


#if defined(SINGLE_STAR_TIMESTEPPING)
        /* save local effective signal velocity of gas for sink particle CFL-like timestep criterion */
        P[n].Sink_SurroundingGasVel = SinkTempInfo[i].Sink_SurroundingGasVel.norm_sq();
        P[n].Sink_SurroundingGasVel += convert_internalenergy_soundspeed2(n,SinkTempInfo[i].Sink_SurroudingGasInternalEnergy);
        P[n].Sink_SurroundingGasVel = sqrt(P[n].Sink_SurroundingGasVel);
#endif

#ifdef SINK_WIND_SPAWN
        /* DAA: for wind spawning, we only need to subtract the BAL wind mass from Sink_Mass (or Sink_Mass_Reservoir) --> wind mass subtracted from P.Mass in sink_spawn_particle_wind_shell()  */
        double dm_wind = (1.-All.Sink_accreted_fraction) / All.Sink_accreted_fraction * dm; /* this is the 'expected' wind mass, will go into reservoir designated for spawning in the next timestep */
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        dm_wind = DMAX(P[n].Sink_Mdot_ROI - P[n].Sink_Mdot, 0.) * dt; /* wind mass loss rate from the alpha disk */
#endif
#ifdef SINGLE_STAR_FB_JETS
        if((P[n].Sink_Mass * UNIT_MASS_IN_SOLAR < 0.01) || P[n].Mass < 3.5*P[n].Sink_Formation_Mass) {dm_wind = 0;} // no jets launched yet if <0.01 msun or if we haven't accreted enough to get a reliable jet direction
#endif
        if(dm_wind > P[n].Mass) {dm_wind = P[n].Mass;}
#if defined(SINK_ALPHADISK_ACCRETION)
        dm = DMIN(dm, P[n].Sink_Mass_Reservoir);
        dm_wind = DMIN(dm_wind, P[n].Sink_Mass_Reservoir - dm);
        P[n].Sink_Mass_Reservoir -= dm_wind;
#else
        if(dm_wind > P[n].Sink_Mass) {dm_wind = P[n].Sink_Mass;}
        P[n].Sink_Mass -= dm_wind;
#endif
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
#if defined(SINGLE_STAR_FB_WINDS)
       if(P[n].ProtoStellarStage == 5) {
           if(P[n].wind_mode == 1) {
                dm_wind = single_star_wind_mdot(n,0) * dt;
                P[n].Sink_Mass -= dm_wind;
            }
        } // wind loss rate previously calculated in stellar_evolution at the end of the previous timestep: remove mass lost via winds
#endif
#if defined(SINGLE_STAR_FB_SNE)
        if(P[n].ProtoStellarStage == 6) { // Star old enough to go out with a boom
            double eps = DMIN(KERNEL_CORE_SIZE*ForceSoftening_KernelRadius(n), P[n].KernelRadius);
#ifdef SINK_GRAVCAPTURE_FIXEDSINKRADIUS
            eps = DMAX(eps, P[n].SinkRadius);
#endif
            double t_clear = eps/single_star_SN_velocity(n); // time needed spawned wind particles to clear the sink
            double m_ejecta_unit = SINGLE_STAR_FB_SNE_N_EJECTA * P[n].Sink_Formation_Mass; // we spawn SINGLE_STAR_FB_SNE_N_EJECTA per ejected shell, each with a cell mass set by the value here
            double SN_mdot = m_ejecta_unit / t_clear; // and we can have maximum 1 shell per t_clear
            dm_wind = DMIN(SN_mdot*dt, P[n].Sink_Mass); // We will spawn particles to model the SN ejecta, but not more than what we can handle at the same time, these particles will have the same mass as gas particles, not like wind particles
            printf("Adding SN ejecta of mass %g from star %llu at time %g, unspawned mass at %g\n", dm_wind, (unsigned long long) P[n].ID, All.Time, (P[n].unspawned_wind_mass+dm_wind));
            P[n].Sink_Mass -= dm_wind; // remove amount of mass lost via winds
            double m_relic = single_star_relic_SN_mass(n); // get the relic mass that should be left behind after the SNe
            if(P[n].Sink_Mass < m_relic + 0.5*m_ejecta_unit) { // less than half a shell mass left in the star: instead of spawning the last shell with very low mass particles we will make the one before slightly more massive
                dm_wind += P[n].Sink_Mass; // add leftover mass to be spawned and zero-out mass
                P[n].Sink_Mass = DMAX(0, m_relic); // zero out mass if nothing is left-over
            }
            if(P[n].Sink_Mass <= m_relic) {
                TreeMomentsStaleFlag = 1; /* sink mass changed: refresh moments, tree structure maintained by MAINTAIN_TREE_IN_REARRANGE */
                Max_Unspawned_MassUnits_fromSink = DMAX(2, Max_Unspawned_MassUnits_fromSink); // a high enough number to ensure that we do spawn winds
            }
        }
#endif
#endif
        P[n].unspawned_wind_mass += dm_wind;
        double n_unspawned = P[n].unspawned_wind_mass / ((SINK_WIND_SPAWN)*target_mass_for_wind_spawning(n)); // number of spawned gas cells that can be made from the mass in the reservoir
        if(n_unspawned> Max_Unspawned_MassUnits_fromSink) {Max_Unspawned_MassUnits_fromSink = n_unspawned;} // track the maximum integer number of elements this sink could spawn
#endif

#ifdef RT_SINK_ANGLEWEIGHT_PHOTON_INJECTION
        P[n].KernelSum_Around_RT_Source = SinkTempInfo[i].Sink_angle_weighted_kernel_sum;
#endif            

        /* dump the results to the 'sink_details' files */
        mass_disk=0; mdot_disk=0; MgasBulge=0; MstarBulge=0; r0 = P[n].KernelRadius * All.cf_atime;
#ifdef SINK_ALPHADISK_ACCRETION
        mass_disk = P[n].Sink_Mass_Reservoir;
        mdot_disk = SinkTempInfo[i].mdot_reservoir;
#endif
#if (defined(SINK_GRAVACCRETION) && (SINK_GRAVACCRETION == 0))
        MgasBulge = SinkTempInfo[i].MgasBulge_in_Kernel;
        MstarBulge = SinkTempInfo[i].MstarBulge_in_Kernel;
#endif

#if defined(SINK_OUTPUT_MOREINFO)
#ifdef SINGLE_STAR_STARFORGE_DEFAULTS
        fprintf(FdSinksDetails, "time=%.16g ID=%llu mtot=%g msink=%g mdisk=%g sink_mdot=%g mdot_disk=%g  dtime=%g ngb_density=%g ngb_internalenergy=%g mgas_kernel=%g mstar_kernel=%g r_kernel=%g pos=%2.16g %2.16g %2.16g vel=%2.16g %2.16g %2.16g angmom_kernel=%g %g %g  sink_angmom=%g %g %g\n",
                All.Time, (unsigned long long)P[n].ID,  P[n].Mass, P[n].Sink_Mass, mass_disk, P[n].Sink_Mdot, mdot_disk, dt, P[n].DensityAroundParticle*All.cf_a3inv, SinkTempInfo[i].Sink_SurroudingGasInternalEnergy,
                SinkTempInfo[i].Mgas_in_Kernel, SinkTempInfo[i].Mstar_in_Kernel, r0, P[n].Pos[0], P[n].Pos[1], P[n].Pos[2],  P[n].Vel[0], P[n].Vel[1], P[n].Vel[2],
                SinkTempInfo[i].Jgas_in_Kernel[0], SinkTempInfo[i].Jgas_in_Kernel[1], SinkTempInfo[i].Jgas_in_Kernel[2], P[n].Sink_Specific_AngMom[0]*P[n].Mass, P[n].Sink_Specific_AngMom[1]*P[n].Mass, P[n].Sink_Specific_AngMom[2]*P[n].Mass ); fflush(FdSinksDetails);
#else
        fprintf(FdSinksDetails, "time=%.16g ID=%llu  mtot=%g msink=%g mdisk=%g sink_mdot=%g mdot_disk=%g dtime=%g ngb_density=%g ngb_internalenergy=%g sfr_kernel=%g mgas_kernel=%g mstar_kernel=%g mgasbulge=%g mstarbulge=%g r_kernel=%g pos=%2.16g %2.16g %2.16g vel=%2.16g %2.16g %2.16g angmomgas_kernel=%g %g %g angmomstar_kernel=%g %g %g\n",
                All.Time, (unsigned long long)P[n].ID,  P[n].Mass, P[n].Sink_Mass, mass_disk, P[n].Sink_Mdot, mdot_disk, dt, P[n].DensityAroundParticle*All.cf_a3inv, SinkTempInfo[i].Sink_SurroudingGasInternalEnergy, SinkTempInfo[i].Sfr_in_Kernel,
                SinkTempInfo[i].Mgas_in_Kernel, SinkTempInfo[i].Mstar_in_Kernel, MgasBulge, MstarBulge, r0, P[n].Pos[0], P[n].Pos[1], P[n].Pos[2],  P[n].Vel[0], P[n].Vel[1], P[n].Vel[2],
                SinkTempInfo[i].Jgas_in_Kernel[0], SinkTempInfo[i].Jgas_in_Kernel[1], SinkTempInfo[i].Jgas_in_Kernel[2], SinkTempInfo[i].Jstar_in_Kernel[0], SinkTempInfo[i].Jstar_in_Kernel[1], SinkTempInfo[i].Jstar_in_Kernel[2] ); fflush(FdSinksDetails);
#endif
#else

#ifdef OUTPUT_ADDITIONAL_RUNINFO
        fprintf(FdSinksDetails, "Timestep Summary: BH=%llu time=%.16g msink=%g mdisk=%g mtot=%g sink_mdot=%g mdot_disk=%g ngb_density=%g ngb_internalenergy=%g  pos=%2.16g %2.16g %2.16g\n", (unsigned long long)P[n].ID, All.Time, P[n].Sink_Mass, mass_disk, P[n].Mass, P[n].Sink_Mdot, mdot_disk,
                P[n].DensityAroundParticle*All.cf_a3inv, SinkTempInfo[i].Sink_SurroudingGasInternalEnergy, P[n].Pos[0], P[n].Pos[1], P[n].Pos[2]); fflush(FdSinksDetails);
#endif
#endif

        bin = P[n].TimeBin;
        TimeBin_Sink_mass[bin] += P[n].Sink_Mass;
        TimeBin_Sink_dynamicalmass[bin] += P[n].Mass;
        TimeBin_Sink_Mdot[bin] += P[n].Sink_Mdot;
        if(P[n].Sink_Mass > 0) {TimeBin_Sink_Medd[bin] += P[n].Sink_Mdot / P[n].Sink_Mass;}

/* check promotion and clipping flags */
#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
        singlestar_subgrid_protostellar_evolution_update_track(n, dm, dt);
        if((P[n].Type!=5) || (P[n].Mass<=0) || !isfinite(P[n].Mass)) {count_sink_elim++;} // our subroutine has promoted or removed this sink: one fewer BH-type particle exists now //
#endif
        
    } // for(i=0; i<N_active_loc_Sink; i++)

#if defined(SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION)
    MPI_Allreduce(&count_sink_elim, &tot_sink_elim, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    All.TotSinks -= tot_sink_elim;
#endif

}


#endif // SINK_PARTICLES
