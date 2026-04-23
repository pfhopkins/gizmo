#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"
/* Function bodies now in cosmic_ray_functions.h (single source of truth).
   Define KOKKOS_INLINE_FUNCTION as empty so functions are non-inline here,
   providing externally-visible symbols for other TUs that link via proto.h. */
#undef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION
#include "cosmic_ray_functions.h"

/*! Routines for cosmic ray 'fluid' modules (as opposed to the explicit CR-PIC methods, which are in the grain+particles section of the code)
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef COSMIC_RAY_FLUID

/* cosmicrayfluid_rsol_corrfac: definition now in cosmic_ray_functions.h */

/* Get_Gas_CosmicRayPressure: definition now in cosmic_ray_functions.h */


#if defined(CRFLUID_EVOLVE_SPECTRUM)
/* routine which defines the actual bin list for the multi-bin spectral CR models. note the number of entries MUST match the hard-coded N_CR_PARTICLE_BINS defined in allvars.h */
void CR_spectrum_define_bins(void)
{
#if (CRFLUID_EVOLVE_SPECTRUM == 2)
    int species_list[N_CR_PARTICLE_SPECIES]={-2, -1, +1, 2,   6, 4, 5,  7}; // {-2=positrons, -1=electrons, 1=protons, 2=B, 3=C, 4=Be7+9, 5=Be10, 6=CNO, 7=antiprotons}
    double charge_v[N_CR_PARTICLE_SPECIES] ={ 1, -1,  1, 5, 7.4, 4, 4, -1}; // list of charge for each of the species above, matched to their codes
#else
    int species_list[N_CR_PARTICLE_SPECIES]={-1, 1};
    double charge_v[N_CR_PARTICLE_SPECIES] ={-1, 1};
#endif
    
#define CR_NUMBER_OF_R_BINS_FOR_LEPTONIC_SPECIES 11 /* needs to be defined to match below, both hard-coded here */
#define CR_NUMBER_OF_R_BINS_FOR_HADRONIC_SPECIES 8  /* needs to be defined to match below, both hard-coded here; note, we by default don't go to extremely low-energy proton bins since those have incredibly rapid Coulomb loss times, which without continuous injection grind the code down and just give zero energy in the bins */
    int k; double R[N_CR_PARTICLE_BINS], Z[N_CR_PARTICLE_BINS]; int spec[N_CR_PARTICLE_BINS], ispec, n0=0;
    double R_lepton[CR_NUMBER_OF_R_BINS_FOR_LEPTONIC_SPECIES]={3.16227766e-03, 1.00000000e-02, 3.16227766e-02, 1.00000000e-01, 3.16227766e-01, 1.00000000e+00, 3.16227766e+00, 1.00000000e+01, 3.16227766e+01, 1.00000000e+02, 3.16227766e+02};
    double R_nuclei[CR_NUMBER_OF_R_BINS_FOR_HADRONIC_SPECIES]={1.00000000e-01, 3.16227766e-01, 1.00000000e+00, 3.16227766e+00, 1.00000000e+01, 3.16227766e+01, 1.00000000e+02, 3.16227766e+02};
    for(ispec=0;ispec<N_CR_PARTICLE_SPECIES;ispec++)
    {
        int is_lepton=0, nmax=CR_NUMBER_OF_R_BINS_FOR_HADRONIC_SPECIES; if(species_list[ispec] < 0) {is_lepton=1; nmax=CR_NUMBER_OF_R_BINS_FOR_LEPTONIC_SPECIES;}
        for(k=0;k<nmax;k++) {Z[n0]=charge_v[ispec]; spec[n0]=species_list[ispec]; if(is_lepton) {R[n0]=R_lepton[k];} else {R[n0]=R_nuclei[k];} n0++;}
    }
    
    /* for ease-of-use purposes rather than adding a bunch of extra flags, we will use spec = -2 to signify positrons. the code will understand what to do with them and treat them with the correct charge, but its just a special
        flag. otherwise the charge should correspond to -1 for electrons, or to the fully-ionized charge of a given nucleus (e.g. its atomic number, 1=p/H, 2=He, etc. [some can have hard-coded behaviors/cross-sections below for various processes designed for their use as tracers, including e.g. 4=Be, 5=B, 6/7/8=C/N/O, etc.)
     can use some slightly-offset values if desired to hard-code special flags, or code a shared weight here [e.g. weight in amu] to represent different isotopes, if desired (e.g. 10Be vs 7Be) */
    int ibin=0, ilast=-200; for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(ibin>=N_CR_PARTICLE_SPECIES) {continue;} else {if(spec[k] != ilast) {All.CR_species_ID_active_list[ibin]=spec[k]; ilast=spec[k]; ibin++;}}} /* creates list of all species here that are active */
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {All.CR_global_rigidity_at_bin_center[k]=R[k]; All.CR_global_charge_in_bin[k]=Z[k]; All.CR_species_ID_in_bin[k]=spec[k];}

#if 1 // (CRFLUID_EVOLVE_SPECTRUM == 2) // now even the simpler network has secondary e-, important for dense regions synchrotron
    /* note that some of our assumptions here and below are hard-coded to the fact below that the species are -ordered- in the order given by 'species list' at the top */
    int temp_species_map[100]; int min_species_id=99999, id_last=-200, n00=0, j; for(k=0;k<N_CR_PARTICLE_SPECIES;k++) {if(species_list[k]<min_species_id) {min_species_id=species_list[k];}}
    int id_map_offset=0; if(min_species_id<0) {id_map_offset=-min_species_id;}
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(All.CR_species_ID_in_bin[k] != id_last) {id_last=All.CR_species_ID_in_bin[k]; temp_species_map[id_last+id_map_offset]=n00; n00++;}}
    /* now species_list[temp_species_map[species_id+id_map_offset]] = species_id, useful as a lookup below */
    for(k=0;k<N_CR_PARTICLE_SPECIES;k++)
    {
        //int species_list={-2, -1, +1, 2, 3, 4, 5}; // {positrons, electrons, protons, B, C, Be7+9, Be10}
        int primary_spec = species_list[k]; /* primary species */
        int secondary_spec[N_CR_PARTICLE_SPECIES]; /* secondary species for this primary -- default to none (-200 key here) */
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {secondary_spec[j]=-200;}
        //if(primary_spec == -2) {secondary_spec[0]=-200;} // positrons -> gamma rays [un-tracked]
        //if(primary_spec == -1) {secondary_spec[0]=-200;} // electrons -> ? [un-tracked]
        if(primary_spec == 1) {secondary_spec[0]=-1; secondary_spec[1]=-2; if(N_CR_PARTICLE_SPECIES>2) {secondary_spec[2]=7;}} // protons -> secondary e- and e+ and anti-p
        if(primary_spec == 2) {secondary_spec[0]=4; secondary_spec[1]=5;} // B -> secondary p and e, Be [un-tracked for now, b/c small contributions]
        if(primary_spec == 3 || primary_spec == 6) {secondary_spec[0]=2; secondary_spec[1]=4; if(N_CR_PARTICLE_SPECIES>2) {secondary_spec[2]=5;}} // C or CNO -> secondary B, Be-7+9, and Be-10 [also e, p, but untracked for now b/c small contributions]
        //if(primary_spec == 4) {secondary_spec[0]=-200;} // Be7+9 -> secondary p and e, [un-tracked for now, b/c small contributions]
        if(primary_spec == 5) {secondary_spec[0]=4;} // Be10 -> B10 (radioactive decay, not from fragmentation: separate vector?)
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {if(secondary_spec[j] > -100) {All.CR_secondary_species_listref[k][j] = temp_species_map[secondary_spec[j]+id_map_offset];} else {All.CR_secondary_species_listref[k][j]=-200;}}
    }
    /* also need to figure out the 'destination bin' for each type of secondary -- we will assume nucleon-nucleon conserves energy per nucleon, while cascades to positrons and secondary electrons are treated slightly differently */
    double E_GeV[N_CR_PARTICLE_BINS], A_wt[N_CR_PARTICLE_BINS]; for(k=0;k<N_CR_PARTICLE_BINS;k++) {E_GeV[k]=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k); A_wt[k]=return_CRbin_CRmass_in_mp(-1,k);}
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        int primary_id = All.CR_species_ID_in_bin[k];
        int primary_listref = temp_species_map[primary_id+id_map_offset];
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++)
        {
            int secondary_listref = All.CR_secondary_species_listref[primary_listref][j];
            if(secondary_listref <= -1) {All.CR_secondary_target_bin[k][j]=-2;}
            else {
                int secondary_id = species_list[secondary_listref];
                int m, target_bin=-1; double diff_min=MAX_REAL_NUMBER, E_target_0=E_GeV[k];
                for(m=0;m<N_CR_PARTICLE_BINS;m++)
                {
                    if(All.CR_species_ID_in_bin[m] != secondary_id) {continue;}
                    double E_target = E_GeV[k] * DMAX(1.,A_wt[m]) / DMAX(1.,A_wt[k]); // fixed energy per nucleon/particle (treating e-/e+ as 1)
                    if(secondary_id < 0) {E_target *= 0.14;} // secondary e+/e- from protons (pion decay) get ~1/8 original p energy, likewise for anti-protons
                    if(secondary_id == 7) {E_target *= 0.08;} // anti-protons similar but get slightly-less energy on average [testing effects right now of this level of hair-splitting!]
                    double diff = log(E_GeV[m]/E_target); diff*=diff; // square of log-diff between energies
                    if(diff < diff_min) {diff_min=diff; target_bin=m; E_target_0=E_target;} // set to this as the 'closest' option
                }
                All.CR_secondary_target_bin[k][j]=-1; // default to no secondary bin (secondary is 'lost')
                if(target_bin >= 0) // check if the bin is valid
                {
                    double E_target=E_target_0, E_bin=E_GeV[target_bin], E_bin_m=E_bin, E_bin_p=E_bin; // if decay to lower energy than we track bins, the products are gone from the spectrum we follow
                    if(target_bin>0) {if(All.CR_species_ID_in_bin[target_bin-1] == secondary_id) {E_bin_m=E_GeV[target_bin-1];}} // define the bin ranges that we will consider for whether the target energy fits
                    if(target_bin<N_CR_PARTICLE_BINS-1) {if(All.CR_species_ID_in_bin[target_bin+1] == secondary_id) {E_bin_p=E_GeV[target_bin+1];}}
                    E_bin_m=sqrt(E_bin_m*E_bin); E_bin_p=sqrt(E_bin_p*E_bin); if(E_bin_m==E_bin) {E_bin_m=E_bin*(E_bin/E_bin_p)*(E_bin/E_bin_p);} else if(E_bin_p==E_bin) {E_bin_p=E_bin*(E_bin/E_bin_m)*(E_bin/E_bin_m);}
                    if(E_target >= 0.9*E_bin_m && E_target <= 1.1*E_bin_p) {All.CR_secondary_target_bin[k][j] = target_bin;} // assign a destination bin
                }
            }
        }
    }
    /* now pre-calculate the fragmentation factors and radioactive factors, all static up to their nH dependence */
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        All.CR_frag_coeff[k]=0; All.CR_rad_decay_coeff[k]=0; double beta_fac=return_CRbin_beta_factor(-1,k), cx_mb_to_coeff=3.0e-17*beta_fac; // cx_mb_to_coeff = (millibarn x c_light) * beta in cgs, to convert units to get \dot[f] = sigma*v*n*f, so prefactor is this times n in cgs
        if(All.CR_species_ID_in_bin[k] == -2)
        {
            double gamma_fac=return_CRbin_gamma_factor(-1,k), gamma_positron=gamma_fac, gamma_minus_1=gamma_positron-1., fac=0; // Dirac expression below considers e- at rest, which is what we're interested in here since it's e+ CRs annihilating (gamma is the gamma of the positron)
            if(gamma_minus_1 > 1.e-2) {fac=((gamma_positron*gamma_positron+4.*gamma_positron+1.)*log(gamma_positron+sqrt(gamma_positron*gamma_positron-1.))/(gamma_positron*gamma_positron-1.) - (gamma_positron+3.)/sqrt(gamma_positron*gamma_positron-1.))/(gamma_positron+1.);} else {fac=1./sqrt(2.*DMAX(gamma_minus_1,1.e-8));} // Dirac expression
            All.CR_frag_coeff[k] = 7.479e-15 * beta_fac * fac; // e+ annihilation with ISM (rest) e- to gamma rays
        }
        if(All.CR_species_ID_in_bin[k] == 1) {if(E_GeV[k] > 0.28) {All.CR_frag_coeff[k] = 6.37e-16;}} // coefficient for hadronic/catastrophic interactions/pionic losses: dEtot/dt = -(coeff) *nnucleoncgs* Etot, or dPtot/dt = -(coeff)*nnucleoncgs* Ptot (since all p effected are in rel limit, and works by deleting N not by lowering individual E. Mannheim & Schlickeiser 1994
        if(All.CR_species_ID_in_bin[k] > 1) /* total fragmentation cross-section/rate, from Mannheim & Schlickeiser 1994: */
        {
            double sigma_frag_tot = 45. * pow(A_wt[k],0.7) * (1.+0.016*sin(1.3-2.63*log(A_wt[k])));
            if(E_GeV[k] < 2.0) {sigma_frag_tot *= (1.-0.62*exp(-E_GeV[k]/0.2)*sin(1.57553/pow(E_GeV[k],0.28)));}
            All.CR_frag_coeff[k] = cx_mb_to_coeff * sigma_frag_tot; // rate depends on 'v', need to include beta here
            if(All.CR_species_ID_in_bin[k] == 7) {double RGV=All.CR_global_rigidity_at_bin_center[k], lnR=log(RGV); All.CR_frag_coeff[k] = cx_mb_to_coeff * 1.5 * (-107.9 + 29.43*lnR - 1.655*lnR*lnR + 189.9/pow(RGV,1./3.));} // larger CX b/c this is anti-proton annihilation; factor ~1.5 accounts for sum of species heavier than H; from summation of cross-sections in Evoli et al. 2017 [arXiv:1711.09616]
        }
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {All.CR_frag_secondary_coeff[k][j]=0;} // initialize this to null
        if(All.CR_frag_coeff[k] > 0) {for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {if(All.CR_secondary_target_bin[k][j] >= 0) // desired secondary products exist
            {
                double x=DMAX(-2.,DMIN(2.,log10(E_GeV[k]))); // used in some of the fitting functions below
                int primary_id = All.CR_species_ID_in_bin[k], secondary_id = All.CR_species_ID_in_bin[All.CR_secondary_target_bin[k][j]];
                //if((primary_id == 5) && (secondary_id == 4)) {} // Be10->B10 (radioactive - frag probability is null) //
                if(primary_id==1) { // p; assume some simple branching ratios for pion production in the relevant regime above
                    if(secondary_id==-1) {All.CR_frag_secondary_coeff[k][j] = (1./3.) * All.CR_frag_coeff[k];} // p->e-
                    if(secondary_id==-2) {All.CR_frag_secondary_coeff[k][j] = (1./3.) * All.CR_frag_coeff[k];} // p->e+
                    if(secondary_id== 7) {if(E_GeV[k]>=2.) {double sqrt_s=1.87654*sqrt(1.+E_GeV[k]/1.87654); All.CR_frag_secondary_coeff[k][j]=cx_mb_to_coeff * 1.4*pow(sqrt_s,0.6)*exp(-pow(17./sqrt_s,1.4));}} // p->pbar [anti-proton production; multiplied by 2x here to include anti-neutrons that decay rapidly to anti-p]. fit to integrated-over-pT results from Reinert & Winkler 2017, arXiv:1712.00002
                }
                if(primary_id==2) { // B; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==4) {All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * 12.0 * pow(E_GeV[k],-0.022);} // B->Be9 (declines weakly from ~14 to ~10 over MeV-TeV, approx here)
                    if(secondary_id==5) {All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * 12.5 * pow(E_GeV[k], 0.018);} // B->Be10 (increases weakly from 11 to 14 over MeV-TeV, approx here)
                }
                if(primary_id==3) { // C; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==2) { // C->B
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10.,1.88490356 -0.05648715*x -0.13108485*x*x +0.11340508*x*x*x +0.08119785*x*x*x*x -0.06574148*x*x*x*x*x -0.01159932*x*x*x*x*x*x +0.00962035*x*x*x*x*x*x*x +0.23395042856711876*exp(-10.8066678706701*(x+1.24740008)*(x+1.24740008)));
                    }
                    if(secondary_id==4) { // C->Be7+9
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10.,1.18321683 +0.11629153*x +0.01653084*x*x -0.11321897*x*x*x -0.03375688*x*x*x*x +0.05771537*x*x*x*x*x +0.00684962*x*x*x*x*x*x -0.00876353*x*x*x*x*x*x*x +0.40590023164209293*exp(-17.1253106513537*(x+1.28525319)*(x+1.28525319)));
                    }
                    if(secondary_id==5) { // C->Be10
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * (0.10757855 + pow(10., 0.53413363 +0.38478345*x -0.51584177*x*x -0.22611016*x*x*x +0.51009595*x*x*x*x +0.04492848*x*x*x*x*x -0.23828966*x*x*x*x*x*x +0.06889871*x*x*x*x*x*x*x));
                    }
                }
                if(primary_id==6) { // CNO effective bin; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==2) { // CNO->B
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10., 1.71801936 - 0.03475011*x -0.09856187*x*x +0.12369455*x*x*x +0.02958446*x*x*x*x -0.05273341*x*x*x*x*x -0.00223893*x*x*x*x*x*x +0.00639451*x*x*x*x*x*x*x + 0.464605776*exp(-17.769530*(x+1.23499649)*(x+1.23499649)) );}
                    if(secondary_id==4) { // CNO->Be7+9 // (could also use CNO->Be9 (not really a point here explicitly tracking Be7), but including all here since closer to what is actually observed)
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10., 1.16648147  +0.09557578*x -0.17970136*x*x +0.06823829*x*x*x +0.04448299*x*x*x*x -0.02883429*x*x*x*x*x -0.00274047*x*x*x*x*x*x +0.00286316*x*x*x*x*x*x*x + 0.476812578*exp(-14.203615*(x+1.21352966)*(x+1.21352966)) );}
                    if(secondary_id==5) { // CNO->Be10
                        All.CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * (0.073388 + pow(10., 0.454778746 + 0.349074384*x -0.684152925*x*x -0.153016497*x*x*x +0.657169204*x*x*x*x +8.06147155e-04*x*x*x*x*x -0.296932460*x*x*x*x*x*x +0.0918014184*x*x*x*x*x*x*x ));}
                }
            }}}
        
        double r_decay = 0; // default to assume no radioactive decay
        if(All.CR_species_ID_in_bin[k] == 5) {r_decay = 1.455e-14;} // Be10 -> B10
        if(r_decay > 0) {All.CR_rad_decay_coeff[k] = r_decay / return_CRbin_gamma_factor(-1,k);} // need to account for the fact that relativistic time dilation extends the lifetimes of highly relativistic sources
    }
#endif
}
#endif


/* CR_energy_spectrum_injection_fraction: body moved to
 * eos/cosmic_ray_fluid/cosmic_ray_functions.h as KOKKOS_INLINE_FUNCTION
 * so the GPU mechanical_fb kernel (B8 Phase 2) can call it on device. */


/* routine which gives diffusion coefficient as a function of energy for the 'constant diffusion coefficient' models:
    current default: -extremely- simple power-law, assuming diffusion coefficient increases with CR energy per unit charge as (E/Z)^(1/2) */
double diffusion_coefficient_constant(int target, int k_CRegy, struct gas_cell_data *cell)
{
    double dimensionless_kappa_relative_to_GV_protons = 1;
#if (N_CR_PARTICLE_BINS > 1)    /* insert physics here */
    int target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
    dimensionless_kappa_relative_to_GV_protons = return_CRbin_beta_factor(target_bin_centering_for_CR_quantities,k_CRegy) * pow( All.CR_global_min_rigidity_in_bin[k_CRegy]*All.CR_global_max_rigidity_in_bin[k_CRegy] , 0.5 * 0.6 ); // assume a quasi-empirical scaling here, and for these correction terms its important that the 'bin center' being used for the zero point here is the geometric mean of the bin edges, hence the 0.5 term b/c geometric mean is sqrt[min*max] //
#endif
    return All.CosmicRayDiffusionCoeff * dimensionless_kappa_relative_to_GV_protons;
}


                                                                                                                              
/* routine which gives diffusion coefficient as a function of CR bin for the self-confinement models [in local equilibrium]. mode sets what we assume about the 'sub-grid'
    parameters f_QLT (rescales quasi-linear theory) or f_cas (rescales turbulence strength)
      <=0: fQLT=1 [most naive quasi-linear theory, ruled out by observations],  fcas=1 [standard Goldreich-Shridar cascade]
        1: fQLT=100, fcas=1
        2: fQLT=1, fcas=100
        3: fQLT=1, fcas-K41 from Hopkins et al. 2020 paper, for pure-Kolmogorov isotropic spectrum
        4: fQLT=1, fcas-IK, IK spectrum instead of GS
   if set mode < 0, will also ignore the dust-damping contribution from Squire et al. 2020.
   coefficient is returned in cgs units
 */
#ifndef CRFLUID_SET_SC_MODEL
#define CRFLUID_SET_SC_MODEL 1 /* set which mode to return from the SC subroutine here, of the various choices for how to e.g. model fCas, fQLT */
#endif
double diffusion_coefficient_self_confinement(int mode, int target, int k_CRegy, double M_A, double L_scale, double b_muG,
    double vA_noion, double rho_cgs, double temperature, double cs_thermal, double nh0, double nHe0, double f_ion, struct particle_data *pp, struct gas_cell_data *cell)
{
    double vol_inv = cell[target].Density*All.cf_a3inv / pp[target].Mass, fturb_multiplier=1, f_QLT=1, R_CR_GV, Z_charge_CR, M_cr_mp; Vec3<double> b0={}, p0={};
    int target_bin_centering_for_CR_quantities = target; // if this = target, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
    target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
    R_CR_GV=return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy); Z_charge_CR=return_CRbin_CR_charge_in_e(target,k_CRegy); M_cr_mp=return_CRbin_CRmass_in_mp(target,k_CRegy);
    int k; double n_cgs=rho_cgs/PROTONMASS_CGS, EPSILON_SMALL=1.e-50, e_CR=0, e_B=0, bhat_dot_CR_Pgrad=0, B2=0;
#ifdef MAGNETIC
    b0=cell[target].BPred*(vol_inv*All.cf_a2inv);
    B2=b0.norm_sq(); e_B=0.5*B2; B2=1./sqrt(B2+MIN_REAL_NUMBER); b0*=B2; // calculate B-field energy and bhat vector
#else
    b0=cell[target].Gradients.CosmicRayPressure[k_CRegy]; // just assume equilibrium here, convert to physical units, pick arbitrary direction here //
    B2=b0.norm_sq(); e_B=cell[target].Pressure*All.cf_a3inv; b0/=sqrt(B2+MIN_REAL_NUMBER); // calculate B-field energy and bhat vector
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
    double R0_m=All.CR_global_min_rigidity_in_bin[k_CRegy], R0_p=All.CR_global_max_rigidity_in_bin[k_CRegy];  // e_CR is in bin, but doesn't take account of bin width; needed only for NLL, want to include CRs 'close' to energy but not all b/c non-resonant, but don't want bin-size dependent, so replace with ~constant * de_cr / dlnR, which works pretty well
    double fac_dXdlnR = 1.0 / log(R0_p/R0_m); // factor to multiply by to account for 'width' of gyro-resonance
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {
        double R0_k=All.CR_global_rigidity_at_bin_center[k];
        if(R0_k>R0_m && R0_k<R0_p) {
            e_CR += cell[target].CosmicRayEnergyPred[k] * fac_dXdlnR * vol_inv; // convert to energy density units
            p0 += cell[target].Gradients.CosmicRayPressure[k] * (fac_dXdlnR * All.cf_a3inv/All.cf_atime); // convert to appropriate physical units
        }} // sum over all bins with their rigidity in the same range, so that we can get a total energy (dominated by p, but should include all for safety)
#else
    e_CR=cell[target].CosmicRayEnergyPred[k_CRegy]*vol_inv; p0=cell[target].Gradients.CosmicRayPressure[k_CRegy]*(All.cf_a3inv/All.cf_atime);
#endif
    bhat_dot_CR_Pgrad = dot(b0, p0); // dot product of bhat and CR pressure gradient, summed over relevant bins
    double beta=return_CRbin_beta_factor(target_bin_centering_for_CR_quantities,k_CRegy), Omega_gyro=beta*(0.00898734*b_muG/R_CR_GV) * UNIT_TIME_IN_CGS, r_L=beta*C_LIGHT_CODE/Omega_gyro, kappa_0=r_L*beta*C_LIGHT_CODE; /* all in physical -code- units */
    double x_LL = DMAX( r_L / L_scale, EPSILON_SMALL ), vA_code=Get_Gas_ion_Alfven_speed_i(target, pp, cell), k_turb=1./L_scale, k_L=1./r_L;

    if(mode==1) {f_QLT = 100;} // multiplier to account for arbitrary deviation from QLT, applies to all damping mechanisms [100 = favored value in our study; or could use fcas = 100]
    fturb_multiplier = pow(M_A,3./2.); // multiplier to account for different turbulent cascade models (fcas = 1)
    if(mode==2) {fturb_multiplier *= 100.;} // arbitrary multiplier (fcas = 100, here)
    if(mode==3) {fturb_multiplier = pow(M_A,3./2.) * 1./(pow(M_A,1./2.)*pow(x_LL,1./6.));} // pure-Kolmogorov (fcas-K41)
    if(mode==4) {fturb_multiplier = pow(M_A,3./2.) / pow(x_LL,1./10.);} // GS anisotropic but perp cascade is IK (fcas-IK) /

    /* ok now we finally have all the terms needed to calculate the various damping rates that determine the equilibrium diffusivity */
    double U0bar_grain=3., rhograin_int_cgs=1., fac_grain=R_CR_GV*sqrt(n_cgs)*U0bar_grain/(b_muG*rhograin_int_cgs), f_grainsize = DMAX(8.e-4*pow(fac_grain*(temperature/1.e4),0.25), 3.e-3*sqrt(fac_grain)), Z_sol=1.; // b=2, uniform logarithmic grain spectrum over a factor of ~100 in grain size; f_grainsize = 0.07*pow(sqrt(fion*n1)*EcrGeV*T4/BmuG,0.25); // MRN size spectrum
#ifdef METALS
    Z_sol = pp[target].Metallicity[0]/0.014;
#endif
    double G_dust = vA_code*k_L * Z_sol * f_grainsize; // also can increase by up to a factor of 2 for regimes where charge collisionally saturated, though this is unlikely to be realized
    double RGV_dust_crit = 0.01 * (vA_code*UNIT_VEL_IN_KMS) / (1. + pow(cs_thermal*UNIT_VEL_IN_KMS/8.1,2)); // critical rigidity below which there are no gyro-resonant dust grains expected with >10 nm sizes
    if(R_CR_GV < RGV_dust_crit) {G_dust *= pow(R_CR_GV/RGV_dust_crit, 2);} // suppression factor for the dust-damping term when outside of this relevant rigidity range (thanks to Margot Fitz Axen for helping identify regimes where this would kick in which we were simulating in GMCs)
    if(mode<0) {G_dust = 0;} // for this choice, neglect the dust-damping term
    double G_ion_neutral = (5.77e-11 * n_cgs * (0.97*nh0 + 0.03*nHe0) * sqrt(temperature)) * UNIT_TIME_IN_CGS / sqrt(M_cr_mp); // ion-neutral damping: need to get thermodynamic quantities [neutral fraction, temperature in Kelvin] to compute here -- // G_ion_neutral = (xiH + xiHe); // xiH = nH * siH * sqrt[(32/9pi) *kB*T*mH/(mi*(mi+mH))]
    double fac_turb = sqrt(k_turb*k_L) * fturb_multiplier; // factor to use below
    double G_turb_plus_linear_landau = (vA_noion + sqrt(M_PI)*cs_thermal/4.) * fac_turb; // linear Landau + turbulent (both have same form, assume k_turb from cascade above)
    double G_adiabatic = 0.5*pp[target].Particle_DivVel*All.cf_a2inv; // adiabatic term [signed like the other linear terms
    double Gamma_NLL = (sqrt(M_PI)/8.) * cs_thermal / r_L; // NLL prefactor: Gamma_NLL is this times (e_A/e_B)
    double f_cas_ET = 7. * (vA_noion / (beta * C_LIGHT_CODE)) * log(1.+L_scale/r_L); /* Alfvenic turbulence following an anisotropic Goldreich-Shridar cascade, per Chandran 2000 */
    double S_ext_turb = f_cas_ET * vA_noion * fac_turb * M_A*M_A * (r_L/L_scale); // extrinsic turbulence cascade term. note consistency means multiplying by pitch-angle and gyro-averaging factors, and fcas above; expression here assumes whatever you do its a balanced cascade (defauly to GS95 scalings)
    double S_ext_gri  = vA_code * fabs(bhat_dot_CR_Pgrad) / (e_B + MIN_REAL_NUMBER); // Flux-steady-state value of the GRI term, normalized to e_B. note unless we rewrite this to the 5th-order polynomial version, assumption here is that return_CRbin_nuplusminus_asymmetry(i,k_CRegy, cell) -> 1 for the term here in steady state
    double Gamma_LIN = -(G_ion_neutral + G_turb_plus_linear_landau + G_dust + G_adiabatic); // sum of all the linear damping/growth terms
    double S_ext = S_ext_turb + S_ext_gri; // total driving term, for the flux-steady assumption

    if(mode==5) {S_ext=S_ext_turb + S_ext_gri; Gamma_LIN=-DMAX(DMIN(fturb_multiplier,1.),100.)*vA_noion*k_turb*(0.+1.*pow(k_L/k_turb,0.25))*((e_CR+EPSILON_SMALL)/(e_B+EPSILON_SMALL));} // resolve fundamental issues with SC+ET models by invoking alternative damping, following Hopkins et al. 2021
    if(mode==6) {double S_lin = 9.0e-19*UNIT_LENGTH_IN_CGS * (1.+M_A) * sqrt(vA_noion*vA_noion + cs_thermal*cs_thermal) * pow(r_L*UNIT_LENGTH_IN_CGS/1.5e12 , -0.66);
        S_ext = 0.1*S_ext_turb + 0.01*S_ext_gri; Gamma_LIN=S_lin - (G_ion_neutral+G_adiabatic+1.e-10*G_dust); Gamma_NLL += vA_noion/r_L;} // resolve fundamental issues with SC+ET models by invoking alternative linear driving, following Hopkins et al. 2021
    if(mode==7) {f_cas_ET=vA_noion/(0.007 * C_LIGHT_CODE); S_ext_turb=f_cas_ET*vA_noion*fac_turb*M_A*M_A*pow(r_L/L_scale,2./3.); S_ext=S_ext_turb+S_ext_gri;} // resolve fundamental issues with SC+ET models by invoking alternative constant driving, following Hopkins et al. 2021

    double fac=0, f0 = Gamma_LIN/(2.*Gamma_NLL + MIN_REAL_NUMBER), f1 = 4.*Gamma_NLL*S_ext / (Gamma_LIN*Gamma_LIN + MIN_REAL_NUMBER);
    if(f0>0) {fac=f0*(1.+sqrt(1.+f1));} else {if(f1>0.1) {fac=f0*(1.-sqrt(1.+f1));} else {fac=(S_ext/(fabs(Gamma_LIN)+MIN_REAL_NUMBER))*(1.-f1/4.);}}
    double gyro_avg_factor = 3./4.; // weighting factor from pitch-angle averaging over nu, gives ~3/4 for nu~|mu-vA/c|^2, etc.
    return (kappa_0 / fac) * (4./(3.*M_PI*gyro_avg_factor)) * (UNIT_VEL_IN_CGS*UNIT_LENGTH_IN_CGS);
}


                                                                                                                              
/* routine which gives diffusion coefficient [in cgs] for extrinsic turbulence models. 'mode' sets whether we assume Alfven modes (mode<0), Fast-mode scattering (mode>0), or both (=0),
     0: 'default' Alfven + Fast modes (both, summing scattering rates linearly)
    -1: 'default' Alfven modes: correctly accounting for an anisotropic Goldreich-Shridar cascade, per Chandran 2000
    -2: Alfven modes in pure Goldreich-Shridhar cascade, ignoring anisotropic effects [*much* higher scattering rate, artificially]
     1: 'default' Fast modes: following Yan & Lazarian 2002, accounting for damping from viscous, ion-neutral, and other effects, and suppression if beta>1
     2: Fast modes following a pure isotropic Kolmogorov cascade down to gyro radius [*much* higher scattering rate, artificially]
 */
double diffusion_coefficient_extrinsic_turbulence(int mode, int target, int k_CRegy, double M_A, double L_scale, double b_muG,
    double vA_noion, double rho_cgs, double temperature, double cs_thermal, double nh0, double nHe0, double f_ion, struct particle_data *pp, struct gas_cell_data *cell)
{
    double f_cas_ET=MAX_REAL_NUMBER, EPSILON_SMALL=1.e-50, h0_kpc=L_scale*UNIT_LENGTH_IN_KPC;
    if(mode <= 0) /* Alfvenic turbulence [default here following Chandran 200, including anisotropy effects] */
    {
        f_cas_ET = 0.007 * C_LIGHT_CODE / vA_noion; /* damping in Alfvenic turbulence following an anisotropic Goldreich-Shridar cascade, per Chandran 2000 */
        if(mode==-2) {f_cas_ET = 1;} /* pure Goldreich-Shridhar cascade, ignoring anisotropic effects per Chandran-00 */ //if(M_A < 1.) {f_cas_ET = pow(1./DMAX(M_A*M_A , x_LL), 1./3.);} /* Lazarian '16 modification for weak cascade in sub-Alfvenic turbulence */
    }
    if(mode >= 0) /* Fast modes [default here following Yan & Lazarian 2002, including damping effects] */
    {
        int target_bin_centering_for_CR_quantities = target; // if this = target, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
        target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
        double R_CR_GV=return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy);
        double n1=rho_cgs/PROTONMASS_CGS, T4=temperature/1.e4, fcasET_colless = 0.04*cs_thermal/vA_noion; /* collisionless [Landau] damping of fast modes */
        double fcasET_viscBrg = 0.03*pow(EPSILON_SMALL + M_A,4./3.)*T4/pow(EPSILON_SMALL + b_muG*h0_kpc*n1*R_CR_GV*T4,1./6.); /* Spitzer/Braginski viscous damping of fast modes */
        double fcasET_viscMol = 0.41*pow(EPSILON_SMALL + M_A,4./3.)*nh0/pow(EPSILON_SMALL + b_muG*h0_kpc*n1*R_CR_GV/(EPSILON_SMALL + T4),1./6.); /* atomic/molecular collisional damping of fast modes */
        double f_cas_ET_fast = fcasET_colless + fcasET_viscBrg + fcasET_viscMol; /* fast modes, accounting for damping, following Yan+Lazarian 2005 */
        double fast_gyrores_dampingsuppression = 1.; // term to account for the fact that small pitch angles become unscattered when neutral fraction is large or beta >~ 1, making kappa blow up rapidly */
        double beta_half = cs_thermal / vA_noion; if(beta_half > 1.) {fast_gyrores_dampingsuppression=0;} else {fast_gyrores_dampingsuppression*=exp(-beta_half*beta_half*beta_half);} // parallel modes strongly damped if beta >~ 1
        double f_neutral_crit = 0.001 * pow(T4,0.25) / (pow(n1*beta_half*beta_half,0.75) * sqrt(h0_kpc)); // neutral fraction above which the parallel modes are strongly damped
        if(nh0 > 2.*f_neutral_crit) {fast_gyrores_dampingsuppression=0;} else {fast_gyrores_dampingsuppression*=exp(-(nh0*nh0*nh0*nh0)/(f_neutral_crit*f_neutral_crit*f_neutral_crit*f_neutral_crit));} // suppression very rapid, as exp(-[fn/f0]^4)
        if(mode==2) {fast_gyrores_dampingsuppression=0; f_cas_ET_fast = 0.0009 * pow(R_CR_GV*h0_kpc*h0_kpc/b_muG,1./3.)/(M_A*M_A);} /* kappa~9e28 * (l_alfven/kpc)^(2/3) * RGV^(1/3) * (B/muG)^(-1/3),  follows Jokipii 1966, with our corrections for spectral shape */
        if(mode==3) {fast_gyrores_dampingsuppression=0; f_cas_ET_fast = 0.003 * (h0_kpc + 0.1*pow(R_CR_GV*h0_kpc*h0_kpc/b_muG,1./3.)/(M_A*M_A) + 2.4e-7*R_CR_GV/b_muG);} /* Snodin et al. 2016 -- different expression for extrinsic MHD-turb diffusivity, using the proper definition of L_Alfven for scaling to the correct limit */
        f_cas_ET = 1./(EPSILON_SMALL + 1./(EPSILON_SMALL+f_cas_ET) + fast_gyrores_dampingsuppression / (EPSILON_SMALL+f_cas_ET_fast)); /* combine fast-mode and Alfvenic scattering */
    }
    return 1.e32 * h0_kpc / (EPSILON_SMALL + M_A*M_A) * f_cas_ET;
}

                                                                                                                              

/*!----------------------------------------------------------------------------------------------------------------------------------------------------
 routines below are more general and/or numerical: they generally do NOT need to be modified even if you are changing the
   physical assumptions, energies, or other properties of the CRs
 ----------------------------------------------------------------------------------------------------------------------------------------------------*/


/* cosmic ray interactions affecting the -thermal- temperature of the gas are included in the actual cooling/heating functions;
    they are solved implicitly above. however we need to account for energy losses of the actual cosmic ray fluid, here. The
    timescale for this is reasonably long, so we can treat it semi-explicitly, as we do here.
    -- We use the estimate for combined hadronic + Coulomb losses from Volk 1996, Ensslin 1997, as updated in Guo & Oh 2008: */
/* CR_cooling_and_losses: both non-EVOLVE_SPECTRUM and EVOLVE_SPECTRUM versions
   are now in cosmic_ray_functions.h (single source of truth) */

                                                                                                                              

/* utility to estimate -locally- (without multi-pass filtering) the local Alfven Mach number */
double Get_AlfvenMachNumber_Local(int i, double vA_idealMHD_codeunits, int use_shear_corrected_vturb_flag, struct gas_cell_data *cell)
{
    int i1,i2; double v2_t=0,dv2_t=0,b2_t=0,db2_t=0,M_A,h0,EPSILON_SMALL=1.e-50; // factor which will represent which cascade model we are going to use
    for(i1=0;i1<3;i1++)
    {
        v2_t += cell[i].VelPred[i1]*cell[i].VelPred[i1];
        for(i2=0;i2<3;i2++) {dv2_t += cell[i].Gradients.Velocity[i1][i2]*cell[i].Gradients.Velocity[i1][i2];}
#ifdef MAGNETIC
        for(i2=0;i2<3;i2++) {db2_t += cell[i].Gradients.B[i1][i2]*cell[i].Gradients.B[i1][i2];}
#endif
    }
#ifdef MAGNETIC
    b2_t = cell[i].Bfield().norm_sq();
#endif
    v2_t=sqrt(v2_t); b2_t=sqrt(b2_t); dv2_t=sqrt(dv2_t); db2_t=sqrt(db2_t); dv2_t/=All.cf_atime; db2_t/=All.cf_atime; b2_t*=All.cf_a2inv; db2_t*=All.cf_a2inv; v2_t/=All.cf_atime; dv2_t/=All.cf_atime;
    h0=P[i].Get_Particle_Size()*All.cf_atime; // physical units

    if(use_shear_corrected_vturb_flag == 1)
    {
        dv2_t = sqrt((1./2.)*((cell[i].Gradients.Velocity[1][0]+cell[i].Gradients.Velocity[0][1]) *
            (cell[i].Gradients.Velocity[1][0]+cell[i].Gradients.Velocity[0][1]) + (cell[i].Gradients.Velocity[2][0]+cell[i].Gradients.Velocity[0][2]) *
            (cell[i].Gradients.Velocity[2][0]+cell[i].Gradients.Velocity[0][2]) + (cell[i].Gradients.Velocity[2][1]+cell[i].Gradients.Velocity[1][2]) * (cell[i].Gradients.Velocity[2][1]+cell[i].Gradients.Velocity[1][2])) +
            (2./3.)*((cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[0][0] + cell[i].Gradients.Velocity[1][1]*cell[i].Gradients.Velocity[1][1] +
            cell[i].Gradients.Velocity[2][2]*cell[i].Gradients.Velocity[2][2]) - (cell[i].Gradients.Velocity[1][1]*cell[i].Gradients.Velocity[2][2] + cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[1][1] +
            cell[i].Gradients.Velocity[0][0]*cell[i].Gradients.Velocity[2][2]))) * All.cf_a2inv;
#ifdef MAGNETIC
        db2_t = sqrt((1./2.)*((cell[i].Gradients.B[1][0]+cell[i].Gradients.B[0][1]) * (cell[i].Gradients.B[1][0]+cell[i].Gradients.B[0][1]) +
            (cell[i].Gradients.B[2][0]+cell[i].Gradients.B[0][2]) * (cell[i].Gradients.B[2][0]+cell[i].Gradients.B[0][2]) +
            (cell[i].Gradients.B[2][1]+cell[i].Gradients.B[1][2]) * (cell[i].Gradients.B[2][1]+cell[i].Gradients.B[1][2])) +
            (2./3.)*((cell[i].Gradients.B[0][0]*cell[i].Gradients.B[0][0] + cell[i].Gradients.B[1][1]*cell[i].Gradients.B[1][1] +
            cell[i].Gradients.B[2][2]*cell[i].Gradients.B[2][2]) - (cell[i].Gradients.B[1][1]*cell[i].Gradients.B[2][2] +
            cell[i].Gradients.B[0][0]*cell[i].Gradients.B[1][1] + cell[i].Gradients.B[0][0]*cell[i].Gradients.B[2][2]))) * All.cf_a3inv;
#endif
        double db_v_equiv = h0 * db2_t * vA_idealMHD_codeunits / (EPSILON_SMALL + b2_t); /* effective delta-v corresponding to delta-B fluctuations */
        double dv_e = sqrt((h0*dv2_t)*(h0*dv2_t) + db_v_equiv*db_v_equiv + EPSILON_SMALL); /* total effective delta-velocity */
        double vA_eff = sqrt(vA_idealMHD_codeunits*vA_idealMHD_codeunits + db_v_equiv*db_v_equiv + EPSILON_SMALL); /* effective Alfven speed including fluctuation-B */
        M_A = (EPSILON_SMALL + dv_e) / (EPSILON_SMALL + vA_eff);
    } else {
        M_A = h0*(EPSILON_SMALL + dv2_t) / (EPSILON_SMALL + vA_idealMHD_codeunits); /* velocity fluctuation-inferred Mach number */
        M_A = DMAX(M_A , h0*(EPSILON_SMALL + db2_t) / (EPSILON_SMALL + b2_t)); /* B-field fluctuation-inferred Mach number [in incompressible B-turb, this will 'catch' where dv is locally low instanteously] */
    }
    M_A = DMAX( EPSILON_SMALL , M_A ); // proper calculation of the local Alfven Mach number
    return M_A;
}

                                                                                                                           


/* parent routine to assign diffusion coefficients. for the most relevant physical models, we do a lot of utility here but do the more interesting
    (and uncertain) physical calculation in the relevant sub-routines above, so you don't need to modify all of this in most cases */
void CalculateAndAssign_CosmicRay_DiffusionAndStreamingCoefficients(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* first define some very general variables, and calculate some useful quantities that will be used for any model */
    int k_CRegy; double DiffusionCoeff, CR_kappa_streaming, CRPressureGradScaleLength, v_streaming;
#if (CRFLUID_DIFFUSION_MODEL > 0)
    double cs_thermal,M_A,L_scale,vA_code,vA_noion,gizmo2gauss,Omega_per_GeV_ifveqc,Bmag,unit_kappa_code,b_muG,E_B,f_ion,temperature,EPSILON_SMALL; int k; k=0;
    unit_kappa_code=UNIT_VEL_IN_CGS*UNIT_LENGTH_IN_CGS; gizmo2gauss=UNIT_B_IN_GAUSS; f_ion=1; temperature=0; EPSILON_SMALL=1.e-50;
    Bmag=2.*cell[i].Pressure*All.cf_a3inv; cs_thermal=sqrt(cell[i].soundspeed2_from_u(cell[i].InternalEnergyPred)); /* quick thermal pressure properties (we'll assume beta=1 if MHD not enabled) */
#ifdef MAGNETIC /* get actual B-field */
    Vec3<double> B = cell[i].Bfield() * All.cf_a2inv; Bmag=B.norm_sq(); // B-field in code units (physical)
#endif
    Bmag=sqrt(DMAX(Bmag,0)); b_muG=Bmag*gizmo2gauss/1.e-6; b_muG=sqrt(b_muG*b_muG + 1.e-6); vA_code=sqrt(Bmag*Bmag/(cell[i].Density*All.cf_a3inv)); vA_noion=vA_code; E_B=0.5*Bmag*Bmag*(pp[i].Mass/(cell[i].Density*All.cf_a3inv));
    Omega_per_GeV_ifveqc=(0.00898734*b_muG) * UNIT_TIME_IN_CGS; /* B-field in units of physical microGauss; set a floor at nanoGauss level. convert to physical code units */
#ifdef COOLING
    double ne=1, nh0=0, nHe0=0, nHepp, nhp, nHeII, mu_meanwt=1, rho=cell[i].Density*All.cf_a3inv, rho_cgs, u0=cell[i].InternalEnergyPred;
    temperature = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp, pp, cell); rho_cgs=rho*UNIT_DENSITY_IN_CGS; // get thermodynamic properties
    f_ion = DMIN(DMAX(DMAX(DMAX(1-nh0, nhp), ne/1.2), 1.e-8), 1.); // account for different measures above (assuming primordial composition)
#endif
    M_A = Get_AlfvenMachNumber_Local(i,vA_noion,0, cell); /* get turbulent Alfven Mach number estimate. 0 or 1 to turn on shear-correction */
    L_scale = pp[i].Get_Particle_Size()*All.cf_atime; /* define turbulent scales [estimation of M_A defined by reference to this scale */
#endif
    
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        v_streaming=0; DiffusionCoeff=0; CR_kappa_streaming=0; CRPressureGradScaleLength=Get_CosmicRayGradientLength(i,k_CRegy, pp, cell); /* set these for the bin as we get started */
#if (CRFLUID_DIFFUSION_MODEL == 0) /* set diffusivity to a universal power-law scaling (constant per-bin)  */
        DiffusionCoeff = diffusion_coefficient_constant(i,k_CRegy, cell); //  this is the input value of the diffusivity, for constant-kappa models
#endif
#if (CRFLUID_DIFFUSION_MODEL == 8) /* set diffusivity to a universal power-law scaling (constant per-bin), plus constant-streaming-speed correction term as implied by some CGM observations  */
        DiffusionCoeff = diffusion_coefficient_constant(i,k_CRegy, cell); //  this is the input value of the diffusivity, for constant-kappa models
        double vst_asymptotic_kms=100., vst00=vst_asymptotic_kms/100., beta=return_CRbin_beta_factor(i,k_CRegy), RGV=return_CRbin_CR_rigidity_in_GV(i,k_CRegy), l00=4./UNIT_LENGTH_IN_KPC;
        double lstar = CRPressureGradScaleLength, l0 = l00*(0.1 + beta*sqrt(RGV))/vst00, diff_corrfac = 1. + lstar/l0;
        DiffusionCoeff *= diff_corrfac;
#endif
#if (CRFLUID_DIFFUSION_MODEL < 0) /* disable CR diffusion, specifically */
        DiffusionCoeff = 0; // no diffusion (but -can- allow streaming)
#endif
#if (CRFLUID_DIFFUSION_MODEL == 3) /* Farber et al. 2018 -- higher coeff in neutral gas, lower in ionized gas */
        DiffusionCoeff = (3.e29/unit_kappa_code) * (1.-f_ion + f_ion/30.); // 30x lower in neutral (note use f_ion directly here, not temperature as they do)
#endif
#if (CRFLUID_DIFFUSION_MODEL == 4) /* Wiener et al. 2017 style pure-streaming but with larger streaming speeds and limited losses, using their scaling for assumption that turbulent+non-linear Landau only dominate damping */
        double ni_m3=f_ion*(rho_cgs/PROTONMASS_CGS)/1.e-3, T6=temperature/1.e6, Lturbkpc=L_scale*UNIT_LENGTH_IN_KPC, Lgradkpc=CRPressureGradScaleLength*UNIT_LENGTH_IN_KPC, h0_fac=0.1*pp[i].Get_Particle_Size()*All.cf_atime*All.cf_a2inv*UNIT_VEL_IN_KMS, dv2_10=cell[i].Gradients.Velocity.frobenius_norm_sq()*h0_fac*h0_fac;
        double ecr_14 = cell[i].CosmicRayEnergyPred[k_CRegy] * (cell[i].Density*All.cf_a3inv/pp[i].Mass) * UNIT_PRESSURE_IN_CGS / 1.0e-14; // CR energy density in CGS units //
        v_streaming = Get_Gas_ion_Alfven_speed_i(i, pp, cell);
        CR_kappa_streaming = GAMMA_COSMICRAY(k_CRegy) * CRPressureGradScaleLength * (v_streaming + (1./UNIT_VEL_IN_KMS)*(4.1*pow(MIN_REAL_NUMBER+ni_m3*T6,0.25)/pow(MIN_REAL_NUMBER+ecr_14*Lgradkpc,0.5) + 1.2*pow(MIN_REAL_NUMBER+dv2_10*ni_m3,0.75)/(MIN_REAL_NUMBER+ecr_14*sqrt(Lturbkpc)))); // convert to effective diffusivity
#endif
#if (CRFLUID_DIFFUSION_MODEL == 5) /* streaming at fast MHD wavespeed [just to see what it does] */
        v_streaming = cell[i].fast_MHD_wavespeed();
        CR_kappa_streaming = GAMMA_COSMICRAY(k_CRegy) * v_streaming * CRPressureGradScaleLength;
#endif
#if (CRFLUID_DIFFUSION_MODEL == 1) || (CRFLUID_DIFFUSION_MODEL == 2) || (CRFLUID_DIFFUSION_MODEL == 7) /* textbook extrinsic turbulence model: kappa~v_CR*r_gyro * B_bulk^2/(B_random[scale~r_gyro]^2) v_CR~c, r_gyro~p*c/(Z*e*B)~1e12 cm * RGV *(3 muG/B)  (RGV~1 is the magnetic rigidity). assuming a Kolmogorov spectrum */
        int scatter_modes = 0; /* default to using both Alfven+damped-fast modes */
#if (CRFLUID_DIFFUSION_MODEL==1)
        scatter_modes = -1; /* Alfven modes only*/
#endif
#if (CRFLUID_DIFFUSION_MODEL==2)
        scatter_modes = 1; /* Fast modes only*/
#endif
#if defined(CRFLUID_SET_ET_MODEL)
        scatter_modes = CRFLUID_SET_ET_MODEL; /* set to user-defined value */
#endif
        DiffusionCoeff = diffusion_coefficient_extrinsic_turbulence(scatter_modes,i,k_CRegy,M_A,L_scale,b_muG,vA_noion,rho_cgs,temperature,cs_thermal,nh0,nHe0,f_ion, pp, cell) / unit_kappa_code;
#endif
#if (CRFLUID_DIFFUSION_MODEL == 6) || (CRFLUID_DIFFUSION_MODEL == 7) /* self-confinement-based diffusivity */
        int target_bin_centering_for_CR_quantities = i; // if this = i, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
        target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
        double Omega_gyro_ifveqc=(0.00898734*b_muG/return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy)) * UNIT_TIME_IN_CGS, r_L=C_LIGHT_CODE/Omega_gyro_ifveqc, kappa_0=r_L*C_LIGHT_CODE; // some handy numbers for limiting extreme-kappa below. all in -physical- code units //
        CR_kappa_streaming = diffusion_coefficient_self_confinement(CRFLUID_SET_SC_MODEL,i,k_CRegy,M_A,L_scale,b_muG,vA_noion,rho_cgs,temperature,cs_thermal,nh0,nHe0,f_ion, pp, cell) / unit_kappa_code;
        if(!isfinite(CR_kappa_streaming)) {CR_kappa_streaming = 1.e30/unit_kappa_code;} /* apply some limiters since its very easy for the routine above to give wildly-large-or-small diffusivity, which wont make a difference compared to just 'small' or 'large', but will mess things up numerically */
        CR_kappa_streaming = DMIN( DMAX( DMIN(DMAX(CR_kappa_streaming,kappa_0) , 1.0e10*GAMMA_COSMICRAY(k_CRegy) * CRPressureGradScaleLength*CRFLUID_REDUCED_C_CODE(k_CRegy)) , 1.e25/unit_kappa_code ) , 1.e34/unit_kappa_code );
#endif

        /* -- ok, we've done what we came to do -- everything below here is pure-numerical, not physics, and should generally not be modified -- */

#if (CRFLUID_DIFFUSION_MODEL == 7) /* 'combined' extrinsic turbulence + self-confinement model: add scattering rates linearly (plus lots of checks to prevent unphysical bounds) */
        CR_kappa_streaming = 1. / (EPSILON_SMALL +  1./(CR_kappa_streaming+EPSILON_SMALL) + 1./(DiffusionCoeff+EPSILON_SMALL) ); DiffusionCoeff=0; if(!isfinite(CR_kappa_streaming)) {CR_kappa_streaming = 1.e30/unit_kappa_code;} // if scattering rates add linearly, this is a rough approximation to the total transport (essentially, smaller of the two dominates)
        CR_kappa_streaming = DMIN( DMAX( CR_kappa_streaming , kappa_0 ) , 1.0e10*GAMMA_COSMICRAY(k_CRegy) * CRPressureGradScaleLength*CRFLUID_REDUCED_C_CODE(k_CRegy) ); CR_kappa_streaming = DMIN( DMAX( CR_kappa_streaming , 1.e25/unit_kappa_code ) , 1.e34/unit_kappa_code );
#endif
        DiffusionCoeff = DiffusionCoeff + CR_kappa_streaming; //  add 'diffusion' and 'streaming' terms since enter numerically the same way
        if((DiffusionCoeff<=0)||(isnan(DiffusionCoeff))) {DiffusionCoeff=0;} /* nan check */
        cell[i].CosmicRayDiffusionCoeff[k_CRegy] = DiffusionCoeff; /* final assignment! */
    } // end CR bin loop
}



/* utility routine which handles the numerically-necessary parts of the CR 'injection' for you; here 'injection_velocity' should be in physical (not comoving) units */
void inject_cosmic_rays(double CR_energy_to_inject, double injection_velocity, int source_type, int target, double *dir, struct gas_cell_data *cell)
{
    if(CR_energy_to_inject <= 0) {return;}
    double f_injected[N_CR_PARTICLE_BINS]; f_injected[0]=1; int k_CRegy;
#if (N_CR_PARTICLE_BINS > 1) /* add a couple steps to make sure injected energy is always normalized properly! */
    double sum_in=0.0; for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,0,target, P, CellP); sum_in+=f_injected[k_CRegy];}
    if(sum_in>0.0) {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]/=sum_in;}} else {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=1./N_CR_PARTICLE_BINS;}}
#endif
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double dEcr = evaluate_cr_transport_reductionfactor(target, k_CRegy, 0, cell) * CR_energy_to_inject * f_injected[k_CRegy]; // normalized properly to sum to unity, and account for RSOL in injection rate [akin to RHD treatment]
        if(dEcr <= 0) {continue;}
#if defined(CRFLUID_EVOLVE_SPECTRUM) // update the evolved slopes with the injection spectrum slope: do a simple energy-weighted mean for the updated/mixed slope here
        double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), egy_slopemode = 1, xm = All.CR_global_min_rigidity_in_bin[k_CRegy] / All.CR_global_rigidity_at_bin_center[k_CRegy], xp = All.CR_global_max_rigidity_in_bin[k_CRegy] / All.CR_global_rigidity_at_bin_center[k_CRegy], xm_e=xm, xp_e=xp; // values needed for bin injection parameters
        if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {egy_slopemode=2; xm_e=xm*xm; xp_e=xp*xp;} // values needed to scale from slope injected to number and back
        double slope_inj = CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,1,target, P, CellP); // spectral slope of injected CRs
#if defined(CRFLUID_ALT_RSOL_FORM) && defined(CRFLUID_ALT_VARIABLE_RSOL) // want to correct injection slope if we're modulating injection with a variable Psi-type rsol function or variable-rsol
        if(k_CRegy>0) {int spec_0=return_CRbin_CR_species_ID(k_CRegy), spec_m=return_CRbin_CR_species_ID(k_CRegy-1), spec_p=-200; if(spec_m==spec_0) {
            double rfac_0=evaluate_cr_transport_reductionfactor(target,k_CRegy,0, cell), rfac_m=evaluate_cr_transport_reductionfactor(target,k_CRegy-1,0), rfac_p=rfac_0, R_0=All.CR_global_rigidity_at_bin_center[k_CRegy], R_m=All.CR_global_rigidity_at_bin_center[k_CRegy-1], R_p=R_0;
            if(k_CRegy<N_CR_PARTICLE_BINS) {spec_p=return_CRbin_CR_species_ID(k_CRegy+1);}
            if(spec_p==spec_0) {rfac_p=evaluate_cr_transport_reductionfactor(target,k_CRegy+1,0, cell); R_p=All.CR_global_rigidity_at_bin_center[k_CRegy+1];
                double xm=log(R_m/R_0),xp=log(R_p/R_0),qm=log(rfac_m/rfac_0),qp=log(rfac_p/rfac_0); slope_inj += (qm*xm + qp*xp) / (xm*xm + xp*xp);} else {slope_inj += log(rfac_0/rfac_m) / log(R_0/R_m);}}} // not clear if actually improves accuracy by substantial margin here, vs letting code self-adjust in next timestep
#endif
        double gamma_one = slope_inj + 1., xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one); // variables below
        double ntot_inj = (dEcr / E_GeV) * ((gamma_one + egy_slopemode) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp_e - xm_gamma_one*xm_e); // injected number in bin
        #pragma omp atomic
        cell[target].CosmicRay_Number_in_Bin[k_CRegy] += ntot_inj; // simply update injected number. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
#endif
        #pragma omp atomic
        cell[target].CosmicRayEnergy[k_CRegy] += dEcr; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        #pragma omp atomic
        cell[target].CosmicRayEnergyPred[k_CRegy] += dEcr; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        double dir_mag=0, flux_mag=dEcr * CRFLUID_REDUCED_C_CODE(k_CRegy); Vec3<double> dir_to_use={}; int k;
#ifdef MAGNETIC
        Vec3<double> Bdir=cell[target].BPred;
        double B_dot_dir=0; for(k=0;k<3;k++) {B_dot_dir+=dir[k]*Bdir[k];} // the 'default' direction is projected onto B
        dir_to_use = B_dot_dir * Bdir; // launch -along- B, projected [with sign determined] by the intially-desired direction
#else
        dir_to_use = {dir[0], dir[1], dir[2]}; // launch in the 'default' direction
        dir_mag = dir_to_use.norm_sq();
        if(dir_mag <= 0) {dir_to_use[0]=0; dir_to_use[1]=0; dir_to_use[2]=1; dir_mag=1;}
        for(k=0;k<3;k++) {
            double dflux=flux_mag*dir_to_use[k]/sqrt(dir_mag);
            #pragma omp atomic
            cell[target].CosmicRayFlux[k_CRegy][k]+=dflux; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
            #pragma omp atomic
            cell[target].CosmicRayFluxPred[k_CRegy][k]+=dflux; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        }
#endif
    }
    return;
}



/* return CR pressure within a given bin */



/* return CR gradient scale length, with various physical limiters applied: not intended for pure numerical gradient-length calculations (where units dont matter), but for preventing some unphysical situations */
double Get_CosmicRayGradientLength(int i, int k_CRegy, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* now we need the -parallel- cosmic ray pressure or energy density scale length */
    int k;
    double CRPressureGradMag = sqrt(1.e-46 + cell[i].Gradients.CosmicRayPressure[k_CRegy].norm_sq()); // sqrt to make absolute value
#ifdef MAGNETIC /* with anisotropic transport, we really want the -parallel- gradient scale-length, so need another factor here */
    Vec3<double> Bvec_tmp = cell[i].Bfield(); double B2_tot = Bvec_tmp.norm_sq(); CRPressureGradMag = dot(Bvec_tmp, cell[i].Gradients.CosmicRayPressure[k_CRegy]); // note, this is signed!
    CRPressureGradMag = sqrt((1.e-40 + CRPressureGradMag*CRPressureGradMag) / (1.e-46 + B2_tot)); // divide B-magnitude to get scalar magnitude, and take sqrt[(G.P)^2] to get absolute value
#endif
    
    /* limit the scale length: if too sharp, need a slope limiter at around the particle size */
    double L_gradient_min = pp[i].Get_Particle_Size() * All.cf_atime;
    /* limit this scale length; if the gradient is too shallow, there is no information beyond a few smoothing lengths, so we can't let streaming go that far */
    double L_gradient_max = DMAX(1000.*L_gradient_min, 500.0*pp[i].KernelRadius*All.cf_atime);

    /* also, physically, cosmic rays cannot stream/diffuse with a faster coefficient than ~v_max*L_mean_free_path, where L_mean_free_path ~ 2.e20 * (cm^-3/n) [collisional here] */
    double nH_cgs = cell[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS;
    double L_mean_free_path = (3.e25 / nH_cgs) / UNIT_LENGTH_IN_CGS;
    L_gradient_max = DMIN(L_gradient_max, L_mean_free_path);
    
    double CRPressureGradScaleLength = Get_Gas_CosmicRayPressure(i, k_CRegy, cell) / CRPressureGradMag * All.cf_atime;
    if(CRPressureGradScaleLength > 0) {CRPressureGradScaleLength = 1.0/(1.0/CRPressureGradScaleLength + 1.0/L_gradient_max);} else {CRPressureGradScaleLength=0;}
    CRPressureGradScaleLength = sqrt(L_gradient_min*L_gradient_min + CRPressureGradScaleLength*CRPressureGradScaleLength);
    return CRPressureGradScaleLength; /* this is returned in -physical- units */
}



/* return_CRbin_CRmass_in_mp, return_CRbin_beta_factor, return_CRbin_gamma_factor,
   return_CRbin_kinetic_energy_in_GeV: definitions now in cosmic_ray_functions.h */

/* gamma_eos_of_crs_in_bin: definition now in cosmic_ray_functions.h */


/* return pre-factor for CR streaming losses, such that loss rate dE/dt = -E * streamfac */
/* CR_get_streaming_loss_rate_coefficient: definition now in cosmic_ray_functions.h */



/* routine to do the drift/kick operations for CRs: mode=0 is kick, mode=1 is drift */
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES)
double CosmicRay_Update_DriftKick(int i, double dt_entr, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    if(dt_entr <= 0) {return 0;} // no update

    int k_CRegy;
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        int k; double eCR, u0; k=0; if(mode==0) {eCR=cell[i].CosmicRayEnergy[k_CRegy]; u0=cell[i].InternalEnergy;} else {eCR=cell[i].CosmicRayEnergyPred[k_CRegy]; u0=cell[i].InternalEnergyPred;} // initial energy
        if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
        if(eCR < 0) {eCR=0;} // limit to physical values
        double closure_f1, closure_f2; closure_f1=1, closure_f2=0; // prefactors for below
        double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(i,k_CRegy, cell); // 3*chi = 3*(1-<mu^2>)/2 closure function //
        closure_f1 = 3.-2.*three_chi; closure_f2 = 1.-three_chi; // prefactors for both terms below //

        // this is the exact solution for the CR flux-update equation over a finite timestep dt: it needs to be solved this way [implicitly] as opposed to explicitly for dt because in the limit of dt_cr_dimless being large, the problem exactly approaches the diffusive solution
        Vec3<double> DtCosmicRayFlux={}, flux={}, CR_veff={}; double CR_vmag=0, q_cr=0, cr_speed=CRFLUID_REDUCED_C_CODE(k_CRegy), rsol_correction_factor=cosmicrayfluid_rsol_corrfac(k_CRegy), V_i=pp[i].Mass/cell[i].Density, P0_cr, fac_for_DtCosmicRayFlux; P0_cr=Get_Gas_CosmicRayPressure(i, k_CRegy, cell);
        cr_speed = DMAX(cell[i].MaxSignalVel , DMIN(CRFLUID_REDUCED_C_CODE(k_CRegy) , 10.*fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy])/(pp[i].Get_Particle_Size()*All.cf_atime)));
        fac_for_DtCosmicRayFlux = -rsol_correction_factor * fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy]) * V_i / (GAMMA_COSMICRAY(k_CRegy)-1.);
        DtCosmicRayFlux = cell[i].Gradients.CosmicRayPressure[k_CRegy];
#ifdef MAGNETIC // do projection onto field lines
        Vec3<double> bhat={}, B0={}; double Bmag2=0, Bmag=0, bbGB=0, DtCRDotBhat=0;
        if(mode==0) {B0=cell[i].B/V_i;} else {B0=cell[i].BPred/V_i;}
        DtCRDotBhat = dot(DtCosmicRayFlux, B0); Bmag2 = B0.norm_sq(); bhat = B0;
        if(Bmag2 > 0) {Bmag=sqrt(Bmag2); bhat /= Bmag;}
        bbGB = -dot(bhat, cell[i].Gradients.B.matvec(bhat)) / Bmag;
        DtCosmicRayFlux = fac_for_DtCosmicRayFlux * (closure_f1*DtCRDotBhat*B0/Bmag2 + bhat*(closure_f2*P0_cr*bbGB));
#endif
        double v_Alfven = three_chi * Get_Gas_ion_Alfven_speed_i(i, pp, cell) * return_CRbin_nuplusminus_asymmetry(i,k_CRegy, cell); /* define naive streaming and Alfven speeds */
        double dt_f_m=DtCosmicRayFlux.norm_sq();
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        double flux_diff=sqrt(dt_f_m), flux_stream=fabs(rsol_correction_factor*v_Alfven*(GAMMA_COSMICRAY(k_CRegy)*eCR)); // estimate contribution to flux from both diffusive and streaming components
        double frac_diff=flux_diff/(flux_diff+flux_stream); // fraction of flux from diffusive term
        double alpha_v=0.,alpha_qN=0.,alpha_qE=1.,alpha_nu=-0.6,alpha_L=0.,alpha_f0=CR_return_spectral_slope_target(i,k_CRegy, cell); // values of coefficients: replace hard-coded alpha_nu with lookup to actual function numerically, below
        double xi = All.CR_global_max_rigidity_in_bin[k_CRegy] / All.CR_global_min_rigidity_in_bin[k_CRegy]; // bin width in our units
        int kCR_p=k_CRegy, kCR_m=k_CRegy-1; // want two neighboring bins with same species
        if(k_CRegy<N_CR_PARTICLE_BINS-1) {if(All.CR_species_ID_in_bin[k_CRegy+1]==All.CR_species_ID_in_bin[k_CRegy]) {kCR_m++; kCR_p++;}} // check if can use this and next, or use this and below
        double xi_pm = sqrt((All.CR_global_min_rigidity_in_bin[kCR_p]*All.CR_global_max_rigidity_in_bin[kCR_p])/(All.CR_global_min_rigidity_in_bin[kCR_m]*All.CR_global_max_rigidity_in_bin[kCR_m])); // bin ratio to next bin for numerical derivative (being careful to follow our convention of defining these at the geometric mean)
        double beta_k =return_CRbin_beta_factor(-1,k_CRegy), beta_p=return_CRbin_beta_factor(-1,kCR_p), beta_m=return_CRbin_beta_factor(-1,kCR_m); // get beta factors needed to go between scattering rates and diffusivities
        alpha_nu = log((beta_p*beta_p/cell[i].CosmicRayDiffusionCoeff[kCR_p]) / (beta_m*beta_m/cell[i].CosmicRayDiffusionCoeff[kCR_m])) / log(xi_pm); // numerically calculate the slope of the scattering-rate dependence for any functional form
        if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {alpha_v=1.; alpha_qE=2.;} // correct to non-relativistic values as needed
        if(beta_k<1. && beta_k>0.) {double one_minus_beta2=1.-beta_k*beta_k; alpha_v=one_minus_beta2*one_minus_beta2; alpha_qE=1.+sqrt(one_minus_beta2);} // these are exact in terms of beta, so good approx here using bin-centered beta values
        alpha_L = -0.5*alpha_nu; // this is an approximate model, since usually in steady state we end up with alpha_L roughly following this scaling -- but note that to leading order in the most important quantity here which is the -ratio- of omega_n to omega_e, the alpha_L term factors out //
        double alpha_mu = alpha_v - (alpha_nu + 0*alpha_L); // use value of alpha-mu for diffusive equilibrium, the regime where this term matters [alpha_L term zero'd here because we're taking really the ratio of omega_1 over omega_delta, more like omega_kappa in the reference]
        double flux_n_over_e_factor_approx = 1. + ((alpha_qN-alpha_qE)*(alpha_v+alpha_mu)/12.)*log(xi)*log(xi); // approximate series expansion, should use full expressions here
        double c0_a=1.+DMAX(DMIN(alpha_f0,0.),-6.)-alpha_L, c0_b=c0_a+2.*alpha_v-alpha_nu, c0_c=-alpha_v+0.5*alpha_nu, ln_xi=log(xi), c0_a_e=c0_a+alpha_qE, c0_b_e=c0_b+alpha_qE, c0_a_n=c0_a+alpha_qN, c0_b_n=c0_b+alpha_qN; // define a bunch of the coefficients we'll need
        double omega_k_e = (c0_a_e/c0_b_e) * ((exp(ln_xi*c0_b_e)-1.)/(exp(ln_xi*c0_a_e)-1.)) * exp(ln_xi*c0_c); // this is the exact value for the omega_e term we need here
        double omega_k_n = (c0_a_n/c0_b_n) * ((exp(ln_xi*c0_b_n)-1.)/(exp(ln_xi*c0_a_n)-1.)) * exp(ln_xi*c0_c); // this is the exact value for the omega_n term we need here
        if(omega_k_e>0.1 && omega_k_e<2. && isfinite(omega_k_e)) {DtCosmicRayFlux *= omega_k_e;} // correct the energy flux (what we evolve by default) by its omega [this absolute correction is less important than the relative correction below, but since we have it, let's use it]
        double flux_n_over_e_factor = omega_k_n / omega_k_e; // exact value
        if((flux_n_over_e_factor<0) || (!isfinite(flux_n_over_e_factor))) {flux_n_over_e_factor = flux_n_over_e_factor_approx;}
        double flux_n_over_e_factor_modulated = 1. + (flux_n_over_e_factor-1.) * frac_diff;
        cell[i].Flux_Number_to_Energy_Correction_Factor[k_CRegy] = DMAX(DMIN(flux_n_over_e_factor_modulated, 2.0), 0.5); // equilibrium streaming solution is alpha_mu->-alpha_v such that bin-centered is exact, so mean correction applies only to flux 'portion' of this
#endif
        if(dt_f_m>0) {DtCosmicRayFlux *= (1.0 + rsol_correction_factor * v_Alfven * (GAMMA_COSMICRAY(k_CRegy) * eCR) / sqrt(dt_f_m));} // (tilde[c]/c) * v_a * (ecr+Pcr), in same direction as gradient wants to 'push' naturally [natural direction of F]

        if(mode==0) {flux=cell[i].CosmicRayFlux[k_CRegy];} else {flux=cell[i].CosmicRayFluxPred[k_CRegy];}
#ifdef MAGNETIC // do projection onto field lines
        double fluxmag=flux.norm_sq(), fluxdot=dot(flux, B0);
        if(fluxmag>0) {fluxmag=sqrt(fluxmag);} else {fluxmag=0;}
        if(fluxdot<0) {fluxmag*=-1;} // points down-field
        // before acting on the 'stiff' sub-system, account for the 'extra' advection term that accounts for 'twisting' of B: note more careful derivation shows this is sub-leading order in v/c, should not be included here
        //double fac_bv=0; for(k=0;k<3;k++) {fac_bv += All.cf_a2inv * bhat[k] * (bhat[0]*cell[i].Gradients.Velocity[k][0] + bhat[1]*cell[i].Gradients.Velocity[k][1] + bhat[2]*cell[i].Gradients.Velocity[k][2]);}
        //if(All.ComovingIntegrationOn) {fac_bv += All.cf_hubble_a;} // adds cosmological/hubble flow term here [not included in peculiar velocity gradient]
        //fluxmag *= exp(-DMAX(-2.,DMIN(2.,rsol_correction_factor*fac_bv*dt_entr))); // limit factor for change here, should be small given Courant factor, then update flux term accordingly, before next step -- acts like a mod of the divv term //
        if(Bmag2>0) {flux = (fluxmag / sqrt(Bmag2)) * B0;} // re-assign to be along field
#endif
        int target_for_CR_beta_factor = i; // if this =1, use energy-weighted mean value in bin for CR beta, otherwise if =-1, use median point of bin
        target_for_CR_beta_factor = -1;
        double beta_fac = return_CRbin_beta_factor(target_for_CR_beta_factor,k_CRegy); // velocity beta, to account for non-relativistic CRs
        double dt_cr_dimless = dt_entr * beta_fac*beta_fac * cr_speed*cr_speed * (1./3.) / (MIN_REAL_NUMBER + fabs(cell[i].CosmicRayDiffusionCoeff[k_CRegy] * rsol_correction_factor));
        dt_cr_dimless = DMIN(dt_cr_dimless , 0.1); // arbitrary limiter here for some additional numerical stability
        if((dt_cr_dimless > 0)&&(dt_cr_dimless < 20.)) {q_cr = exp(-dt_cr_dimless);} // factor for CR interpolation
        flux = q_cr*flux + (1.-q_cr)*DtCosmicRayFlux; // updated flux
        CR_veff = flux/(eCR+MIN_REAL_NUMBER); CR_vmag = CR_veff.norm_sq(); // effective streaming speed
        if((CR_vmag <= 0) || (isnan(CR_vmag))) // check for valid numbers
        {
            flux = {}; CR_veff = {}; // zero if invalid
        } else {
            double CR_vmax = CRFLUID_REDUCED_C_CODE(k_CRegy); // enforce a hard upper limit here, though shouldn't be needed with modern formulation
            CR_vmag = sqrt(CR_vmag); if(CR_vmag > CR_vmax) {flux *= CR_vmax/CR_vmag; CR_veff *= CR_vmax/CR_vmag;} // limit flux to free-streaming speed [as with RT]
        }
        if(mode==0) {cell[i].CosmicRayFlux[k_CRegy]=flux;} else {cell[i].CosmicRayFluxPred[k_CRegy]=flux;}
    
        /* update scalar CR energy. first update the CR energies from fluxes. since this is positive-definite, some additional care is needed */
        double dCR_dt = cell[i].DtCosmicRayEnergy[k_CRegy], eCR_tmp = eCR;
        double dCR = dCR_dt*dt_entr, dCRmax = 1.e10*(eCR_tmp+MIN_REAL_NUMBER);
#if defined(GALSF)
        dCRmax = DMAX(2.0*eCR_tmp , 0.1*u0*pp[i].Mass);
#endif
        if(dCR > dCRmax) {dCR=dCRmax;} // don't allow excessively large values
        if(dCR < -eCR_tmp) {dCR=-eCR_tmp;} // don't allow it to go negative
        double eCR_0, eCR_00; eCR_00 = eCR_tmp; eCR_tmp += dCR; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy]=eCR_tmp;} else {cell[i].CosmicRayEnergyPred[k_CRegy]=eCR_tmp;} // updated energy
        eCR_0 = eCR_tmp; // save this value for below
        
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        // add update for CR number if evolved explicitly //
        if(mode==0) // only update on kicks, since we worth with a drift-conserved slope determining the ratio of N and E
        {
            double dN = cell[i].DtCosmicRay_Number_in_Bin[k_CRegy]*dt_entr, n0 = cell[i].CosmicRay_Number_in_Bin[k_CRegy], n_new = n0+dN;
            double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), xm = All.CR_global_min_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k], xp = All.CR_global_max_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k], xm_e=xm, xp_e=xp;
            if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {xm_e = xm*xm; xp_e = xp*xp;} // extra power of p in energy equation accounted for here, all that's needed
            double N_min = eCR_tmp / (E_GeV * xp_e * (1.-1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at highest energy
            double N_max = eCR_tmp / (E_GeV * xm_e * (1.+1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at lowest energy
            n_new = DMIN(DMAX(n_new,N_min),N_max); if((n_new<0) || (isnan(n_new))) {n_new=0;}
            cell[i].CosmicRay_Number_in_Bin[k_CRegy] = n_new; // alright, updated CR number for evolution equations
        }
#endif
    
#if defined(COOLING_OPERATOR_SPLIT)
        /* now need to account for the adiabatic heating/cooling of the 'fluid', here, with gamma=GAMMA_COSMICRAY(k_CRegy) */
        double dCR_div = CR_calculate_adiabatic_gasCR_exchange_term(i, dt_entr, (GAMMA_COSMICRAY(k_CRegy)-1.)*eCR_tmp, mode, pp, cell); // this will handle the update below - separate subroutine b/c we want to allow it to appear in a couple different places
        double uf = DMAX(u0 - dCR_div/pp[i].Mass , All.MinEgySpec); // final updated value of internal energy per above
        if(mode==0) {cell[i].InternalEnergy = uf;} else {cell[i].InternalEnergyPred = uf;} // update gas
#if !defined(CRFLUID_EVOLVE_SPECTRUM)
        if(mode==0) {cell[i].CosmicRayEnergy[k_CRegy] += dCR_div;} else {cell[i].CosmicRayEnergyPred[k_CRegy] += dCR_div;} // update CRs: note if explicitly evolving spectrum, this is done separately below //
#endif
#endif

    } // loop over CR bins complete
    
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    if(cell[i].DtCREgyNewInjectionFromShocks > 0) /* now perform the actual CR injection using the rates estimated in the hydro solver */
    {
        Vec3<double> dir = -cell[i].Gradients.Pressure; /* initial flux direction down pressure gradient */
        inject_cosmic_rays(cell[i].DtCREgyNewInjectionFromShocks * dt_entr, 1000./UNIT_VEL_IN_KMS, 2, i, &dir[0], cell); /* inject the energy */
        cell[i].DtCREgyNewInjectionFromShocks = 0; // reset to nil, we've successfully injected the energy
    }
#endif
#if defined(SINK_CR_INJECTION_AT_TERMINATION)
    if(cell[i].Sink_CR_Energy_Available_For_Injection > 0) {
        /* need to determine whether or not sufficient deceleration has occurred in order to inject CRs from our 'reservoir */
        double vmag = (pp[i].Vel / All.cf_atime).norm_sq(); int k; /* we will base this on a simple estimate of the velocity and how much things have decelerated */
        if(vmag>0) {vmag=sqrt(vmag);}
        double v_outflow_fast_forinjection = All.Sink_outflow_velocity;
#ifdef SINK_TEST_WIND_MIXED_FASTSLOW
        v_outflow_fast_forinjection = (SINK_TEST_WIND_MIXED_FASTSLOW)/UNIT_VEL_IN_KMS;
#endif
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        v_outflow_fast_forinjection = 0.05 * C_LIGHT_CODE;
#endif
        if((pp[i].ID != All.SpawnedWindCellID) || (vmag < ((double)(SINK_CR_INJECTION_AT_TERMINATION))*v_outflow_fast_forinjection)) {
            Vec3<double> dir = -cell[i].Gradients.Pressure; /* initial flux direction down pressure gradient */
            inject_cosmic_rays(cell[i].Sink_CR_Energy_Available_For_Injection, v_outflow_fast_forinjection, 5, i, &dir[0], cell); /* inject the energy */
            cell[i].Sink_CR_Energy_Available_For_Injection = 0;  // reset its value to nil, now that it has been injected
        }
    }
#endif
    
    return 1;
}
#endif





#if defined(CRFLUID_EVOLVE_SPECTRUM)

/* CR_cooling_and_losses_multibin: definition now in cosmic_ray_functions.h */
/* initialize CR quantities needed specifically for our multi-bin spectral methods */
void CR_initialize_multibin_quantities(void)
{
    if(ThisTask==0) {printf("Initializing global cosmic ray spectral variables:\n"); fflush(stdout);}
    int k; CR_spectrum_define_bins(); // call this to define the actual list of bins, needs to be done before basically anything else!
    double R0[N_CR_PARTICLE_BINS], Z[N_CR_PARTICLE_BINS];
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {R0[k]=All.CR_global_rigidity_at_bin_center[k]; Z[k]=return_CRbin_CR_charge_in_e(-1,k);}
    double R0_bins[N_CR_PARTICLE_BINS], R0_bin_m[N_CR_PARTICLE_BINS], R0_bin_p[N_CR_PARTICLE_BINS];
    int cr_species_key; for(cr_species_key=0;cr_species_key<N_CR_PARTICLE_SPECIES;cr_species_key++) // loop over whether we consider nuclei or electrons, first //
    {
       int n_active=0, bins_sorted[N_CR_PARTICLE_BINS];
       for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(All.CR_species_ID_in_bin[k]==All.CR_species_ID_active_list[cr_species_key]) {bins_sorted[n_active]=k; R0_bins[n_active]=R0[k]; n_active++;}} // bin is valid: charge matches that desired
       if(n_active<=0) {continue;} // nothing to do here
       if(n_active<N_CR_PARTICLE_BINS) {for(k=n_active;k<N_CR_PARTICLE_BINS;k++) {R0_bins[k]=MAX_REAL_NUMBER;}} // set a dummy value here for sorting purposes below
       //qsort(bins_sorted, n_active, sizeof(int), compare_CR_rigidity_for_sort); // sort on energies from smallest-to-largest [this is hard-coded by requiring the list go in monotonic increasing order for e and p, regardless of how the e and p are themselves ordered //
       for(k=0;k<n_active;k++)
       {
           int j = bins_sorted[k], j_m=j, j_p=j; // target bin and bin below/above
           if(k > 0) {j_m = bins_sorted[k-1];} // define previous bin 'down'
           if(k < n_active-1) {j_p = bins_sorted[k+1];} // define next bin 'up'
           R0_bin_m[j] = sqrt(R0[j]*R0[j_m]); R0_bin_p[j] = sqrt(R0[j]*R0[j_p]); // take the bin edges at the geometric means (halfway between midpoints in log-space)
           if(j_m==j) {R0_bin_m[j] = R0[j] * pow(R0[j]/R0_bin_p[j], 2);} // lowest bin gets 'padded' in the small-R direction [extends 2x as far in log-space]
           if(j_p==j) {R0_bin_p[j] = R0[j] * pow(R0[j]/R0_bin_m[j], 2);} // highest bin gets 'padded' in the large-R direction [extends 2x as far in log-space]
       }
    }
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {All.CR_global_min_rigidity_in_bin[k] = R0_bin_m[k]; All.CR_global_max_rigidity_in_bin[k] = R0_bin_p[k];} // set the variables we just defined

    /* ok, now we need to build the lookup tables */
    double gamma_limit = 120.;
    int n_gamma_sample = 10000, n_table = N_CR_SPECTRUM_LUT-1;
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        double xm = All.CR_global_min_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k]; // dimensionless bin minimum
        double xp = All.CR_global_max_rigidity_in_bin[k] / All.CR_global_rigidity_at_bin_center[k]; // dimensionless bin maximum
        double p_power_in_e = 1., xm_e = xm, xp_e = xp; // define variables used below
        if(CR_check_if_bin_is_nonrelativistic(k)) {p_power_in_e = 2.; xm_e = xm*xm; xp_e = xp*xp;} // extra power of p in momentum equation accounted for here, all that's needed
        double gamma_min = -gamma_limit, gamma_max = gamma_limit, d_gamma = (gamma_max-gamma_min) / ((double)n_gamma_sample); int j;

        double gamma = gamma_min, gamma_prev = gamma_min, R_index_prev=0; j=0; int alldone_key=0; // define variables for use in loop below
        while(gamma <= gamma_max)
        {
            double gamma_one = gamma + 1., xm_g = pow(xm, gamma_one), xp_g = pow(xp, gamma_one); // key variables needed
            double R = (gamma_one / (gamma_one + p_power_in_e)) * (xp_g*xp_e - xm_g*xm_e) / (xp_g - xm_g); // ratio of kinetic energy to number times bin-mean energy: this is the key function we need to invert
            double R_index = (log(R / xm_e) / log(xp_e / xm_e)) * n_table; // index one would obtain from this value of R
            if((int)R_index >= j)
            {
                double slope_gamma = gamma + (gamma-gamma_prev) * (((double)j) - R_index) / (R_index - R_index_prev); // linearly interpolate between this and previous step to 'exact' gamma giving desired R
                if(j==1 && slope_gamma < gamma_min) {All.CR_global_slope_lut[k][j-1] = slope_gamma + gamma_min;}
                All.CR_global_slope_lut[k][j] = slope_gamma; // set the look-up-table value for use later
                j++; // move to look for next target bin
            }
            if(j >= n_table) {alldone_key=1; break;} // dont need to continue loop, we've filled in all values desired here //
            R_index_prev = R_index; gamma_prev = gamma; // save for next loop
            gamma += d_gamma; // augment
            // check for bad values of gamma that we wish to avoid because they can cause divergences, and just slightly dodge around them //
            double tol = 0.1*d_gamma, badval; // tolerance around the bad values
            badval=-1; if(fabs(gamma - badval) < tol) {if(gamma<badval) {gamma=badval-tol;} else {gamma=badval+tol;}}
            badval=-2; if(fabs(gamma - badval) < tol) {if(gamma<badval) {gamma=badval-tol;} else {gamma=badval+tol;}}
            badval=-3; if(fabs(gamma - badval) < tol) {if(gamma<badval) {gamma=badval-tol;} else {gamma=badval+tol;}}
        }
        if(alldone_key==0 && j<n_table) {int j0=j-1; for(j=j0+1;j<N_CR_SPECTRUM_LUT;j++) {All.CR_global_slope_lut[k][j]=All.CR_global_slope_lut[k][j0]+(j-j0)*DMIN(gamma_limit,fabs(gamma_limit-All.CR_global_slope_lut[k][j0]));}}
            else {All.CR_global_slope_lut[k][N_CR_SPECTRUM_LUT-1]=gamma_limit;} // set upper end of table value to unity
    }
    
    if(ThisTask==0) {for(k=0;k<N_CR_PARTICLE_BINS;k++) { // print outputs for users
        printf("\n .. bin=%d, charge=%g e, mass=%g mp, rigidity Rmin=%g R0=%g Rmax=%g GV, energy=%g GeV, relativistic?=%d [1=Y/0=N] beta=%g gamma=%g \n",
           k,return_CRbin_CR_charge_in_e(-1,k),return_CRbin_CRmass_in_mp(-1,k),All.CR_global_min_rigidity_in_bin[k],All.CR_global_rigidity_at_bin_center[k],
           All.CR_global_max_rigidity_in_bin[k],return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k),1-CR_check_if_bin_is_nonrelativistic(k),return_CRbin_beta_factor(-1,k),
           return_CRbin_gamma_factor(-1,k)); fflush(stdout);
        printf(" .. LUT for CR slopes in this bin: \n"); printf("  .. j  .. R_egy/num .. gamma \n");
        int j; for(j=0;j<N_CR_SPECTRUM_LUT;j++) {printf("  .. %4d  %5.4g %10.3g \n",j,((double)j)/((double)n_table),All.CR_global_slope_lut[k][j]); fflush(stdout);}
    }}
    
    return;
}



/* CR_return_slope_from_number_and_energy_in_bin: definition now in cosmic_ray_functions.h */


/* CR_return_new_bin_edge_from_rate: definition now in cosmic_ray_functions.h */

                                                                                                                              
/* CR_coulomb_energy_integrand: definition now in cosmic_ray_functions.h */


/* CR_reaccel_energy_integrand: definition now in cosmic_ray_functions.h */


/* CR_compton_energy_integrand: definition now in cosmic_ray_functions.h */


/* CR_check_if_bin_is_nonrelativistic: definition now in cosmic_ray_functions.h */


/* CR_return_effective_number_in_bin_in_codeunits,
   CR_return_spectral_slope_target: definitions now in cosmic_ray_functions.h */


/* return true number of CRs in bin, in actual units */
double CR_return_true_number_in_bin(int target, int k_bin, struct gas_cell_data *cell)
{
    return CR_return_effective_number_in_bin_in_codeunits(target,k_bin, cell) * UNIT_ENERGY_IN_CGS / (1.0e9*ELECTRONVOLT_IN_ERGS); // does the unit conversion from Ecode/GeV to dimensionless number
}


/* subroutine to return the effective CR number from the bin slope and energy */
double CR_get_number_in_bin_from_slope(int target, int k_bin, double energy, double slope)
{
    double etot = energy;
    double gamma_one = 1. + slope;
    double E_bin_center = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_bin);
    double xm = All.CR_global_min_rigidity_in_bin[k_bin] / All.CR_global_rigidity_at_bin_center[k_bin];
    double xp = All.CR_global_max_rigidity_in_bin[k_bin] / All.CR_global_rigidity_at_bin_center[k_bin];
    double xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one);
    int NR_key = CR_check_if_bin_is_nonrelativistic(k_bin);
    double gamma_fac = 0; // factor to get total number in bin. we use the slope we obtain to convert between number and energy, given the bin-centered values
    if(NR_key) {gamma_fac = ((gamma_one + 2.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // non-relativistic map between E and N
        else {gamma_fac = ((gamma_one + 1.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp - xm_gamma_one*xm);} // relativistic map between E and N
    return (etot / E_bin_center) * gamma_fac; // dimensional normalization. note the units are arbitrary for the E_bin_NtoE term, as long as we are consistent [we can evolve constant x N, instead of N, for convenience]
 }


/* return mean kinetic energy per CR for CRs in bin */
double CR_return_mean_energy_in_bin_in_GeV(int target, int k_bin, struct gas_cell_data *cell)
{
    double etot = cell[target].CosmicRayEnergyPred[k_bin]; // total energy in bin
    double ntot_GeV = CR_return_effective_number_in_bin_in_codeunits(target, k_bin, cell); // total number in units of GeV
    return etot / ntot_GeV; // this returns the mean weighted over the CR spectrum
}


/* CR_return_mean_rigidity_in_bin_in_GV: definition now in cosmic_ray_functions.h */


/* sort function if we need to sort CR bins by rigidity from lowest to highest for future compatibility */
int compare_CR_rigidity_for_sort(const void *a, const void *b)
{
    double x = All.CR_global_rigidity_at_bin_center[ *(int *) a];
    double y = All.CR_global_rigidity_at_bin_center[ *(int *) b];
    if (x < y) {return -1;} else if(x > y) {return 1;}
    return 0;
}


/* return_CRbin_kinetic_energy_in_GeV_binvalsNRR: definition now in cosmic_ray_functions.h */


#endif


/* return_CRbin_M1speed: definition now in cosmic_ray_functions.h */


/* evaluate_cr_transport_reductionfactor: definition now in cosmic_ray_functions.h */


/* return_cosmic_ray_anisotropic_closure_function_threechi: definition now in cosmic_ray_functions.h */


/* return_CRbin_CR_rigidity_in_GV, return_CRbin_CR_charge_in_e,
   return_CRbin_CR_species_ID: definitions now in cosmic_ray_functions.h */



/* Get_Gas_ion_Alfven_speed_i: definition now in cosmic_ray_functions.h */


/* return_CRbin_nuplusminus_asymmetry: definition now in cosmic_ray_functions.h */


#endif // closes block for entire file for COSMIC_RAY_FLUID


#ifdef COSMIC_RAY_SUBGRID_LEBRON // block for simplified sub-grid CR model
/* function to return injection rate of CRs -time-averaged in total energy, extremely boiled-down version */
double cr_get_source_injection_rate(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double Edot = 0;
#ifdef GALSF
#ifdef GALSF_FB_MECHANICAL
    if(pp[i].Type == 4)
    {
        double star_age=evaluate_stellar_age_Gyr(i), RSNe=0, agemin=0.003401, agebrk=0.01037, agemax=0.03753;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
        agemin=0.0037; agebrk=0.7e-2; agemax=0.044; double f1=3.9e-4, f2=5.1e-4, f3=1.8e-4; // inputs for newer SNe rate (and newer Ia rate below)
        if(star_age<agemin) {RSNe=0;} else if(star_age<=agebrk) {RSNe=f1*pow(star_age/agemin,log(f2/f1)/log(agebrk/agemin));} else if(star_age<=agemax) {RSNe=f2*pow(star_age/agebrk,log(f3/f2)/log(agemax/agebrk));} else {RSNe=0;} // core-collapse; updated with same stellar evolution models for wind mass loss [see there for references]. simple 2-part power-law provides extremely-accurate fit. models predict a totally negligible metallicity-dependence.
        double t_Ia_min=agemax, norm_Ia=1.6e-3; if(star_age>t_Ia_min) {RSNe += norm_Ia * 7.94e-5 * pow(star_age,-1.1) / fabs(pow(t_Ia_min/0.1,-0.1) - 0.61);} // Ia DTD following Maoz & Graur 2017, ApJ, 848, 25
        //if(star_age < 0.04) {RSNe = 3.0e-4;} else {RSNe = DMIN(3.e-4 , RSNe);} /* replace this with a 'time smoothed' version over the last ~100+ Myr */
#else
        if(star_age>agemin) {if(star_age<=agebrk) {RSNe=5.408e-4;} else {if(star_age<=agemax) {RSNe=2.516e-4;}}} // core-collapse rate [super-simple 2-piece constant] //
        if(star_age>agemax) {RSNe=5.3e-8 + 1.6e-5*exp(-0.5*((star_age-0.05)/0.01)*((star_age-0.05)/0.01));} // Ia (prompt Gaussian+delay, Manucci+06)
#endif
        Edot = All.CosmicRay_SNeFraction * (RSNe*UNIT_TIME_IN_MYR) * (pp[i].Mass*UNIT_MASS_IN_SOLAR) * (1.0e51/UNIT_ENERGY_IN_CGS);
    }
#endif
#ifdef SINK_PARTICLES
    if(pp[i].Type == 5) {
        double mdot_eff = pp[i].Sink_Mdot; // code units
        mdot_eff = DMIN( mdot_eff , pp[i].Sink_Mass / (100./UNIT_TIME_IN_MYR) ); // if time-averaging over ~Gyr, can't have time-averaged injection rate above Mbh/<t> more or less (modulo order-one corrections for all this)
        Edot = evaluate_sink_cosmicray_efficiency(pp[i].Sink_Mdot,pp[i].Sink_Mass,i) * mdot_eff * C_LIGHT_CODE*C_LIGHT_CODE; // injection in code units
    }
#endif
#endif
    if(Edot > 0) {return Edot * cr_get_source_shieldfac(i, pp, cell);} else {return 0;}
}

/* function to return shielding/loss factor correction */
double cr_get_source_shieldfac(int i, struct particle_data *pp, struct gas_cell_data *cell)
{
    double cr_atten_fac = 1;
    if(pp[i].KernelRadius > 0 && pp[i].NumNgb > 0 && All.Time > All.TimeBegin)
    {
        double dx=pp[i].KernelRadius/pp[i].NumNgb, rho; // code units
        Vec3<double> gradrho = pp[i].GradRho;
        if(pp[i].Type==0) {rho=cell[i].Density;} else {rho=pp[i].DensityAroundParticle;}
        if(rho > 0)
        {
            double gradrho_mag = gradrho.norm();
            if(gradrho_mag > 0) {dx += rho/gradrho_mag;} // code units
            double R_loss = ((6.37 + 3.09)*1.e-16*UNIT_TIME_IN_CGS) * (rho*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS); // physical units
            double psi_loss_i = (R_loss / All.CosmicRay_Subgrid_Vstream_0) / sqrt(1. + R_loss*All.CosmicRay_Subgrid_Kappa_0/(All.CosmicRay_Subgrid_Vstream_0*All.CosmicRay_Subgrid_Vstream_0)); // physical units
            double dtau = 0.5 * psi_loss_i * (dx*All.cf_atime); // physical units in dx, so dimensionless here
            cr_atten_fac = exp(-DMIN(dtau, 50.));
        }
    }
    return cr_atten_fac;
}
#endif // closes block for entire file for COSMIC_RAY_SUBGRID_LEBRON



/* Get_CosmicRayEnergyDensity_cgs, Get_CosmicRayIonizationRate_cgs,
   CR_gas_heating: definitions now in cosmic_ray_functions.h */


/* subroutine to calculate which part of the adiabatic PdV work from the RP gets assigned to the CRs vs the gas; since the CRs are always smooth by definition under this operation this follows simply from the local cell divergence and the effective CR eos */
double CR_calculate_adiabatic_gasCR_exchange_term(int i, double dt_entr, double gamma_minus_eCR_tmp, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    double u0, d_CR; if(mode==0) {u0=cell[i].InternalEnergy;} else {u0=cell[i].InternalEnergyPred;} // initial energy
    if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
    
    double divv_p=-dt_entr*pp[i].Particle_DivVel*All.cf_a2inv, divv_f=divv_p, divv_u=0; // get locally-estimated gas velocity divergence for cells - if using non-Lagrangian method, need to modify. take negative of this [for sign of change to energy] and multiply by timestep
#ifdef COSMIC_RAY_FLUID
    divv_f=-dt_entr*cell[i].Face_DivVel_ForAdOps*All.cf_a2inv;
#endif
    if(All.ComovingIntegrationOn) {double divv_h=-dt_entr*(3.*All.cf_hubble_a); divv_p+=divv_h; divv_f+=divv_h;} // include hubble-flow terms
    double P_cr = gamma_minus_eCR_tmp * cell[i].Density * All.cf_a3inv / pp[i].Mass, P_tot = cell[i].Pressure * All.cf_a3inv; // define the pressure from CRs and total pressure (physical units)
#ifdef MAGNETIC
    double B2 = (cell[i].Bfield() * All.cf_a2inv).norm_sq();
    P_tot += 0.5*B2; // add magnetic pressure [B^2/2], in physical code units, since it contributes to the PdV work but not included in 'pressure' total above
#endif
    double fac_P = DMAX(0, DMIN(1, P_cr/(P_tot + 1.e-10*P_cr + MIN_REAL_NUMBER))); // fraction of total pressure from CRs
    double Ui = u0 * pp[i].Mass; // factor for multiplication below, and initial thermal energy
    double dtI_hydro = cell[i].DtInternalEnergy * pp[i].Mass * dt_entr; // change given by hydro-step computed delta_InternalEnergy
    double min_IEgy = pp[i].Mass * All.MinEgySpec; // minimum internal energy - in total units -
    
    if(divv_p*dtI_hydro > 0 || divv_f*dtI_hydro > 0) // same sign from hydro and from smooth-flow-estimator, suggests we are in a smooth flow, so we'll use stronger assumptions about the effective 'entropy' here
    {
        if(divv_p*dtI_hydro <= 0) {divv_u=divv_f;} // if divv_f agrees in sign here, use it
        if(divv_f*dtI_hydro <= 0) {divv_u=divv_p;} // if divv_p agrees in sign here, use it
        if(divv_p*divv_f > 0) {if(fabs(divv_p) > fabs(divv_f)) {divv_u=divv_p;} else {divv_u=divv_f;}} // if both agree in sign here, use -larger- since more accurately captures CR-dominated limit
        d_CR = gamma_minus_eCR_tmp * divv_u; // expected PdV CR energy change
        if(fabs(d_CR) > fabs(dtI_hydro)) {d_CR = dtI_hydro;} // do not allow this to exceed the sum (since all terms have the same sign here, in a well-ordered smooth flow)
        if(fabs(d_CR) < fac_P*fabs(dtI_hydro)) {d_CR = fac_P*dtI_hydro;} // but also do not allow CR term to be -below- CR pressure fraction times total term, since that should be attributed to the CR (as this is all a quasi-adiabatic term)
    } else { // both divv terms agree with each other, but dis-agree with the sign of the total change. can't assume anything about smoothness-of-the-flow
        if(fabs(divv_p) > fabs(divv_f)) {divv_u=divv_f;} else {divv_u=divv_p;} // pick the divv estimator with the smaller absolute magnitude, since it deviates
        d_CR = gamma_minus_eCR_tmp * divv_u; // expected PdV CR energy change
        double f_limiter, fac_test=fabs(d_CR)/fabs(dtI_hydro); if(fac_test>fac_P) {d_CR*=fac_P/fac_test;} // don't let CR change exceed their pressure fraction
        if(d_CR > 0) {if(Ui <= min_IEgy) {f_limiter = 1.e-20;} else {f_limiter=0.5;} // gas will be 'cooled', limit so don't overshoot when Pcr is large
            if(d_CR > f_limiter*(Ui-min_IEgy)) {d_CR = f_limiter*(Ui-min_IEgy);} // limit fractional loss to gas
        } else {f_limiter = 1000.; if(fabs(d_CR)>f_limiter*Ui) {d_CR=-f_limiter*Ui;}} // gas will be heated, limit fractional gain
    }
#if defined(CRFLUID_EVOLVE_SPECTRUM) && !defined(COOLING_OPERATOR_SPLIT)
    cell[i].Face_DivVel_ForAdOps = -d_CR / (All.cf_a2inv * gamma_minus_eCR_tmp * dt_entr + MIN_REAL_NUMBER); // this is the 'effective' divergence here (in code units) which matches exactly the change in CR energy when the above limiters etc are applied. we can save this for use in the other CR subroutines
#endif
    return d_CR; // return final value
}
