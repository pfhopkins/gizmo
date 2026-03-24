#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "../../declarations/allvars.h"
#include "../../core/proto.h"

/*! Routines for cosmic ray 'fluid' modules (as opposed to the explicit CR-PIC methods, which are in the grain+particles section of the code)
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#ifdef COSMIC_RAY_FLUID


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
    int ibin=0, ilast=-200; for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(ibin>=N_CR_PARTICLE_SPECIES) {continue;} else {if(spec[k] != ilast) {CR_species_ID_active_list[ibin]=spec[k]; ilast=spec[k]; ibin++;}}} /* creates list of all species here that are active */
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {CR_global_rigidity_at_bin_center[k]=R[k]; CR_global_charge_in_bin[k]=Z[k]; CR_species_ID_in_bin[k]=spec[k];}

#if 1 // (CRFLUID_EVOLVE_SPECTRUM == 2) // now even the simpler network has secondary e-, important for dense regions synchrotron
    /* note that some of our assumptions here and below are hard-coded to the fact below that the species are -ordered- in the order given by 'species list' at the top */
    int temp_species_map[100]; int min_species_id=99999, id_last=-200, n00=0, j; for(k=0;k<N_CR_PARTICLE_SPECIES;k++) {if(species_list[k]<min_species_id) {min_species_id=species_list[k];}}
    int id_map_offset=0; if(min_species_id<0) {id_map_offset=-min_species_id;}
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(CR_species_ID_in_bin[k] != id_last) {id_last=CR_species_ID_in_bin[k]; temp_species_map[id_last+id_map_offset]=n00; n00++;}}
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
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {if(secondary_spec[j] > -100) {CR_secondary_species_listref[k][j] = temp_species_map[secondary_spec[j]+id_map_offset];} else {CR_secondary_species_listref[k][j]=-200;}}
    }
    /* also need to figure out the 'destination bin' for each type of secondary -- we will assume nucleon-nucleon conserves energy per nucleon, while cascades to positrons and secondary electrons are treated slightly differently */
    double E_GeV[N_CR_PARTICLE_BINS], A_wt[N_CR_PARTICLE_BINS]; for(k=0;k<N_CR_PARTICLE_BINS;k++) {E_GeV[k]=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k); A_wt[k]=return_CRbin_CRmass_in_mp(-1,k);}
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        int primary_id = CR_species_ID_in_bin[k];
        int primary_listref = temp_species_map[primary_id+id_map_offset];
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++)
        {
            int secondary_listref = CR_secondary_species_listref[primary_listref][j];
            if(secondary_listref <= -1) {CR_secondary_target_bin[k][j]=-2;}
            else {
                int secondary_id = species_list[secondary_listref];
                int m, target_bin=-1; double diff_min=MAX_REAL_NUMBER, E_target_0=E_GeV[k];
                for(m=0;m<N_CR_PARTICLE_BINS;m++)
                {
                    if(CR_species_ID_in_bin[m] != secondary_id) {continue;}
                    double E_target = E_GeV[k] * DMAX(1.,A_wt[m]) / DMAX(1.,A_wt[k]); // fixed energy per nucleon/particle (treating e-/e+ as 1)
                    if(secondary_id < 0) {E_target *= 0.14;} // secondary e+/e- from protons (pion decay) get ~1/8 original p energy, likewise for anti-protons
                    if(secondary_id == 7) {E_target *= 0.08;} // anti-protons similar but get slightly-less energy on average [testing effects right now of this level of hair-splitting!]
                    double diff = log(E_GeV[m]/E_target); diff*=diff; // square of log-diff between energies
                    if(diff < diff_min) {diff_min=diff; target_bin=m; E_target_0=E_target;} // set to this as the 'closest' option
                }
                CR_secondary_target_bin[k][j]=-1; // default to no secondary bin (secondary is 'lost')
                if(target_bin >= 0) // check if the bin is valid
                {
                    double E_target=E_target_0, E_bin=E_GeV[target_bin], E_bin_m=E_bin, E_bin_p=E_bin; // if decay to lower energy than we track bins, the products are gone from the spectrum we follow
                    if(target_bin>0) {if(CR_species_ID_in_bin[target_bin-1] == secondary_id) {E_bin_m=E_GeV[target_bin-1];}} // define the bin ranges that we will consider for whether the target energy fits
                    if(target_bin<N_CR_PARTICLE_BINS-1) {if(CR_species_ID_in_bin[target_bin+1] == secondary_id) {E_bin_p=E_GeV[target_bin+1];}}
                    E_bin_m=sqrt(E_bin_m*E_bin); E_bin_p=sqrt(E_bin_p*E_bin); if(E_bin_m==E_bin) {E_bin_m=E_bin*(E_bin/E_bin_p)*(E_bin/E_bin_p);} else if(E_bin_p==E_bin) {E_bin_p=E_bin*(E_bin/E_bin_m)*(E_bin/E_bin_m);}
                    if(E_target >= 0.9*E_bin_m && E_target <= 1.1*E_bin_p) {CR_secondary_target_bin[k][j] = target_bin;} // assign a destination bin
                }
            }
        }
    }
    /* now pre-calculate the fragmentation factors and radioactive factors, all static up to their nH dependence */
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        CR_frag_coeff[k]=0; CR_rad_decay_coeff[k]=0; double beta_fac=return_CRbin_beta_factor(-1,k), cx_mb_to_coeff=3.0e-17*beta_fac; // cx_mb_to_coeff = (millibarn x c_light) * beta in cgs, to convert units to get \dot[f] = sigma*v*n*f, so prefactor is this times n in cgs
        if(CR_species_ID_in_bin[k] == -2)
        {
            double gamma_fac=return_CRbin_gamma_factor(-1,k), gamma_positron=gamma_fac, gamma_minus_1=gamma_positron-1., fac=0; // Dirac expression below considers e- at rest, which is what we're interested in here since it's e+ CRs annihilating (gamma is the gamma of the positron)
            if(gamma_minus_1 > 1.e-2) {fac=((gamma_positron*gamma_positron+4.*gamma_positron+1.)*log(gamma_positron+sqrt(gamma_positron*gamma_positron-1.))/(gamma_positron*gamma_positron-1.) - (gamma_positron+3.)/sqrt(gamma_positron*gamma_positron-1.))/(gamma_positron+1.);} else {fac=1./sqrt(2.*DMAX(gamma_minus_1,1.e-8));} // Dirac expression
            CR_frag_coeff[k] = 7.479e-15 * beta_fac * fac; // e+ annihilation with ISM (rest) e- to gamma rays
        }
        if(CR_species_ID_in_bin[k] == 1) {if(E_GeV[k] > 0.28) {CR_frag_coeff[k] = 6.37e-16;}} // coefficient for hadronic/catastrophic interactions/pionic losses: dEtot/dt = -(coeff) *nnucleoncgs* Etot, or dPtot/dt = -(coeff)*nnucleoncgs* Ptot (since all p effected are in rel limit, and works by deleting N not by lowering individual E. Mannheim & Schlickeiser 1994
        if(CR_species_ID_in_bin[k] > 1) /* total fragmentation cross-section/rate, from Mannheim & Schlickeiser 1994: */
        {
            double sigma_frag_tot = 45. * pow(A_wt[k],0.7) * (1.+0.016*sin(1.3-2.63*log(A_wt[k])));
            if(E_GeV[k] < 2.0) {sigma_frag_tot *= (1.-0.62*exp(-E_GeV[k]/0.2)*sin(1.57553/pow(E_GeV[k],0.28)));}
            CR_frag_coeff[k] = cx_mb_to_coeff * sigma_frag_tot; // rate depends on 'v', need to include beta here
            if(CR_species_ID_in_bin[k] == 7) {double RGV=CR_global_rigidity_at_bin_center[k], lnR=log(RGV); CR_frag_coeff[k] = cx_mb_to_coeff * 1.5 * (-107.9 + 29.43*lnR - 1.655*lnR*lnR + 189.9/pow(RGV,1./3.));} // larger CX b/c this is anti-proton annihilation; factor ~1.5 accounts for sum of species heavier than H; from summation of cross-sections in Evoli et al. 2017 [arXiv:1711.09616]
        }
        for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {CR_frag_secondary_coeff[k][j]=0;} // initialize this to null
        if(CR_frag_coeff[k] > 0) {for(j=0;j<N_CR_PARTICLE_SPECIES;j++) {if(CR_secondary_target_bin[k][j] >= 0) // desired secondary products exist
            {
                double x=DMAX(-2.,DMIN(2.,log10(E_GeV[k]))); // used in some of the fitting functions below
                int primary_id = CR_species_ID_in_bin[k], secondary_id = CR_species_ID_in_bin[CR_secondary_target_bin[k][j]];
                //if((primary_id == 5) && (secondary_id == 4)) {} // Be10->B10 (radioactive - frag probability is null) //
                if(primary_id==1) { // p; assume some simple branching ratios for pion production in the relevant regime above
                    if(secondary_id==-1) {CR_frag_secondary_coeff[k][j] = (1./3.) * CR_frag_coeff[k];} // p->e-
                    if(secondary_id==-2) {CR_frag_secondary_coeff[k][j] = (1./3.) * CR_frag_coeff[k];} // p->e+
                    if(secondary_id== 7) {if(E_GeV[k]>=2.) {double sqrt_s=1.87654*sqrt(1.+E_GeV[k]/1.87654); CR_frag_secondary_coeff[k][j]=cx_mb_to_coeff * 1.4*pow(sqrt_s,0.6)*exp(-pow(17./sqrt_s,1.4));}} // p->pbar [anti-proton production; multiplied by 2x here to include anti-neutrons that decay rapidly to anti-p]. fit to integrated-over-pT results from Reinert & Winkler 2017, arXiv:1712.00002
                }
                if(primary_id==2) { // B; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==4) {CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * 12.0 * pow(E_GeV[k],-0.022);} // B->Be9 (declines weakly from ~14 to ~10 over MeV-TeV, approx here)
                    if(secondary_id==5) {CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * 12.5 * pow(E_GeV[k], 0.018);} // B->Be10 (increases weakly from 11 to 14 over MeV-TeV, approx here)
                }
                if(primary_id==3) { // C; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==2) { // C->B
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10.,1.88490356 -0.05648715*x -0.13108485*x*x +0.11340508*x*x*x +0.08119785*x*x*x*x -0.06574148*x*x*x*x*x -0.01159932*x*x*x*x*x*x +0.00962035*x*x*x*x*x*x*x +0.23395042856711876*exp(-10.8066678706701*(x+1.24740008)*(x+1.24740008)));
                    }
                    if(secondary_id==4) { // C->Be7+9
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10.,1.18321683 +0.11629153*x +0.01653084*x*x -0.11321897*x*x*x -0.03375688*x*x*x*x +0.05771537*x*x*x*x*x +0.00684962*x*x*x*x*x*x -0.00876353*x*x*x*x*x*x*x +0.40590023164209293*exp(-17.1253106513537*(x+1.28525319)*(x+1.28525319)));
                    }
                    if(secondary_id==5) { // C->Be10
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * (0.10757855 + pow(10., 0.53413363 +0.38478345*x -0.51584177*x*x -0.22611016*x*x*x +0.51009595*x*x*x*x +0.04492848*x*x*x*x*x -0.23828966*x*x*x*x*x*x +0.06889871*x*x*x*x*x*x*x));
                    }
                }
                if(primary_id==6) { // CNO effective bin; fitting function to the results compiled and re-fit in Moskalenko & Mashnik 2003 (used for GALPROP), doing a solar-abundance-ratio weighted average over CNO, and summing over the relevant isotopes
                    if(secondary_id==2) { // CNO->B
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10., 1.71801936 - 0.03475011*x -0.09856187*x*x +0.12369455*x*x*x +0.02958446*x*x*x*x -0.05273341*x*x*x*x*x -0.00223893*x*x*x*x*x*x +0.00639451*x*x*x*x*x*x*x + 0.464605776*exp(-17.769530*(x+1.23499649)*(x+1.23499649)) );}
                    if(secondary_id==4) { // CNO->Be7+9 // (could also use CNO->Be9 (not really a point here explicitly tracking Be7), but including all here since closer to what is actually observed)
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * pow(10., 1.16648147  +0.09557578*x -0.17970136*x*x +0.06823829*x*x*x +0.04448299*x*x*x*x -0.02883429*x*x*x*x*x -0.00274047*x*x*x*x*x*x +0.00286316*x*x*x*x*x*x*x + 0.476812578*exp(-14.203615*(x+1.21352966)*(x+1.21352966)) );}
                    if(secondary_id==5) { // CNO->Be10
                        CR_frag_secondary_coeff[k][j] = cx_mb_to_coeff * (0.073388 + pow(10., 0.454778746 + 0.349074384*x -0.684152925*x*x -0.153016497*x*x*x +0.657169204*x*x*x*x +8.06147155e-04*x*x*x*x*x -0.296932460*x*x*x*x*x*x +0.0918014184*x*x*x*x*x*x*x ));}
                }
            }}}
        
        double r_decay = 0; // default to assume no radioactive decay
        if(CR_species_ID_in_bin[k] == 5) {r_decay = 1.455e-14;} // Be10 -> B10
        if(r_decay > 0) {CR_rad_decay_coeff[k] = r_decay / return_CRbin_gamma_factor(-1,k);} // need to account for the fact that relativistic time dilation extends the lifetimes of highly relativistic sources
    }
#endif
}
#endif


/* routine which determines the fraction of injected CR energy per 'bin' of CR energy.
    so it can be used, we send the CR bin, the source type [0=SNe; 1=stellar winds; 2-4=unused now; 5=sink/AGN;],
    the target gas cell, the shock velocity (used to scale if desired),
    and a flag indicating the option to return the spectral slope/index */
double CR_energy_spectrum_injection_fraction(int k_CRegy, int source_type, double shock_vel, int return_index_in_bin, int target)
{
    double f_bin = 1./N_CR_PARTICLE_BINS; /* uniformly distributed */
#if (N_CR_PARTICLE_BINS > 1)    /* insert physics here */
#if (N_CR_PARTICLE_BINS == 2) /* one-bin protons, one electrons */
    double f_bin_v[2]={0.95 , 0.05}; f_bin=f_bin_v[k_CRegy]; // 5% of injection into e-, roughly motivated by observed spectra and nearby SNRs
#endif
#if (N_CR_PARTICLE_BINS > 2) /* multi-bin spectrum for p and e-: inset assumptions about injection spectrum here! */
    double f_elec = 0.02; // fraction of the energy to put into e- as opposed to p+ at injection [early experiments with 'observed'  fraction ~ 1% give lower e-/p+ actually observed in the end, so tentative favoring closer to equal at injection? but not run to z=0, so U_rad high from CMB; still experimenting here]
    double inj_slope = 4.25; // injection slope with j(p) ~ p^(-inj_slope), so dN/dp ~ p^(2-inj_slope)
    double R_break_e = 1.0; // location of spectral break for injection e- spectrum, in GV
    double inj_slope_lowE_e = 4.2; // injection slope with j(p) ~ p^(-inj_slope), so dN/dp ~ p^(2-inj_slope), for electrons below R_break_e
#if !defined(CRFLUID_ALT_RSOL_FORM)
    inj_slope_lowE_e=4.25; // slightly better fit with this scheme, though a bit marginal compared to using the above defaults (also tried R_break_e=2.0; inj_slope=4.30; , similar, but no significant improvement)
#endif
    double R=return_CRbin_CR_rigidity_in_GV(-1,k_CRegy); int species=return_CRbin_CR_species_ID(k_CRegy); // get bin-centered R and species type
    //if(species < 0 && R < R_break_e) {inj_slope = inj_slope_lowE_e;} // follow model injection spectra favored in Strong et al. 2011 (A+A, 534, A54), who argue the low-energy e- injection spectrum must break to a lower slope by ~1 independent of propagation and re-acceleration model
    if(species > -200 && R < R_break_e) {inj_slope = inj_slope_lowE_e;} // follow model injection spectra favored in Strong et al. 2011 (A+A, 534, A54), who argue the low-energy e- injection spectrum must break to a lower slope by ~1 independent of propagation and re-acceleration model
    double EGeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy); // get bin-centered E_GeV for normalizing total energy in bin
    f_bin = EGeV * pow(R/R_break_e , 3.-inj_slope) * log(CR_global_max_rigidity_in_bin[k_CRegy] / CR_global_min_rigidity_in_bin[k_CRegy]); // normalize accounting for slope, isotropic spectrum, logarithmic bin width [which can vary], and energy per N

    if(return_index_in_bin) {return 2.-inj_slope;} // this is the index corresponding to our dN/dp ~ p^gamma
    double f_norm = 1.e-20; // default to very little energy
    if(species == -1) {f_norm = f_elec;} // e-
    if(species == +1) {f_norm = 1.-f_elec;} // p
    if(species == -2) {f_norm = 1.e-10 * f_elec;} // e+ (assuming negligible e+ injection to start)
    if(species > 1 && species != 7) // heavy elements need to scale injection rates appropriately
    {
        double Zfac=0, Zfac_ISM=P[target].Metallicity[0]/All.SolarAbundances[0], mu_wt=return_CRbin_CRmass_in_mp(-1,k_CRegy), Z_cr=fabs(return_CRbin_CR_charge_in_e(-1,k_CRegy)), Mism_over_Mej=1; // scale heavier elements to the metallicity of the gas into which CRs are being accelerated
        Zfac = Zfac_ISM; // assume abundance of ejecta is identical to ambient ISM into which its being ejected
        if(source_type == 1) {Zfac = (Mism_over_Mej*Zfac_ISM + 1.4*DMIN(Zfac_ISM,1.))/(Mism_over_Mej*HYDROGEN_MASSFRAC + HYDROGEN_MASSFRAC);} // stellar outflows. using FIRE-3 yields this is exact after IMF-integrating for Z_ism=Z_star, for CNO; basically get slight enhancement, but not much, b/c these are OB winds; even including AGB, would only move factor to 3.4 from 1.4, which is halved, so not much effect at all.
        if(source_type == 0) {if(shock_vel>5000./UNIT_VEL_IN_KMS) {Zfac=(Mism_over_Mej*Zfac_ISM + 9.70)/(Mism_over_Mej*HYDROGEN_MASSFRAC + 0.025);} else {Zfac=(Mism_over_Mej*Zfac_ISM + 13.645)/(Mism_over_Mej*HYDROGEN_MASSFRAC + 0.441);}} // shock_vel here tells us if its a 1a [faster] or CCSNe. for CCSNe, use Iwamoto 1999 and Nomoto 2006 to get abundances of ejecta in CNO, integrated over IMF and species of interest. for both, assume acceleration efficiency is maximized at highest mach numbers after shock actually develops, so swept-up ISM mass is ~ejecta mass
        // now scale to mass fraction for solar abundances which gives the units we work with above
        if(species == 2) {Zfac *= 3.7e-9;} // B (for standard elements initialize to solar ratios assuming similar energy/nucleon)
        if(species == 3) {Zfac *= 2.4e-3;} // C
        if(species == 4) {Zfac *= 1.4e-10;} // Be7+9 (stable)
        if(species == 5) {Zfac *= 1.4e-20;} // Be10 (radioactive)
        if(species == 6) {Zfac *= 0.0094;} // CNO (combined bin)
        f_norm = Zfac * pow(mu_wt/Z_cr , inj_slope-3.) / mu_wt; // approximate injection factor for a constant-beta distribution at a given R_GV needed below
    }
    f_bin *= f_norm; // normalize injection depending on the species (e- or p+, etc)
#endif
#endif
    if(return_index_in_bin) {return 0;}
    return f_bin;
}


/* routine which gives diffusion coefficient as a function of energy for the 'constant diffusion coefficient' models:
    current default: -extremely- simple power-law, assuming diffusion coefficient increases with CR energy per unit charge as (E/Z)^(1/2) */
double diffusion_coefficient_constant(int target, int k_CRegy)
{
    double dimensionless_kappa_relative_to_GV_protons = 1;
#if (N_CR_PARTICLE_BINS > 1)    /* insert physics here */
    int target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
    dimensionless_kappa_relative_to_GV_protons = return_CRbin_beta_factor(target_bin_centering_for_CR_quantities,k_CRegy) * pow( CR_global_min_rigidity_in_bin[k_CRegy]*CR_global_max_rigidity_in_bin[k_CRegy] , 0.5 * 0.6 ); // assume a quasi-empirical scaling here, and for these correction terms its important that the 'bin center' being used for the zero point here is the geometric mean of the bin edges, hence the 0.5 term b/c geometric mean is sqrt[min*max] //
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
    double vA_noion, double rho_cgs, double temperature, double cs_thermal, double nh0, double nHe0, double f_ion)
{
    double vol_inv = CellP[target].Density*All.cf_a3inv / P[target].Mass, fturb_multiplier=1, f_QLT=1, R_CR_GV, Z_charge_CR, M_cr_mp; Vec3<double> b0={}, p0={};
    int target_bin_centering_for_CR_quantities = target; // if this = target, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
    target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
    R_CR_GV=return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy); Z_charge_CR=return_CRbin_CR_charge_in_e(target,k_CRegy); M_cr_mp=return_CRbin_CRmass_in_mp(target,k_CRegy);
    int k; double n_cgs=rho_cgs/PROTONMASS_CGS, EPSILON_SMALL=1.e-50, e_CR=0, e_B=0, bhat_dot_CR_Pgrad=0, B2=0;
#ifdef MAGNETIC
    for(k=0;k<3;k++) {b0[k]=CellP[target].BPred[k]*vol_inv*All.cf_a2inv;}
    B2=b0.norm_sq(); e_B=0.5*B2; B2=1./sqrt(B2+MIN_REAL_NUMBER); b0*=B2; // calculate B-field energy and bhat vector
#else
    for(k=0;k<3;k++) {b0[k]=CellP[target].Gradients.CosmicRayPressure[k_CRegy][k];} // just assume equilibrium here, convert to physical units, pick arbitrary direction here //
    B2=b0.norm_sq(); e_B=CellP[target].Pressure*All.cf_a3inv; b0/=sqrt(B2+MIN_REAL_NUMBER); // calculate B-field energy and bhat vector
#endif
#ifdef CRFLUID_EVOLVE_SPECTRUM
    double R0_m=CR_global_min_rigidity_in_bin[k_CRegy], R0_p=CR_global_max_rigidity_in_bin[k_CRegy];  // e_CR is in bin, but doesn't take account of bin width; needed only for NLL, want to include CRs 'close' to energy but not all b/c non-resonant, but don't want bin-size dependent, so replace with ~constant * de_cr / dlnR, which works pretty well
    double fac_dXdlnR = 1.0 / log(R0_p/R0_m); // factor to multiply by to account for 'width' of gyro-resonance
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {
        double R0_k=CR_global_rigidity_at_bin_center[k];
        if(R0_k>R0_m && R0_k<R0_p) {
            e_CR += CellP[target].CosmicRayEnergyPred[k] * fac_dXdlnR * vol_inv; // convert to energy density units
            int mk; for(mk=0;mk<3;mk++) {p0[mk] += CellP[target].Gradients.CosmicRayPressure[k][mk] * fac_dXdlnR * All.cf_a3inv/All.cf_atime;} // convert to appropriate physical units
        }} // sum over all bins with their rigidity in the same range, so that we can get a total energy (dominated by p, but should include all for safety)
#else
    e_CR=CellP[target].CosmicRayEnergyPred[k_CRegy]*vol_inv; for(k=0;k<3;k++) {p0[k]=CellP[target].Gradients.CosmicRayPressure[k_CRegy][k]*All.cf_a3inv/All.cf_atime;}
#endif
    bhat_dot_CR_Pgrad = dot(b0, p0); // dot product of bhat and CR pressure gradient, summed over relevant bins
    double beta=return_CRbin_beta_factor(target_bin_centering_for_CR_quantities,k_CRegy), Omega_gyro=beta*(0.00898734*b_muG/R_CR_GV) * UNIT_TIME_IN_CGS, r_L=beta*C_LIGHT_CODE/Omega_gyro, kappa_0=r_L*beta*C_LIGHT_CODE; /* all in physical -code- units */
    double x_LL = DMAX( r_L / L_scale, EPSILON_SMALL ), vA_code=Get_Gas_ion_Alfven_speed_i(target), k_turb=1./L_scale, k_L=1./r_L;

    if(mode==1) {f_QLT = 100;} // multiplier to account for arbitrary deviation from QLT, applies to all damping mechanisms [100 = favored value in our study; or could use fcas = 100]
    fturb_multiplier = pow(M_A,3./2.); // multiplier to account for different turbulent cascade models (fcas = 1)
    if(mode==2) {fturb_multiplier *= 100.;} // arbitrary multiplier (fcas = 100, here)
    if(mode==3) {fturb_multiplier = pow(M_A,3./2.) * 1./(pow(M_A,1./2.)*pow(x_LL,1./6.));} // pure-Kolmogorov (fcas-K41)
    if(mode==4) {fturb_multiplier = pow(M_A,3./2.) / pow(x_LL,1./10.);} // GS anisotropic but perp cascade is IK (fcas-IK) /

    /* ok now we finally have all the terms needed to calculate the various damping rates that determine the equilibrium diffusivity */
    double U0bar_grain=3., rhograin_int_cgs=1., fac_grain=R_CR_GV*sqrt(n_cgs)*U0bar_grain/(b_muG*rhograin_int_cgs), f_grainsize = DMAX(8.e-4*pow(fac_grain*(temperature/1.e4),0.25), 3.e-3*sqrt(fac_grain)), Z_sol=1.; // b=2, uniform logarithmic grain spectrum over a factor of ~100 in grain size; f_grainsize = 0.07*pow(sqrt(fion*n1)*EcrGeV*T4/BmuG,0.25); // MRN size spectrum
#ifdef METALS
    Z_sol = P[target].Metallicity[0]/0.014;
#endif
    double G_dust = vA_code*k_L * Z_sol * f_grainsize; // also can increase by up to a factor of 2 for regimes where charge collisionally saturated, though this is unlikely to be realized
    double RGV_dust_crit = 0.01 * (vA_code*UNIT_VEL_IN_KMS) / (1. + pow(cs_thermal*UNIT_VEL_IN_KMS/8.1,2)); // critical rigidity below which there are no gyro-resonant dust grains expected with >10 nm sizes
    if(R_CR_GV < RGV_dust_crit) {G_dust *= pow(R_CR_GV/RGV_dust_crit, 2);} // suppression factor for the dust-damping term when outside of this relevant rigidity range (thanks to Margot Fitz Axen for helping identify regimes where this would kick in which we were simulating in GMCs)
    if(mode<0) {G_dust = 0;} // for this choice, neglect the dust-damping term
    double G_ion_neutral = (5.77e-11 * n_cgs * (0.97*nh0 + 0.03*nHe0) * sqrt(temperature)) * UNIT_TIME_IN_CGS / sqrt(M_cr_mp); // ion-neutral damping: need to get thermodynamic quantities [neutral fraction, temperature in Kelvin] to compute here -- // G_ion_neutral = (xiH + xiHe); // xiH = nH * siH * sqrt[(32/9pi) *kB*T*mH/(mi*(mi+mH))]
    double fac_turb = sqrt(k_turb*k_L) * fturb_multiplier; // factor to use below
    double G_turb_plus_linear_landau = (vA_noion + sqrt(M_PI)*cs_thermal/4.) * fac_turb; // linear Landau + turbulent (both have same form, assume k_turb from cascade above)
    double G_adiabatic = 0.5*P[target].Particle_DivVel*All.cf_a2inv; // adiabatic term [signed like the other linear terms
    double Gamma_NLL = (sqrt(M_PI)/8.) * cs_thermal / r_L; // NLL prefactor: Gamma_NLL is this times (e_A/e_B)
    double f_cas_ET = 7. * (vA_noion / (beta * C_LIGHT_CODE)) * log(1.+L_scale/r_L); /* Alfvenic turbulence following an anisotropic Goldreich-Shridar cascade, per Chandran 2000 */
    double S_ext_turb = f_cas_ET * vA_noion * fac_turb * M_A*M_A * (r_L/L_scale); // extrinsic turbulence cascade term. note consistency means multiplying by pitch-angle and gyro-averaging factors, and fcas above; expression here assumes whatever you do its a balanced cascade (defauly to GS95 scalings)
    double S_ext_gri  = vA_code * fabs(bhat_dot_CR_Pgrad) / (e_B + MIN_REAL_NUMBER); // Flux-steady-state value of the GRI term, normalized to e_B. note unless we rewrite this to the 5th-order polynomial version, assumption here is that return_CRbin_nuplusminus_asymmetry(i,k_CRegy) -> 1 for the term here in steady state
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
    double vA_noion, double rho_cgs, double temperature, double cs_thermal, double nh0, double nHe0, double f_ion)
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
void CR_cooling_and_losses(int target, double n_elec, double nHcgs, double dtime_cgs)
{
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    CR_cooling_and_losses_multibin(target, n_elec, nHcgs, dtime_cgs, 0); // call the multi-binned version of this routine, passing the relevant parameters, and exit
    return;
#endif

    if(dtime_cgs <= 0) {return;} /* catch */
    int k_CRegy; double f_ion=DMAX(DMIN(Get_Gas_Ionized_Fraction(target),1.),0.);
    double a_hadronic = 6.37e-16, b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*(1.-f_ion))*HYDROGEN_MASSFRAC; /* some coefficients; a_hadronic is the default coefficient, b_coulomb_ion_per_GeV the default Coulomb+ionization (the two scale nearly-identically) normalization divided by GeV, b/c we need to divide the energy per CR  */
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double CR_coolrate,Z,species_ID; CR_coolrate=0; Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy)); species_ID=return_CRbin_CR_species_ID(k_CRegy);
        if(species_ID > 0) /* protons and nuclei here */
        {
#if (N_CR_PARTICLE_BINS > 2) /* note these are currently energy-loss expressions; for truly multi-bin, probably better to work with dp/dt, instead of dE/dt */
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), beta=return_CRbin_beta_factor(target,k_CRegy);
            CR_coolrate += b_coulomb_ion_per_GeV * ((Z*Z)/(beta*E_GeV)) * nHcgs; // all protons Coulomb-interact, can be rapid for low-E
            if(E_GeV>=0.28) {CR_coolrate += a_hadronic * nHcgs;} // only GeV CRs or higher trigger above threshold for collisions
#else
            CR_coolrate = (0.87*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * nHcgs; /* for N<=2, assume a universal spectral shape, the factor here corrects for the fraction above-threshold for hadronic interactions, and 0.53 likewise for averaging  */
#endif
        } else { /* electrons here: note for electrons and positrons, always in the relativistic limit, don't need to worry about beta << 1 limits */
            /* bremsstrahlung [following Blumenthal & Gould, 1970]: dEkin/dt=4*alpha_finestruct*r_classical_elec^2*c * SUM[n_Z,ion * Z * (Z+1) * (ln[2*gamma_elec]-1/3) * E_kin */
            double E_GeV=return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), E_rest=0.000511, gamma=1.+E_GeV/E_rest;
            CR_coolrate += n_elec * nHcgs * 1.39e-16 * DMAX(log(2.*gamma)-0.33,0);
            /* synchrotron and inverse compton scale as dE/dt=(4/3)*sigma_Thompson*c*gamma_elec^2*(U_mag+U_rad), where U_mag and U_rad are the magnetic and radiation energy densities, respectively. Ignoring Klein-Nishina corrections here, as they are negligible at <40 GeV and only a ~15% correction up to ~1e5 GeV */
            double b_muG = get_cell_Bfield_in_microGauss(target), U_mag_ev=0.0248342*b_muG*b_muG, U_rad_ev = get_cell_Urad_in_eVcm3(target);
            CR_coolrate += 5.2e-20 * gamma * (U_mag_ev + U_rad_ev); // U_mag_ev=(B^2/8pi)/(eV/cm^(-3)), here; U_rad=U_rad/(eV/cm^-3) //
        }
        
        /* here, cooling is being treated as energy loss 'within the bin'. with denser bins, should allow for movement -between- bins, using the spectral method below */
        CR_coolrate *= CosmicRayFluid_RSOL_Corrfac(k_CRegy); // account for RSOL terms as needed
        double q_CR_cool = exp(-CR_coolrate * dtime_cgs); if(CR_coolrate * dtime_cgs > 20.) {q_CR_cool = 0;}
        CellP[target].CosmicRayEnergyPred[k_CRegy] *= q_CR_cool; CellP[target].CosmicRayEnergy[k_CRegy] *= q_CR_cool;
        int k; for(k=0;k<3;k++) {CellP[target].CosmicRayFlux[k_CRegy][k] *= q_CR_cool; CellP[target].CosmicRayFluxPred[k_CRegy][k] *= q_CR_cool;}
    }
    return;
}

                                                                                                                              

/* utility to estimate -locally- (without multi-pass filtering) the local Alfven Mach number */
double Get_AlfvenMachNumber_Local(int i, double vA_idealMHD_codeunits, int use_shear_corrected_vturb_flag)
{
    int i1,i2; double v2_t=0,dv2_t=0,b2_t=0,db2_t=0,M_A,h0,EPSILON_SMALL=1.e-50; // factor which will represent which cascade model we are going to use
    for(i1=0;i1<3;i1++)
    {
        v2_t += CellP[i].VelPred[i1]*CellP[i].VelPred[i1];
        for(i2=0;i2<3;i2++) {dv2_t += CellP[i].Gradients.Velocity[i1][i2]*CellP[i].Gradients.Velocity[i1][i2];}
#ifdef MAGNETIC
        b2_t += Get_Gas_BField(i,i1) * Get_Gas_BField(i,i1);
        for(i2=0;i2<3;i2++) {db2_t += CellP[i].Gradients.B[i1][i2]*CellP[i].Gradients.B[i1][i2];}
#endif
    }
    v2_t=sqrt(v2_t); b2_t=sqrt(b2_t); dv2_t=sqrt(dv2_t); db2_t=sqrt(db2_t); dv2_t/=All.cf_atime; db2_t/=All.cf_atime; b2_t*=All.cf_a2inv; db2_t*=All.cf_a2inv; v2_t/=All.cf_atime; dv2_t/=All.cf_atime;
    h0=Get_Particle_Size(i)*All.cf_atime; // physical units

    if(use_shear_corrected_vturb_flag == 1)
    {
        dv2_t = sqrt((1./2.)*((CellP[i].Gradients.Velocity[1][0]+CellP[i].Gradients.Velocity[0][1]) *
            (CellP[i].Gradients.Velocity[1][0]+CellP[i].Gradients.Velocity[0][1]) + (CellP[i].Gradients.Velocity[2][0]+CellP[i].Gradients.Velocity[0][2]) *
            (CellP[i].Gradients.Velocity[2][0]+CellP[i].Gradients.Velocity[0][2]) + (CellP[i].Gradients.Velocity[2][1]+CellP[i].Gradients.Velocity[1][2]) * (CellP[i].Gradients.Velocity[2][1]+CellP[i].Gradients.Velocity[1][2])) +
            (2./3.)*((CellP[i].Gradients.Velocity[0][0]*CellP[i].Gradients.Velocity[0][0] + CellP[i].Gradients.Velocity[1][1]*CellP[i].Gradients.Velocity[1][1] +
            CellP[i].Gradients.Velocity[2][2]*CellP[i].Gradients.Velocity[2][2]) - (CellP[i].Gradients.Velocity[1][1]*CellP[i].Gradients.Velocity[2][2] + CellP[i].Gradients.Velocity[0][0]*CellP[i].Gradients.Velocity[1][1] +
            CellP[i].Gradients.Velocity[0][0]*CellP[i].Gradients.Velocity[2][2]))) * All.cf_a2inv;
#ifdef MAGNETIC
        db2_t = sqrt((1./2.)*((CellP[i].Gradients.B[1][0]+CellP[i].Gradients.B[0][1]) * (CellP[i].Gradients.B[1][0]+CellP[i].Gradients.B[0][1]) +
            (CellP[i].Gradients.B[2][0]+CellP[i].Gradients.B[0][2]) * (CellP[i].Gradients.B[2][0]+CellP[i].Gradients.B[0][2]) +
            (CellP[i].Gradients.B[2][1]+CellP[i].Gradients.B[1][2]) * (CellP[i].Gradients.B[2][1]+CellP[i].Gradients.B[1][2])) +
            (2./3.)*((CellP[i].Gradients.B[0][0]*CellP[i].Gradients.B[0][0] + CellP[i].Gradients.B[1][1]*CellP[i].Gradients.B[1][1] +
            CellP[i].Gradients.B[2][2]*CellP[i].Gradients.B[2][2]) - (CellP[i].Gradients.B[1][1]*CellP[i].Gradients.B[2][2] +
            CellP[i].Gradients.B[0][0]*CellP[i].Gradients.B[1][1] + CellP[i].Gradients.B[0][0]*CellP[i].Gradients.B[2][2]))) * All.cf_a3inv;
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
void CalculateAndAssign_CosmicRay_DiffusionAndStreamingCoefficients(int i)
{
    /* first define some very general variables, and calculate some useful quantities that will be used for any model */
    int k_CRegy; double DiffusionCoeff, CR_kappa_streaming, CRPressureGradScaleLength, v_streaming;
#if (CRFLUID_DIFFUSION_MODEL > 0)
    double cs_thermal,M_A,L_scale,vA_code,vA_noion,gizmo2gauss,Omega_per_GeV_ifveqc,Bmag,unit_kappa_code,b_muG,E_B,f_ion,temperature,EPSILON_SMALL; int k; k=0;
    unit_kappa_code=UNIT_VEL_IN_CGS*UNIT_LENGTH_IN_CGS; gizmo2gauss=UNIT_B_IN_GAUSS; f_ion=1; temperature=0; EPSILON_SMALL=1.e-50;
    Bmag=2.*CellP[i].Pressure*All.cf_a3inv; cs_thermal=sqrt(convert_internalenergy_soundspeed2(i,CellP[i].InternalEnergyPred)); /* quick thermal pressure properties (we'll assume beta=1 if MHD not enabled) */
#ifdef MAGNETIC /* get actual B-field */
    Vec3<double> B={}; Bmag=0; for(k=0;k<3;k++) {B[k]=Get_Gas_BField(i,k)*All.cf_a2inv;} Bmag=B.norm_sq(); // B-field in code units (physical)
#endif
    Bmag=sqrt(DMAX(Bmag,0)); b_muG=Bmag*gizmo2gauss/1.e-6; b_muG=sqrt(b_muG*b_muG + 1.e-6); vA_code=sqrt(Bmag*Bmag/(CellP[i].Density*All.cf_a3inv)); vA_noion=vA_code; E_B=0.5*Bmag*Bmag*(P[i].Mass/(CellP[i].Density*All.cf_a3inv));
    Omega_per_GeV_ifveqc=(0.00898734*b_muG) * UNIT_TIME_IN_CGS; /* B-field in units of physical microGauss; set a floor at nanoGauss level. convert to physical code units */
#ifdef COOLING
    double ne=1, nh0=0, nHe0=0, nHepp, nhp, nHeII, mu_meanwt=1, rho=CellP[i].Density*All.cf_a3inv, rho_cgs, u0=CellP[i].InternalEnergyPred;
    temperature = ThermalProperties(u0, rho, i, &mu_meanwt, &ne, &nh0, &nhp, &nHe0, &nHeII, &nHepp); rho_cgs=rho*UNIT_DENSITY_IN_CGS; // get thermodynamic properties
    f_ion = DMIN(DMAX(DMAX(DMAX(1-nh0, nhp), ne/1.2), 1.e-8), 1.); // account for different measures above (assuming primordial composition)
#endif
    M_A = Get_AlfvenMachNumber_Local(i,vA_noion,0); /* get turbulent Alfven Mach number estimate. 0 or 1 to turn on shear-correction */
    L_scale = Get_Particle_Size(i)*All.cf_atime; /* define turbulent scales [estimation of M_A defined by reference to this scale */
#endif
    
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        v_streaming=0; DiffusionCoeff=0; CR_kappa_streaming=0; CRPressureGradScaleLength=Get_CosmicRayGradientLength(i,k_CRegy); /* set these for the bin as we get started */
#if (CRFLUID_DIFFUSION_MODEL == 0) /* set diffusivity to a universal power-law scaling (constant per-bin)  */
        DiffusionCoeff = diffusion_coefficient_constant(i,k_CRegy); //  this is the input value of the diffusivity, for constant-kappa models
#endif
#if (CRFLUID_DIFFUSION_MODEL == 8) /* set diffusivity to a universal power-law scaling (constant per-bin), plus constant-streaming-speed correction term as implied by some CGM observations  */
        DiffusionCoeff = diffusion_coefficient_constant(i,k_CRegy); //  this is the input value of the diffusivity, for constant-kappa models
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
        double ni_m3=f_ion*(rho_cgs/PROTONMASS_CGS)/1.e-3, T6=temperature/1.e6, Lturbkpc=L_scale*UNIT_LENGTH_IN_KPC, Lgradkpc=CRPressureGradScaleLength*UNIT_LENGTH_IN_KPC, h0_fac=0.1*Get_Particle_Size(i)*All.cf_atime*All.cf_a2inv*UNIT_VEL_IN_KMS, dv2_10=CellP[i].Gradients.Velocity.frobenius_norm_sq()*h0_fac*h0_fac;
        double ecr_14 = CellP[i].CosmicRayEnergyPred[k_CRegy] * (CellP[i].Density*All.cf_a3inv/P[i].Mass) * UNIT_PRESSURE_IN_CGS / 1.0e-14; // CR energy density in CGS units //
        v_streaming = Get_Gas_ion_Alfven_speed_i(i);
        CR_kappa_streaming = GAMMA_COSMICRAY(k_CRegy) * CRPressureGradScaleLength * (v_streaming + (1./UNIT_VEL_IN_KMS)*(4.1*pow(MIN_REAL_NUMBER+ni_m3*T6,0.25)/pow(MIN_REAL_NUMBER+ecr_14*Lgradkpc,0.5) + 1.2*pow(MIN_REAL_NUMBER+dv2_10*ni_m3,0.75)/(MIN_REAL_NUMBER+ecr_14*sqrt(Lturbkpc)))); // convert to effective diffusivity
#endif
#if (CRFLUID_DIFFUSION_MODEL == 5) /* streaming at fast MHD wavespeed [just to see what it does] */
        v_streaming = Get_Gas_Fast_MHD_wavespeed_i(i);
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
        DiffusionCoeff = diffusion_coefficient_extrinsic_turbulence(scatter_modes,i,k_CRegy,M_A,L_scale,b_muG,vA_noion,rho_cgs,temperature,cs_thermal,nh0,nHe0,f_ion) / unit_kappa_code;
#endif
#if (CRFLUID_DIFFUSION_MODEL == 6) || (CRFLUID_DIFFUSION_MODEL == 7) /* self-confinement-based diffusivity */
        int target_bin_centering_for_CR_quantities = i; // if this = i, evaluate quantities like R_GV at the CR-energy weighted mean of the bin, if =-1, evaluate them at the bin center instead: important for some subtle effects especially if using numerical derivatives for correction terms
        target_bin_centering_for_CR_quantities = -1; // the correction terms depend on these being evaluated at their bin-centered locations
        double Omega_gyro_ifveqc=(0.00898734*b_muG/return_CRbin_CR_rigidity_in_GV(target_bin_centering_for_CR_quantities,k_CRegy)) * UNIT_TIME_IN_CGS, r_L=C_LIGHT_CODE/Omega_gyro_ifveqc, kappa_0=r_L*C_LIGHT_CODE; // some handy numbers for limiting extreme-kappa below. all in -physical- code units //
        CR_kappa_streaming = diffusion_coefficient_self_confinement(CRFLUID_SET_SC_MODEL,i,k_CRegy,M_A,L_scale,b_muG,vA_noion,rho_cgs,temperature,cs_thermal,nh0,nHe0,f_ion) / unit_kappa_code;
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
        CellP[i].CosmicRayDiffusionCoeff[k_CRegy] = DiffusionCoeff; /* final assignment! */
    } // end CR bin loop
}



/* utility routine which handles the numerically-necessary parts of the CR 'injection' for you; here 'injection_velocity' should be in physical (not comoving) units */
void inject_cosmic_rays(double CR_energy_to_inject, double injection_velocity, int source_type, int target, double *dir)
{
    if(CR_energy_to_inject <= 0) {return;}
    double f_injected[N_CR_PARTICLE_BINS]; f_injected[0]=1; int k_CRegy;
#if (N_CR_PARTICLE_BINS > 1) /* add a couple steps to make sure injected energy is always normalized properly! */
    double sum_in=0.0; for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,0,target); sum_in+=f_injected[k_CRegy];}
    if(sum_in>0.0) {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]/=sum_in;}} else {for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++) {f_injected[k_CRegy]=1./N_CR_PARTICLE_BINS;}}
#endif
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double dEcr = evaluate_cr_transport_reductionfactor(target, k_CRegy, 0) * CR_energy_to_inject * f_injected[k_CRegy]; // normalized properly to sum to unity, and account for RSOL in injection rate [akin to RHD treatment]
        if(dEcr <= 0) {continue;}
#if defined(CRFLUID_EVOLVE_SPECTRUM) // update the evolved slopes with the injection spectrum slope: do a simple energy-weighted mean for the updated/mixed slope here
        double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), egy_slopemode = 1, xm = CR_global_min_rigidity_in_bin[k_CRegy] / CR_global_rigidity_at_bin_center[k_CRegy], xp = CR_global_max_rigidity_in_bin[k_CRegy] / CR_global_rigidity_at_bin_center[k_CRegy], xm_e=xm, xp_e=xp; // values needed for bin injection parameters
        if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {egy_slopemode=2; xm_e=xm*xm; xp_e=xp*xp;} // values needed to scale from slope injected to number and back
        double slope_inj = CR_energy_spectrum_injection_fraction(k_CRegy,source_type,injection_velocity,1,target); // spectral slope of injected CRs
#if defined(CRFLUID_ALT_RSOL_FORM) && defined(CRFLUID_ALT_VARIABLE_RSOL) // want to correct injection slope if we're modulating injection with a variable Psi-type rsol function or variable-rsol
        if(k_CRegy>0) {int spec_0=return_CRbin_CR_species_ID(k_CRegy), spec_m=return_CRbin_CR_species_ID(k_CRegy-1), spec_p=-200; if(spec_m==spec_0) {
            double rfac_0=evaluate_cr_transport_reductionfactor(target,k_CRegy,0), rfac_m=evaluate_cr_transport_reductionfactor(target,k_CRegy-1,0), rfac_p=rfac_0, R_0=CR_global_rigidity_at_bin_center[k_CRegy], R_m=CR_global_rigidity_at_bin_center[k_CRegy-1], R_p=R_0;
            if(k_CRegy<N_CR_PARTICLE_BINS) {spec_p=return_CRbin_CR_species_ID(k_CRegy+1);}
            if(spec_p==spec_0) {rfac_p=evaluate_cr_transport_reductionfactor(target,k_CRegy+1,0); R_p=CR_global_rigidity_at_bin_center[k_CRegy+1];
                double xm=log(R_m/R_0),xp=log(R_p/R_0),qm=log(rfac_m/rfac_0),qp=log(rfac_p/rfac_0); slope_inj += (qm*xm + qp*xp) / (xm*xm + xp*xp);} else {slope_inj += log(rfac_0/rfac_m) / log(R_0/R_m);}}} // not clear if actually improves accuracy by substantial margin here, vs letting code self-adjust in next timestep
#endif
        double gamma_one = slope_inj + 1., xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one); // variables below
        double ntot_inj = (dEcr / E_GeV) * ((gamma_one + egy_slopemode) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp_e - xm_gamma_one*xm_e); // injected number in bin
        #pragma omp atomic
        CellP[target].CosmicRay_Number_in_Bin[k_CRegy] += ntot_inj; // simply update injected number. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
#endif
        #pragma omp atomic
        CellP[target].CosmicRayEnergy[k_CRegy] += dEcr; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        #pragma omp atomic
        CellP[target].CosmicRayEnergyPred[k_CRegy] += dEcr; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        double dir_mag=0, flux_mag=dEcr * CRFLUID_REDUCED_C_CODE(k_CRegy); Vec3<double> dir_to_use={}; int k;
#ifdef MAGNETIC
        Vec3<double> Bdir={}; for(k=0;k<3;k++) {Bdir[k]=CellP[target].BPred[k];}
        double B_dot_dir=0; for(k=0;k<3;k++) {B_dot_dir+=dir[k]*Bdir[k];} // the 'default' direction is projected onto B
        dir_to_use = B_dot_dir * Bdir; // launch -along- B, projected [with sign determined] by the intially-desired direction
#else
        for(k=0;k<3;k++) {dir_to_use[k]=dir[k];} // launch in the 'default' direction
        dir_mag = dir_to_use.norm_sq();
        if(dir_mag <= 0) {dir_to_use[0]=0; dir_to_use[1]=0; dir_to_use[2]=1; dir_mag=1;}
        for(k=0;k<3;k++) {
            double dflux=flux_mag*dir_to_use[k]/sqrt(dir_mag);
            #pragma omp atomic
            CellP[target].CosmicRayFlux[k_CRegy][k]+=dflux; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
            #pragma omp atomic
            CellP[target].CosmicRayFluxPred[k_CRegy][k]+=dflux; // update injected CR energy. needs to be done thread-safely, but since the above routines dont depend on this, it should be safe to do here.
        }
#endif
    }
    return;
}



/* return CR pressure within a given bin */
double INLINE_FUNC Get_Gas_CosmicRayPressure(int i, int k_CRegy)
{
    if((P[i].Mass > 0) && (CellP[i].Density>0) && (CellP[i].CosmicRayEnergyPred[k_CRegy] > 0))
    {
        return (GAMMA_COSMICRAY(k_CRegy)-1.) * (CellP[i].CosmicRayEnergyPred[k_CRegy] * CellP[i].Density) / P[i].Mass; // cosmic ray pressure = (4/3-1) * e_cr = 1/3 * (E_cr/Vol) //
    } else {return 0;}
}



/* return CR gradient scale length, with various physical limiters applied: not intended for pure numerical gradient-length calculations (where units dont matter), but for preventing some unphysical situations */
double Get_CosmicRayGradientLength(int i, int k_CRegy)
{
    /* now we need the -parallel- cosmic ray pressure or energy density scale length */
    double CRPressureGradMag = 0.0;
    int k; for(k=0;k<3;k++) {CRPressureGradMag += CellP[i].Gradients.CosmicRayPressure[k_CRegy][k]*CellP[i].Gradients.CosmicRayPressure[k_CRegy][k];}
    CRPressureGradMag = sqrt(1.e-46 + CRPressureGradMag); // sqrt to make absolute value
#ifdef MAGNETIC /* with anisotropic transport, we really want the -parallel- gradient scale-length, so need another factor here */
    double B2_tot=0.0, b=0; CRPressureGradMag=0; for(k=0;k<3;k++) {b=Get_Gas_BField(i,k); B2_tot+=b*b; CRPressureGradMag+=b*CellP[i].Gradients.CosmicRayPressure[k_CRegy][k];} // note, this is signed!
    CRPressureGradMag = sqrt((1.e-40 + CRPressureGradMag*CRPressureGradMag) / (1.e-46 + B2_tot)); // divide B-magnitude to get scalar magnitude, and take sqrt[(G.P)^2] to get absolute value
#endif
    
    /* limit the scale length: if too sharp, need a slope limiter at around the particle size */
    double L_gradient_min = Get_Particle_Size(i) * All.cf_atime;
    /* limit this scale length; if the gradient is too shallow, there is no information beyond a few smoothing lengths, so we can't let streaming go that far */
    double L_gradient_max = DMAX(1000.*L_gradient_min, 500.0*P[i].KernelRadius*All.cf_atime);

    /* also, physically, cosmic rays cannot stream/diffuse with a faster coefficient than ~v_max*L_mean_free_path, where L_mean_free_path ~ 2.e20 * (cm^-3/n) [collisional here] */
    double nH_cgs = CellP[i].Density * All.cf_a3inv * UNIT_DENSITY_IN_NHCGS;
    double L_mean_free_path = (3.e25 / nH_cgs) / UNIT_LENGTH_IN_CGS;
    L_gradient_max = DMIN(L_gradient_max, L_mean_free_path);
    
    double CRPressureGradScaleLength = Get_Gas_CosmicRayPressure(i,k_CRegy) / CRPressureGradMag * All.cf_atime;
    if(CRPressureGradScaleLength > 0) {CRPressureGradScaleLength = 1.0/(1.0/CRPressureGradScaleLength + 1.0/L_gradient_max);} else {CRPressureGradScaleLength=0;}
    CRPressureGradScaleLength = sqrt(L_gradient_min*L_gradient_min + CRPressureGradScaleLength*CRPressureGradScaleLength);
    return CRPressureGradScaleLength; /* this is returned in -physical- units */
}



/* routine to quickly estimate the atomic mass of the CR particles (in proton masses): making assumptions here [e.g. no positrons] -- could always hard-code some choice here for more complicated scenarios */
double return_CRbin_CRmass_in_mp(int target, int k_CRegy)
{
    int species = return_CRbin_CR_species_ID(k_CRegy);
    if(species < 0) {return 0.000544618;} // electrons or positrons
        else {
            if(species == 1) {return  1.0;} // p
            if(species == 7) {return  1.0;} // pbar (anti-p)
            if(species == 2) {return 10.8;} // B: mostly 11, but non-negligible 10, with 10B/11B ~ 0.58, so should really account for both. won't make huge difference here but some care needed in propagation models and abundance models and corrections
            if(species == 3) {return 12.0;} // C
            if(species == 4) {return  9.0;} // Be7-9: for now going with 9 as the reference, but could do 7 as well...
            if(species == 5) {return  10.;} // Be10
            if(species == 6) {return 14.8;} // CNO bin
            /* if don't have a rule for the species, default to 2x Z for heavy species */
            double Z_abs = fabs(return_CRbin_CR_charge_in_e(target,k_CRegy));
            if(Z_abs > 1.5) {return 2.*Z_abs;} else {return 1;} // nuclei, making a simple assumption about the nuclear weight [can modify easily with custom flag!]
    }
}


/* routine which returns the dimensionless velocity beta = v/c of the CRs (this is not the RSOL, but true c, for e.g. cooling, energies, etc) */
double return_CRbin_beta_factor(int target, int k_CRegy)
{
    double m_cr_mp=return_CRbin_CRmass_in_mp(target,k_CRegy); // mass in proton masses
    double q = return_CRbin_CR_rigidity_in_GV(target,k_CRegy)*1.06579*fabs(return_CRbin_CR_charge_in_e(target,k_CRegy))/m_cr_mp; // dimensionless factor to convert from R to beta
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    return beta;
}


/* routine which returns the dimensionless lorentz factor gamma=1/sqrt[1-beta^2] of the CRs (this is not the RSOL, but true c, for e.g. cooling, energies, etc) */
double return_CRbin_gamma_factor(int target, int k_CRegy)
{
    double m_cr_mp=return_CRbin_CRmass_in_mp(target,k_CRegy); // mass in proton masses
    double q = return_CRbin_CR_rigidity_in_GV(target,k_CRegy)*1.06579*fabs(return_CRbin_CR_charge_in_e(target,k_CRegy))/m_cr_mp; // dimensionless factor to convert from R to beta
    return sqrt(1.+q*q);
}


/* routine which returns the effective adiabatic index of CRs, P_cr = (gamma_eos - 1) * e_kinetic */
double gamma_eos_of_crs_in_bin(int k_CRegy)
{
    return (4. + 1./return_CRbin_gamma_factor(-1,k_CRegy)) / 3.;
}

/* routine which returns the CR kinetic energy in GeV, for a given rigidity, etc. */
double return_CRbin_kinetic_energy_in_GeV(int target, int k_CRegy)
{
    double m_cr_mp=return_CRbin_CRmass_in_mp(target,k_CRegy); // mass in proton masses
    double R_GV = return_CRbin_CR_rigidity_in_GV(target,k_CRegy), Z = fabs(return_CRbin_CR_charge_in_e(target,k_CRegy));
    double q = R_GV*Z*1.06579/m_cr_mp; // dimensionless factor to convert from R to beta
    double gamma = sqrt(1.+q*q), beta = q/gamma;
    double KE_fac = DMAX(0.,(1.-sqrt(DMAX(0.,1.-beta*beta)))) / DMAX(beta,MIN_REAL_NUMBER); if(beta < 0.01) {KE_fac = 0.5*beta;} // non-relativistic expansion used when gamma very close to one, to prevent numerical errors
    return R_GV * Z * KE_fac;
}


/* routine which returns the CR number density at a given bin in cm^-3 */
double return_CRbin_numberdensity_in_cgs(int target, int k_CRegy)
{
    double e_CR_tot = CellP[target].CosmicRayEnergyPred[k_CRegy] * (CellP[target].Density*All.cf_a3inv/P[target].Mass) * (UNIT_PRESSURE_IN_CGS);
    double ECR_per  = return_CRbin_kinetic_energy_in_GeV(target,k_CRegy) * 0.00160218; /* converts to energy in erg */
    return e_CR_tot / ECR_per;
}


/* handy function that just returns the radiation energy density in eV/cm^-3, physical units. purely here to save us time re-writing this */
double get_cell_Urad_in_eVcm3(int i)
{
    double erad = 0.26*All.cf_a3inv/All.cf_atime; // default with the CMB energy density, which we assume here is a baseline minimum
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY) // use actual explicitly-evolved radiation field, if possible
    int kfreq; double e_units = (CellP[i].Density*All.cf_a3inv/P[i].Mass) * UNIT_PRESSURE_IN_EV;
    for(kfreq=0;kfreq<N_RT_FREQ_BINS;kfreq++) {erad+=CellP[i].Rad_E_gamma_Pred[kfreq]*e_units;}
#else
    double uRad_MW = 0.31 + 0.66; /* dust (0.31) and stars (0.66) for Milky way ISRF from Draine (2011); want this to decay as we approach the IGM (where CMB totally dominates) */
    double prefac_rad=1, rho_cgs=CellP[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS; if(All.ComovingIntegrationOn) {double rhofac = rho_cgs / (1000.*COSMIC_BARYON_DENSITY_CGS);
        if(rhofac < 0.2) {prefac_rad=0;} else {if(rhofac > 200.) {prefac_rad=1;} else {prefac_rad=exp(-1./(rhofac*rhofac));}}} // in cosmological runs, turn off stellar background for any gas with density unless it's >1000 times the cosmic mean density
    prefac_rad *= rho_cgs/(0.01*PROTONMASS_CGS + rho_cgs); // cut off below low densities, ~1e-2
    erad += uRad_MW * prefac_rad;
#endif
    return erad;
}



/* return pre-factor for CR streaming losses, such that loss rate dE/dt = -E * streamfac */
double CR_get_streaming_loss_rate_coefficient(int target, int k_CRegy)
{
    double dt = GET_PARTICLE_TIMESTEP_IN_PHYSICAL(target), streamfac = 0;
    if((CellP[target].CosmicRayEnergy[k_CRegy] <= MIN_REAL_NUMBER) || (CellP[target].CosmicRayEnergy[k_CRegy] <= MIN_REAL_NUMBER) || (dt <= 0)) {return 0;}
    double vA = Get_Gas_ion_Alfven_speed_i(target); /* define Alfven speed, which is what appears in the proper formulation here */
    double v_flux_eff=0; int k; for(k=0;k<3;k++) {v_flux_eff += CellP[target].CosmicRayFluxPred[k_CRegy][k] * CellP[target].CosmicRayFluxPred[k_CRegy][k];} // need magnitude of flux vector
    if(v_flux_eff > 0) {v_flux_eff=sqrt(v_flux_eff) / (MIN_REAL_NUMBER + CellP[target].CosmicRayEnergyPred[k_CRegy]);} else {v_flux_eff=0;} // effective speed of CRs = |F|/E
    int target_for_cr_gamma = target; // if this = -1, use the gamma factor at the bin-center for evaluating this, if this = target, use the mean gamma of the bin, weighted by the CR energy -- won't give exactly the same result here
    double gamma_0=return_CRbin_gamma_factor(target_for_cr_gamma,k_CRegy), gamma_fac=gamma_0/(gamma_0-1.), beta_fac=return_CRbin_beta_factor(target_for_cr_gamma,k_CRegy); // lorentz factor here, needed in next line, because the loss term here scales with -total- energy, not kinetic energy
    if(beta_fac<0.1) {gamma_fac=2./(beta_fac*beta_fac) -0.5 - 0.125*beta_fac*beta_fac;} // avoid accidental nan
    streamfac = (vA * (beta_fac*beta_fac) / fabs(3.*CellP[target].CosmicRayDiffusionCoeff[k_CRegy])) * ((gamma_fac) * return_CRbin_nuplusminus_asymmetry(target,k_CRegy) * v_flux_eff/CosmicRayFluid_RSOL_Corrfac(k_CRegy) - (3.*(GAMMA_COSMICRAY(k_CRegy)-1.) + (gamma_fac)) * vA * (2./3.) * return_cosmic_ray_anisotropic_closure_function_threechi(target,k_CRegy)); // this is (vA/[3kappa])*(F - 2*chifac*vA*(ecr+3*Pcr))/ecr, using the 'full F' [corrected back from rsol, b/c rsol correction moves outside this for loss terms]
    double sfac_max = fabs(0.1 * CellP[target].CosmicRayEnergy[k_CRegy] / (MIN_REAL_NUMBER + dt));
    if(fabs(streamfac) > sfac_max) {streamfac *= sfac_max / fabs(streamfac);}
    return streamfac; // probably want to limit to make sure above doesn't take on too extreme a value... also above, initially only had positive term since this removes energy from CRs when streaming super-Alfvenically, but when streaming sub-Alfvenically, could this become a source term with energy going into CRs? seems problematic if vA very high, but then scattering would work inefficiently... so plausible, but really need to be careful again about magnitude...
}



/* routine to do the drift/kick operations for CRs: mode=0 is kick, mode=1 is drift */
#if !defined(CRFLUID_EVOLVE_SCATTERINGWAVES)
double CosmicRay_Update_DriftKick(int i, double dt_entr, int mode)
{
    if(dt_entr <= 0) {return 0;} // no update

    int k_CRegy;
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        int k; double eCR, u0; k=0; if(mode==0) {eCR=CellP[i].CosmicRayEnergy[k_CRegy]; u0=CellP[i].InternalEnergy;} else {eCR=CellP[i].CosmicRayEnergyPred[k_CRegy]; u0=CellP[i].InternalEnergyPred;} // initial energy
        if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
        if(eCR < 0) {eCR=0;} // limit to physical values
        double closure_f1, closure_f2; closure_f1=1, closure_f2=0; // prefactors for below
        double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(i,k_CRegy); // 3*chi = 3*(1-<mu^2>)/2 closure function //
        closure_f1 = 3.-2.*three_chi; closure_f2 = 1.-three_chi; // prefactors for both terms below //

        // this is the exact solution for the CR flux-update equation over a finite timestep dt: it needs to be solved this way [implicitly] as opposed to explicitly for dt because in the limit of dt_cr_dimless being large, the problem exactly approaches the diffusive solution
        Vec3<double> DtCosmicRayFlux={}, flux={}, CR_veff={}; double CR_vmag=0, q_cr=0, cr_speed=CRFLUID_REDUCED_C_CODE(k_CRegy), rsol_correction_factor=CosmicRayFluid_RSOL_Corrfac(k_CRegy), V_i=P[i].Mass/CellP[i].Density, P0_cr, fac_for_DtCosmicRayFlux; P0_cr=Get_Gas_CosmicRayPressure(i,k_CRegy);
        cr_speed = DMAX(CellP[i].MaxSignalVel , DMIN(CRFLUID_REDUCED_C_CODE(k_CRegy) , 10.*fabs(CellP[i].CosmicRayDiffusionCoeff[k_CRegy])/(Get_Particle_Size(i)*All.cf_atime)));
        fac_for_DtCosmicRayFlux = -rsol_correction_factor * fabs(CellP[i].CosmicRayDiffusionCoeff[k_CRegy]) * V_i / (GAMMA_COSMICRAY(k_CRegy)-1.);
        for(k=0;k<3;k++) {DtCosmicRayFlux[k] = CellP[i].Gradients.CosmicRayPressure[k_CRegy][k];}
#ifdef MAGNETIC // do projection onto field lines
        Vec3<double> bhat={}, B0={}; double Bmag2=0, Bmag=0, bbGB=0, DtCRDotBhat=0;
        if(mode==0) {B0=CellP[i].B/V_i;} else {B0=CellP[i].BPred/V_i;}
        DtCRDotBhat = dot(DtCosmicRayFlux, B0); Bmag2 = B0.norm_sq(); bhat = B0;
        if(Bmag2 > 0) {Bmag=sqrt(Bmag2); bhat /= Bmag;}
        for(k=0;k<3;k++) {int k2; for(k2=0;k2<3;k2++) {bbGB += -bhat[k]*bhat[k2]*CellP[i].Gradients.B[k][k2]/Bmag;}}
        for(k=0;k<3;k++) {
            DtCosmicRayFlux[k] = fac_for_DtCosmicRayFlux * (closure_f1*DtCRDotBhat*B0[k]/Bmag2 + closure_f2*P0_cr*bbGB);
        }
#endif
        double v_Alfven = three_chi * Get_Gas_ion_Alfven_speed_i(i) * return_CRbin_nuplusminus_asymmetry(i,k_CRegy); /* define naive streaming and Alfven speeds */
        double dt_f_m=DtCosmicRayFlux.norm_sq();
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        double flux_diff=sqrt(dt_f_m), flux_stream=fabs(rsol_correction_factor*v_Alfven*(GAMMA_COSMICRAY(k_CRegy)*eCR)); // estimate contribution to flux from both diffusive and streaming components
        double frac_diff=flux_diff/(flux_diff+flux_stream); // fraction of flux from diffusive term
        double alpha_v=0.,alpha_qN=0.,alpha_qE=1.,alpha_nu=-0.6,alpha_L=0.,alpha_f0=CR_return_spectral_slope_target(i,k_CRegy); // values of coefficients: replace hard-coded alpha_nu with lookup to actual function numerically, below
        double xi = CR_global_max_rigidity_in_bin[k_CRegy] / CR_global_min_rigidity_in_bin[k_CRegy]; // bin width in our units
        int kCR_p=k_CRegy, kCR_m=k_CRegy-1; // want two neighboring bins with same species
        if(k_CRegy<N_CR_PARTICLE_BINS-1) {if(CR_species_ID_in_bin[k_CRegy+1]==CR_species_ID_in_bin[k_CRegy]) {kCR_m++; kCR_p++;}} // check if can use this and next, or use this and below
        double xi_pm = sqrt((CR_global_min_rigidity_in_bin[kCR_p]*CR_global_max_rigidity_in_bin[kCR_p])/(CR_global_min_rigidity_in_bin[kCR_m]*CR_global_max_rigidity_in_bin[kCR_m])); // bin ratio to next bin for numerical derivative (being careful to follow our convention of defining these at the geometric mean)
        double beta_k =return_CRbin_beta_factor(-1,k_CRegy), beta_p=return_CRbin_beta_factor(-1,kCR_p), beta_m=return_CRbin_beta_factor(-1,kCR_m); // get beta factors needed to go between scattering rates and diffusivities
        alpha_nu = log((beta_p*beta_p/CellP[i].CosmicRayDiffusionCoeff[kCR_p]) / (beta_m*beta_m/CellP[i].CosmicRayDiffusionCoeff[kCR_m])) / log(xi_pm); // numerically calculate the slope of the scattering-rate dependence for any functional form
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
        CellP[i].Flux_Number_to_Energy_Correction_Factor[k_CRegy] = DMAX(DMIN(flux_n_over_e_factor_modulated, 2.0), 0.5); // equilibrium streaming solution is alpha_mu->-alpha_v such that bin-centered is exact, so mean correction applies only to flux 'portion' of this
#endif
        if(dt_f_m>0) {DtCosmicRayFlux *= (1.0 + rsol_correction_factor * v_Alfven * (GAMMA_COSMICRAY(k_CRegy) * eCR) / sqrt(dt_f_m));} // (tilde[c]/c) * v_a * (ecr+Pcr), in same direction as gradient wants to 'push' naturally [natural direction of F]

        if(mode==0) {flux=CellP[i].CosmicRayFlux[k_CRegy];} else {flux=CellP[i].CosmicRayFluxPred[k_CRegy];}
#ifdef MAGNETIC // do projection onto field lines
        double fluxmag=flux.norm_sq(), fluxdot=dot(flux, B0);
        if(fluxmag>0) {fluxmag=sqrt(fluxmag);} else {fluxmag=0;}
        if(fluxdot<0) {fluxmag*=-1;} // points down-field
        // before acting on the 'stiff' sub-system, account for the 'extra' advection term that accounts for 'twisting' of B: note more careful derivation shows this is sub-leading order in v/c, should not be included here
        //double fac_bv=0; for(k=0;k<3;k++) {fac_bv += All.cf_a2inv * bhat[k] * (bhat[0]*CellP[i].Gradients.Velocity[k][0] + bhat[1]*CellP[i].Gradients.Velocity[k][1] + bhat[2]*CellP[i].Gradients.Velocity[k][2]);}
        //if(All.ComovingIntegrationOn) {fac_bv += All.cf_hubble_a;} // adds cosmological/hubble flow term here [not included in peculiar velocity gradient]
        //fluxmag *= exp(-DMAX(-2.,DMIN(2.,rsol_correction_factor*fac_bv*dt_entr))); // limit factor for change here, should be small given Courant factor, then update flux term accordingly, before next step -- acts like a mod of the divv term //
        if(Bmag2>0) {flux = (fluxmag / sqrt(Bmag2)) * B0;} // re-assign to be along field
#endif
        int target_for_CR_beta_factor = i; // if this =1, use energy-weighted mean value in bin for CR beta, otherwise if =-1, use median point of bin
        target_for_CR_beta_factor = -1;
        double beta_fac = return_CRbin_beta_factor(target_for_CR_beta_factor,k_CRegy); // velocity beta, to account for non-relativistic CRs
        double dt_cr_dimless = dt_entr * beta_fac*beta_fac * cr_speed*cr_speed * (1./3.) / (MIN_REAL_NUMBER + fabs(CellP[i].CosmicRayDiffusionCoeff[k_CRegy] * rsol_correction_factor));
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
        if(mode==0) {for(k=0;k<3;k++) {CellP[i].CosmicRayFlux[k_CRegy][k]=flux[k];}} else {for(k=0;k<3;k++) {CellP[i].CosmicRayFluxPred[k_CRegy][k]=flux[k];}}
    
        /* update scalar CR energy. first update the CR energies from fluxes. since this is positive-definite, some additional care is needed */
        double dCR_dt = CellP[i].DtCosmicRayEnergy[k_CRegy], eCR_tmp = eCR;
        double dCR = dCR_dt*dt_entr, dCRmax = 1.e10*(eCR_tmp+MIN_REAL_NUMBER);
#if defined(GALSF)
        dCRmax = DMAX(2.0*eCR_tmp , 0.1*u0*P[i].Mass);
#endif
        if(dCR > dCRmax) {dCR=dCRmax;} // don't allow excessively large values
        if(dCR < -eCR_tmp) {dCR=-eCR_tmp;} // don't allow it to go negative
        double eCR_0, eCR_00; eCR_00 = eCR_tmp; eCR_tmp += dCR; if((eCR_tmp<0)||(isnan(eCR_tmp))) {eCR_tmp=0;} // check against energy going negative or nan
        if(mode==0) {CellP[i].CosmicRayEnergy[k_CRegy]=eCR_tmp;} else {CellP[i].CosmicRayEnergyPred[k_CRegy]=eCR_tmp;} // updated energy
        eCR_0 = eCR_tmp; // save this value for below
        
#if defined(CRFLUID_EVOLVE_SPECTRUM)
        // add update for CR number if evolved explicitly //
        if(mode==0) // only update on kicks, since we worth with a drift-conserved slope determining the ratio of N and E
        {
            double dN = CellP[i].DtCosmicRay_Number_in_Bin[k_CRegy]*dt_entr, n0 = CellP[i].CosmicRay_Number_in_Bin[k_CRegy], n_new = n0+dN;
            double E_GeV = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_CRegy), xm = CR_global_min_rigidity_in_bin[k] / CR_global_rigidity_at_bin_center[k], xp = CR_global_max_rigidity_in_bin[k] / CR_global_rigidity_at_bin_center[k], xm_e=xm, xp_e=xp;
            if(CR_check_if_bin_is_nonrelativistic(k_CRegy)) {xm_e = xm*xm; xp_e = xp*xp;} // extra power of p in energy equation accounted for here, all that's needed
            double N_min = eCR_tmp / (E_GeV * xp_e * (1.-1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at highest energy
            double N_max = eCR_tmp / (E_GeV * xm_e * (1.+1.e-4)); // even with arbitrarily large slopes we cannot exceed this limit: all CRs 'piled up' at lowest energy
            n_new = DMIN(DMAX(n_new,N_min),N_max); if((n_new<0) || (isnan(n_new))) {n_new=0;}
            CellP[i].CosmicRay_Number_in_Bin[k_CRegy] = n_new; // alright, updated CR number for evolution equations
        }
#endif
    
#if defined(COOLING_OPERATOR_SPLIT)
        /* now need to account for the adiabatic heating/cooling of the 'fluid', here, with gamma=GAMMA_COSMICRAY(k_CRegy) */
        double dCR_div = CR_calculate_adiabatic_gasCR_exchange_term(i, dt_entr, (GAMMA_COSMICRAY(k_CRegy)-1.)*eCR_tmp, mode); // this will handle the update below - separate subroutine b/c we want to allow it to appear in a couple different places
        double uf = DMAX(u0 - dCR_div/P[i].Mass , All.MinEgySpec); // final updated value of internal energy per above
        if(mode==0) {CellP[i].InternalEnergy = uf;} else {CellP[i].InternalEnergyPred = uf;} // update gas
#if !defined(CRFLUID_EVOLVE_SPECTRUM)
        if(mode==0) {CellP[i].CosmicRayEnergy[k_CRegy] += dCR_div;} else {CellP[i].CosmicRayEnergyPred[k_CRegy] += dCR_div;} // update CRs: note if explicitly evolving spectrum, this is done separately below //
#endif
#endif

    } // loop over CR bins complete
    
#if defined(CRFLUID_INJECTION_AT_SHOCKS)
    if(CellP[i].DtCREgyNewInjectionFromShocks > 0) /* now perform the actual CR injection using the rates estimated in the hydro solver */
    {
        double dir[3]; int k; for(k=0;k<3;k++) {dir[k] = -CellP[i].Gradients.Pressure[k];} /* initial flux direction down pressure gradient */
        inject_cosmic_rays(CellP[i].DtCREgyNewInjectionFromShocks * dt_entr, 1000./UNIT_VEL_IN_KMS, 2, i, dir); /* inject the energy */
        CellP[i].DtCREgyNewInjectionFromShocks = 0; // reset to nil, we've successfully injected the energy
    }
#endif
#if defined(SINK_CR_INJECTION_AT_TERMINATION)
    if(CellP[i].Sink_CR_Energy_Available_For_Injection > 0) {
        /* need to determine whether or not sufficient deceleration has occurred in order to inject CRs from our 'reservoir */
        double vmag = (P[i].Vel / All.cf_atime).norm_sq(); int k; /* we will base this on a simple estimate of the velocity and how much things have decelerated */
        if(vmag>0) {vmag=sqrt(vmag);}
        double v_outflow_fast_forinjection = All.Sink_outflow_velocity;
#ifdef SINK_TEST_WIND_MIXED_FASTSLOW
        v_outflow_fast_forinjection = (SINK_TEST_WIND_MIXED_FASTSLOW)/UNIT_VEL_IN_KMS;
#endif
#ifdef SINK_RIAF_SUBEDDINGTON_MODEL
        v_outflow_fast_forinjection = 0.05 * C_LIGHT_CODE;
#endif
        if((P[i].ID != All.SpawnedWindCellID) || (vmag < ((double)(SINK_CR_INJECTION_AT_TERMINATION))*v_outflow_fast_forinjection)) {
            double dir[3]; for(k=0;k<3;k++) {dir[k] = -CellP[i].Gradients.Pressure[k];} /* initial flux direction down pressure gradient */
            inject_cosmic_rays(CellP[i].Sink_CR_Energy_Available_For_Injection, v_outflow_fast_forinjection, 5, i, dir); /* inject the energy */
            CellP[i].Sink_CR_Energy_Available_For_Injection = 0;  // reset its value to nil, now that it has been injected
        }
    }
#endif
    
    return 1;
}
#endif





#if defined(CRFLUID_EVOLVE_SPECTRUM)

/* this routine does the CR cooling/losses and "heating"/re-acceleration for multi-bin CR spectra: i.e. exchanging CR number
    between bins in the multi-bin approximation and modifying the spectral slope within each bin */
void CR_cooling_and_losses_multibin(int target, double n_elec, double nHcgs, double dtime_cgs, int mode_driftkick)
{
    /*! loss_mode:
     0 - hadronic+catastrophic: simply remove energy from the bin (N and E decrease together, preserving spectral slope)
     1 - adiabatic+bremstrahhlung: pure multiplicative: Edot ~ E (so instantaneously conserves slope, shifts pmin,p0,pmax, need to calculate flux)
     2 - coulomb+ionization: scale identically, for low-E protons important, messy dependence, similar to e.g. fitting function from Girichidis, dp/dt ~ (1 + p^(-1.9))
     3 - inverse compton+synchrotron: Edot ~ E^2, also pdot~p^2 [ultra-rel b/c e-]: modifies slope
     */
    if(dtime_cgs <= 0 || nHcgs <= 0) {return;} /* catch */
    
    /* define a bunch of general-use variables below */
    //double *bin_slopes; bin_slopes = CellP[target].CosmicRay_PwrLaw_Slopes_in_Bin; // if directly evolving slope
    double *ntot_evolved; ntot_evolved = CellP[target].CosmicRay_Number_in_Bin; double bin_slopes[N_CR_PARTICLE_BINS]; // if directly evolving number
    double *Ucr, Ucr_tot=0; if(mode_driftkick==0) {Ucr=CellP[target].CosmicRayEnergy;} else {Ucr=CellP[target].CosmicRayEnergyPred;}
    int k; for(k=0;k<N_CR_PARTICLE_BINS;k++) {Ucr_tot+=Ucr[k];} // check total energy since some fluid cells can have no CRs
    if(Ucr_tot < MIN_REAL_NUMBER) {return;} // catch - nothing to do here //
    double t=0, dt=0, bbGv=0, E_rest_e_GeV=0.000511, f_ion=DMAX(DMIN(Get_Gas_Ionized_Fraction(target),1.),0.), b_muG=get_cell_Bfield_in_microGauss(target), U_mag_ev=0.0248342*b_muG*b_muG, U_rad_ev=get_cell_Urad_in_eVcm3(target);

#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
    double vA=Get_Gas_Alfven_speed_i(target), vA_ion, vA_touse, kappa_max=0, kappa_i[N_CR_PARTICLE_BINS], delta_diffcoeff[N_CR_PARTICLE_BINS], reaccel_coeff[N_CR_PARTICLE_BINS]; vA_ion=vA/sqrt(f_ion); vA_touse=vA;
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {kappa_i[k] = CellP[target].CosmicRayDiffusionCoeff[k]; if(kappa_i[k]>kappa_max) {kappa_max=kappa_i[k];}} // will use diffusion coefficient below
    double reaccel_coeff_0 = 2. * vA_touse*vA_touse / UNIT_TIME_IN_CGS; // below we'll adopt the standard ansatz that the momentum-space diffusion coefficient is related to the spatial coefficient by the usual heuristic expression Dxx*Dpp = <dv^2>
    if(All.Time <= All.TimeBegin || kappa_max*UNIT_LENGTH_IN_CGS*UNIT_VEL_IN_CGS < 1.e25) {reaccel_coeff_0 = 0;} // make sure kappa is initialized and has a physically meaningful value where the prescription here makes sense, or else this will give nonsense values
    reaccel_coeff_0 = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs, reaccel_coeff_0)); // our courant-type condition on the timestep should ensure this as well, but just in case, we need to enforce a condition here //
#endif
    
    //double adiabatic_coeff_divv = (1./3.) * (CellP[target].Face_DivVel_ForAdOps*All.cf_a2inv) / UNIT_TIME_IN_CGS ; // coefficient for adiabatic work [compression/expansion terms]. convert to physical units [a2inv], and then cgs for units here. SIGN is flipped from usual convention since we assume convention where positive coefficients = losses, for convenience with everything else below.
    double adiabatic_coeff[N_CR_PARTICLE_BINS], adiabatic_coeff_divv = (1./3.) * (P[target].Particle_DivVel*All.cf_a2inv) / UNIT_TIME_IN_CGS; // need to use full DivVel to get the right answer here for CR EOS scaling at high-p
    if(All.ComovingIntegrationOn) {adiabatic_coeff_divv += (1./3.) * (3.*All.cf_hubble_a) / UNIT_TIME_IN_CGS;} // adiabatic term from Hubble expansion (needed for cosmological integrations. also converted to physical, cgs, and sign convention we use here.
    // here we calculate the anisotropic stress correction terms, as needed to properly evaluate the CR energy loss. note we don't need to do this for the 'post hydro' correction step since that is the correction for the isotropic pressure used in the Riemann problem; the additional terms here come entirely outside of that.
#if defined(MAGNETIC)
    Vec3<double> bhat={}; for(k=0;k<3;k++) {bhat[k]=Get_Gas_BField(target,k)*All.cf_a2inv;}
    double Bmag2=bhat.norm_sq(); if(Bmag2>0) {bhat/=sqrt(Bmag2);}
    for(k=0;k<3;k++) {int k2; for(k2=0;k2<3;k2++) {bbGv += bhat[k]*bhat[k2]*CellP[target].Gradients.Velocity[k][k2] * All.cf_a2inv / UNIT_TIME_IN_CGS;}} // bhat bhat : grad u -> needed to obtain double-dot-product appropriately
    bbGv = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs , bbGv)); // our timestep limiter should ensure this, but problem is it responds to the -previous- timestep, so we need to impose an additional check here to prevent a numerical divergence when/if the gradients are inaccurate
#endif
    double adiabatic_min = -0.5*P[target].Mass*DMAX(DMIN(CellP[target].InternalEnergyPred,CellP[target].InternalEnergy)-All.MinEgySpec,0.) / (Ucr_tot*dtime_cgs + MIN_REAL_NUMBER); if(adiabatic_coeff_divv < adiabatic_min) {adiabatic_coeff_divv = adiabatic_min;} // limit adiabatic -gains- of CRs (careful about sign convention here, negative means gain!) as this leads to too-large thermal losses, prevented by limiters in our step computing the exchange between CRs and gas in adiabatic calc above //
    adiabatic_coeff_divv = DMAX(-0.9/dtime_cgs, DMIN(0.9/dtime_cgs , adiabatic_coeff_divv)); // our timestep limiter should ensure this, but problem is it responds to the -previous- timestep, so we need to impose an additional check here to prevent a numerical divergence when/if the gradients are inaccurate

    double Ucr_i[N_CR_PARTICLE_BINS], A_wt[N_CR_PARTICLE_BINS], Z[N_CR_PARTICLE_BINS], x_m[N_CR_PARTICLE_BINS], x_p[N_CR_PARTICLE_BINS], R0[N_CR_PARTICLE_BINS], E_GeV[N_CR_PARTICLE_BINS], bin_centered_rate_coeff[N_CR_PARTICLE_BINS], streaming_coeff[N_CR_PARTICLE_BINS], brems_coeff[N_CR_PARTICLE_BINS], M1SpeedCorrFac[N_CR_PARTICLE_BINS]={1}; int NR_key[N_CR_PARTICLE_BINS];
    double hadronic_coeff = 6.37e-16 * nHcgs; // coefficient for hadronic/catastrophic interactions: dEtot/dt = -(coeff) * Etot, or dPtot/dt = -(coeff) * Ptot (since all p effected are in rel limit, and works by deleting N not by lowering individual E. Mannheim & Schlickeiser 1994
    double coulomb_coeff = 3.09e-16 * nHcgs * ((n_elec + 0.57*(1.-f_ion))*HYDROGEN_MASSFRAC); // default Coulomb+ionization (the two scale nearly-identically) normalization divided by GeV, b/c we need to divide the energy per CR. needs to be multiplied by ((Z*Z)/(beta*E_GeV)). Mannheim & Schlickeiser 1994
    double brems_coeff_0 = 1.39e-16  * n_elec * nHcgs; // coefficient for Bremsstrahlung [following Blumenthal & Gould, 1970]: dEkin/dt=4*alpha_finestruct*r_classical_elec^2*c * SUM[n_Z,ion * Z * (Z+1) * (ln[2*gamma_elec]-1/3) * E_kin . this needs to be multiplied by [DMAX(log(2.*gamma)-0.33,0)]; becomes dE/dt = -(coeff) * E, or dP/dt = -(coeff) * P [since all e- in rel limit].
    double synchIC_coeff_0 = 5.2e-20 * (U_mag_ev + U_rad_ev); // synchrotron and inverse compton scale as dE/dt=(4/3)*sigma_Thompson*c*gamma_elec^2*(U_mag+U_rad), where U_mag and U_rad are the magnetic and radiation energy densities, respectively. Ignoring Klein-Nishina corrections here, as they are negligible at <40 GeV and only a ~15% correction up to ~1e5 GeV. U_mag_ev=(B^2/8pi)/(eV/cm^(-3)), here; U_rad=U_rad/(eV/cm^-3). needs to be multiplied by gamma
    double e_ion_coeff = 3.60e-16 * nHcgs * (1.-f_ion), e_coulomb_coeff = 6.40e-16 * nHcgs * n_elec; // electron coulomb + ionization terms - note very similar to proton ionization (slightly different normalization b/c of log terms, and always in relativistic limit. see e.g. Ginzburg and Syrovatskii, 1964; Gould and Burbidge, 1965, Ramaty and Lingenfelter, 1966. note coulomb coeff has a very weak ~0.01*log[E_cr] scaling, but this scaling is basically offset entirely by the beta-dependence of the scaling at energies of interest, and is very weak, so we ignore it. e.g. Gould 72: Edot = (3/2)*sigma_T*ne*me*c^3/(beta^2) * (log[me*c^2*beta*sqrt[gamma-1]/(hbar*sqrt(4pi*e^2*ne/me))] + log[2]*(beta^2 / 2 + 1/gamma) + 1/2 + (gamma-1)^2/(16*gamma^2)
    
    double dt_min=dtime_cgs, dt_min_e=dt_min, dt_min_p=dt_min, dt_tmp, CourFac=0.4; // courant-like factor for use in subcycling here //
    int sign_flip_adiabatic_terms = 0, sign_key_for_adiabatic_loop = 1; // key that tells us if the adiabatic+brems+streaming terms have a strong sign flip, in which case we need to do 2 loops instead of 1
    double min_R_e=MAX_REAL_NUMBER, min_R_p=MAX_REAL_NUMBER, max_R_e=MIN_REAL_NUMBER, max_R_p=MIN_REAL_NUMBER; // record the minimum/max bin for e and p, for use in timestepping and limiting below (need to know if we're in the smallest bin)
    for(k=0;k<N_CR_PARTICLE_BINS;k++) /* initialize a bunch of variables for the different bins that we'll refer to below */
    {
        Ucr_i[k] = Ucr[k]; // save initial energy for reference at the end of this loop
        Z[k] = return_CRbin_CR_charge_in_e(-1,k); // want bin-centered values, so give index = -1
        R0[k] = CR_global_rigidity_at_bin_center[k]; // want bin-centered values, so give index = -1
        E_GeV[k] = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k); // want bin-centered values, so give index = -1
        NR_key[k] = CR_check_if_bin_is_nonrelativistic(k); // key to decide whether to use relativistic or non-relativistic scalings
        A_wt[k] = return_CRbin_CRmass_in_mp(-1,k); // return the proper nuclear weight here, in units of the proton mass
        x_m[k] = CR_global_min_rigidity_in_bin[k] / R0[k]; // ratio of min-to-mid-bin CR rigidity or momentum, used for scaling everything below
        x_p[k] = CR_global_max_rigidity_in_bin[k] / R0[k]; // ratio of max-to-mid-bin CR rigidity or momentum, used for scaling everything below
        bin_slopes[k] = CR_return_slope_from_number_and_energy_in_bin(Ucr[k], ntot_evolved[k], E_GeV[k], k); // initialize slopes to use below from LUT, if not directly evolving them
        if(CR_species_ID_in_bin[k] < 0) {if(R0[k]<min_R_e) {min_R_e=R0[k];}} else {if(R0[k]<min_R_p) {min_R_p=R0[k];}} // record minimum rigidity to use in limits applied below
        if(CR_species_ID_in_bin[k] < 0) {if(R0[k]>max_R_e) {max_R_e=R0[k];}} else {if(R0[k]>max_R_p) {max_R_p=R0[k];}} // record minimum rigidity to use in limits applied below
        double three_chi = return_cosmic_ray_anisotropic_closure_function_threechi(target,k); // get the closure function
        adiabatic_coeff[k] = three_chi*adiabatic_coeff_divv + (1.-three_chi)*bbGv; // allows for energy+species-dependent CR anisotropy
        
        if(CR_species_ID_in_bin[k] < 0) {brems_coeff[k] = brems_coeff_0 * DMAX(log(2.*(1.+E_GeV[k]/E_rest_e_GeV))-0.33,0);} else {brems_coeff[k]=0;}
        streaming_coeff[k] = CR_get_streaming_loss_rate_coefficient(target,k) / UNIT_TIME_IN_CGS;

        M1SpeedCorrFac[k] = evaluate_cr_transport_reductionfactor(target, k, 1); /* implement PFH correction term, similar to RHD with RSOL, to account for RSOL in residence time for attenuation */

        // calculate the timestep limit from all possible bins, maximum step. note no constraint from hadronic here b/c cannot 'cross the bin'
        bin_centered_rate_coeff[k] = 0;
        double adiab_brem_coeff = adiabatic_coeff[k] + streaming_coeff[k] + brems_coeff[k]; bin_centered_rate_coeff[k]+=adiab_brem_coeff; // do constraint from adiabatic + Bremsstrahlung
        if(adiab_brem_coeff < 0) {sign_key_for_adiabatic_loop=-1;} // note the net sign of this term for use below
        if(k>0) {if(adiab_brem_coeff * (adiabatic_coeff[k-1] + streaming_coeff[k-1] + brems_coeff[k-1]) < 0) {sign_flip_adiabatic_terms = 1;}} // have a sign flip, and its not negligible in magnitude //
        if(CR_species_ID_in_bin[k] < 0) // e- or e+ (leptonic): do constraint from synchrotron + IC
        {
            double IC_sync_coeff = (E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0; bin_centered_rate_coeff[k]+=IC_sync_coeff;
            double ion_coeff = (e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k]; bin_centered_rate_coeff[k]+=ion_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
        } else { // p: do constraint from Coulomb + ionization
            if(NR_key[k]==1) // bin is in non-relativistic limit, use those expressions
            {
                double Coul_coeff = ((0.88 * A_wt[k]*A_wt[k]) / (R0[k]*R0[k]*R0[k] * fabs(Z[k]))) * coulomb_coeff; bin_centered_rate_coeff[k]+=Coul_coeff; // non-relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc. should be multiplied by atomic weight A^2 as well.
            } else { // bin is in relativistic limit, use those expressions
                double Coul_coeff = (fabs(Z[k]/R0[k])) * coulomb_coeff; bin_centered_rate_coeff[k]+=Coul_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
            }
        }
    }
    
    if(sign_flip_adiabatic_terms==1) {sign_key_for_adiabatic_loop=-1;} // have sign-flips, so adiabatic term -must- have the oppose sign. otherwise -no- sign flips, so just follow the last sign recorded above

#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
    for(k=0;k<N_CR_PARTICLE_BINS;k++) // additional variables need to be initialized here, after the previous loop definitions
    {
        int minbin_flag=0, maxbin_flag=0;
        if(k>0) {if(CR_species_ID_in_bin[k-1] != CR_species_ID_in_bin[k]) {minbin_flag=1;}} else {minbin_flag=1;} // previous bin doesn't exist or has opposite sign: lowest-E bin for type
        if(k<N_CR_PARTICLE_BINS-1) {if(CR_species_ID_in_bin[k+1] != CR_species_ID_in_bin[k]) {maxbin_flag=1;}} else {maxbin_flag=1;} // next bin doesn't exist or has opposite sign: highest-E bin for type
        /* need to estimate the log-slope of the spatial diffusion coefficient vs momentum: delta = dln[kappa] / dln[R], for each species type */
        if(minbin_flag) {
            delta_diffcoeff[k] = log((kappa_i[k+1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER)) / log((R0[k+1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER)); // linear estimate from next bin
        } else if(maxbin_flag) {
            delta_diffcoeff[k] = log((kappa_i[k] + MIN_REAL_NUMBER) / (kappa_i[k-1] + MIN_REAL_NUMBER)) / log((R0[k] + MIN_REAL_NUMBER) / (R0[k-1] + MIN_REAL_NUMBER)); // linear estimate from previous bin
        } else {
            double y_m = log((kappa_i[k-1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER)), y_p = log((kappa_i[k+1] + MIN_REAL_NUMBER) / (kappa_i[k] + MIN_REAL_NUMBER));
            double x_m = log((R0[k-1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER)), x_p = log((R0[k+1] + MIN_REAL_NUMBER) / (R0[k] + MIN_REAL_NUMBER));
            delta_diffcoeff[k] = (y_p * x_m*x_m - y_m * x_p*x_p) / (x_m * x_p * (x_m - x_p)); // second-order log-slope estimate
        }
        double delta_limit=0.0315059; delta_diffcoeff[k] = DMIN(DMAX(delta_diffcoeff[k], delta_limit),2.-delta_limit); // need to restrict delta value for the quasi-linear theory model below to make any sense (otherwise just gives unphysical answers because relevant integrals all diverge)
        double alpha_denom = delta_diffcoeff[k] * (4.-delta_diffcoeff[k]) * (4.-delta_diffcoeff[k]*delta_diffcoeff[k]); // quasi-linear isotropic turbulent-type assumption for coefficient relating Dpp and Dxx
        reaccel_coeff[k] = -delta_diffcoeff[k] * reaccel_coeff_0 / ((kappa_i[k] + MIN_REAL_NUMBER) * (alpha_denom + MIN_REAL_NUMBER)); // up to a factor of (R/R0)^(-delta) * (1 - slope_gamma/2), this gives the rate in 1/seconds of the effective momentum-space diffusion. note sign here: generally implies energy gain.
        if(fabs(reaccel_coeff[k]) > fabs(delta_diffcoeff[k]*adiabatic_coeff[k])) {reaccel_coeff[k] *= fabs(delta_diffcoeff[k]*adiabatic_coeff[k]) / fabs(reaccel_coeff[k]);}
        bin_centered_rate_coeff[k] += reaccel_coeff[k] * (1. - 0.5*DMIN(bin_slopes[k], 2.)); // limit this to ensure we don't miss if this should be larger
    }
#endif

    
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        if((Ucr[k] < 1.e-10 * Ucr_tot) || (bin_centered_rate_coeff[k]==0)) {continue;} // don't bother with timestep limits if the bin contains totally negligible fraction of CR energy
        if(All.ComovingIntegrationOn) {if(All.Time < 0.12) {if(CR_species_ID_in_bin[k] < 0) {if(R0[k] >= 0.999*max_R_e) {continue;}}}} // don't need to apply the same timestep in the smallest bin, so long as regulated by the bin above it, since we'll just cool to 0 here in finite time and that's ok
        double abs_bin_coeff_limit = 0.05 * fabs(bin_centered_rate_coeff[k]) * M1SpeedCorrFac[k]; // set threshold for fraction of rate where we need to worry about detailed subcycling: sub-dominant processes not important here. find few percent works well here.
        double adiab_brem_coeff = fabs(adiabatic_coeff[k] + streaming_coeff[k] + brems_coeff[k]) * M1SpeedCorrFac[k]; // do constraint from adiabatic + Bremsstrahlung
        if(adiab_brem_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * log(x_p[k]/x_m[k]) / adiab_brem_coeff; if(CR_species_ID_in_bin[k]<0) {dt_min_e=DMIN(dt_min_e, dt_tmp);} else {dt_min_p=DMIN(dt_min_p, dt_tmp);}}
        if(CR_species_ID_in_bin[k] < 0) // e- or e+ (leptonic): do constraint from synchrotron + IC
        {
           double IC_sync_coeff = (E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0 * M1SpeedCorrFac[k];
           if(IC_sync_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * (1./x_m[k] - 1./x_p[k]) / IC_sync_coeff; dt_min_e=DMIN(dt_min_e, dt_tmp);}
           double ion_coeff = (e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k] * M1SpeedCorrFac[k]; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
           if(ion_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * DMIN(x_p[k]-x_m[k], x_m[k]) / ion_coeff; dt_min_e=DMIN(dt_min_e, dt_tmp);}
        } else { // p: do constraint from Coulomb + ionization
           if(NR_key[k]==1) // bin is in non-relativistic limit, use those expressions
           {
               double Coul_coeff = ((0.88 * A_wt[k]*A_wt[k]) / (R0[k]*R0[k]*R0[k] * fabs(Z[k]))) * coulomb_coeff * M1SpeedCorrFac[k]; // non-relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc. should be multiplied by atomic weight A^2 as well.
               if((CR_species_ID_in_bin[k] == 1) || (E_GeV[k]/A_wt[k] > 1.)) {if(Coul_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * (x_p[k]*x_p[k]*x_p[k] - x_m[k]*x_m[k]*x_m[k]) / (3.*Coul_coeff); dt_min_p=DMIN(dt_min_p, dt_tmp);}}
           } else { // bin is in relativistic limit, use those expressions
               double Coul_coeff = (fabs(Z[k])/R0[k]) * coulomb_coeff * M1SpeedCorrFac[k]; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
               if(Coul_coeff > abs_bin_coeff_limit) {dt_tmp = CourFac * DMIN(x_p[k]-x_m[k], x_m[k]) / Coul_coeff; dt_min_p=DMIN(dt_min_p, dt_tmp);}
           }
        }
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
        double rcoeff_bin = fabs(reaccel_coeff[k]) * (1.-0.5*DMIN(bin_slopes[k],1.)) * M1SpeedCorrFac[k]; // limit b/c this might change in step, then estimate rate coefficient at bin center
        if(rcoeff_bin > abs_bin_coeff_limit) {dt_tmp = CourFac * (pow(x_p[k],delta_diffcoeff[k]) - pow(x_m[k],delta_diffcoeff[k])) / rcoeff_bin;} // set timestep limit
        if(CR_species_ID_in_bin[k]<0) {dt_min_e=DMIN(dt_min_e, dt_tmp);} else {dt_min_p=DMIN(dt_min_p, dt_tmp);} // applies to both e and p
#endif
    }
    if(All.ComovingIntegrationOn) {if(Ucr_tot < 1.e-6*P[target].Mass*CellP[target].InternalEnergy) {dt_min_e*=10.; dt_min_p*=10.; dt_min_e=DMAX(dt_min_e,0.01*dtime_cgs); dt_min_p=DMAX(dt_min_p,0.01*dtime_cgs);}} // allow larger slope errors when the CR energy is a negligible fraction of total
    
    double dt_target = DMIN(dtime_cgs, DMIN(dt_min_e,dt_min_p)); // timescale for subcycling, using constraint above. limit for extreme cases where we might run into expense problems
    if(dt_target < 1.e-4*dtime_cgs)
    {
        if(dt_target < 1.e-8*dtime_cgs)
        {
            printf("WARNING: timestep for subcycling wants to exceed limit: dt_min_e/p=%g/%g dt_tot=%g \n",dt_min_e,dt_min_p,dtime_cgs);
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
            printf(" ID=%llu mode=%d nH=%g ne=%g vA=%g vA_ion=%g dt=%g Utot=%g hadronic_coeff=%g coulomb_coeff=%g brems_coeff_0=%g synchIC_coeff_0=%g (U_mag_eV=%g U_rad_eV=%g) dtmin=%g signflip=%d signkey=%d \n",(unsigned long long)P[target].ID,mode_driftkick,nHcgs,n_elec,vA,vA_ion,dtime_cgs,Ucr_tot,hadronic_coeff,coulomb_coeff,brems_coeff_0,synchIC_coeff_0,U_mag_ev,U_rad_ev,DMIN(dt_min_e,dt_min_p),sign_flip_adiabatic_terms,sign_key_for_adiabatic_loop);
            for(k=0;k<N_CR_PARTICLE_BINS;k++) {printf("  k=%d U=%g Z=%g R0=%g E=%g NR=%d xm=%g xp=%g slope=%g kappa=%g adiabatic_coeff=%g brem_c=%g stream_c=%g reacc_c=%g delta_DxxSlope=%g IC_sync_c=%g el_ion_coul_c=%g p_ion_coul_c_R/NR=%g/%g binratec=%g \n",k,Ucr[k],Z[k],R0[k],E_GeV[k],NR_key[k],x_m[k],x_p[k],bin_slopes[k],kappa_i[k],adiabatic_coeff[k],brems_coeff[k],streaming_coeff[k],reaccel_coeff[k],delta_diffcoeff[k],(E_GeV[k]/E_rest_e_GeV) * synchIC_coeff_0,(e_coulomb_coeff + (1.+0.07*log(E_GeV[k])) * e_ion_coeff) / R0[k], (fabs(Z[k])/R0[k]) * coulomb_coeff,0.88/ (R0[k]*R0[k]*R0[k] * fabs(Z[k])) * coulomb_coeff,bin_centered_rate_coeff[k]);}
#else
            printf(" ID=%llu mode=%d nH=%g ne=%g dt=%g Utot=%g hadronic_coeff=%g coulomb_coeff=%g brems_coeff_0=%g synchIC_coeff_0=%g (U_mag_eV=%g U_rad_eV=%g) dtmin=%g signflip=%d signkey=%d \n",(unsigned long long)P[target].ID,mode_driftkick,nHcgs,n_elec,dtime_cgs,Ucr_tot,hadronic_coeff,coulomb_coeff,brems_coeff_0,synchIC_coeff_0,U_mag_ev,U_rad_ev,DMIN(dt_min_e,dt_min_p),sign_flip_adiabatic_terms,sign_key_for_adiabatic_loop);
            for(k=0;k<N_CR_PARTICLE_BINS;k++) {printf("  k=%d U=%g Z=%g R0=%g E=%g NR=%d xm=%g xp=%g slope=%g adiabatic_coeff=%g bremc=%g strmc=%g binratec=%g \n",k,Ucr[k],Z[k],R0[k],E_GeV[k],NR_key[k],x_m[k],x_p[k],bin_slopes[k],adiabatic_coeff[k],brems_coeff[k],streaming_coeff[k],bin_centered_rate_coeff[k]);}
#endif
        }
        double dt_min_tmp = 1.e-4 * dtime_cgs; // enforce this since our integrators can deal with it, just at some loss of accuracy
        if(dt_target < dt_min_tmp) {dt_min_e=DMAX(dt_min_e, dt_min_tmp); dt_min_p=DMAX(dt_min_p, dt_min_tmp);}
    }

    int cr_species_key; for(cr_species_key=0;cr_species_key<N_CR_PARTICLE_SPECIES;cr_species_key++) // loop over whether we consider nuclei or electrons, first //
    {
        int species_ID=CR_species_ID_active_list[cr_species_key]; int n_active=0, bins_sorted[N_CR_PARTICLE_BINS]; double R0_bins[N_CR_PARTICLE_BINS];
        for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(CR_species_ID_in_bin[k]==species_ID) {bins_sorted[n_active]=k; R0_bins[n_active]=R0[k]; n_active++;}} // bin is valid: charge matches that desired
        if(n_active<=0) {continue;} // nothing to do here
        if(n_active<N_CR_PARTICLE_BINS) {for(k=n_active;k<N_CR_PARTICLE_BINS;k++) {R0_bins[k]=MAX_REAL_NUMBER;}} // set a dummy value here for sorting purposes below
        //qsort(bins_sorted, n_active, sizeof(int), compare_CR_rigidity_for_sort); // sort on energies from smallest-to-largest [this is hard-coded by requiring the list go in monotonic increasing order for e and p, regardless of how the e and p are themselves ordered //

        t = 0; // reset this before we enter the time integration loop below! //
        if(species_ID<0) {dt_target = DMIN(dtime_cgs, dt_min_e);} else {dt_target = DMIN(dtime_cgs, dt_min_p);} // allow separate timesteps for e and p, since there is no direct exchange between them in the subcycle below
        while(t < dtime_cgs)
        {
            dt = DMIN(dt_target , dtime_cgs - t); // set the subcycle step size
            if(dt <= 0) {break;} // we have reached the end of the timestep - exit loop

            int loss_mode; // this will determine which type[s] of losses [differentiated by their qualitative scalings] we are considering //
            for(loss_mode=0;loss_mode<=5;loss_mode++)
            {
                if(loss_mode==0) {if(species_ID==-1) {continue;}} // loss-mode=0 [Hadronic] uses only nuclei and positrons, skip to next in loop
                if(loss_mode==3) {if(species_ID>0) {continue;}} // loss-mode=3 [Compton+Synchrotron] uses only electrons+positrons
                if(loss_mode==4) {if(sign_flip_adiabatic_terms == 0) {continue;}} // adiabatic+streaming+brems term walked in same order, so we don't need to do an additional loop here
#if !defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                if(loss_mode==5) {continue;}
#endif
                
                int order = -1; // default to -descending- energy order [for energy-loss]. but for adiabatic terms may need to use -ascending- order if energy increases
                if(loss_mode==1) {if(sign_key_for_adiabatic_loop < 0) {order = 1;}} // adiabatic: if net energy -gain- in this step [only step where its possible], then switch to ascending order
                if(loss_mode==5) {order = 1;} // reacceleration always produces increasing energies (in the parameter space of interest)
                
                double dn_flux=0, de_flux=0;
                for(k=0;k<n_active;k++)
                {
                    int j = bins_sorted[k]; // target bin for flux, will move 'up the ladder' passing fluxes to higher energies
                    if(order == -1) {j = bins_sorted[n_active-1-k];} // will move 'down the ladder' passing fluxes to lower energies
                    
                    if(loss_mode==0) // hadronic+catastrophic losses.
                    {
#if 1 // (CRFLUID_EVOLVE_SPECTRUM == 2) // now even the simpler network has secondary e-, important for dense regions synchrotron
                        /* this is where we also include losses from fragmentation and radioactive decay, for heavier nuclei */
                        double frag_coeff=CR_frag_coeff[j]*nHcgs, rad_coeff=CR_rad_decay_coeff[j], total_catastrophic_coeff=frag_coeff+rad_coeff;
                        if(total_catastrophic_coeff > 0) // have some losses here, account for those
                        {
                            double fac_n=DMIN(total_catastrophic_coeff*dt*M1SpeedCorrFac[j], 60.), fac_e=fac_n; // loss rate for number & energy [equal if loss rate is energy-independent. correction term is small: ~0.95 if loss rate ~1/E [e.g. radioactive], or ~1.05 if rate ~E
                            /*
                            double fac_e=DMIN(total_catastrophic_coeff*dt*M1SpeedCorrFac[j], 60.); // loss rate for energy
                            double slope_gamma=bin_slopes[j], xm=x_m[j], xp=x_p[j], xm_g=pow(xm,slope_gamma), xp_g=pow(xp,slope_gamma), xm_g1=xm_g*xm, xp_g1=xp_g*xp, xm_g2=xm_g1*xm, xp_g2=xp_g1*xp, ecorrfac=1;
                            ecorrfac = (slope_gamma*(2.+slope_gamma))/((1.+slope_gamma)*(1.+slope_gamma)) * ((xm_g1-xp_g1)*(xm_g1-xp_g1)) / ((xm_g-xp_g)*(xm_g2-xp_g2));
                            if(ecorrfac>0 && isfinite(ecorrfac)) {fac_e *= ecorrfac;}
                            */
                            if(CR_secondary_target_bin[j][0] > -2) /* now check for whether or not there are any secondary products */
                            {
                                double dfac_e, dfac_n; if(fac_e<0.07) {dfac_e=fac_e-0.5*fac_e*fac_e+fac_e*fac_e*fac_e/6.;} else {dfac_e=1.-exp(-fac_e);}
                                if(fac_n<0.07) {dfac_n=fac_n-0.5*fac_n*fac_n+fac_n*fac_n*fac_n/6.;} else {dfac_n=1.-exp(-fac_n);}
                                double slope_inj = bin_slopes[j]; // slope dN/dp of CRs doing the injection -- should be conserved in production
                                int m; for(m=0;m<N_CR_PARTICLE_SPECIES;m++) {
                                    int j_s = CR_secondary_target_bin[j][m]; /* destination bin for secondary product 'm' */
                                    if(j_s < -1) {break;} /* no more secondaries exist, cease this loop */
                                    double secondary_coeff = CR_frag_secondary_coeff[j][m]*nHcgs; /* rate of secondary production via fragmentation scales with this */
                                    if(m==0) {secondary_coeff += rad_coeff;} /* 100% of radioactive products go into the first secondary bin */
                                    if(j_s < 0 || secondary_coeff <= 0) {continue;} /* no production in this particular bin/channel */
                                    double frac_secondary = DMAX(0.,DMIN(1., secondary_coeff / total_catastrophic_coeff)); /* restrict to sensible bounds */
                                    if(CR_species_ID_in_bin[j_s] < 0) {frac_secondary *= 1./HYDROGEN_MASSFRAC;} // crude correction for He secondary e-/e+ production terms

                                    double U_donor = frac_secondary*dfac_e*Ucr[j] * DMAX(1.,A_wt[j_s])/DMAX(1.,A_wt[j]); // need to account for the different total energy assuming fixed energy per nucleon here
                                    if(CR_species_ID_in_bin[j_s] < 0) {U_donor *= 0.1;} // secondary e+/e- from protons (pion decay) get ~0.1 original p energy -- needs to match assumption above
                                    if(CR_species_ID_in_bin[j_s] == 7) {U_donor *= 0.08;} // pbar get ~0.08 original p energy -- needs to match assumption above
                                    double N_donor = frac_secondary*dfac_n*ntot_evolved[j]; // absolute number being transferred between bins
                                    
                                    if(CR_species_ID_in_bin[j]==6 && CR_species_ID_in_bin[j_s]==5) {U_donor *= 0.9;} // 10Be assumption needs tiny correction b/c of mean molecular weight of CNO bin putting it slightly in the wrong place (giving problematic slopes)
                                    int split_two_bin=0, j2=-1, js2=-1; /* for some species where we have a big energy jump in the parent and not secondary (e.g. hadrons -> leptons) we get 'jumps' in the spectrum, which produce artificial features; attempt to smooth these out by distributing over a pair of bins */
                                    if((CR_species_ID_in_bin[j_s]<0 || CR_species_ID_in_bin[j_s]==7) && (k<n_active-1) && k>0) {
                                        j2=bins_sorted[n_active-1-(k+1)]; if(CR_species_ID_in_bin[j2]==CR_species_ID_in_bin[j]) {
                                            js2=CR_secondary_target_bin[j2][m]; if(CR_species_ID_in_bin[j_s]==CR_species_ID_in_bin[js2]) {
                                                if(js2 < j_s-1) {split_two_bin=1; U_donor *= 0.5; N_donor *= 0.5;}}}}
                                    
                                    /* instead of conserving U and N separately, which can cause problems in this step with unphysical slopes owing to discreteness, conserve U and dN/dp */
                                    double E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s),egy_slopemode_s=1,xm_s=CR_global_min_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s],xp_s=CR_global_max_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s],xm_e_s=xm_s, xp_e_s=xp_s;
                                    if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                    double gamma_one_s=slope_inj+1., xm_gamma_one_s=pow(xm_s,gamma_one_s), xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                    N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin

                                    Ucr[j_s] += U_donor; // update energy in secondary bin
                                    ntot_evolved[j_s] += N_donor; // update number in secondary bin
                                    bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s); // get the updated slope for this bin
                                    
                                    if(split_two_bin==1) {j_s -= 1;
                                        double facU=0.5/0.5; U_donor *= facU; //facN=0.5/0.5; /* share over a second bin */
                                        Ucr[j_s] += U_donor;
                                        /* repeat exercise to obtain N_donor */
                                        E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s); egy_slopemode_s=1; xm_s=CR_global_min_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s]; xp_s=CR_global_max_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s]; xm_e_s=xm_s; xp_e_s=xp_s;
                                        if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                        gamma_one_s=slope_inj+1.; xm_gamma_one_s=pow(xm_s,gamma_one_s); xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                        N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin

                                        //ntot_evolved[j_s] += facN * N_donor;
                                        ntot_evolved[j_s] += N_donor; // update number in secondary bin
                                        bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s);
                                    }
                                    
                                    if(split_two_bin==0 && CR_species_ID_in_bin[j]==1 && k==0 && (CR_species_ID_in_bin[j_s]<0 || CR_species_ID_in_bin[j_s]==7)) { /* now code extending the CR spectrum of secondary production to energies higher than our max limit, assuming continued power-law extrapolation of the CR spectrum */
                                        double Rx0=CR_global_rigidity_at_bin_center[j_s], U00=U_donor, xm_0=CR_global_min_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s], xp_0=CR_global_max_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s];
                                        int spec_0=CR_species_ID_in_bin[j_s], slope_0=2.+slope_inj; slope_0=DMAX(-4.,DMIN(slope_0,0.)); if(spec_0==7) {slope_0=DMIN(slope_0,-0.7);}
                                        j_s++;
                                        while(j_s<N_CR_PARTICLE_BINS && CR_species_ID_in_bin[j_s]==spec_0) {
                                            double Rx1=CR_global_rigidity_at_bin_center[j_s];
                                            E_GeV_s=return_CRbin_kinetic_energy_in_GeV_binvalsNRR(j_s); egy_slopemode_s=1; xm_s=CR_global_min_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s]; xp_s=CR_global_max_rigidity_in_bin[j_s]/CR_global_rigidity_at_bin_center[j_s]; xm_e_s=xm_s; xp_e_s=xp_s;
                                            if(CR_check_if_bin_is_nonrelativistic(j_s)) {egy_slopemode_s=2; xm_e_s=xm_s*xm_s; xp_e_s=xp_s*xp_s;} // values needed to scale from slope injected to number and back
                                            gamma_one_s=slope_inj+1.; xm_gamma_one_s=pow(xm_s,gamma_one_s); xp_gamma_one_s=pow(xp_s,gamma_one_s); // variables below
                                            U_donor = U00 * pow(Rx1/Rx0,slope_0) * log(xp_s/xm_s)/log(xp_0/xm_0);
                                            N_donor = (U_donor/E_GeV_s) * ((gamma_one_s + egy_slopemode_s) / (gamma_one_s)) * (xp_gamma_one_s - xm_gamma_one_s) / (xp_gamma_one_s*xp_e_s - xm_gamma_one_s*xm_e_s); // injected number in bin
                                            Ucr[j_s] += U_donor;
                                            ntot_evolved[j_s] += N_donor;
                                            bin_slopes[j_s] = CR_return_slope_from_number_and_energy_in_bin(Ucr[j_s],ntot_evolved[j_s],E_GeV[j_s],j_s); j_s++;
                                        }}
                                }
                            }
                            Ucr[j] *= exp(-fac_e); ntot_evolved[j] *= exp(-fac_n);
                        }
#else
                        if(E_GeV[j] > 0.28) {double fac=exp(-DMIN(hadronic_coeff*dt*M1SpeedCorrFac[j], 60.)); Ucr[j]*=fac; ntot_evolved[j]*=fac;} // only >~GeV trigger threshold for collisions //
#endif
                        dn_flux=0; de_flux=0; // make sure these are zero'd for next step
                        continue; // these -destroy- CRs, decreasing N and E identically [ignoring secondary production for now], leaving slope un-modified and -no- bin-to-bin fluxes: easy to solve, just modify total energy+number identically in the bin //
                    }
                    
                    double etot = Ucr[j]; // total energy in bin, one of our key evolved variables
                    double E_bin_NtoE = E_GeV[j]; // use this to normalize the number in a convenient unit (just to keep easier to track, units are arbitrary). save it so we don't forget below.
                    double slope_gamma = bin_slopes[j]; // get the initial slope if we evolve it and can pass it as a conserved quantity
                    double ntot = ntot_evolved[j]; // total number in bin [in our effective units] //

                    double xm = x_m[j], xp = x_p[j], gamma_one = slope_gamma+1., xe_gamma_one = 0, xm_gamma_one = 0, xp_gamma_one = 0; // just for convenience below
                    /* // old mode below, where we directly evolve slope and derived number from it
                    double ntot = 0; // total number in bin. we use the slope we obtain to convert between number and energy, given the bin-centered values
                    xm_gamma_one = pow(xm, gamma_one); xp_gamma_one = pow(xp, gamma_one); // variables needed for conversion below
                    if(NR_key[j]) {ntot = ((gamma_one + 2.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // non-relativistic map between E and N
                        else {ntot = ((gamma_one + 1.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp - xm_gamma_one*xm);} // relativistic map between E and N
                    ntot *= (etot / E_bin_NtoE); // dimensional normalization. note the units are arbitrary for the E_bin_NtoE term, as long as we are consistent [we can evolve constant x N, instead of N, for convenience]
                    */
                    
                    double dn_flux_fromprevbin = dn_flux, de_flux_fromprevbin = de_flux; // fluxes coming from the last bin. note these have -already- been cooled, so don't want to double-count that operation, hence we add them at the end of the sub-step
                    double extra_var_topass_for_xe = 0; // dummy variable to pass to subroutine below
                    dn_flux = 0; de_flux = 0; // reset these, they will be re-defined below but in case we skip the relevant loop they need to be zeroed
                    if((etot <= 0 && de_flux_fromprevbin <= 0) || (ntot <= 0 && dn_flux_fromprevbin <= 0)) {continue;} // no CRs to actually work with here [nothing injected yet]!
                    
                    double rate_prefac = 0; // coefficient for the bin-centered loss rate from this mode [-negative- loss rate = energy-gain]
                    if(loss_mode==1 || loss_mode==4) // adiabatic + brems + streaming
                    {
                        if(loss_mode==1) {rate_prefac = adiabatic_coeff[j];} // adiabatic always in mode=1
                        if((sign_flip_adiabatic_terms==0) || (loss_mode==4)) {rate_prefac += brems_coeff[j] + streaming_coeff[j];} // no sign-flip, or in mode=4 so we do add the strictly-loss terms
                    }
                    if(loss_mode==2) // coulomb + ion
                    {
                        if(CR_species_ID_in_bin[j] < 0) // electron or positron ionization losses here
                        {
                            rate_prefac = (e_coulomb_coeff + (1.+0.07*log(E_GeV[j])) * e_ion_coeff) / R0[j]; // always in relativistic limit here. 1/R0 is b/c this coefficient is defined normalized to the bin center in GeV, to make the equations dimensionless
                        } else { // proton or nuclei ionization + Coulomb losses here
                            rate_prefac = (fabs(Z[j])/R0[j]) * coulomb_coeff; // relativistic expression. note this is equation for p evolution, where R is rigidity, so need to be careful with Z factors, etc.
                            if(NR_key[j]) {double A=A_wt[j]; rate_prefac *= 0.88 *A*A / (R0[j]*R0[j]*fabs(Z[j]*Z[j]));} // non-relativistic expression (times constant (A/(R0*Z))^2 //
                        }
                    }
                    if(loss_mode==3) {rate_prefac = (E_GeV[j]/E_rest_e_GeV) * synchIC_coeff_0;} // IC + synch
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                    if(loss_mode==5) {if(slope_gamma>2.) {rate_prefac=0;} else {rate_prefac=reaccel_coeff[j]*(1.-0.5*slope_gamma); extra_var_topass_for_xe=delta_diffcoeff[j];}} // we will treat gamma as constant over this integration sub-step
#endif
                    
                    int do_cooling_ops = 1; // option to skip the cooling subcycle part of the step here:
                    if(fabs(rate_prefac) < 0.01*fabs(bin_centered_rate_coeff[j])+MIN_REAL_NUMBER) {do_cooling_ops=0;} // this loss mode is negligible here, skip it [pure optimization]
                    if(fabs(bin_centered_rate_coeff[j])*dt*M1SpeedCorrFac[j] < 1.e-16 * Ucr[j]) {do_cooling_ops=0;} // total loss rate is so small we won't be able to track it to floating-point accuracy, skip it
                    if(loss_mode==1 || loss_mode==4) {if((rate_prefac < 0 && order == -1) || (rate_prefac > 0 && order == 1)) {do_cooling_ops=0;}} // adiabatic: make sure we are operating in correct order (will ignore bin if we have a sign switch, which is ok, since it must be nearly-null in that case
                    
                    if(do_cooling_ops) // enter the 'cooling' (bin-to-bin flux calculations) portion of the subcycle
                    {
                        double rate_dt = rate_prefac * dt * M1SpeedCorrFac[j]; // dimensionless step size with rate prefactor times timestep
                        double x_to_edge = CR_return_new_bin_edge_from_rate(rate_dt, xm, xp, loss_mode, NR_key[j], extra_var_topass_for_xe); // returns the dimensionless 'x' representing the particles furthest from bin edge that 'reach' the edge by end of this sub-step
                        gamma_one = slope_gamma+1.; xe_gamma_one = pow(x_to_edge, gamma_one); xm_gamma_one = pow(xm, gamma_one); xp_gamma_one = pow(xp, gamma_one);
                        
                        double dn_lostfrombin = 0; // calculate the number flux integrating out to that new bin edge
                        if(rate_prefac < 0) {dn_lostfrombin = ntot * (1 - xe_gamma_one / xp_gamma_one) / (1 - xm_gamma_one / xp_gamma_one);}
                            else {dn_lostfrombin = ntot * (xe_gamma_one / xm_gamma_one - 1.) / (xp_gamma_one / xm_gamma_one - 1.);}
                        
                        double etot_final_inbin = etot; // calculate the final energy of all particles still inside bin bounds at end of the sub-step
                        double etot_final_allparticlesfrombin = etot; // calculate the final energy of all particles that began sub-step within bin bounds [difference is net flux of energy out-of-bin]
                        
                        if(loss_mode==1 || loss_mode==4) // adiabatic + brems
                        {
                            double norm_fac = exp(-rate_dt), x1, x0, x1_gamma_one, x0_gamma_one; // now make sure we flip the sign convention back to normal for rate
                            if(rate_dt < 0) {x1=x_to_edge; x1_gamma_one=xe_gamma_one; x0=xm; x0_gamma_one=xm_gamma_one;} else {x1=xp; x1_gamma_one=xp_gamma_one; x0=x_to_edge; x0_gamma_one=xe_gamma_one;} // this is everything that remains in bin. so if decaying, runs from x_m' to xp; if growing [rate < 0] runs from xm to x_p'
                            if(NR_key[j])
                            {
                                norm_fac *= norm_fac; // exp(2*rate*dt) because of KE going as p^2
                                etot_final_inbin = etot * norm_fac * (x1_gamma_one*x1*x1 - x0_gamma_one*x0*x0) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm); // extra power of x from extra power of p in KE
                            } else {
                                etot_final_inbin = etot * norm_fac * (x1_gamma_one*x1 - x0_gamma_one*x0) / (xp_gamma_one*xp - xm_gamma_one*xm); // relativistic limit expression
                            }
                            etot_final_allparticlesfrombin = etot * norm_fac; // easy because all particles lose the same fractional energy
                        }
                        if(loss_mode==2) // coulomb + ion
                        {
                            if(NR_key[j])
                            {
                                double norm_fac = etot * (gamma_one+2.) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm); // get the pre-factor for the integral from numbers we already have
                                double xmin_remains = pow(3.*rate_dt, 1./3.); // sets absolute minimum value of x for which any energy remains at all, needs to be integrated only from this bound upwards
                                int n_int=10; double xmin=DMAX(xm,xmin_remains), u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xmin*exp_fac, f0=CR_coulomb_energy_integrand(xmin,rate_dt,gamma_one), f1, exp_fac_2=exp_fac*exp_fac, x0=xmin, x1;
                                while(x < xp)
                                {
                                    x1 = DMIN(x*exp_fac, xp);
                                    f1 = CR_coulomb_energy_integrand( x1 , rate_dt, gamma_one);
                                    double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                    u_int += u_int_fac;
                                    if(x0 < x_to_edge) {if(x1 > x_to_edge) {double f1_e=CR_coulomb_energy_integrand(x_to_edge,rate_dt,gamma_one); u_int_xe+=log(x_to_edge/x0)*(f1_e-f0)/(log((f1_e+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}}
                                    x *= exp_fac_2; f0 = f1; x0 = x1;
                                }
                                etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
                            } else {
                                double d_egy_fac = rate_dt * ntot * E_bin_NtoE; // re-normalizes ntot correctly back in our chosen units to the -same- units as etot, for use below
                                etot_final_inbin = etot * (xp_gamma_one*xp - xe_gamma_one*x_to_edge) / (xp_gamma_one*xp - xm_gamma_one*xm)
                                    - d_egy_fac * (xp_gamma_one - xe_gamma_one) / (xp_gamma_one - xm_gamma_one); // relativistic limit expression: accounts both for particles leaving the bin [first term] and total energy lost by all particles that stay in the bin [second term]
                                etot_final_allparticlesfrombin = etot - d_egy_fac; // easy because all particles lose the same absolute energy
                            }
                        }
                        if(loss_mode==3) // IC + synch
                        {
                            double norm_fac = etot * (gamma_one+1.) / (xp_gamma_one*xp - xm_gamma_one*xm); // get the pre-factor for the integral from numbers we already have
                            int n_int=10; double u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xm*exp_fac, f0=CR_compton_energy_integrand(xm,rate_dt,gamma_one), f1, exp_fac_2=exp_fac*exp_fac, x0=xm, x1;
                            while(x < xp)
                            {
                                x1 = DMIN(x*exp_fac, xp);
                                f1 = CR_compton_energy_integrand( x1 , rate_dt, gamma_one);
                                double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                u_int += u_int_fac;
                                if(x0 < x_to_edge) {if(x1 > x_to_edge) {double f1_e=CR_compton_energy_integrand(x_to_edge,rate_dt,gamma_one); u_int_xe+=log(x_to_edge/x0)*(f1_e-f0)/(log((f1_e+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}}
                                x *= exp_fac_2; f0 = f1; x0 = x1;
                            }
                            etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
                        }
                        if(loss_mode==5) // diffusive re-acceleration
                        {
#if defined(CRFLUID_ALT_REACCEL_ONLY_DIFFUSIVE)
                            double norm_fac = etot * (gamma_one+1.) / (xp_gamma_one*xp - xm_gamma_one*xm); // get the pre-factor for the integral from numbers we already have
                            if(NR_key[j]) {norm_fac = etot * (gamma_one+2.) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // use correct NR form
                            double delta_j = delta_diffcoeff[j], rate_dt_abs = fabs(rate_dt); // value of delta-slope needed below
                            int n_int=10; double xmin=xm, u_int=0, u_int_xe=0, dlnx=log(xp/xm)/((double)n_int), exp_fac=exp(dlnx/2.), x=xmin*exp_fac, f0=CR_reaccel_energy_integrand(xmin,rate_dt_abs,gamma_one,delta_j,NR_key[j]), f1, exp_fac_2=exp_fac*exp_fac, x0=xmin, x1;
                            while(x < xp)
                            {
                                x1 = DMIN(x*exp_fac, xp);
                                f1 = CR_reaccel_energy_integrand( x1 , rate_dt_abs , gamma_one , delta_j , NR_key[j]);
                                double u_int_fac = log(x1/x0) * (f1 - f0) / (log((f1+MIN_REAL_NUMBER)/(f0+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);
                                u_int += u_int_fac;
                                if(x1 > x_to_edge) {if(x0 < x_to_edge) {double f0_e=CR_reaccel_energy_integrand(x_to_edge, rate_dt_abs, gamma_one, delta_j , NR_key[j]); u_int_xe += log(x1/x_to_edge)*(f1-f0_e)/(log((f1+MIN_REAL_NUMBER)/(f0_e+MIN_REAL_NUMBER)) + MIN_REAL_NUMBER);} else {u_int_xe+=u_int_fac;}} // // integrate from x_edge to f1, or x0+x1 both > x_to_edge, so whole bin is outside range
                                x *= exp_fac_2; f0 = f1; x0 = x1;
                            }
                            etot_final_inbin = (u_int - u_int_xe) * norm_fac; etot_final_allparticlesfrombin = u_int * norm_fac; // normalize these appropriately
#endif
                        }
                        
                        dn_lostfrombin = DMIN(dn_lostfrombin , (1.0-1.e-8)*ntot); // limit to prevent 0's which will give nan's
                        etot_final_inbin = DMAX(etot_final_inbin , 1.e-8*etot); // limit to prevent 0's which will give nan's
                        dn_flux = dn_lostfrombin; // set total outgoing flux of number of particles to next bin
                        de_flux = DMAX(etot_final_allparticlesfrombin - etot_final_inbin, 0); // determine total outgoing flux of energy to next bin

                        ntot -= dn_lostfrombin; // update number
                        etot = etot_final_inbin; // update energy
                    } // close the bin-to-bin flux calculation portion of the subcycle

                    ntot += dn_flux_fromprevbin; // update the bin with the -already cooled- CRs from the previous bin
                    etot += de_flux_fromprevbin; // update the bin with the -already cooled- CRs from the previous bin
                    if(ntot > 0 && etot > 0) {slope_gamma = CR_return_slope_from_number_and_energy_in_bin(etot, ntot, E_bin_NtoE, j);} else {slope_gamma=0;} // get the updated slope for this bin

                    ntot_evolved[j] = ntot; // set finalized updated number in bin
                    Ucr[j] = etot; // set finalized updated energy in bin
                    bin_slopes[j] = slope_gamma; // set finalized updated slope [or N, if that's the conserved quantity we evolve]
                } // loop over active bins
            } // loop over loss mode
            t += dt; // update time
        } // loop over charge
    } // loop over time
    
    if(mode_driftkick==0) {for(k=0;k<N_CR_PARTICLE_BINS;k++) {double fac=1; if(Ucr_i[k]>0 && Ucr[k]>0) {fac=DMAX(1.e-10,DMIN(1.e10,Ucr[k]/Ucr_i[k]));} CellP[target].CosmicRayEnergyPred[k] *= fac;}}
    /* update fluxes. these will self-adapt quickly but should feel losses as well, so we do that here */
    int kdir; for(k=0;k<N_CR_PARTICLE_BINS;k++) {double fac=1; if(Ucr_i[k]>0 && Ucr[k]>0) {fac=DMAX(1.e-2,DMIN(1.e2,Ucr[k]/Ucr_i[k]));}
        for(kdir=0;kdir<3;kdir++) {if(mode_driftkick==0) {CellP[target].CosmicRayFlux[k][kdir] *= fac; CellP[target].CosmicRayFluxPred[k][kdir] *= fac;} else {CellP[target].CosmicRayFluxPred[k][kdir] *= fac;}}}
    return; // all done!
}




/* initialize CR quantities needed specifically for our multi-bin spectral methods */
void CR_initialize_multibin_quantities(void)
{
    if(ThisTask==0) {printf("Initializing global cosmic ray spectral variables:\n"); fflush(stdout);}
    int k; CR_spectrum_define_bins(); // call this to define the actual list of bins, needs to be done before basically anything else!
    double R0[N_CR_PARTICLE_BINS], Z[N_CR_PARTICLE_BINS];
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {R0[k]=CR_global_rigidity_at_bin_center[k]; Z[k]=return_CRbin_CR_charge_in_e(-1,k);}
    double R0_bins[N_CR_PARTICLE_BINS], R0_bin_m[N_CR_PARTICLE_BINS], R0_bin_p[N_CR_PARTICLE_BINS];
    int cr_species_key; for(cr_species_key=0;cr_species_key<N_CR_PARTICLE_SPECIES;cr_species_key++) // loop over whether we consider nuclei or electrons, first //
    {
       int n_active=0, bins_sorted[N_CR_PARTICLE_BINS];
       for(k=0;k<N_CR_PARTICLE_BINS;k++) {if(CR_species_ID_in_bin[k]==CR_species_ID_active_list[cr_species_key]) {bins_sorted[n_active]=k; R0_bins[n_active]=R0[k]; n_active++;}} // bin is valid: charge matches that desired
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
    for(k=0;k<N_CR_PARTICLE_BINS;k++) {CR_global_min_rigidity_in_bin[k] = R0_bin_m[k]; CR_global_max_rigidity_in_bin[k] = R0_bin_p[k];} // set the variables we just defined

    /* ok, now we need to build the lookup tables */
    double gamma_limit = 120.;
    int n_gamma_sample = 10000, n_table = N_CR_SPECTRUM_LUT-1;
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        double xm = CR_global_min_rigidity_in_bin[k] / CR_global_rigidity_at_bin_center[k]; // dimensionless bin minimum
        double xp = CR_global_max_rigidity_in_bin[k] / CR_global_rigidity_at_bin_center[k]; // dimensionless bin maximum
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
                if(j==1 && slope_gamma < gamma_min) {CR_global_slope_lut[k][j-1] = slope_gamma + gamma_min;}
                CR_global_slope_lut[k][j] = slope_gamma; // set the look-up-table value for use later
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
        if(alldone_key==0 && j<n_table) {int j0=j-1; for(j=j0+1;j<N_CR_SPECTRUM_LUT;j++) {CR_global_slope_lut[k][j]=CR_global_slope_lut[k][j0]+(j-j0)*DMIN(gamma_limit,fabs(gamma_limit-CR_global_slope_lut[k][j0]));}}
            else {CR_global_slope_lut[k][N_CR_SPECTRUM_LUT-1]=gamma_limit;} // set upper end of table value to unity
    }
    
    if(ThisTask==0) {for(k=0;k<N_CR_PARTICLE_BINS;k++) { // print outputs for users
        printf("\n .. bin=%d, charge=%g e, mass=%g mp, rigidity Rmin=%g R0=%g Rmax=%g GV, energy=%g GeV, relativistic?=%d [1=Y/0=N] beta=%g gamma=%g \n",
           k,return_CRbin_CR_charge_in_e(-1,k),return_CRbin_CRmass_in_mp(-1,k),CR_global_min_rigidity_in_bin[k],CR_global_rigidity_at_bin_center[k],
           CR_global_max_rigidity_in_bin[k],return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k),1-CR_check_if_bin_is_nonrelativistic(k),return_CRbin_beta_factor(-1,k),
           return_CRbin_gamma_factor(-1,k)); fflush(stdout);
        printf(" .. LUT for CR slopes in this bin: \n"); printf("  .. j  .. R_egy/num .. gamma \n");
        int j; for(j=0;j<N_CR_SPECTRUM_LUT;j++) {printf("  .. %4d  %5.4g %10.3g \n",j,((double)j)/((double)n_table),CR_global_slope_lut[k][j]); fflush(stdout);}
    }}
    
    return;
}



/* input value 'R' = ratio of total CR energy in the bin ('e_tot') to the total CR number times the energy of the CRs with the bin-centered rigidity ('n_tot' x 'E_cr_bin_center_list'), which is a dimensionless function of the slope, whether the bin is relativistic or not, and the bin edges relative to the bin center */
double CR_return_slope_from_number_and_energy_in_bin(double energy_in_code_units, double number_effective_in_code_units, double bin_centered_energy_in_GeV, int k_bin)
{
    if((energy_in_code_units <= 0.) || isnan(energy_in_code_units) || (number_effective_in_code_units <= 0.) || isnan(number_effective_in_code_units)) {return CR_global_slope_lut[k_bin][0];}
    double R = energy_in_code_units / (number_effective_in_code_units * bin_centered_energy_in_GeV + MIN_REAL_NUMBER);
    int n_table = N_CR_SPECTRUM_LUT; // table size
    double xm = CR_global_min_rigidity_in_bin[k_bin] / CR_global_rigidity_at_bin_center[k_bin], xp = CR_global_max_rigidity_in_bin[k_bin] / CR_global_rigidity_at_bin_center[k_bin], xm_e = xm, xp_e = xp;
    if(CR_check_if_bin_is_nonrelativistic(k_bin)) {xm_e=xm*xm; xp_e=xp*xp;} // sets bounds that this value can possibly obtain
    if(R >= xp_e) {return CR_global_slope_lut[k_bin][n_table-1];} // set to maximum
    if(R <= xm_e) {return CR_global_slope_lut[k_bin][0];} // set to minimum
    double n_interp = (log(R / xm_e) / log(xp_e / xm_e)) * (n_table-1); // fraction of the way between min and max in our log-spaced table, returns 0-1
    int n0 = (int) floor(n_interp), n1=n0+1;
    if(n1 > n_table-1) {n1=n_table-1;}
    double slope_gamma = CR_global_slope_lut[k_bin][n0] + (n_interp-(double)n0) * (CR_global_slope_lut[k_bin][n1]-CR_global_slope_lut[k_bin][n0]);
    // check for specific bad values that will nan out and avoid them - this introduces really minimal errors
    double tol = 0.0001, badval; // tolerance around the bad values
    badval=-1; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    badval=-2; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    badval=-3; if(fabs(slope_gamma - badval) < tol) {if(slope_gamma<badval) {slope_gamma=badval-tol;} else {slope_gamma=badval+tol;}}
    return slope_gamma; // ok, safe to return
}


/* return solution for the distance in dimensionless terms from the bin edge for CRs which can propagate to the edge with the dimensionless rate factor given */
double CR_return_new_bin_edge_from_rate(double rate_dt_dimless, double x_m_bin, double x_p_bin, int loss_mode, int NR_key, double additional_variable_dummy)
{
    if(loss_mode <= 0) {return 0;} // hadronic+catastrophic: should never be called [just should never get to this point since hadronic should be done]
    
    double x_e = 0;
    if(loss_mode == 1 || loss_mode == 4) // adiabatic+brems+streaming
        {if(rate_dt_dimless < 0) {x_e = x_p_bin * exp(rate_dt_dimless);} else {x_e = x_m_bin * exp(rate_dt_dimless);}} // return xp' or xm' depending on sign: solution to dx/dtau=(+/-)x

    if(loss_mode == 2) // coulomb+ionization
    {
        if(NR_key) // non-relativistic Coulomb+ionization terms
            {x_e = pow(x_m_bin*x_m_bin*x_m_bin + 3.*rate_dt_dimless , 1./3.);} // solution to dx/dtau=-1/x^2
        else // relativistic Coulomb+ionization terms
            {x_e = x_m_bin + rate_dt_dimless;} // solution to dx/dtau=-1
    }

    if(loss_mode == 3) // IC+synchrotron
        {if(rate_dt_dimless*x_m_bin < 1.) {x_e = x_m_bin / (1. - rate_dt_dimless * x_m_bin);} else {x_e = x_p_bin;}} // solution for IC+synchrotron: dx/dtau=-x^2
    
    if(loss_mode == 5) // re-acceleration
        {x_e = pow( DMAX(pow(x_p_bin, additional_variable_dummy) + rate_dt_dimless, 0.), 1./additional_variable_dummy);} // solution to dlnx/dtau = x^-delta; here using additional_variable_dummy = delta_diffcoeff[k]
    
    if(x_e < x_m_bin) {x_e=x_m_bin;}
    if(x_e > x_p_bin) {x_e=x_p_bin;}
    return x_e; // catch
}

                                                                                                                              
/* integrand needed for numerical evaluation of non-relativistic Coulomb loss terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
double CR_coulomb_energy_integrand(double x, double tau, double slope)
{
    return pow(x,slope) * pow(x*x*x - 3.*tau , 2./3.);
}


/* integrand needed for numerical evaluation of re-acceleration energy gain terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
double CR_reaccel_energy_integrand(double x, double tau, double slope, double delta_slope, int NR_key)
{
    double R_slope = 1./delta_slope;
    if(NR_key) {R_slope *= 2.;}
    return pow(x,slope) * pow( pow(x, delta_slope) + tau , R_slope );
}


/* integrand needed for numerical evaluation of Compton loss terms; this should be G in int_x0^x1 [G] dlnx_old = E_new, so G = x_old * (dN/dx)_old * E_new, in dimensionless bin-centered units: it's crucial here that 'slope' input is ~df/dlogx = "gamma_one" in the notation above, as opposed to "slope_gamma" */
double CR_compton_energy_integrand(double x, double tau, double slope)
{
    return pow(x,slope+1.) / (1. + tau*x);
}


/* quick boolean to determine if we should use relativistic or non-relativistic scalings for a given bin, in our cooling subroutines where we need this to be true across the bin */
int CR_check_if_bin_is_nonrelativistic(int k_bin) // relativistic binflag: all e-, protons/nuclei with R_GV > 1.87655 * A/Z = 1.87655 for protons : need this globally
{
    if(return_CRbin_CR_species_ID(k_bin) <= 0) {return 0;} // assume e-, e+ relativistic here
    else {
        double m_cr_mp = return_CRbin_CRmass_in_mp(-1,k_bin), Zabs = fabs(return_CRbin_CR_charge_in_e(-1,k_bin)); // mass in proton masses, charge in e
        if(CR_global_rigidity_at_bin_center[k_bin] < 1.87655*m_cr_mp/Zabs) {return 1;} else {return 0;} // use a simple momentum criterion, dividing exactly for protons as desired here
    }
    //if((CR_species_ID_in_bin[k_bin] > 0) && (CR_global_rigidity_at_bin_center[k_bin] < 1.87655)) {return 1;} // simpler version if just using e- and p; for now, only protons with division at p=2m0*c, so NR and R expressions equate, qualify as non-relativistic
    return 0; // default to relativistic otherwise
}


/* return number of CRs in bin, in our strange code units, where the 'number' has units of E_code / GeV. to convert to real number need to multiply by this but that can cause lots of overflow issues so we work with this unit */
double CR_return_effective_number_in_bin_in_codeunits(int target, int k_bin)
{
    return CellP[target].CosmicRay_Number_in_Bin[k_bin];
}


/* return the CR bin spectral slope, depending on what we use as our evolved variable -- note this is the slope of the 1D CR distribution function, df/dp ~ p^alpha, so e.g. N_cr = integral[dp * df/dp] */
double CR_return_spectral_slope_target(int target, int k_bin)
{
    //return CellP[target].CosmicRay_PwrLaw_Slopes_in_Bin[k_bin]; // evolving slopes directly
    return CR_return_slope_from_number_and_energy_in_bin(CellP[target].CosmicRayEnergy[k_bin], CellP[target].CosmicRay_Number_in_Bin[k_bin], return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_bin), k_bin); // calculate if not evolving directly
}


/* return true number of CRs in bin, in actual units */
double CR_return_true_number_in_bin(int target, int k_bin)
{
    return CR_return_effective_number_in_bin_in_codeunits(target,k_bin) * UNIT_ENERGY_IN_CGS / (1.0e9*ELECTRONVOLT_IN_ERGS); // does the unit conversion from Ecode/GeV to dimensionless number
}


/* subroutine to return the effective CR number from the bin slope and energy */
double CR_get_number_in_bin_from_slope(int target, int k_bin, double energy, double slope)
{
    double etot = energy;
    double gamma_one = 1. + slope;
    double E_bin_center = return_CRbin_kinetic_energy_in_GeV_binvalsNRR(k_bin);
    double xm = CR_global_min_rigidity_in_bin[k_bin] / CR_global_rigidity_at_bin_center[k_bin];
    double xp = CR_global_max_rigidity_in_bin[k_bin] / CR_global_rigidity_at_bin_center[k_bin];
    double xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one);
    int NR_key = CR_check_if_bin_is_nonrelativistic(k_bin);
    double gamma_fac = 0; // factor to get total number in bin. we use the slope we obtain to convert between number and energy, given the bin-centered values
    if(NR_key) {gamma_fac = ((gamma_one + 2.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp*xp - xm_gamma_one*xm*xm);} // non-relativistic map between E and N
        else {gamma_fac = ((gamma_one + 1.) / (gamma_one)) * (xp_gamma_one - xm_gamma_one) / (xp_gamma_one*xp - xm_gamma_one*xm);} // relativistic map between E and N
    return (etot / E_bin_center) * gamma_fac; // dimensional normalization. note the units are arbitrary for the E_bin_NtoE term, as long as we are consistent [we can evolve constant x N, instead of N, for convenience]
 }


/* return mean kinetic energy per CR for CRs in bin */
double CR_return_mean_energy_in_bin_in_GeV(int target, int k_bin)
{
    double etot = CellP[target].CosmicRayEnergyPred[k_bin]; // total energy in bin
    double ntot_GeV = CR_return_effective_number_in_bin_in_codeunits(target, k_bin); // total number in units of GeV
    return etot / ntot_GeV; // this returns the mean weighted over the CR spectrum
}


/* return mean rigidity in GV per CR for CRs in bin */
double CR_return_mean_rigidity_in_bin_in_GV(int target, int k_bin)
{
    double slope = CR_return_spectral_slope_target(target, k_bin);
    double gamma_one = 1. + slope;
    double R0 = CR_global_rigidity_at_bin_center[k_bin], xm = CR_global_min_rigidity_in_bin[k_bin] / R0, xp = CR_global_max_rigidity_in_bin[k_bin] / R0, xm_gamma_one = pow(xm, gamma_one), xp_gamma_one = pow(xp, gamma_one);
    double gamma_fac = (gamma_one/(gamma_one+1.)) * (xp_gamma_one*xp - xm_gamma_one*xm) / (xp_gamma_one - xm_gamma_one); // dimensionless term from weighted integral over the bin
    return R0 * gamma_fac; // returns value appropriately weighted over the bin
}


/* sort function if we need to sort CR bins by rigidity from lowest to highest for future compatibility */
int compare_CR_rigidity_for_sort(const void *a, const void *b)
{
    double x = CR_global_rigidity_at_bin_center[ *(int *) a];
    double y = CR_global_rigidity_at_bin_center[ *(int *) b];
    if (x < y) {return -1;} else if(x > y) {return 1;}
    return 0;
}


/* routine to return bin-centered energy using the slope approximation used in the fast cooling sub-cycling, of each bin being in the relativistic or non-relativistic regime */
double return_CRbin_kinetic_energy_in_GeV_binvalsNRR(int k_CRegy)
{
    double R_GV = CR_global_rigidity_at_bin_center[k_CRegy];
    double Zabs = fabs(return_CRbin_CR_charge_in_e(-1,k_CRegy));
    if(CR_check_if_bin_is_nonrelativistic(k_CRegy))
    {
        double fac = 0.5328928536929374; // converts from R_GV to E in GeV for E = p^2/(2m), assuming Z=1, m=mp
        double m_cr_mp = return_CRbin_CRmass_in_mp(-1,k_CRegy); // mass in proton masses
        return fac * R_GV*R_GV * (Zabs*Zabs / m_cr_mp); // E in GeV in non-relativistic limit
    }
    return R_GV * Zabs; // E in GeV in relativistic limit
}


#endif


/* optional code to allow the RSOL to depend on bin energy, still testing this */
double return_CRbin_M1speed(int k_CRegy)
{
    double vmax = CRFLUID_SPEEDOFLIGHT_REDUCTION * C_LIGHT_CODE;
#if defined(CRFLUID_ALT_VARIABLE_RSOL) && (N_CR_PARTICLE_BINS > 1)    /* experimental block here */
    double R = CR_global_rigidity_at_bin_center[k_CRegy];
    double f = All.CosmicRayDiffusionCoeff * UNIT_LENGTH_IN_KPC * pow(R , 0.8);
    if(f > vmax) {return f;}
#endif
    return vmax;
}


/* estimate amount by which flux of CRs has been reduced relative to solution with c_reduced = c_true, for RSOL with M1 */
double evaluate_cr_transport_reductionfactor(int target, int k_CRegy, int mode)
{
#if defined(CRFLUID_ALT_RSOL_FORM)
    return CosmicRayFluid_RSOL_Corrfac(k_CRegy); // uniform reduction factor for all terms
#else
    double kappa = CellP[target].CosmicRayDiffusionCoeff[k_CRegy]; /* diffusion coefficient [physical units] */
    double fluxmag=0, Bmag=0, gradmag=0, Lgrad=0, veff=0, P0=Get_Gas_CosmicRayPressure(target,k_CRegy); int m;
    for(m=0;m<3;m++) {
        double f_0=CellP[target].CosmicRayFluxPred[k_CRegy][m], g_0=CellP[target].Gradients.CosmicRayPressure[k_CRegy][m], B_0=f_0;
#ifdef MAGNETIC
        B_0 = Get_Gas_BField(target,m);
#endif
        Bmag += B_0*B_0; fluxmag += f_0*B_0; gradmag += g_0*B_0;
    }
    if(Bmag>0) {fluxmag=fabs(fluxmag)/sqrt(Bmag); gradmag=fabs(gradmag)/sqrt(Bmag);}
    if(gradmag>0) {Lgrad = All.cf_atime * P0 / gradmag;}
    if(fluxmag>0 && CellP[target].CosmicRayEnergyPred[k_CRegy] > MIN_REAL_NUMBER) {veff = fluxmag / CellP[target].CosmicRayEnergyPred[k_CRegy];}
    double vmax = CRFLUID_REDUCED_C_CODE(k_CRegy);
    int use_injectionmod=0;
    if(mode==0) {use_injectionmod=1;} else {if(return_CRbin_CR_species_ID(k_CRegy) < 0) {use_injectionmod=1;}} // for injection, or leptons, where loss=injection at high-RGV (high-diffcoeff, so high veff), more accurate to match suppression factors this way
    if(use_injectionmod) {veff = vmax;} else {veff = DMIN(veff, vmax);} // we're injecting, so the relevant speed here is just the injection speed (note we speed-limit flux to this for timestepping reasons)
    if(use_injectionmod) {Lgrad = 5./UNIT_LENGTH_IN_KPC;} // set initial gradient length to a constant to reduce noise [looking at newer tests, don't -really- need this, but doesn't hurt either]
    double v_max = DMIN( C_LIGHT_CODE , kappa / (MIN_REAL_NUMBER + Lgrad) ); // attempt at a limiter function here to determine if being flux-limited in the equations below //
    double RSOL_over_v_desired = veff / (MIN_REAL_NUMBER + v_max);
    if(isfinite(RSOL_over_v_desired) && (RSOL_over_v_desired > 0) && (RSOL_over_v_desired < MAX_REAL_NUMBER)) {if(RSOL_over_v_desired < 1) {return RSOL_over_v_desired;}}
    return 1;
#endif
}


/*<! closure function needed for arbitrarily anisotropic CR distribution function, from Hopkins '21 */
double return_cosmic_ray_anisotropic_closure_function_threechi(int target, int k_CRegy)
{
    double fluxmag2=0,ecr,ecrv,f,mu1_2,mu2; int k; ecr=CellP[target].CosmicRayEnergyPred[k_CRegy];
    for(k=0;k<3;k++) {f=CellP[target].CosmicRayFluxPred[k_CRegy][k]; fluxmag2+=f*f;}
#if defined(CRFLUID_ALT_RSOL_FORM)
    double v_eff_cr = CRFLUID_REDUCED_C_CODE(k_CRegy); // universal reduction factor
#else
    double kappa=CellP[target].CosmicRayDiffusionCoeff[k_CRegy], Lgrad_inv=0, P=0; for(k=0;k<3;k++) {P=CellP[target].Gradients.CosmicRayPressure[k_CRegy][k]; Lgrad_inv+=P*P;}
    Lgrad_inv = (sqrt(Lgrad_inv) / Get_Gas_CosmicRayPressure(target, k_CRegy)) / All.cf_atime;
    double v_eff_cr = DMIN(DMAX(CRFLUID_REDUCED_C_CODE(k_CRegy) , kappa*Lgrad_inv) , C_LIGHT_CODE); // more complicated factor b/c transport not universally slowed-down
#endif
    ecrv=ecr*v_eff_cr; f=fluxmag2/(ecrv*ecrv); mu1_2=DMAX(0,DMIN(1,f));
    if(!isfinite(mu1_2) || fluxmag2 < MIN_REAL_NUMBER || ecrv < MIN_REAL_NUMBER || !isfinite(f) || f < MIN_REAL_NUMBER || !isfinite(ecrv) || !isfinite(fluxmag2)) {mu1_2=0;} else {if(isfinite(f) && f > 1) {mu1_2=1;}} // safety checks for initialization where may have 0's, etc.
    mu1_2 = DMAX(0,DMIN(1,mu1_2)); // one more safety check
    mu2 = (3.+4.*mu1_2) / (5.+2.*sqrt(4.-3.*mu1_2)); // actual closure relation
    return 3.*(1.-mu2)/2.; // definition of chi variable
}


/* routine which returns the typical absolute value of the rigidity of a given CR [in GV] in a given 'bin' of our multi-bin approximation */
double return_CRbin_CR_rigidity_in_GV(int target, int k_CRegy)
{
    double R = 1;
#if (N_CR_PARTICLE_BINS > 1)    /* insert physics here */
#if (N_CR_PARTICLE_BINS == 2) /* one-bin protons, one electrons */
    double Rv[2]={1.8 , 0.6}; R=Rv[k_CRegy]; // approximate peak energies of each from Cummings et al. 2016 Fig 15
#endif
#if (N_CR_PARTICLE_BINS > 2) /* arbitrary numbers of bins for CR spectra, assumes binning same in e- and p */
    if(target >= 0) {R=CR_return_mean_rigidity_in_bin_in_GV(target,k_CRegy);} else {R=CR_global_rigidity_at_bin_center[k_CRegy];} // this is pre-defined globally for this bin list
#endif
#endif
    return R;
}

/* routine which returns the typical charge of CRs in a given 'bin' of our multi-bin approximation */
double return_CRbin_CR_charge_in_e(int target, int k_CRegy)
{
#if (N_CR_PARTICLE_BINS > 2)
    return CR_global_charge_in_bin[k_CRegy]; // this is pre-defined globally for this bin list
#endif
#if (N_CR_PARTICLE_BINS == 2) /* one-bin protons, one electrons */
    if(k_CRegy==0) {return 1} else {return -1;} // proton, then e- bin
#endif
    return 1; // default one-bin model is only protons
}

/* routine which returns the species ID of the CRs allowing for different code modes */
int return_CRbin_CR_species_ID(int k_CRegy)
{
#if defined(CRFLUID_EVOLVE_SPECTRUM)
    return CR_species_ID_in_bin[k_CRegy]; // this is pre-defined globally for this bin list
#endif
#if (N_CR_PARTICLE_BINS == 2) /* one-bin protons, one electrons */
    if(k_CRegy==0) {return 1} else {return -1;}; // proton, then e- bin
#endif
    return 1; // default to protons
}



/* calculate the -ion- Alfven speed in a given element, relevant for very short-wavelength modes with frequency larger than the ion-neutral collision timescale (relevant for CRs in particular) */
double Get_Gas_ion_Alfven_speed_i(int i)
{
#if !defined(MAGNETIC)
    return Get_Gas_thermal_soundspeed_i(i); // if no B-fields, just assume Alfven speed equal to thermal sound speed
#endif
    double vA = Get_Gas_Alfven_speed_i(i); // normal ideal-MHD Alfven speed
    double f_ion = Get_Gas_Ionized_Fraction(i);
    double mu_eff_ion = 1. + (24.305 - 1.)/(1. + pow(f_ion/1.e-3,2)); // -very- crude approximation to transition to heavy-ion dominance at very low ion fractions
    vA /= sqrt(1.e-15 + mu_eff_ion * f_ion); // Alfven speed of interest is that of the ions alone, not the ideal MHD Alfven speed [corrected for weight with crude approximation above - note that in grain-charge dominated regime, this becomes deeply ambiguous] //
    return vA;
}


/* calculate |nu_+ - nu_-|/|nu_+ + nu_-|, i.e. the asymmetry between scattering by forward-vs-backward propagating waves. in SC limit, this is 1, in ET limit with fully-isotropic scattering, this is 0. relevant here because we want in principle to be able to interpolate between the two, and that's important in particular in the strong ion-neutral damping limit. */
double return_CRbin_nuplusminus_asymmetry(int i, int k_CRegy)
{
    return 1; // default to unity; basically always true with reasonable SC growth rates, even for extremely low ionization fractions //
}


#endif // closes block for entire file for COSMIC_RAY_FLUID


#ifdef COSMIC_RAY_SUBGRID_LEBRON // block for simplified sub-grid CR model
/* function to return injection rate of CRs -time-averaged in total energy, extremely boiled-down version */
double cr_get_source_injection_rate(int i)
{
    double Edot = 0;
#ifdef GALSF
#ifdef GALSF_FB_MECHANICAL
    if(P[i].Type == 4)
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
        Edot = All.CosmicRay_SNeFraction * (RSNe*UNIT_TIME_IN_MYR) * (P[i].Mass*UNIT_MASS_IN_SOLAR) * (1.0e51/UNIT_ENERGY_IN_CGS);
    }
#endif
#ifdef SINK_PARTICLES
    if(P[i].Type == 5) {
        double mdot_eff = P[i].Sink_Mdot; // code units
        mdot_eff = DMIN( mdot_eff , P[i].Sink_Mass / (100./UNIT_TIME_IN_MYR) ); // if time-averaging over ~Gyr, can't have time-averaged injection rate above Mbh/<t> more or less (modulo order-one corrections for all this)
        Edot = evaluate_sink_cosmicray_efficiency(P[i].Sink_Mdot,P[i].Sink_Mass,i) * mdot_eff * C_LIGHT_CODE*C_LIGHT_CODE; // injection in code units
    }
#endif
#endif
    if(Edot > 0) {return Edot * cr_get_source_shieldfac(i);} else {return 0;}
}

/* function to return shielding/loss factor correction */
double cr_get_source_shieldfac(int i)
{
    double cr_atten_fac = 1;
    if(P[i].KernelRadius > 0 && P[i].NumNgb > 0 && All.Time > All.TimeBegin)
    {
        double dx=P[i].KernelRadius/P[i].NumNgb, rho; // code units
        Vec3<double> gradrho={}; int k; for(k=0;k<3;k++) {gradrho[k]=P[i].GradRho[k];}
        if(P[i].Type==0) {rho=CellP[i].Density;} else {rho=P[i].DensityAroundParticle;}
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



/* return total CR energy density associated with a cell */
double INLINE_FUNC Get_CosmicRayEnergyDensity_cgs(int i)
{
    if(i<0) {return 0;}
#ifdef COSMIC_RAY_FLUID
    double u_cr=0; int k; for(k=0;k<N_CR_PARTICLE_BINS;k++) {u_cr += CellP[i].CosmicRayEnergyPred[k];}
    return u_cr * (CellP[i].Density*All.cf_a3inv / P[i].Mass) * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef COSMIC_RAY_SUBGRID_LEBRON
    return CellP[i].SubGrid_CosmicRayEnergyDensity*All.cf_a3inv * UNIT_PRESSURE_IN_CGS;
#endif
#ifdef RT_ISRF_BACKGROUND
    double column = evaluate_NH_from_GradRho(P[i].GradRho,P[i].KernelRadius,CellP[i].Density,P[i].NumNgb,1,i) * UNIT_SURFDEN_IN_CGS, sigma_0=2.23e-3; // sigma_0 is N_H = 1e21 cm^-2 in g cm^-2
    double u_cr_0 = sqrt(All.InterstellarRadiationFieldStrength) * 1.6e-12; // unattenuated energy density; prescription for scaling here assumes ISRF ~ sigma_SFR but zeta_CR ~ t_depletion^-1 ~ sigma_SFR^0.5 assuming the Kennicutt 1998 relation
    if(column < sigma_0) {return u_cr_0;} else {
        double atten_fac = exp(DMAX(-column/100.,-90)) * (sigma_0/(column+MIN_REAL_NUMBER)); // attenuates as N_H^{-1} above column of 1e21 cm^-2, with an exponential truncation above 100 g cm^-2; fairly uncertain, closer to predictions of diffusive transport models
	    return atten_fac * u_cr_0;
    }
#endif    
    return 1.6e-12; // eV/cm-3, approximate from Cummings et al. 2016 V1 data
}


/* return total CR ionization rate zeta_cr in s^-1 associated with a cell */
double Get_CosmicRayIonizationRate_cgs(int i)
{
    double zeta_cr = 0;
#if defined(COSMIC_RAY_FLUID) && (N_CR_PARTICLE_BINS > 2)
    double ecr_units=(CellP[i].Density*All.cf_a3inv/P[i].Mass)*UNIT_PRESSURE_IN_CGS; int k;
    for(k=0;k<N_CR_PARTICLE_BINS;k++)
    {
        double T_GeV=return_CRbin_kinetic_energy_in_GeV(-1,k), beta=return_CRbin_beta_factor(-1,k), Z=return_CRbin_CR_charge_in_e(-1,k), gamma=return_CRbin_gamma_factor(-1,k);
        zeta_cr += 3.43e-18 * (Z*Z/T_GeV) * ((1.-0.069*beta*beta+0.14*log(beta*gamma))/beta) * (CellP[i].CosmicRayEnergyPred[k]*ecr_units); // cross sections from standard Bethe-Blocke formulation, valid at all CR energies we consider explicitly
    }
#else
    zeta_cr = 1.e-5 * Get_CosmicRayEnergyDensity_cgs(i); // scales following Cummings et al. 2016 to 1.6e-17 per eV/cm^3
#if !defined(COSMIC_RAY_FLUID) && !defined(COSMIC_RAY_SUBGRID_LEBRON) && !defined(RT_ISRF_BACKGROUND)
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (CellP[i].Density*All.cf_a3inv*UNIT_DENSITY_IN_CGS) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}}
    zeta_cr *= prefac_CR; // in cosmological runs where we're not following CRs in any sense, turn off CR ionization for any gas with density unless it's >1000 times the cosmic mean density
#endif
#endif
#ifdef METALS
    zeta_cr += 1.e-21 * P[i].Metallicity[0]/All.SolarAbundances[0]; // include radioactive decay of K-40 and other species, which scales with metallicity
    zeta_cr += 1.e-19 * P[i].Metallicity[10]/All.SolarAbundances[10]; // Contribution from short-lived radioisotopes, here scaling to Fe abundance - Adams et al 2014 Apj 789 86
#endif
    return zeta_cr;
}


/* cosmic ray heating of gas, from Guo & Oh 2008, following Mannheim & Schlickeiser 1994.
 We assume all the electron losses go into radiation (and ionization), as the electron coulomb losses into gas are lower than protons by factor of energy and me/mp.
 For protons, we assume 1/6 of the hadronic losses (based on branching ratios) and all of the Coulomb losses thermalize.
 Do want to make sure that the rates we assume here are consistent with those used in the CR cooling routine above. */
double CR_gas_heating(int target, double n_elec, double nH0, double nHcgs)
{
#if defined(CRFLUID_ALT_DISABLE_LOSSES)
    return 0;
#endif
    double a_hadronic, b_coulomb_ion_per_GeV, f_heat_hadronic;
    a_hadronic = 6.37e-16; b_coulomb_ion_per_GeV = 3.09e-16*(n_elec + 0.57*nH0)*HYDROGEN_MASSFRAC; f_heat_hadronic=1./6.; /* some coefficients; a_hadronic is the default coefficient, b_coulomb_ion_per_GeV the default divided by GeV, b/c we need to divide the energy per CR. note there is an extra factor in principle for the ionization term here compared to its version in the CR losses module above: this represents the fraction of CR energy going into the thermal energy of the gas, as opposed to ionization energy, but this is close to unity */
#if defined(COSMIC_RAY_FLUID) || defined(COSMIC_RAY_SUBGRID_LEBRON)
#if (N_CR_PARTICLE_BINS > 2)
    double e_heat=0, e_CR_units_0=(CellP[target].Density*All.cf_a3inv/P[target].Mass) * UNIT_PRESSURE_IN_CGS / nHcgs; int k_CRegy;
    for(k_CRegy=0;k_CRegy<N_CR_PARTICLE_BINS;k_CRegy++)
    {
        double e_cr_units = CellP[target].CosmicRayEnergyPred[k_CRegy] * e_CR_units_0;
        if(return_CRbin_CR_species_ID(k_CRegy) > 0)
        {
            double E_GeV = return_CRbin_kinetic_energy_in_GeV(target,k_CRegy), beta = return_CRbin_beta_factor(target,k_CRegy), Z=fabs(return_CRbin_CR_charge_in_e(target,k_CRegy));
            double T_eff_fullion = 0.59*(5./3.-1.)*U_TO_TEMP_UNITS*CellP[target].InternalEnergyPred, xm = 0.0286*sqrt(T_eff_fullion/2.e6);
            e_heat += b_coulomb_ion_per_GeV * ((Z*Z*beta*beta)/((beta*beta*beta+xm*xm*xm)*E_GeV)) * e_cr_units; // all protons Coulomb-heat, can be rapid for low-E
            if(E_GeV>=0.28) {e_heat += f_heat_hadronic * a_hadronic * e_cr_units;} // only GeV CRs or higher trigger above threshold for collisions
        }
    }
    return e_heat;
#else
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * Get_CosmicRayEnergyDensity_cgs(target) / nHcgs; /* for N<=2, assume a universal spectral shape, the factor here corrects for the fraction above-threshold for hadronic interactions, and 0.53 likewise for averaging  */
#endif
#elif defined(COOL_LOW_TEMPERATURES) // no CR module, but low-temperature cooling is on, we account for the CRs as a heating source, assuming a MW-like background scaled cosmologically to avoid over-heating IGM at high redshifts //
    double prefac_CR=1.; if(All.ComovingIntegrationOn) {double rhofac = (PROTONMASS_CGS*nHcgs/HYDROGEN_MASSFRAC) / (1000.*COSMIC_BARYON_DENSITY_CGS); if(rhofac < 0.2) {prefac_CR=0;} else {if(rhofac > 200.) {prefac_CR=1;} else {prefac_CR=exp(-1./(rhofac*rhofac));}}} // in cosmological runs, turn off CR heating for any gas with density unless it's >1000 times the cosmic mean density
#ifdef RT_ISRF_BACKGROUND
    prefac_CR *= Get_CosmicRayEnergyDensity_cgs(target)/1.6e-12; // prescription for scaling here assumes ISRF ~ sigma_SFR but zeta_CR ~ t_depletion^-1 ~ sigma_SFR^0.5 assuming the Kennicutt 1998 relation
#endif
    return (0.87*f_heat_hadronic*a_hadronic + 0.53*b_coulomb_ion_per_GeV) * (1.6e-12*prefac_CR) / (1.e-2 + nHcgs); // assume MW-like CR background modulated by above factor (1.6e-12*prefac_CR)=eCR_cgs here //
#else
    return 0;
#endif
}


/* subroutine to calculate which part of the adiabatic PdV work from the RP gets assigned to the CRs vs the gas; since the CRs are always smooth by definition under this operation this follows simply from the local cell divergence and the effective CR eos */
double CR_calculate_adiabatic_gasCR_exchange_term(int i, double dt_entr, double gamma_minus_eCR_tmp, int mode)
{
    double u0, d_CR; if(mode==0) {u0=CellP[i].InternalEnergy;} else {u0=CellP[i].InternalEnergyPred;} // initial energy
    if(u0<All.MinEgySpec) {u0=All.MinEgySpec;} // enforced throughout code
    
    double divv_p=-dt_entr*P[i].Particle_DivVel*All.cf_a2inv, divv_f=divv_p, divv_u=0; // get locally-estimated gas velocity divergence for cells - if using non-Lagrangian method, need to modify. take negative of this [for sign of change to energy] and multiply by timestep
#ifdef COSMIC_RAY_FLUID
    divv_f=-dt_entr*CellP[i].Face_DivVel_ForAdOps*All.cf_a2inv;
#endif
    if(All.ComovingIntegrationOn) {double divv_h=-dt_entr*(3.*All.cf_hubble_a); divv_p+=divv_h; divv_f+=divv_h;} // include hubble-flow terms
    double P_cr = gamma_minus_eCR_tmp * CellP[i].Density * All.cf_a3inv / P[i].Mass, P_tot = CellP[i].Pressure * All.cf_a3inv; // define the pressure from CRs and total pressure (physical units)
#ifdef MAGNETIC
    double B2=0; int k; for(k=0;k<3;k++) {double B=Get_Gas_BField(i,k)*All.cf_a2inv; B2+=B*B;}
    P_tot += 0.5*B2; // add magnetic pressure [B^2/2], in physical code units, since it contributes to the PdV work but not included in 'pressure' total above
#endif
    double fac_P = DMAX(0, DMIN(1, P_cr/(P_tot + 1.e-10*P_cr + MIN_REAL_NUMBER))); // fraction of total pressure from CRs
    double Ui = u0 * P[i].Mass; // factor for multiplication below, and initial thermal energy
    double dtI_hydro = CellP[i].DtInternalEnergy * P[i].Mass * dt_entr; // change given by hydro-step computed delta_InternalEnergy
    double min_IEgy = P[i].Mass * All.MinEgySpec; // minimum internal energy - in total units -
    
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
    CellP[i].Face_DivVel_ForAdOps = -d_CR / (All.cf_a2inv * gamma_minus_eCR_tmp * dt_entr + MIN_REAL_NUMBER); // this is the 'effective' divergence here (in code units) which matches exactly the change in CR energy when the above limiters etc are applied. we can save this for use in the other CR subroutines
#endif
    return d_CR; // return final value
}

