#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"

/* this file handles the FIRE short-range radiation-pressure and
    photo-ionization terms. written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */

#if defined(GALSF_FB_FIRE_RT_LOCALRP)
#include "radfb_local_gpu.h"
#endif

#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
#include <vector>
#include "../mesh/gpu_neighbor_list.h"
#include "../mesh/ghost_writeback.h"
#include "../mesh/ghost_symlist_lifecycle.h"
#include "../system/gpu_particles_arena.h"
#endif

#if defined(GALSF_FB_FIRE_RT_LOCALRP) /* first the radiation pressure coupled in the immediate vicinity of the star */
/*!   -- this subroutine is not openmp parallelized at present, so there's not any issue about conflicts over shared memory. if you make it openmp, make sure you protect the writes to shared memory here! -- */
void radiation_pressure_winds_consolidated(void)
{
    radiation_pressure_winds_gpu(P, CellP, NumPart,
                                 ActiveParticleList.data(), (int)ActiveParticleList.size());
}

#endif /* closes defined(GALSF_FB_FIRE_RT_LOCALRP)  */





#if defined(GALSF_FB_FIRE_RT_HIIHEATING)
/* Routines for simple FIRE local photo-ionization heating feedback model. This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO. */
/*!   -- this subroutine is not openmp parallelized at present, so there's not any issue about conflicts over shared memory. if you make it openmp, make sure you protect the writes to shared memory here! -- */


/* NOTE: HII_heating_singledomain outer loop cannot be wrapped in Kokkos::parallel_for
 * because the inner greedy search calls ngb_treefind_variable_targeted, which writes
 * to the global Ngblist buffer.  Making Ngblist thread_local in allvars.cc would
 * unblock this; deferred until that change is made. */
void HII_heating_singledomain(void)    /* this version of the HII routine only communicates with particles on the same processor */
{
#ifdef RT_CHEM_PHOTOION
    return; // the work here is done in the actual RT routines if this switch is enabled //
#endif
    if(All.HIIRegion_fLum_Coupled<=0) {return;}
    if(All.Time<=0) {return;}
    PRINT_STATUS("Local HII-Region photo-heating/ionization calculation");
    Vec3<double> pos={}; MyFloat h_i, dt, rho; int startnode, numngb, j, n, i, NITER_HIIFB, MAX_N_ITERATIONS_HIIFB, jnearest,already_ionized,do_ionize,dummy;
    double total_N_ionizing_part=0,total_Ndot_ionizing=0,total_m_ionized=0,total_N_ionized=0,avg_RHII=0,mionizable=0,mionized=0,mion_actual=0;
    double RHII,RHIIMAX,R_search,rnearest,stellum,prob,rho_j,prandom,m_available,m_effective,RHII_initial,RHIImultiplier;
    double uion; uion = HIIRegion_Temp / (0.59 * (5./3.-1.) * U_TO_TEMP_UNITS); /* assume fully-ionized gas with gamma=5/3; this is a global variable below */
    MAX_N_ITERATIONS_HIIFB = 5; NITER_HIIFB = 0;

    /* Modern path: prebuilt CSR neighbor list. Skip ghosts inside the inner loop
     * to preserve the "singledomain" semantic (only ionize gas on this rank's
     * domain — no MPI export). */
    {
        std::vector<int> hii_src_idx;
        std::vector<double> hii_src_radii;
        hii_src_idx.reserve(64);
        hii_src_radii.reserve(64);

        /* Pre-pass: identify candidate sources + per-source max search radius
         * (covers the full possible RHII expansion range, capped at 1.26 RHIIMAX). */
        for (int ip : ActiveParticleList) {
#ifdef SINK_HII_HEATING
            if(!((P[ip].Type == 5)||(((P[ip].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[ip].Type == 2)||(P[ip].Type==3))))))) continue;
#else
            if(!((P[ip].Type == 4)||((All.ComovingIntegrationOn==0)&&((P[ip].Type == 2)||(P[ip].Type==3))))) continue;
#endif
            if(P[ip].Mass <= 0 || !isfinite(P[ip].Mass)) continue;
            double dt_i = get_particle_feedback_timestep_in_physical(ip);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
            if(P[ip].Type == 5) dt_i = P[ip].dt_since_last_gas_search;
#endif
            if(dt_i <= 0) continue;
            double stellum_i = All.HIIRegion_fLum_Coupled * particle_ionizing_luminosity_in_cgs(ip);
#ifdef CHIMES_HII_REGIONS
            stellum_i = chimes_ion_luminosity(evaluate_stellar_age_Gyr(ip)*1000., P[ip].Mass*UNIT_MASS_IN_SOLAR) * 4.68e-11;
#endif
            if(stellum_i <= 0) continue;
            if(P[ip].KernelRadius <= 0 || P[ip].DensityAroundParticle <= 0) continue;
            double h_local = P[ip].KernelRadius;
            double RHIIMAX_l = 2. * 240.0*pow(stellum_i, 0.5) / (All.cf_atime * UNIT_LENGTH_IN_CGS);
            if(RHIIMAX_l < 2.0*h_local) RHIIMAX_l = 2.0*h_local;
            if(RHIIMAX_l > 10.0*h_local) RHIIMAX_l = 10.0*h_local;
            double R_for_NL = 1.26 * RHIIMAX_l;
            if(R_for_NL < 0.5*h_local) R_for_NL = 0.5*h_local;
            hii_src_idx.push_back(ip);
            hii_src_radii.push_back(R_for_NL);
        }

        int num_src = (int)hii_src_idx.size();
        int imported_ghosts = 0;
        gpu_neighbor_list_t gnl = {};
        int local_count = 0;
        if(num_src > 0) {
            if(ghost_get_num_ghosts() <= 0) {
                gizmo_density_prep_ghosts(gizmo_ghost_safety_factor());
                imported_ghosts = 1;
            }
            local_count = ghost_get_num_local();
            int num_all = local_count + ghost_get_num_ghosts();
            if(num_all <= 0) num_all = NumPart;
            gpu_particles_arena_acquire(num_all, P, CellP);
            struct particle_data *P_gpu = gpu_particles_arena_P();
            gpu_ngb_list_build(P_gpu, num_all,
                               hii_src_idx.data(), num_src,
                               NGB_SEARCH_ONEWAY, 1 /* gas only */,
                               &gnl, NULL, 1.0, hii_src_radii.data());
        }

        /* Sequential per-source loop — preserves greedy ionization ordering across
         * sources (later sources observe DelayTimeHII / InternalEnergy from earlier). */
        for(int aa = 0; aa < num_src; aa++) {
            i = hii_src_idx[aa];
            dt = get_particle_feedback_timestep_in_physical(i);
#ifdef SINK_INTERACT_ON_GAS_TIMESTEP
            if(P[i].Type == 5) dt = P[i].dt_since_last_gas_search;
#endif
            stellum = All.HIIRegion_fLum_Coupled * particle_ionizing_luminosity_in_cgs(i);
#ifdef CHIMES_HII_REGIONS
            stellum = chimes_ion_luminosity(evaluate_stellar_age_Gyr(i)*1000., P[i].Mass*UNIT_MASS_IN_SOLAR) * 4.68e-11;
#endif
            pos = P[i].Pos; rho = P[i].DensityAroundParticle; h_i = P[i].KernelRadius;
            RHII = 4.78e-9*pow(stellum, 0.333)*pow(rho*All.cf_a3inv*UNIT_DENSITY_IN_CGS, -0.66667);
            RHII /= All.cf_atime * UNIT_LENGTH_IN_CGS;
            RHIIMAX = 2. * 240.0*pow(stellum, 0.5) / (All.cf_atime*UNIT_LENGTH_IN_CGS);
            if(RHIIMAX < 2.0*h_i) RHIIMAX = 2.0*h_i;
            if(RHIIMAX > 10.0*h_i) RHIIMAX = 10.0*h_i;
            mionizable = VOLUME_NORM_COEFF_FOR_NDIMS*rho*RHII*RHII*RHII;
            double M_ionizing_emitted = (3.05e10 * PROTONMASS_CGS) * stellum * (dt * UNIT_TIME_IN_CGS);
            mionizable = DMIN(mionizable, M_ionizing_emitted/UNIT_MASS_IN_CGS);
            if(RHII > RHIIMAX) RHII = RHIIMAX;
            if(RHII < 0.3*h_i) RHII = 0.3*h_i;
            RHII_initial = RHII;
            total_N_ionizing_part += 1;
            total_Ndot_ionizing += stellum * (3.05e10/HYDROGEN_MASSFRAC);

            prandom = get_random_number(P[i].ID + 7);
            if(prandom < 5.0*mionizable/P[i].Mass) {
                mionized = 0.0; jnearest = -1; rnearest = MAX_REAL_NUMBER; NITER_HIIFB = 0;
                int nl_start = gnl.offsets[aa], nl_end = gnl.offsets[aa+1];
                int nl_n = nl_end - nl_start;
                int *ngb_list_touse = (int *)alloca((nl_n > 0 ? nl_n : 1) * sizeof(int));
                int more_iters = 1;
                do {
                    double RHII_2 = RHII*RHII;
                    jnearest = -1; rnearest = MAX_REAL_NUMBER;
                    R_search = RHII;
                    if(h_i > 0.5*R_search) R_search = 0.5*h_i;
                    double R_search_2 = R_search * R_search;

                    /* Walk prebuilt NL slice, filter ghosts (singledomain) + radius. */
                    numngb = 0;
                    for(int nn = nl_start; nn < nl_end; nn++) {
                        int j_cand = gnl.neighbors[nn];
                        if(j_cand >= local_count) continue; /* skip ghosts */
                        if(P[j_cand].Type != 0 || P[j_cand].Mass <= 0) continue;
                        Vec3<double> dpc = pos - P[j_cand].Pos; nearest_xyz(dpc, 1);
                        if(dpc.norm_sq() > R_search_2) continue;
                        ngb_list_touse[numngb++] = j_cand;
                    }

                    if(numngb > 0) {
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
                        qsort(ngb_list_touse, numngb, sizeof(int), compare_densities_for_sort);
#endif
                        for(n = 0; n < numngb; n++) {
                            if(mionized >= mionizable) break;
                            j = ngb_list_touse[n];
                            if(P[j].Mass <= 0 || P[j].Type != 0) continue;
                            if(CellP[j].DelayTimeHII > 0) continue;
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2) && !defined(CHIMES_HII_REGIONS)
                            if(CellP[j].Ne > 0.8) continue;
#endif
                            if(P[j].Type == 0 && P[j].Mass > 0) {
                                Vec3<double> dr = pos - P[j].Pos; nearest_xyz(dr, 1);
                                double r2 = dr.norm_sq();
                                if(r2 > RHII_2) continue;
                                double r = sqrt(r2), u = 0;
                                already_ionized = 0; rho_j = CellP[j].density_for_energy();
                                if(CellP[j].InternalEnergy < CellP[j].InternalEnergyPred) u = CellP[j].InternalEnergy; else u = CellP[j].InternalEnergyPred;
                                if(CellP[j].DelayTimeHII > 0) already_ionized = 1;
#if !defined(CHIMES_HII_REGIONS)
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
                                if((CellP[j].Ne > 0.8) || (u > 50.*uion)) already_ionized = 1;
#else
                                if(u > uion) already_ionized = 1;
#endif
#endif
                                if(already_ionized) continue;
                                do_ionize = 0; prob = 0;
                                if((r <= RHII) && (already_ionized == 0) && (mionized < mionizable)) {
                                    m_effective = P[j].Mass*(CellP[j].Density/rho);
                                    m_available = mionizable - mionized;
                                    if(m_effective <= m_available) { do_ionize = 1; prob = 1.001; }
                                    else { prob = m_available/m_effective; if(prandom < prob) do_ionize = 1; }
                                    if(do_ionize == 1) {
                                        already_ionized = do_the_local_ionization(j, dt, i);
                                        total_N_ionized += 1;
                                        mion_actual += P[j].Mass;
                                        avg_RHII += P[j].Mass*r*All.cf_atime*UNIT_LENGTH_IN_KPC;
                                    }
                                    mionized += prob*m_effective;
                                }
#if (GALSF_FB_FIRE_STELLAREVOLUTION > 2)
                                if((CellP[j].Density < rnearest) && (already_ionized == 0)) { rnearest = CellP[j].Density; jnearest = j; }
#else
                                if((r < rnearest) && (already_ionized == 0)) { rnearest = r; jnearest = j; }
#endif
                            }
                        }
                    }

                    /* Backstop: ionize jnearest if still photons */
                    if((mionized < mionizable) && (jnearest >= 0)) {
                        j = jnearest; m_effective = P[j].Mass*(CellP[j].Density/rho); m_available = mionizable - mionized;
                        prob = m_available/m_effective; do_ionize = 0;
                        if(prandom < prob) do_ionize = 1;
                        if(do_ionize == 1) {
                            already_ionized = do_the_local_ionization(j, dt, i);
                            total_N_ionized += 1;
                            mion_actual += P[j].Mass;
                            Vec3<double> dr = pos - P[j].Pos; nearest_xyz(dr, 1); double r2 = dr.norm_sq();
                            avg_RHII += P[j].Mass*sqrt(r2)*All.cf_atime*UNIT_LENGTH_IN_KPC;
                        }
                        mionized += prob*m_effective;
                    }

                    /* RHII expansion (mirrors legacy logic exactly) */
                    RHIImultiplier = 1.10;
                    more_iters = 0;
                    if(mionized < 0.95*mionizable) {
                        if((RHII >= DMAX(30.0*RHII_initial, RHIIMAX)) || (NITER_HIIFB >= MAX_N_ITERATIONS_HIIFB)) {
                            mionized = 1.001*mionizable;
                        } else {
                            if(mionized <= 0) RHIImultiplier = 2.0;
                            else {
                                RHIImultiplier = pow(mionized/mionizable, -0.333);
                                if(RHIImultiplier > 5.0) RHIImultiplier = 5.0;
                                if(RHIImultiplier < 1.26) RHIImultiplier = 1.26;
                            }
                            RHII *= RHIImultiplier; if(RHII > 1.26*RHIIMAX) RHII = 1.26*RHIIMAX;
                            more_iters = 1;
                        }
                    }
                    NITER_HIIFB++;
                } while(more_iters && mionized < mionizable);
                if(mion_actual > 0) total_m_ionized += mion_actual;
            }
        }

        if(num_src > 0) {
            gpu_ngb_list_free(&gnl, NULL);
            gpu_particles_arena_invalidate();
        }
        if(imported_ghosts) ghost_exchange_cleanup();
    }


    double totMPI_N_ionizing_part=0,totMPI_Ndot_ionizing=0,totMPI_m_ionized=0,totMPI_avg_RHII=0,totMPI_N_ionized=0;
    MPI_Reduce(&total_N_ionizing_part, &totMPI_N_ionizing_part, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_Ndot_ionizing, &totMPI_Ndot_ionizing, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_m_ionized, &totMPI_m_ionized, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_N_ionized, &totMPI_N_ionized, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_RHII, &totMPI_avg_RHII, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(ThisTask == 0)
    {
        if(totMPI_N_ionizing_part>0)
        {
            if(totMPI_m_ionized>0) {totMPI_avg_RHII /= totMPI_m_ionized;}
            PRINT_STATUS(" ..Nsources=%g with dN/dt=%g/s ionized N=%g (M=%g sol) cells in <R_HII>=%g kpc",totMPI_N_ionizing_part,totMPI_Ndot_ionizing,totMPI_N_ionized,totMPI_m_ionized*UNIT_MASS_IN_SOLAR,totMPI_avg_RHII);
            fprintf(FdHIIHeating,"%.16g %g %g %g %g %g \n",All.Time,totMPI_N_ionizing_part,totMPI_Ndot_ionizing,totMPI_N_ionized,totMPI_m_ionized*UNIT_MASS_IN_SOLAR,totMPI_avg_RHII); fflush(FdHIIHeating);
        }
        if(All.HighestActiveTimeBin == All.HighestOccupiedTimeBin) {fflush(FdHIIHeating);}
    } // ThisTask == 0
    CPU_Step[CPU_HIIHEATING] += measure_time();
} // void HII_heating_singledomain(void)



int do_the_local_ionization(int target, double dt, int source)
{
#if defined(CHIMES_HII_REGIONS) // set a number of chimes-specific quantities here //
    int k,age_bin=0; double stellar_age_myr=1000.*evaluate_stellar_age_Gyr(source), log_age_Myr=log10(stellar_age_myr); // determine stellar age bin
    if(log_age_Myr<CHIMES_LOCAL_UV_AGE_LOW) {age_bin=0;} else if(log_age_Myr < CHIMES_LOCAL_UV_AGE_MID) {age_bin = (int) floor(((log_age_Myr - CHIMES_LOCAL_UV_AGE_LOW) / CHIMES_LOCAL_UV_DELTA_AGE_LOW) + 1);}
    else {age_bin = (int) floor((((log_age_Myr - CHIMES_LOCAL_UV_AGE_MID) / CHIMES_LOCAL_UV_DELTA_AGE_HI) + ((CHIMES_LOCAL_UV_AGE_MID - CHIMES_LOCAL_UV_AGE_LOW) / CHIMES_LOCAL_UV_DELTA_AGE_LOW)) + 1); if(age_bin > CHIMES_LOCAL_UV_NBINS - 1) {age_bin = CHIMES_LOCAL_UV_NBINS - 1;}}
    // reset all of the HII-region chimes quantities to null
    for(k=0;k<CHIMES_LOCAL_UV_NBINS;k++) {CellP[target].Chimes_fluxPhotIon_HII[k]=0; CellP[target].Chimes_G0_HII[k]=0;}
    // set the quantities desired for this age bin specifically: need a softened radius, for use here //
    double stellar_mass=P[source].Mass*UNIT_MASS_IN_SOLAR; Vec3<double> dp = P[source].Pos - P[target].Pos;
    nearest_xyz(dp); dp *= All.cf_atime*UNIT_LENGTH_IN_CGS; double r2 = dp.norm_sq(); // separation in cgs
    double eps_cgs=KERNEL_FAC_FROM_FORCESOFT_TO_PLUMMER*ForceSoftening_KernelRadius(source)*All.cf_atime*UNIT_LENGTH_IN_CGS; // plummer equivalent softening
    r2+=eps_cgs*eps_cgs; // gravitational Softening (cgs units)
    CellP[target].Chimes_fluxPhotIon_HII[age_bin] = (1.0 - All.Chimes_f_esc_ion) * chimes_ion_luminosity(stellar_age_myr, stellar_mass) / r2; // cgs flux of H-ionising photons per second seen by the star particle
    CellP[target].Chimes_G0_HII[age_bin] = (1.0 - All.Chimes_f_esc_G0) * chimes_G0_luminosity(stellar_age_myr, stellar_mass) / r2; // cgs flux in the 6-13.6 eV band

#else

#if (GALSF_FB_FIRE_STELLAREVOLUTION <= 2) 
    CellP[target].InternalEnergy = DMAX(CellP[target].InternalEnergy , HIIRegion_Temp / (0.59 * (5./3.-1.) * U_TO_TEMP_UNITS)); /* assume fully-ionized gas with gamma=5/3 */
    CellP[target].InternalEnergyPred = CellP[target].InternalEnergy; /* full reset of the internal energy */
#else
    double delta_U_of_ionization = (20.-13.6) * ((ELECTRONVOLT_IN_ERGS / PROTONMASS_CGS) / UNIT_SPECEGY_IN_CGS) * (1.-DMAX(0.,DMIN(1.,CellP[target].Ne/1.5))); /* energy injected per unit mass, in code units, by ionization, assuming each atom absorbs, and mean energy of absorbed photons is given by x=18 eV here (-13.6 for energy of ionization) */
    double Theat_star = 1.38 * 3.2, Z_sol = 1.; // typical IMF-averaged temp of ionizing star=32,000 K, with effective ionization temperature parameter psi=1.38 (temp of ionized e's in units of stellar temp). Then metallicity in solar units.
#ifdef METALS
    Z_sol = P[target].Metallicity[0]/All.SolarAbundances[0]; // set metallicity
#if defined(GALSF_ISMDUSTCHEM_MODEL) 
    Z_sol = (P[target].Metallicity[4]-CellP[target].ISMDustChem_Dust_Metal[4])/All.SolarAbundances[4]; // cooling for HII equilibrium depends predominantly on gas-phase O so need to account for any locked up in dust
#endif
#endif
    double t4_eqm = DMIN( 1.5*Theat_star , 3.85/DMAX(log(DMAX(390.*Z_sol/Theat_star,1.001)),0.01) ); // equilibrium H2 region temperature in 1e4 K: 1.5*Theat = eqm temp for pure-H region, while second expression assumes eqm cooling with O+C, etc, but breaks down at low-Z when metals don't dominate cooling.
    double u_eqm = (t4_eqm * HIIRegion_Temp) / (0.59 * (5./3.-1.) * U_TO_TEMP_UNITS); // converted to specific internal energy, assuming full ionization
    double u_post_ion_no_cooling = CellP[target].InternalEnergy + delta_U_of_ionization; // energy after ionization, before any cooling
    double u_final = DMIN( u_post_ion_no_cooling , u_eqm ); // don't heat to higher temperature than intial energy of ionization allows
    CellP[target].InternalEnergy = u_final; CellP[target].InternalEnergyPred = u_final; /* add it */
    /* assign typical strong HII region flux + enough flux to maintain cell fully-ionized, regardless (x'safety-factor'): note this is repeated in the 'self-shield' routine but has to be in both because otherwise this wont appear on the first timestep after which a gas cell is flagged as ionized by this subroutine */
    double n1000 = CellP[target].Density*All.cf_a3inv*UNIT_DENSITY_IN_NHCGS / 1000.; // density in 1000 cm^-3
    double flux_compactHII = DMAX(0.85*pow(n1000,1./3.) , 1) * 2.6e5*n1000; // set to typical value in HII region or minimum needed to maintain f_neutral < 1e-5-ish, whichever is larger
    CellP[target].Rad_Flux_UV += flux_compactHII; CellP[target].Rad_Flux_EUV += flux_compactHII;
#endif
    CellP[target].Ne = 1.0 + 2.0*yhelium(target, P); /* set the cell to fully ionized */

#endif

    CellP[target].DelayTimeHII = DMIN(dt, 10./UNIT_TIME_IN_MYR); /* tell the code to flag this in the cooling subroutine */
    return 1;
}

#endif
