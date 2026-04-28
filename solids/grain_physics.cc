#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../solids/grain_physics_gpu.h"

/*

 This module contains the self-contained sub-routines needed for
 grain-specific physics in proto-planetary/proto-stellar/planetary cases,
 GMC and ISM/CGM/IGM dust dynamics, dust dynamics in cool-star atmospheres,
 winds, and SNe remnants, as well as terrestrial turbulence and
 particulate-laden turbulence. Anywhere where particles coupled to gas
 via coulomb, aerodynamic, or lorentz forces are interesting.

 This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.

 */



#ifdef GRAIN_FLUID

#if defined(GRAIN_BACKREACTION)
void grain_backrx(void);
#endif

extern void grain_drag_evaluate_gpu(struct particle_data *, struct gas_cell_data *, int *, int);

/* function to apply the drag on the grains from surrounding gas properties */
void apply_grain_dragforce(void)
{
    CPU_Step[CPU_MISC] += measure_time();
    int i, k; PRINT_STATUS("Beginning particulate/grain/PIC force evaluation.");

    /* GPU path: gather active particles, dispatch Kokkos kernel, scatter results */
    {
        int N_active = 0;
        for(int ii : ActiveParticleList) N_active++;
        int *grain_indices = (int *) malloc((N_active > 0 ? N_active : 1) * sizeof(int));
        int aa = 0;
        for(int ii : ActiveParticleList) grain_indices[aa++] = ii;
        grain_drag_evaluate_gpu(P, CellP, grain_indices, N_active);
        free(grain_indices);
    }
#if defined(GRAIN_BACKREACTION)
    grain_backrx(); /* call parent routine to assign the back-reaction force among neighbors */
#endif
    PRINT_STATUS(" ..particulate/grain/PIC force evaluation done.");
    CPU_Step[CPU_DRAGFORCE] += measure_time();
}




/* this is a template for fully-automated parallel (hybrid MPI+OpenMP/Pthreads) neighbor communication
 written in a completely modular fashion. this works as long as what you are trying to do isn't too
 complicated (from a communication point-of-view). You specify a few key variables, and then
 define the variables that need to be passed, and write the actual sub-routine that does the actual
 'work' between neighbors, but all of the parallelization, looping, communication blocks,
 etc, are all handled for you. */

#define CORE_FUNCTION_NAME grain_backrx_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(((1 << P[i].Type) & (GRAIN_PTYPES))&&(P[i].TimeBin>=0)&&(P[i].Mass>0)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */


/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME
{
    int NodeList[NODELISTLENGTH]; Vec3<MyDouble> Pos; MyDouble KernelRadius; /* these must always be defined */
#if defined(GRAIN_BACKREACTION)
    Vec3<double> Grain_DeltaMomentum; double Gas_Density, Grain_AccelTimeMin;
#endif
}
*DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration)
{   /* "i" is the particle from which data will be assigned, to structure "in" */
    in->Pos=P[i].Pos; /* good example - always needed */
    in->KernelRadius = P[i].KernelRadius; /* also always needed for search (can change the radius "P[i].KernelRadius" but in->KernelRadius must be defined */
#if defined(GRAIN_BACKREACTION)
    in->Grain_DeltaMomentum = P[i].Grain_DeltaMomentum;
    in->Gas_Density = P[i].Gas_Density;
    in->Grain_AccelTimeMin = P[i].Grain_AccelTimeMin;
#endif
}

/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME
{ /* define variables below as e.g. "double X;" */
}
*DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -back to- the 'searching' element */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration)
{  /* "i" is the particle to which data from structure "out" will be assigned. mode=0 for local communication,
    =1 for data sent back from other processors. you must account for this. */
    /* example: ASSIGN_ADD(P[i].X,out->X,mode); which is short for: if(mode==0) {P[i].X=out->X;} else {P[i].X+=out->X;} */
}


/* this subroutine does the actual neighbor-element calculations (this is the 'core' of the loop, essentially) */
/*!   -- this subroutine writes to shared memory [updating the neighbor values]: need to protect these writes for openmp below. modified values for the minimum timestep are read, so both read and write need to be protected. */
int grain_backrx_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, n; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */

    if(local.KernelRadius <= 0) {return 0;} /* don't bother doing a loop if this isnt going to do anything */
    int kernel_shared_BITFLAG = 1; /* grains 'see' gas in this loop */

    /* Now start the actual neighbor computation for this particle */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0) {
        while(startnode >= 0) {
            numngb_inbox = ngb_treefind_variable_threads_targeted(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist, kernel_shared_BITFLAG);
            if(numngb_inbox < 0) {return -2;} /* no neighbors! */
            for(n = 0; n < numngb_inbox; n++) /* neighbor loop */
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if((P[j].Mass <= 0)||(P[j].KernelRadius <= 0)) {continue;} /* make sure neighbor is valid */
                int k; Vec3<double> dp=local.Pos-P[j].Pos; /* position offset */
                nearest_xyz(dp); double r2=dp.norm_sq(); /* box-wrap appropriately and calculate distance */
#ifdef BOX_BND_PARTICLES
                if(P[j].ID > 0) {r2 = -1;} /* ignore frozen boundary particles */
#endif
                if((r2>0)&&(r2<local.KernelRadius*local.KernelRadius)) /* only keep elements inside search radius */
                {
                    double hinv,hinv3,hinv4,wk_i=0,dwk_i=0,r=sqrt(r2); kernel_hinv(local.KernelRadius,&hinv,&hinv3,&hinv4);
                    kernel_main(r*hinv, hinv3, hinv4, &wk_i, &dwk_i, 0); /* kernel quantities that may be needed */
#if defined(GRAIN_BACKREACTION)
                    double wt = -wk_i / local.Gas_Density, dv2=0; /* degy=wt*delta_egy; */
                    for(k=0;k<3;k++) {
                        double dv = wt*local.Grain_DeltaMomentum[k]; // momentum to be sent to this neighbor element
                        dv2+=dv*dv; // save squared sum
                        #pragma omp atomic
                        P[j].Vel[k] += dv; // add the velocity (checking for thread safety in doing so!)
                        #pragma omp atomic
                        CellP[j].VelPred[k] += dv; // add the velocity (checking for thread safety in doing so!)
                        #pragma omp atomic
                        P[j].dp[k] += dv * P[j].Mass; // track momentum change for tree node update
                    }
                    
                    double taccel_min_prev = 0, taccel_min_new = 0;
                    #pragma omp atomic read
                    taccel_min_prev = P[j].Grain_AccelTimeMin; // this can be modified below so needs to be done in a thread-safe manner here //
                    taccel_min_new = DMIN(DMIN(2.*All.ErrTolIntAccuracy*P[j].Get_Particle_Size()*All.cf_atime*All.cf_atime/sqrt(dv2+MIN_REAL_NUMBER) , 4.*local.Grain_AccelTimeMin), taccel_min_prev);
                    if(taccel_min_new < taccel_min_prev)
                    {
                        #pragma omp atomic write
                        P[j].Grain_AccelTimeMin = taccel_min_new;
                    }
                    /* for(k=0;k<3;k++) {degy-=P[j].Mass*dv*(VelPred_j[k]+0.5*dv);} CellP[j].InternalEnergy += degy; */ // ignoring these terms -- if re-add them be sure to do so thread-safely //
#endif
                }
            } // numngb_inbox loop
        } // while(startnode)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    }
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
}


void grain_backrx(void)
{
    PRINT_STATUS(" ..assigning grain back-reaction to gas");
    {
        /* Build LOCAL active grain list (i-type restricted by GRAIN_PTYPES). */
        int num_active = 0;
        for(int i : ActiveParticleList) {
            if(((1 << P[i].Type) & (GRAIN_PTYPES)) && (P[i].TimeBin >= 0) && (P[i].Mass > 0) && (P[i].KernelRadius > 0)) num_active++;
        }
        int *nl_active = (int *) mymalloc("grainbackrx_nl_active",
            (num_active > 0 ? num_active : 1) * sizeof(int));
        double *nl_radii = (double *) mymalloc("grainbackrx_nl_radii",
            (num_active > 0 ? num_active : 1) * sizeof(double));
        {int aa = 0; for(int i : ActiveParticleList) {
            if(((1 << P[i].Type) & (GRAIN_PTYPES)) && (P[i].TimeBin >= 0) && (P[i].Mass > 0) && (P[i].KernelRadius > 0)) {
                nl_active[aa] = i; nl_radii[aa] = (double)P[i].KernelRadius; aa++;
            }
        }}
        grain_backrx_evaluate_gpu(P, CellP, NumPart, nl_active, num_active, nl_radii);
        myfree(nl_radii); myfree(nl_active);
        CPU_Step[CPU_DRAGFORCE] += measure_time();
    }
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */







#ifdef GRAIN_COLLISIONS
/* return_grain_cross_section_per_unit_mass / prob_of_grain_interaction /
   calculate_interact_kick were host-only functions with implicit global-P
   access and GSL RNG. They have been moved to:
     - solids/grain_helper_functions.h (return_grain_cross_section_per_unit_mass_P,
       prob_of_grain_interaction_tab)
     - sidm/sidm_helper_functions.h (calculate_interact_kick_rng)
   as KOKKOS_INLINE_FUNCTION forms that thread particle_data and the
   GeoFactorTable as explicit args and use the counter-based gpu_rng.
   CPU callers pass `GeoFactorTable` directly; the B2 AGSForce GPU kernel
   passes a SharedSpace mirror. */
#endif






#if defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)

#define CORE_FUNCTION_NAME interpolate_fluxes_opacities_gasgrains_evaluate /* name of the 'core' function doing the actual inter-neighbor operations. this MUST be defined somewhere as "int CORE_FUNCTION_NAME(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)" */
#define CONDITIONFUNCTION_FOR_EVALUATION if(((1 << P[i].Type) & (GRAIN_PTYPES+1))&&(P[i].TimeBin>=0)&&(P[i].Mass>0)) /* function for which elements will be 'active' and allowed to undergo operations. can be a function call, e.g. 'density_is_active(i)', or a direct function call like 'if(P[i].Mass>0)' */
#include "../system/code_block_xchange_initialize.h" /* pre-define all the ALL_CAPS variables we will use below, so their naming conventions are consistent and they compile together, as well as defining some of the function calls needed */

/* this structure defines the variables that need to be sent -from- the 'searching' element */
struct INPUT_STRUCT_NAME {
    int NodeList[NODELISTLENGTH], Type; MyDouble Mass, KernelRadius; Vec3<MyDouble> Pos, Vel; MyDouble Grain_Size, Grain_Abs_Coeff[N_RT_FREQ_BINS]; /* these must always be defined */
} *DATAIN_NAME, *DATAGET_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -from- the 'searching' element */
static inline void INPUTFUNCTION_NAME(struct INPUT_STRUCT_NAME *in, int i, int loop_iteration) {   /* "i" is the particle from which data will be assigned, to structure "in" */
    in->Type=P[i].Type; in->Mass=P[i].Mass; in->KernelRadius=P[i].KernelRadius; int k; in->Pos=P[i].Pos; in->Vel=P[i].Vel;
    if((1 << P[i].Type) & (GRAIN_PTYPES))
    {
        in->Grain_Size=P[i].Grain_Size; int k_freq;
        double R_grain_code=P[i].Grain_Size/UNIT_LENGTH_IN_CGS, rho_grain_code=All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS, rho_gas_code=P[i].Gas_Density*All.cf_a3inv; /* internal grain density in code units */
        for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
        {
            double Q_abs_eff = return_grain_extinction_efficiency_Q(i, k_freq); /* need this to calculate the absorption efficiency in each band */
            in->Grain_Abs_Coeff[k_freq] = Q_abs_eff * 3. / (4. * C_LIGHT_CODE_REDUCED * rho_grain_code * R_grain_code * rho_gas_code);
        }
    }
}

/* this structure defines the variables that need to be sent -back to- the 'searching' element */
struct OUTPUT_STRUCT_NAME { /* define variables below as e.g. "double X;" */
    Vec3<MyDouble> Interpolated_Radiation_Acceleration; /* flux values to return to grains */
    MyDouble Interpolated_Opacity[N_RT_FREQ_BINS]; /* opacity values interpolated to gas positions */
    MyDouble InterpolatedGeometricDustCrossSection; /* geometric opacity (no wavelength dependence) interpolated to gas positions */
} *DATARESULT_NAME, *DATAOUT_NAME; /* dont mess with these names, they get filled-in by your definitions automatically */

/* this subroutine assigns the values to the variables that need to be sent -back to- the 'searching' element */
static inline void OUTPUTFUNCTION_NAME(struct OUTPUT_STRUCT_NAME *out, int i, int mode, int loop_iteration) {  /* "i" is the particle to which data from structure "out" will be assigned. mode=0 for local communication, =1 for data sent back from other processors. you must account for this. */
    int k,k_freq;
    if(P[i].Type==0) {ASSIGN_ADD(CellP[i].InterpolatedGeometricDustCrossSection,out->InterpolatedGeometricDustCrossSection,mode);
        for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++) {ASSIGN_ADD(CellP[i].Interpolated_Opacity[k_freq],out->Interpolated_Opacity[k_freq],mode);}}
    if((1 << P[i].Type) & (GRAIN_PTYPES)) {P[i].GravAccel += out->Interpolated_Radiation_Acceleration/All.cf_a2inv;} /* this simply adds to the 'gravitational' acceleration for kicks */ // currently incompatible with hermite integrator -- need to update to Other_Accel
}

/* this subroutine does the actual neighbor-element calculations (this is the 'core' of the loop, essentially) */
int interpolate_fluxes_opacities_gasgrains_evaluate(int target, int mode, int *exportflag, int *exportnodecount, int *exportindex, int *ngblist, int loop_iteration)
{
    int startnode, numngb_inbox, listindex = 0, j, n; struct INPUT_STRUCT_NAME local; struct OUTPUT_STRUCT_NAME out; memset(&out, 0, sizeof(struct OUTPUT_STRUCT_NAME)); /* define variables and zero memory and import data for local target*/
    if(mode == 0) {INPUTFUNCTION_NAME(&local, target, loop_iteration);} else {local = DATAGET_NAME[target];} /* imports the data to the correct place and names */

    /* Now start the actual neighbor computation for this particle */
    if(mode == 0) {startnode = All.MaxPart; /* root node */} else {startnode = DATAGET_NAME[target].NodeList[0]; startnode = Nodes[startnode].u.d.nextnode;    /* open it */}
    while(startnode >= 0) {
        while(startnode >= 0) {
            if(local.Type == 0)
            {
                numngb_inbox = ngb_treefind_pairs_threads_targeted(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist, GRAIN_PTYPES); /* gas searches for grains which can see -it- */
            } else {
                numngb_inbox = ngb_treefind_variable_threads(local.Pos, local.KernelRadius, target, &startnode, mode, exportflag, exportnodecount, exportindex, ngblist); /* grains search for gas -they- can see */
            }
            if(numngb_inbox < 0) {return -2;} /* no neighbors! */
            for(n = 0; n < numngb_inbox; n++) /* neighbor loop */
            {
                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if((P[j].Mass <= 0)||(P[j].KernelRadius <= 0)) {continue;} /* make sure neighbor is valid */
                int k,k_freq; double h_to_use; Vec3<double> dp=local.Pos-P[j].Pos; /* position offset */
                nearest_xyz(dp); double r2=dp.norm_sq(); /* box-wrap appropriately and calculate distance */
                if(local.Type == 0) {h_to_use = P[j].KernelRadius;} else {h_to_use = local.KernelRadius;}
                if((r2>0)&&(r2<h_to_use*h_to_use)) /* only keep elements inside search radius */
                {
                    double wt=0,hinv,hinv3,hinv4,wk_i=0,dwk_i=0,r=sqrt(r2); kernel_hinv(h_to_use,&hinv,&hinv3,&hinv4);
                    kernel_main(r*hinv, hinv3, hinv4, &wk_i, &dwk_i, 0); /* kernel quantities that may be needed */

                    if(local.Type==0) /* sitting on a -gas- element, want to interpolate opacity to it */
                    {
                        wt = P[j].Mass * (wk_i / P[j].Gas_Density); /* dimensionless weight of this grain element as 'seen' by the gas: = (grain_part_mass/gas_part_mass) * (gas_part_mass * Wk / gas_density [=sum gas_part_mass * Wk]) - using sum as seen by grain and H for grain ensures grain can't be 'double counted' -- sum over all gas elements gives correct total mass of dust distributed over the entire system, as desired */
                        double R_grain_code=P[j].Grain_Size/UNIT_LENGTH_IN_CGS, rho_grain_code=All.Grain_Internal_Density/UNIT_DENSITY_IN_CGS; /* internal grain density in code units */
                        double geometric_dustcrosssection_factor = wt * 3. / (4. * rho_grain_code * R_grain_code); // interpolated opacity contribution in the absence of any absorption efficiency parameter
                        out.InterpolatedGeometricDustCrossSection += geometric_dustcrosssection_factor; // add this alone here
                        for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
                        {
                            double Q_abs_eff = return_grain_extinction_efficiency_Q(j, k_freq); /* need this to calculate the absorption efficiency in each band */
                            out.Interpolated_Opacity[k_freq] += Q_abs_eff * geometric_dustcrosssection_factor; // add the relevant weight for each band
                        }
                    } else { /* sitting on a -grain- element, want to interpolate flux to it and calculate radiation pressure force */
                        wt = CellP[j].Density*All.cf_a3inv * wk_i; /* weight of element to 'i, with appropriate coefficient from above */
                        double radacc[3]={0},vel_i[3]={0},dtEgamma_work_done=0; for(k=0;k<3;k++) {vel_i[k]=RSOL_CORRECTION_FACTOR_FOR_VELOCITY_TERMS*local.Vel[k]/All.cf_atime;} /* velocity of interest here is the grain velocity (radiation in lab frame) */
                        for(k_freq=0;k_freq<N_RT_FREQ_BINS;k_freq++)
                        {
                            double f_kappa_abs=0.5,vdot_h[3]={0},flux_i[3]={0},flux_mag=0,erad_i=0,flux_corr=1;
                            f_kappa_abs = 0.5; // rt_absorb_frac_albedo(i,k_freq, P, CellP); -- this is set to 1/2 anyways but would require extra passing, ignore for now //
#if defined(RT_EVOLVE_FLUX) || (defined(RT_USE_GRAVTREE_SAVE_RAD_FLUX) && defined(RT_USE_GRAVTREE_SAVE_RAD_ENERGY))
                            erad_i = CellP[j].Rad_E_gamma_Pred[k_freq];
                            {Vec3<double> v_i{vel_i[0],vel_i[1],vel_i[2]}; Vec3<double> vdh = erad_i * (v_i + CellP[j].ET[k_freq].matvec(v_i)); vdot_h[0]=vdh[0]; vdot_h[1]=vdh[1]; vdot_h[2]=vdh[2];} // P_rad term + eI term //
                            for(k=0;k<3;k++) {flux_i[k]=CellP[j].Rad_Flux_Pred[k_freq][k]; flux_mag+=flux_i[k]*flux_i[k];}
                            if(flux_mag>0 && isfinite(flux_mag)) {flux_mag=sqrt(flux_mag);} else {flux_mag=MIN_REAL_NUMBER; flux_i[0]=flux_i[1]=0; flux_i[2]=flux_mag;}
#elif (defined(RT_OTVET) || defined(RT_FLUXLIMITEDDIFFUSION))
                            erad_i = CellP[j].Rad_E_gamma_Pred[k_freq];
                            for(k=0;k<3;k++) {flux_i[k] = -CellP[j].Gradients.Rad_E_gamma_ET[k_freq][k]; flux_mag+=flux_i[k]*flux_i[k];}
                            if(flux_mag>0) {for(k=0;k<3;k++) {flux_i[k]/=sqrt(flux_mag);}} else {flux_i[0]=0;flux_i[1]=0;flux_i[2]=1;}
                            flux_mag = erad_i*C_LIGHT_CODE_REDUCED; for(k=0;k<3;k++) {flux_i[k]*=flux_mag;}
#endif
                            if(!isfinite(flux_mag) || flux_mag<=MIN_REAL_NUMBER) {flux_mag=MIN_REAL_NUMBER; flux_i[0]=flux_i[1]=0; flux_i[2]=flux_mag;}
                            double flux_thin = erad_i * C_LIGHT_CODE_REDUCED; if(!isfinite(flux_thin) || flux_thin<=0) {flux_thin=0;}
                            flux_corr = DMIN(1., 100.*flux_thin/flux_mag);
                            for(k=0;k<3;k++)
                            {
                                radacc[k] += local.Grain_Abs_Coeff[k_freq] * wt * (flux_corr*flux_i[k] - vdot_h[k]); // note these 'vdoth' terms shouldn't be included in FLD, since its really assuming the entire right-hand-side of the flux equation reaches equilibrium with the pressure tensor, which gives the expression in rt_utilities
                                dtEgamma_work_done += (2.*f_kappa_abs-1.) * radacc[k] * vel_i[k] * local.Mass; // PdV work done by photons [absorbed ones are fully-destroyed, so their loss of energy and momentum is already accounted for by their deletion in this limit -- note that we have to be careful about the RSOL factors here! //
                            }
                        }
                        for(int kk=0;kk<3;kk++) {out.Interpolated_Radiation_Acceleration[kk] += radacc[kk];} /* prepare to send back to grain */
                    }
                }
            } // numngb_inbox loop
        } // while(startnode)
        if(mode == 1) {listindex++; if(listindex < NODELISTLENGTH) {startnode = DATAGET_NAME[target].NodeList[listindex]; if(startnode >= 0) {startnode = Nodes[startnode].u.d.nextnode; /* open it */}}} /* continue to open leaves if needed */
    }
    if(mode == 0) {OUTPUTFUNCTION_NAME(&out, target, 0, loop_iteration);} else {DATARESULT_NAME[target] = out;} /* collects the result at the right place */
    return 0;
}

void interpolate_fluxes_opacities_gasgrains(void)
{
    PRINT_STATUS(" ..assigning opacities to gas from the grain distribution, and interpolating radiation fields to grains");
    {
        int num_gas = 0, num_grain = 0;
        for(int i : ActiveParticleList) {
            if(P[i].TimeBin < 0 || P[i].Mass <= 0 || P[i].KernelRadius <= 0) continue;
            if(P[i].Type == 0) num_gas++;
            if((1 << P[i].Type) & (GRAIN_PTYPES)) num_grain++;
        }
        int ng = (num_gas > 0 ? num_gas : 1), nr = (num_grain > 0 ? num_grain : 1);
        int *nl_active_gas   = (int *)    mymalloc("gasgrainrt_nl_gas_active",   ng * sizeof(int));
        double *nl_radii_gas = (double *) mymalloc("gasgrainrt_nl_gas_radii",    ng * sizeof(double));
        int *nl_active_grain   = (int *)    mymalloc("gasgrainrt_nl_grain_active", nr * sizeof(int));
        double *nl_radii_grain = (double *) mymalloc("gasgrainrt_nl_grain_radii",  nr * sizeof(double));
        {int ag = 0, ar = 0; for(int i : ActiveParticleList) {
            if(P[i].TimeBin < 0 || P[i].Mass <= 0 || P[i].KernelRadius <= 0) continue;
            if(P[i].Type == 0) { nl_active_gas[ag] = i; nl_radii_gas[ag] = (double)P[i].KernelRadius; ag++; }
            if((1 << P[i].Type) & (GRAIN_PTYPES)) { nl_active_grain[ar] = i; nl_radii_grain[ar] = (double)P[i].KernelRadius; ar++; }
        }}
        interpolate_fluxes_opacities_gasgrains_evaluate_gpu(P, CellP, NumPart,
            nl_active_gas, num_gas, nl_radii_gas,
            nl_active_grain, num_grain, nl_radii_grain);
        myfree(nl_radii_grain); myfree(nl_active_grain);
        myfree(nl_radii_gas); myfree(nl_active_gas);
        CPU_Step[CPU_DRAGFORCE] += measure_time();
    }
}
#include "../system/code_block_xchange_finalize.h" /* de-define the relevant variables and macros to avoid compilation errors and memory leaks */



double return_grain_extinction_efficiency_Q(int i, int k_freq)
{
    double Q = 1; /* default to geometric opacity */
#if defined(GRAIN_RDI_TESTPROBLEM)
    Q *= All.Grain_Q_at_MaxGrainSize; // this needs to be set by-hand, Q for the maximum sized grains. irrelevant for the scale-free problem (degenerate with flux), but important here */
#if !defined(GRAIN_RDI_TESTPROBLEM_ACCEL_DEPENDS_ON_SIZE)
    Q *= P[i].Grain_Size / All.Grain_Size_Max;
#endif
#else
    /* INSERT PHYSICS HERE -- this is where you want to specify the optical properties of grains relative to the frequency bins being evolved. could code up something for -ALL- the bins we do, but that's a lot, so we'll do these as-needed, for runs with different frequencies */
#if defined(RADTRANSFER) || defined(RT_USE_GRAVTREE)
    double nu_min_ev=All.RHD_bins_nu_min_ev[k_freq], nu_max_ev=All.RHD_bins_nu_max_ev[k_freq]; // get the radiation frequency range in eV
    double x_min = 5.068e4 * P[i].Grain_Size * nu_min_ev, x_max = 5.068e4 * P[i].Grain_Size * nu_max_ev; // this is the 'x' parameter for Q, defined as 2*pi*a_grain/lambda_light
    /* don't have a detailed model here for dielectric properties of grains, so instead use an extremely simple model as follows */
    double x_eff=sqrt(x_min*x_max); if(x_min<=MIN_REAL_NUMBER) {x_eff=0.5*x_max;} // take geometric mean or linear mean if former not well-defined
    return DMIN(x_eff, 1.); // simple model where Q~x for x<<1, Q=1 for x>>1
#else
    if(ThisTask==0) {PRINT_WARNING("Code does not have entered grain absorption efficiency/optical properties for your specific wavelength being evolved. Please enter that information in the routine 'return_grain_extinction_efficiency_Q'. For now will assume geometric absorption (Q=1). \n");}
#endif
#endif
    return Q;
}



#endif //defined(RT_OPACITY_FROM_EXPLICIT_GRAINS)




#endif
