/* elastic_physics_functions.h — KOKKOS_INLINE_FUNCTION inline body for
 * get_negative_pressure_tensilecorrfac, the tensile-instability kernel
 * correction factor used by the hydro pair body under solid-EOS Configs
 * (EOS_TILLOTSON / EOS_ELASTIC / EOS_ANEOS).
 *
 * Why this header: hydro_functions.h:hydro_accumulate_neighbor (a
 * __host__ __device__ pair body, in the corridor hot path) calls this
 * function on the device pass. Without an inline body visible at the call
 * site, the nvc++ device pass sees only the host-only forward declaration
 * in core/proto.h and emits warning #20011-D ("calling a __host__ function
 * from a __host__ __device__ function is not allowed"). That is a silent
 * physics error on GPU.
 *
 * elastic_physics.cc continues to provide the non-inline host external
 * symbol via the standard #undef KOKKOS_INLINE_FUNCTION / re-include
 * pattern; other host call sites (proto.h forward-declared) link against
 * that one strong symbol.
 *
 * Written by Phil Hopkins (phopkins@caltech.edu) for GIZMO.
 */
#pragma once

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

/* Requires the caller to have already included mesh/kernel.h (provides
 * kernel_main) and declarations/allvars.h (provides All.DesNumNgb).
 * NOT included here because hydro_functions.h has already included it via its
 * own include chain, so this header does not need to repeat it. (kernel.h has
 * carried a #pragma once since the drift-helper work; it is safe either way.) */


#ifdef EOS_ELASTIC
/* pressure_dependent_yield_strength and the fracture-damage updates, used by the stress update below
   under the EOS_DAMAGE_POROSITY bits. Safe to include here: it carries an include
   guard and does not pull in Kokkos. */
#include "damage_porosity_functions.h"

/* time derivative of the deviatoric stress tensor from a velocity gradient (Hooke's law with the Jaumann rotation) */
KOKKOS_INLINE_FUNCTION
void elastic_stress_rate_P(int i, const double dv0[3][3], struct gas_cell_data *cell)
{
    int j,k,l,NDim=NUMDIMS;
    double R[3][3], S[3][3], dS=0;
    if(!solid_material_is_elastic(cell[i].CompositionType)) {return;} /* the gas of a gas+solid build carries no deviatoric stress */
    double mu = solid_material_of(cell[i].CompositionType).ShearModulus * cell[i].strength_fraction(); /* thermally softened (molten: no stress rate) */
    for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {S[j][k] = cell[i].Elastic_Stress_Tensor_Pred[j][k];}}
    for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {R[j][k] = 0.5*(dv0[j][k] - dv0[k][j]);}}
    double trace_vel=0; for(j=0;j<NDim;j++) {trace_vel += dv0[j][j];}
    for(j=0;j<NDim;j++) // velocity index
    {
        for(k=0;k<NDim;k++) // gradient index
        {
            dS = mu * (dv0[j][k] + dv0[k][j]); // symmetric strain component
            if(k==j) {dS -= 2.*mu*trace_vel/NDim;} // trace component
            for(l=0;l<NDim;l++) {dS += R[j][l]*S[l][k] - S[j][l]*R[l][k];} // Jaumann rotation W.S - S.W, with R = W = (dv_j/dx_k - dv_k/dx_j)/2: the stress turns with the material
            cell[i].Dt_Elastic_Stress_Tensor[j][k] = dS; // save it to variable
        }
    }
}

/* the radial return (Wilkins 1964): scale a deviator with second invariant J2 back onto the yield surface sqrt(NUMDIMS J2) = Y.
   It only ever lowers the stress. Returns 1 if it acted. */
KOKKOS_INLINE_FUNCTION
int elastic_radial_return(double S[3][3], double J2, double Y_yield)
{
    if(!(J2 > 0)) {return 0;}
    double f_Y = Y_yield*Y_yield/(NUMDIMS*J2); // (yield stress / equivalent stress)^2
    if(!(f_Y < 1)) {return 0;}
    double return_factor = sqrt(f_Y); for(int j=0;j<NUMDIMS;j++) {for(int k=0;k<NUMDIMS;k++) {S[j][k] *= return_factor;}}
    return 1;
}

/* thermal softening of a stored stress: while the material stays partly solid the stress follows the strength fraction both ways,
   S *= f_now / f_last (the elastic strain is kept; the yield return that follows removes what exceeds the new strength); once it has
   melted (f_last = 0) the stress is gone and only new strain builds it again. No energy moves: the stress work already sits in the
   internal energy. f_last becomes f_now. Returns the factor applied. */
KOKKOS_INLINE_FUNCTION
double elastic_strength_rescale(double S[3][3], double *f_last, double f_now)
{
    double rescale = 1;
    if(*f_last > 0 && f_now != *f_last) {rescale = f_now / *f_last; for(int j=0;j<NUMDIMS;j++) {for(int k=0;k<NUMDIMS;k++) {S[j][k] *= rescale;}}}
    *f_last = f_now;
    return rescale;
}

/* routine to update the deviatoric stress tensor */
KOKKOS_INLINE_FUNCTION
void elastic_body_update_driftkick_P(int i, double dt_entr, int mode,
                                     struct particle_data *pp, struct gas_cell_data *cell)
{
    int j,k,NDim=NUMDIMS;
    double dv0[3][3], S_new[3][3], mu, Y0, J2=0, I1=0;
    if(!solid_material_is_elastic(cell[i].CompositionType)) {return;} /* the gas of a gas+solid build carries no deviatoric stress */
    const struct solid_material &material = solid_material_of(cell[i].CompositionType);
    mu = material.ShearModulus; Y0 = material.YieldStrength;

    if(mode < 2) // drift or kick operation
    {
#ifdef EOS_ELASTIC_MELT
        /* thermal softening at the state this copy is updated with (the committed energy at the kick, the predicted one at the drift;
           temperature and pressure from the last equation-of-state evaluation, projected again after the next one) */
        double f_strength = solid_strength_fraction(cell[i].CompositionType, (mode == 0) ? cell[i].InternalEnergy : cell[i].InternalEnergyPred, cell[i].Temperature, cell[i].Pressure);
        double f_last = (mode == 0) ? cell[i].StrengthFraction : cell[i].StrengthFractionPred;
#else
        double f_strength = 1, f_last = 1;
#endif
        for(j=0;j<NDim;j++) {
            for(k=0;k<NDim;k++) {
                // determine which variable we are updating (mode=0/1 is kick/drift)
                if(mode==0) {S_new[j][k]=cell[i].Elastic_Stress_Tensor[j][k];} else {S_new[j][k]=cell[i].Elastic_Stress_Tensor_Pred[j][k];}
                S_new[j][k] += dt_entr * cell[i].Dt_Elastic_Stress_Tensor[j][k]; // apply time evolution
                if(k==j) {I1 += S_new[j][k];} // first invariant of the tensor
                J2 += 0.5 * S_new[j][k]*S_new[j][k]; // second invariant of the tensor
            }}
        // now apply the yield criterion: the constant von Mises strength, or (EOS_DAMAGE_POROSITY value 2, materials with
        // InternalFriction > 0) the pressure-dependent strength of intact rock
        double Y_eff = Y0;
#if defined(EOS_DAMAGE_POROSITY) && DAMAGE_POROSITY_BIT_PRESSURE_STRENGTH
        if(material.InternalFriction > 0) {Y_eff = pressure_dependent_yield_strength(material, cell[i].Pressure);}
#endif
        double rescale = elastic_strength_rescale(S_new, &f_last, f_strength); /* first follow the softened modulus, then return */
        J2 *= rescale * rescale;
#ifdef EOS_ELASTIC_MELT
        if(mode == 0) {cell[i].StrengthFraction = f_last;} else {cell[i].StrengthFractionPred = f_last;}
#endif
        Y_eff *= f_strength; /* thermally softened (molten: the return anneals the stress to zero) */
        volatile int yield_return_applied = elastic_radial_return(S_new, J2, Y_eff);
        /* recent-yield memory, at the kick only: decay it over RECENT_YIELD_MEMORY_SOUND_CROSSINGS sound-crossing times of the
           (physical) kernel radius, then add the overstress the return just removed, in units of the material's yield strength
           (for von Mises plasticity, summed over kicks, this is proportional to the plastic strain in yield strains, so the update
           converges with the timestep); the hydro pair uses it to switch the tensile-instability correction off in material that
           is flowing plastically. */
        if(mode == 0)
        {
            double h_phys = pp[i].KernelRadius * All.cf_atime, c_s = cell[i].SoundSpeed;
            if(h_phys > 0 && c_s > 0 && dt_entr > 0) {cell[i].RecentYieldMemory *= exp(-dt_entr * c_s / (RECENT_YIELD_MEMORY_SOUND_CROSSINGS * h_phys));}
            if(!(f_strength > 0)) {cell[i].RecentYieldMemory = DMAX(cell[i].RecentYieldMemory, 2.*TENSILE_CORRECTION_YIELD_MEMORY_ONSET);} /* molten: no strength, so the correction stays fully off (its stress is zero, so no return acts to say so) */
            else if(yield_return_applied)
            {
                double Y_norm = (Y0 > 0) ? f_strength * Y0 : Y_eff; /* a fixed strength scale (the constant von Mises Y, also for a pressure-dependent material, whose current strength can vanish in tension); without one, the current strength */
                if(Y_norm > 0) {
                    double overstress = (sqrt(NDim*J2) - Y_eff) / Y_norm;
                    if(isfinite(overstress) && overstress > 0) {cell[i].RecentYieldMemory += overstress;}
                } else {cell[i].RecentYieldMemory = DMAX(cell[i].RecentYieldMemory, 2.*TENSILE_CORRECTION_YIELD_MEMORY_ONSET);} /* no strength at all: the correction is fully off */
            }
        }
#if defined(EOS_DAMAGE_POROSITY) && DAMAGE_POROSITY_BIT_FRACTURE
        /* bit 0: fracture damage, updated at the kick (mode==0) only, by the law the material names; both laws take the cold shear
           modulus mu their flaw thresholds were calibrated with, not the thermally softened one. Fully molten material (f = 0) has no
           flaws to grow or activate: its damage and active-flaw count are kept unchanged until it re-solidifies. */
        if(mode == 0 && f_strength > 0)
        {
            double D_new, A_new;
            int composition = cell[i].CompositionType;
            volatile int shear_activated = (solid_material_of(composition).FractureActivation == FRACTURE_ACTIVATION_SHEAR);
            if(shear_activated)
            {
                damage_update_grady_kipp(cell[i].Damage, cell[i].ActiveCracks,
                                         J2, mu, cell[i].SoundSpeed, dt_entr,
                                         composition,
                                         &D_new, &A_new);
            } else {
                /* the damage-reduced total stress the flaws see: -P I + (1-D) S (the EOS pressure already carries (1-D) of any tension) */
                double D_now = cell[i].Damage, stress[3][3] = {{0}};
                for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {stress[j][k] = (1. - D_now) * S_new[j][k];} stress[j][j] -= cell[i].Pressure;}
                double rho_i = cell[i].Density;
                /* the material bulk modulus: Tillotson's A; a tabulated material has no constant, so its local rho c_bulk^2 */
                double bulk_modulus = rho_i * (cell[i].SoundSpeed*cell[i].SoundSpeed - elastic_longitudinal_soundspeed2_excess(composition, rho_i, cell[i].strength_fraction()));
#ifdef EOS_TILLOTSON
                if(eos_branch_of(composition) == EOS_BRANCH_TILLOTSON) {bulk_modulus = All.Tillotson_EOS_params[tillotson_subindex(composition)][4];}
#endif
                /* the fixed volume holding the element's flaws (its compact reference volume; times the out-of-plane extent in 1D/2D) and its flaw-count bound */
                double flaw_volume = pp[i].Mass / solid_material_of(composition).ReferenceDensity;
#if ((EOS_DAMAGE_POROSITY) & 4)
                flaw_volume *= solid_material_of(composition).InitialDistention; /* a porous element's bulk reference volume */
#endif
#if (NUMDIMS < 3)
                flaw_volume *= All.FractureOutOfPlaneExtent;
#endif
                double s_max = All.SolidMaterialFlawsPerMass[composition] * pp[i].Mass;
                uint64_t identity = gizmo_rng_particle_identity((uint64_t)pp[i].ID, (uint64_t)pp[i].ID_child_number, (uint64_t)pp[i].ID_generation); /* persistent: split children differ */
                if(damage_update_tensile_flaws(D_now, cell[i].ActiveCracks, stress, NDim, bulk_modulus, mu, rho_i, pp[i].Mass / rho_i,
                                               flaw_volume, s_max, identity, dt_entr, composition, &D_new, &A_new))
                {
                    printf("ERROR: tensile flaws: element ID %llu (child %llu) has more than %d flaws (s_max %g; its mass grew past the startup bound)\n",
                           (unsigned long long)pp[i].ID, (unsigned long long)pp[i].ID_child_number, 10 * TENSILE_FLAWS_MAX_PER_ELEMENT, s_max);
                    endrun(1154);
                }
            }
            cell[i].Damage       = (MyFloat)D_new;
            cell[i].ActiveCracks = (MyFloat)A_new;
        }
#endif
        // write out to variable //
        for(j=0;j<NDim;j++) {
            for(k=0;k<NDim;k++) {
                if(mode==0) {cell[i].Elastic_Stress_Tensor[j][k]=S_new[j][k];} else {cell[i].Elastic_Stress_Tensor_Pred[j][k]=S_new[j][k];}
            }}

    } else {

        // ok all below is for mode = 2, which is the actual calculation of the time derivative of the stress tensor
        for(j=0;j<NDim;j++) {for(k=0;k<NDim;k++) {dv0[j][k] = cell[i].Gradients.Velocity[j][k];}}
        elastic_stress_rate_P(i, dv0, cell);

    }

    return; // all done here
}

#ifdef ELASTIC_STATE_DEPENDENT_STRENGTH
/* after an accepted equation-of-state refresh (refresh_eos_and_project, and the device drift and cooling tails): a material whose strength
   depends on the evolving state -- on pressure (EOS_DAMAGE_POROSITY model 2) or on temperature or energy (thermal softening,
   EOS_ELASTIC_MELT) -- projects the stress copies that state stands for (STATE_COPY_* in core/proto.h) onto its strength at the refreshed
   state: the stored stress first follows the strength fraction (elastic_strength_rescale), then the radial return. Every stress the forces
   read then satisfies the yield condition at the state they read (the stress update itself saw the previous refresh). The return only
   removes stress: a rise in pressure or strength never restores what a return took. */
KOKKOS_INLINE_FUNCTION
void elastic_project_after_eos_P(int i, struct gas_cell_data *cell, int state_copies)
{
    int composition = cell[i].CompositionType;
    if(!solid_material_is_elastic(composition)) {return;}
    const struct solid_material &material = solid_material_of(composition);
    double Y_strength = material.YieldStrength; int state_dependent = 0;
#if defined(EOS_DAMAGE_POROSITY) && DAMAGE_POROSITY_BIT_PRESSURE_STRENGTH
    if(material.InternalFriction > 0) {Y_strength = pressure_dependent_yield_strength(material, cell[i].Pressure); state_dependent = 1;}
#endif
#ifdef EOS_ELASTIC_MELT
    if(All.SolidStrengthModel[composition].MeltForm != MELT_FORM_NONE) {state_dependent = 1;}
#endif
    if(!state_dependent) {return;} /* the constant law: nothing depends on the refreshed state */
    double S[3][3];
    for(int copy = 0; copy < 2; copy++)
    {
        int committed = (copy == 0);
        if(!(state_copies & (committed ? STATE_COPY_COMMITTED : STATE_COPY_PREDICTED))) {continue;}
        Mat3<MyDouble> &stress = committed ? cell[i].Elastic_Stress_Tensor : cell[i].Elastic_Stress_Tensor_Pred;
        for(int j=0;j<3;j++) {for(int k=0;k<3;k++) {S[j][k] = stress[j][k];}}
        double f_now = 1, f_last = 1;
#ifdef EOS_ELASTIC_MELT
        f_now = solid_strength_fraction(composition, committed ? cell[i].InternalEnergy : cell[i].InternalEnergyPred, cell[i].Temperature, cell[i].Pressure);
        f_last = committed ? cell[i].StrengthFraction : cell[i].StrengthFractionPred;
#endif
        int changed = (elastic_strength_rescale(S, &f_last, f_now) != 1);
#ifdef EOS_ELASTIC_MELT
        if(committed) {cell[i].StrengthFraction = f_last;} else {cell[i].StrengthFractionPred = f_last;}
        if(committed && !(f_now > 0)) {cell[i].RecentYieldMemory = DMAX(cell[i].RecentYieldMemory, 2.*TENSILE_CORRECTION_YIELD_MEMORY_ONSET);} /* molten: the tensile correction stays off (the memory is committed history) */
#endif
        double J2 = 0;
        for(int j=0;j<NUMDIMS;j++) {for(int k=0;k<NUMDIMS;k++) {J2 += 0.5 * S[j][k]*S[j][k];}}
        if(elastic_radial_return(S, J2, f_now * Y_strength)) {changed = 1;}
        if(changed) {for(int j=0;j<3;j++) {for(int k=0;k<3;k++) {stress[j][k] = S[j][k];}}}
    }
}
#endif
#endif /* EOS_ELASTIC */

#if defined(EOS_ELASTIC) || defined(EOS_TILLOTSON) || defined(EOS_ANEOS)
/* routine to get and define the correction factor needed to prevent tensile instability for negative pressures, for arbitrary kernels & dimensions */
KOKKOS_INLINE_FUNCTION
double get_negative_pressure_tensilecorrfac(double r, double h_i, double h_j)
{
    double dx_ips=0, wk_0=0, dwk_tmp=0, wk_r=0, r_over_heff=0;
#if (NUMDIMS==1)
    dx_ips = 2. / All.DesNumNgb; // 1D inter-node separation for desired NNgb, relative to radius of compact support
#elif(NUMDIMS==2)
    dx_ips = sqrt(M_PI / All.DesNumNgb); // 2D inter-node separation for desired NNgb, relative to radius of compact support
#else
    dx_ips = pow(4.*M_PI/3. / All.DesNumNgb, 1./3.); // 3D inter-node separation for desired NNgb, relative to radius of compact support
#endif
    kernel_main(dx_ips, 1., 1., &wk_0, &dwk_tmp, -1); // use kernels because of their stability properties: here weight for 'mean separation'
    r_over_heff = r / DMAX(h_i, h_j);
    kernel_main(r_over_heff, 1., 1., &wk_r, &dwk_tmp, -1); // here weight for actual half-separation
    return 0.2 * pow(wk_r / wk_0, 4); // correction factor for n=4 from Monaghan et al. 2000, Gray et al. 2001
}
#endif
