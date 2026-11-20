/* gr/gr_kick_functions.h -- the GR integration of the conserved variables: kick, drift and predict.
 *
 * The hydro loop returns the rates of the conserved variables (DtEnergy = dtau/dt, DtMomentum = dS_j/dt, DtB, DtVolume);
 * the kick advances (D, S_j, tau, sqrt(gamma)B) with those rates and the metric source terms, inverts the result
 * (con2prim) and converts it into the rates of the primitive velocity and specific energy (HydroAccel, DtInternalEnergy)
 * that the generic kick/drift of the code apply. The drift re-inverts the kicked and predicted conserved variables with
 * the metric at the new position (implicit position correction along the path), and the predict advances the predicted
 * primitives from the conserved rates. Lupi 2023, MNRAS, 519, 1115.
 *
 * Every function is a KOKKOS_INLINE_FUNCTION on the particle/cell pointers it is given (the drift runs on the device in
 * drift_particle_impl, the kick on the host in do_the_kick): no globals but All. A failed inversion prints and calls
 * endrun (the device variant traps, the host one requests a controlled stop).
 *
 * The metric source terms are in gr/gr_source_functions.h, the inversions (including the fixed-u one of the drift) in
 * gr/gr_con2prim_functions.h.
 *
 * Port of gr/gr_kickdrift.cc of the C/C++ GIZMO branches (A. Lupi, G. Fedrigo). */
#ifndef GR_KICK_FUNCTIONS_H
#define GR_KICK_FUNCTIONS_H
#ifdef GENERAL_RELATIVITY

#include "gr_source_functions.h" /* gr_add_source_term */

/* the primitives of cell i from its (kicked or predicted) state: Eulerian velocity of vcoord, mass and volume, specific energy, field B*V */
KOKKOS_INLINE_FUNCTION double gr_primitives_of_cell(int i, const Vec3<double> &vcoord, double mass, double Vi, double u, struct gr_primitives *Prims, struct particle_data *pp, struct gas_cell_data *cell)
{
    const struct gr_metric *gr = &(cell[i].gr);
    Prims->v = Vphys(vcoord, gr);
    double Wlorentz = lorentz_factor(gr->gamma, Prims->v, gr->limiter);
    Prims->rho0 = mass/Wlorentz/gr->gdet/Vi;
    Prims->u = u;
    Prims->p = gr_pressure_rho0_u(Prims->rho0, Prims->u);
    return Wlorentz;
}

/* the kick of the conserved variables over dt_kick (mode 0: first half-step, 1: second half-step), converted into the rates
   HydroAccel = dv/dt (coordinate velocity) and DtInternalEnergy = du/dt that the generic kick applies */
KOKKOS_INLINE_FUNCTION void gr_kick_impl(int i, double dt_kick, double mass_old, double mass_new, int mode, struct particle_data *pp, struct gas_cell_data *cell)
{
    const struct gr_metric *gr = &(cell[i].gr);
    struct gr_primitives Prims;
    struct gr_conservatives Cons;
    /* until the rates are computed at the end they hold the raw flux sums of the hydro loop: zero them, so that an early return (failed
       source term or inversion, excised cell) leaves nothing for the generic kick to apply */
    cell[i].HydroAccel = Vec3<double>{0,0,0}; cell[i].DtInternalEnergy = 0;
    double Vi, Vnew; /* the volume before and after the kick, from the volume variation of the Riemann problems */
    if(mode == 1) {Vnew = mass_new/cell[i].Density; Vi = Vnew*exp(-cell[i].DtVolume*dt_kick/Vnew);}
    else          {Vi = mass_old/cell[i].Density; Vnew = Vi*exp(cell[i].DtVolume*dt_kick/Vi);}
#ifdef GR_TEST_PARTICLE
    mass_old = mass_new = Vi = Vnew = 1;
#endif
    double Wlorentz = gr_primitives_of_cell(i, pp[i].Vel, mass_old, Vi, cell[i].InternalEnergy, &Prims, pp, cell);
    gr_get_conservative_variables(&Prims, &Cons, Wlorentz, gr);
    /* source terms, then the fluxes (with MFM and HLLC the sources are applied in both half-kicks, before the fluxes, as in Lupi 2023) */
#if !(defined(HYDRO_MESHLESS_FINITE_MASS) && (GENERAL_RELATIVITY==1))
    if(mode == 0)
#endif
    {if(gr_add_source_term(i, &Prims, &Cons, dt_kick, 1-mode, pp, cell) < 0) {return;}}
    if(pp[i].Mass == 0) {return;}
    Cons.D = mass_new/Vnew;
    Cons.tau = (Cons.tau*Vi + cell[i].DtEnergy*dt_kick)/Vnew;
    Cons.S = (Vi*Cons.S + dt_kick*cell[i].DtMomentum)/Vnew;
    double res_con2prim = gr_get_primitive_variables(&Cons, &Prims, gr, gr->limiter);
    if(res_con2prim < 0)
    {
        printf("GR: failed inversion (code %g) in the kick: ID %llu at %g %g %g, kernel radius %g\n", res_con2prim, (unsigned long long) pp[i].ID, pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2], pp[i].KernelRadius);
        endrun(9992405); return;
    }
    if(mode == 0) {cell[i].InversionKeptU = 0;} /* (a new step) */
    /* (flag not in the original code) no admissible u exists for (D, S_j, B^i, tau) and the inversion kept the previous u (as in the
       original code): the cell is exempt from the energy time-step limit (timestep.cc), since its u carries no information */
    if(Prims.u_kept) {cell[i].InversionKeptU = 1;}
#if !(defined(HYDRO_MESHLESS_FINITE_MASS) && (GENERAL_RELATIVITY==1))
    if(mode == 1) {if(gr_add_source_term(i, &Prims, &Cons, dt_kick, 1-mode, pp, cell) < 0) {return;} if(pp[i].Mass == 0) {return;}}
#endif
    /* the rates from the old and new primitives */
    double rate_fac = 1./dt_kick;
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    rate_fac *= mass_new/pp[i].Mass; /* the generic kick applies the rates with the predicted mass P.Mass and divides by mass_new */
#endif
    cell[i].DtInternalEnergy = (Prims.u - cell[i].InternalEnergy)*rate_fac;
    cell[i].HydroAccel = (Vcoord(Prims.v, gr) - pp[i].Vel)*rate_fac;
}

/* the drift of the conserved variables: the kicked and predicted states are re-inverted with the metric at the new position,
   with an implicit correction of the position along the path for MFM (the particle moves with the velocity of the drifted state); for MFV
   the cell moves with its mesh-generating point (ParticleVel) and is re-inverted once at the position it reached. The correction is not
   mirrored in the read-only motion predictor (predict_particle_motion, used only by the gravity walks, which GR never runs) nor in the
   speed bound of the tree nodes: the re-inversion holds S_j fixed, so the coordinate velocity changes only through the metric along the
   path, |dv| ~ |v| dt |d g|, and the corrected displacement stays within the TREE_DRIFT_VELOCITY_PREFAC = 2 widening while dt |d g| < 2 */
KOKKOS_INLINE_FUNCTION void gr_drift_impl(int i, double dt_drift, struct particle_data *pp, struct gas_cell_data *cell)
{
    struct gr_primitives Prims;
    struct gr_conservatives Cons, ConsPred;
    double Vi = pp[i].Mass/cell[i].Density, res_con2prim, Wlorentz;
    Vec3<double> v_old = pp[i].Vel, v_new;
#ifdef GR_TEST_PARTICLE
    pp[i].Mass = Vi = 1;
    if(gr_hole_displacement(pp[i].Pos).norm() <= 1.001) {pp[i].Mass = 0;}
#endif
    /* conserved variables of the predicted and of the kicked state, with the metric of the old position */
    Wlorentz = gr_primitives_of_cell(i, cell[i].VelPred, pp[i].Mass, Vi, cell[i].InternalEnergyPred, &Prims, pp, cell);
    gr_get_conservative_variables(&Prims, &ConsPred, Wlorentz, &(cell[i].gr));
    Wlorentz = gr_primitives_of_cell(i, pp[i].Vel, pp[i].Mass, Vi, cell[i].InternalEnergy, &Prims, pp, cell);
    gr_get_conservative_variables(&Prims, &Cons, Wlorentz, &(cell[i].gr));
    /* implicit drift loop, accounting for the metric change along the path */
    double error = MAX_REAL_NUMBER; int iter = 0;
    do {
        Vec3<double> x_err = pp[i].Pos;
        set_metric_coefficients(pp[i].Pos, pp[i].Potential, &(cell[i].gr)); /* metric at the predicted location */
#ifdef GR_DRIFT_HOLD_INTERNAL_ENERGY
        {struct gr_conservatives Ctmp; res_con2prim = gr_drift_reinvert(&Cons, cell[i].InternalEnergy, GAMMA_DEFAULT, &(cell[i].gr), cell[i].gr.limiter, &Prims, &Ctmp);}
#else
        res_con2prim = gr_get_primitive_variables(&Cons, &Prims, &(cell[i].gr), cell[i].gr.limiter);
#endif
        if(res_con2prim < 0)
        {
            printf("GR: failed inversion (code %g) in the drift: ID %llu at %g %g %g\n", res_con2prim, (unsigned long long) pp[i].ID, pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2]);
            endrun(9992406); return;
        }
        v_new = Vcoord(Prims.v, &(cell[i].gr));
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
        break; /* MFV: the cell is a mesh point moving with ParticleVel (any mesh motion), decoupled from the fluid velocity: a single
                  re-inversion at the position it reached, no implicit position correction */
#endif
        pp[i].Pos += (0.5*dt_drift)*(v_new - v_old); v_old = v_new; /* position with the corrected velocity */
        x_err = pp[i].Pos - x_err;
        error = fmax(fmax(fabs(x_err[0]), fabs(x_err[1])), fabs(x_err[2]));
        iter++;
        if(iter > GR_IMPLICIT_MAX_NITER)
        {
            printf("GR WARNING (drift): %d iterations: pos %2.15g %2.15g %2.15g, err %g %g %g / %g\n", GR_IMPLICIT_MAX_NITER, pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2], x_err[0], x_err[1], x_err[2], error);
            break;
        }
    } while(error > GR_IMPLICIT_TOLERANCE);
    /* metric at the new location, kicked primitives (corrector) */
    set_metric_coefficients(pp[i].Pos, pp[i].Potential, &(cell[i].gr));
#ifdef GR_DRIFT_HOLD_INTERNAL_ENERGY
    res_con2prim = gr_drift_reinvert(&Cons, cell[i].InternalEnergy, GAMMA_DEFAULT, &(cell[i].gr), cell[i].gr.limiter, &Prims, &Cons);
#else
    res_con2prim = gr_get_primitive_variables(&Cons, &Prims, &(cell[i].gr), cell[i].gr.limiter);
#endif
    if(res_con2prim < 0) {printf("GR: failed inversion (code %g) in the drift corrector: ID %llu\n", res_con2prim, (unsigned long long) pp[i].ID); endrun(9992407); return;}
    pp[i].Vel = Vcoord(Prims.v, &(cell[i].gr)); /* velocity consistent with the metric change */
    cell[i].InternalEnergy = Prims.u;
    /* predicted primitives */
#ifdef GR_DRIFT_HOLD_INTERNAL_ENERGY
    {Vec3<double> vguess = Vphys(cell[i].VelPred, &(cell[i].gr)); Prims.v = vguess;
     res_con2prim = gr_drift_reinvert(&ConsPred, cell[i].InternalEnergyPred, GAMMA_DEFAULT, &(cell[i].gr), cell[i].gr.limiter, &Prims, &ConsPred);}
#else
    res_con2prim = gr_get_primitive_variables(&ConsPred, &Prims, &(cell[i].gr), cell[i].gr.limiter);
#endif
    if(res_con2prim < 0) {printf("GR: failed inversion (code %g) in the drift (predicted state): ID %llu\n", res_con2prim, (unsigned long long) pp[i].ID); endrun(9992408); return;}
    Vec3<double> vpred_new = Vcoord(Prims.v, &(cell[i].gr));
#if defined(HYDRO_MESHLESS_FINITE_VOLUME) && (HYDRO_FIX_MESH_MOTION>=5)
    cell[i].ParticleVel += vpred_new - cell[i].VelPred;
#endif
    cell[i].VelPred = vpred_new;
    cell[i].InternalEnergyPred = Prims.u;
}

/* the prediction of the primitives over dt_hydrokick from the conserved rates (and the metric sources), replacing the generic
   VelPred/InternalEnergyPred prediction: HydroAccel and DtInternalEnergy are zeroed so that the generic drift adds nothing */
KOKKOS_INLINE_FUNCTION void gr_predict_impl(int i, double dt_hydrokick, struct particle_data *pp, struct gas_cell_data *cell)
{
    const struct gr_metric *gr = &(cell[i].gr);
    struct gr_primitives Prims;
    struct gr_conservatives ConsPred;
    double Vi = pp[i].Mass/cell[i].Density, mass_new = pp[i].Mass, res_con2prim;
#ifdef GR_TEST_PARTICLE
    if(pp[i].Mass > 0) {pp[i].Mass = 1.0;} /* (an excised test particle keeps its zero mass) */
    mass_new = pp[i].Mass; Vi = 1.0;
#endif
#ifdef PARTICLE_EXCISION
    if(pp[i].Mass == 0) {return;}
#endif
    double Wlorentz = gr_primitives_of_cell(i, cell[i].VelPred, pp[i].Mass, Vi, cell[i].InternalEnergyPred, &Prims, pp, cell);
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    mass_new = fmax(mass_new + cell[i].DtMass*dt_hydrokick, 0.5*cell[i].MassTrue);
#endif
    gr_get_conservative_variables(&Prims, &ConsPred, Wlorentz, gr);
    double Vnew = Vi*exp(cell[i].DtVolume*dt_hydrokick/Vi);
    if(gr_add_source_term(i, &Prims, &ConsPred, dt_hydrokick, 0, pp, cell) < 0) {return;} /* (also inverts the sourced state into Prims) */
    if(pp[i].Mass == 0) {return;}
    ConsPred.D = mass_new/Vnew;
    ConsPred.tau = (ConsPred.tau*Vi + cell[i].DtEnergy*dt_hydrokick)/Vnew;
    ConsPred.S = (Vi*ConsPred.S + dt_hydrokick*cell[i].DtMomentum)/Vnew;
    res_con2prim = gr_get_primitive_variables(&ConsPred, &Prims, gr, gr->limiter);
    if(res_con2prim < 0) {printf("GR: failed inversion (code %g) in the predict (fluxes): ID %llu\n", res_con2prim, (unsigned long long) pp[i].ID); endrun(9992409); return;}
    cell[i].VelPred = Vcoord(Prims.v, gr);
    cell[i].InternalEnergyPred = Prims.u;
    /* zero the rates, to avoid double counting in the generic drift */
    cell[i].HydroAccel = Vec3<double>{0,0,0};
    cell[i].DtInternalEnergy = 0;
}

#endif /* GENERAL_RELATIVITY */
#endif /* GR_KICK_FUNCTIONS_H */
