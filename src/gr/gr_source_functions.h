/* gr/gr_source_functions.h -- the metric source terms of the momentum and energy equations (Valencia formulation), device-callable.
 *
 * gr_source_term() evaluates them for a cell from the stress-energy tensor of its primitives; gr_add_source_term() adds them to the
 * conserved variables over a time interval (predictor, plus an implicit corrector), for the kick and the predict (gr/gr_kick_functions.h).
 * With the energy-holding drift, where the drift re-inverts the moved cells (GR_DRIFT_PATH_SOURCE), the part of the energy source that the
 * re-inversion of the drifted state produces (gr_drift_tau_path_rate) is removed from it.
 *
 * Port of gr/gr_source.h of the C/C++ GIZMO branches (A. Lupi, G. Fedrigo). */
#ifndef GR_SOURCE_FUNCTIONS_H
#define GR_SOURCE_FUNCTIONS_H
#ifdef GENERAL_RELATIVITY


#ifdef GR_DRIFT_PATH_SOURCE
/* rate of change of tau along the drift of cell i at fixed (D, S_j, sqrt(gamma) B^i, u), (d tau/d x^k) w^k with w the coordinate
   velocity the cell is displaced with: the fluid velocity of the state for MFM, the velocity of the mesh-generating point (ParticleVel) for
   MFV. The Valencia energy source is the total rate along the path, of which this part is produced by the re-inversion of the drifted
   state, so it is removed from the source applied in the kick. Centred finite differences of the slaved inversion along the direction of
   motion. Only where the drift re-inverts the moved cells (GR_DRIFT_PATH_SOURCE): elsewhere the kick applies the whole source. */
KOKKOS_INLINE_FUNCTION double gr_drift_tau_path_rate(int i, const struct gr_conservatives *cons, const struct gr_primitives *prims, struct particle_data *pp, struct gas_cell_data *cell)
{
    /* chain rule in the coordinates: the drift displaces the particle by dt vc^k, so the rate is (d tau/d x^k) vc^k with vc^k = dx^k/dt
       the coordinate velocity of the state; the finite-difference step is the coordinate displacement GR_EPS_TOLERANCE vc/|vc| (|vc| the
       Euclidean norm, a parametrisation of the step that cancels out: no spatial metric is involved) */
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    const Vec3<double> vc = cell[i].ParticleVel; /* MFV: the cell is a mesh point, displaced with its own velocity */
#else
    const Vec3<double> vc = Vcoord(prims->v, &(cell[i].gr));
#endif
    const double vnorm = vc.norm();
    if(vnorm == 0) {return 0;}
    const Vec3<double> dx = (GR_EPS_TOLERANCE/vnorm)*vc;
    const double gamma_eos = GAMMA_DEFAULT; /* GR: ideal gas with a constant adiabatic index */
    double tau_pm[2];
    for(int k = 0; k < 2; k++)
    {
        struct gr_metric gr_k; struct gr_primitives prims_k = *prims; struct gr_conservatives cons_k;
        set_metric_coefficients((k == 0) ? pp[i].Pos + dx : pp[i].Pos - dx, &gr_k);
        if(gr_drift_reinvert(cons, prims->u, gamma_eos, &gr_k, cell[i].gr.limiter, &prims_k, &cons_k) < 0) {return 0;}
        tau_pm[k] = cons_k.tau;
    }
    return (tau_pm[0] - tau_pm[1])*(0.5*vnorm/GR_EPS_TOLERANCE); /* [f(x+dx) - f(x-dx)] / (2 |dx|) * |vc| */
}
#endif

/* metric source terms of the momentum and energy equations at cell i, from the stress-energy tensor of the primitives
   (v may be limited in place where required): S_j: (alpha sqrt(gamma)/2) T^{mu nu} d_j g_{mu nu};
   tau: alpha sqrt(gamma) [T^{mu nu} K_ij-terms - (T^00 beta^j + T^0j) d_j alpha], with K_ij = -alpha Gamma^0_ij */
KOKKOS_INLINE_FUNCTION void gr_source_term(int i, struct gr_primitives *prims, struct gr_conservatives *Src, struct particle_data *pp, struct gas_cell_data *cell)
{
    const struct gr_metric *gr = &(cell[i].gr);
    const double alpha = gr->alpha, alphainv2 = 1./(alpha*alpha);
    double Pstar = prims->p, rhoh = prims->rho0*(1 + prims->u) + prims->p;
    double Wfac = lorentz_factor(gr->gamma, prims->v, gr->limiter);
    /* four-velocity u^mu = (W/alpha, W (v - beta/alpha)); v is the velocity seen by the Eulerian observer */
    double u0 = Wfac/alpha;
    Vec3<double> u_sp = Wfac*(prims->v - gr->beta/alpha);
    /* T^{mu nu} = rho h u u - b b + P* g^{mu nu}, with g^00 = -1/alpha^2, g^0i = beta^i/alpha^2, g^ij = gamma^ij - beta^i beta^j/alpha^2 */
    struct gr_sym4 T;
    T.tt = rhoh*u0*u0 - Pstar*alphainv2;
    T.ti = (rhoh*u0)*u_sp + (Pstar*alphainv2)*gr->beta;
    T.ij = rhoh*outer_product(u_sp) + Pstar*(gr->ugamma - alphainv2*outer_product(gr->beta));
    /* metric derivatives at the cell */
    SymmetricTensor2<double> Kij; struct gr_sym4 dg[3]; Vec3<double> d_alpha; Mat3<double> d_beta;
    get_metric_derivatives(pp[i].Pos, gr, Kij, dg, d_alpha, d_beta);
    Src->D = 0.; /* no mass source */
    Src->S = (0.5*alpha*gr->gdet)*Vec3<double>{gr_sym4_contract(T, dg[0]), gr_sym4_contract(T, dg[1]), gr_sym4_contract(T, dg[2])};
    double t = T.tt*Kij.quad(gr->beta) + 2.*dot(T.ti, Kij.matvec(gr->beta)) + Kij.double_contract(T.ij) - dot(T.tt*gr->beta + T.ti, d_alpha);
    Src->tau = alpha*gr->gdet*t;
    Src->p = 0;
}

/* adds the source terms over dtime to the conserved variables (predictor, plus the implicit corrector when do_implicit) and
   returns the updated primitives in prims. Returns 0, or -1 after a failed inversion (reported, endrun requested) */
KOKKOS_INLINE_FUNCTION int gr_add_source_term(int i, struct gr_primitives *prims, struct gr_conservatives *cons, double dtime, int do_implicit, struct particle_data *pp, struct gas_cell_data *cell)
{
#ifdef BOX_BND_PARTICLES
    if(pp[i].ID == 0) {return 0;} /* no GR gravity for frozen boundary particles */
#endif
    const struct gr_metric *gr = &(cell[i].gr);
    struct gr_conservatives Src;
    gr_source_term(i, prims, &Src, pp, cell);
#ifdef GR_DRIFT_PATH_SOURCE
    Src.tau -= gr_drift_tau_path_rate(i, cons, prims, pp, cell); /* the part of the energy source the drift accounts for */
#endif
    /* predictor */
    cons->S += dtime*Src.S;
    cons->tau += dtime*Src.tau;
    double res_con2prim = gr_get_primitive_variables(cons, prims, gr, gr->limiter);
    if(res_con2prim < 0)
    {
        printf("GR: failed inversion (code %g) adding the sources: ID %llu at %g %g %g\n", res_con2prim, (unsigned long long) pp[i].ID, pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2]);
        endrun(9993000); return -1;
    }
    if(do_implicit)
    {
        /* corrector: re-evaluate the sources with the updated primitives, iterated to convergence of the velocity */
        struct gr_conservatives Src_corr = Src;
        Vec3<double> p_err;
        double error = MAX_REAL_NUMBER; int iter = 0;
        do
        {
            p_err = cons->S/cons->D;
            Src = Src_corr;
            gr_source_term(i, prims, &Src_corr, pp, cell);
#ifdef GR_DRIFT_PATH_SOURCE
            Src_corr.tau -= gr_drift_tau_path_rate(i, cons, prims, pp, cell);
#endif
            cons->S += dtime*(Src_corr.S - Src.S);
            cons->tau += dtime*(Src_corr.tau - Src.tau);
            res_con2prim = gr_get_primitive_variables(cons, prims, gr, gr->limiter);
            if(res_con2prim < 0)
            {
                printf("GR: failed inversion (code %g) in the source corrector: ID %llu at %g %g %g\n", res_con2prim, (unsigned long long) pp[i].ID, pp[i].Pos[0], pp[i].Pos[1], pp[i].Pos[2]);
                endrun(9993001); return -1;
            }
            p_err = cons->S/cons->D - p_err;
            error = fmax(fmax(p_err[0]*p_err[0], p_err[1]*p_err[1]), p_err[2]*p_err[2]);
            error = sqrt(cons->D*cons->D*error/cons->S.norm_sq());
            iter++;
            if(iter > GR_IMPLICIT_MAX_NITER)
            {
                printf("GR WARNING: %d iterations adding the sources: S/D %2.15g %2.15g %2.15g rho0 %g u %g error %g\n", GR_IMPLICIT_MAX_NITER, cons->S[0]/cons->D, cons->S[1]/cons->D, cons->S[2]/cons->D, prims->rho0, prims->u, error);
                break;
            }
        } while(error > GR_IMPLICIT_TOLERANCE);
    }
    return 0;
}

#endif /* GENERAL_RELATIVITY */
#endif /* GR_SOURCE_FUNCTIONS_H */
