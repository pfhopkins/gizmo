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
/* rate of change of tau along the drift of cell i at fixed (D, S_j, u), (d tau/d x^k) w^k with w the coordinate velocity the cell is
   displaced with: the fluid velocity of the state for MFM, the velocity of the mesh-generating point (ParticleVel) for MFV. The Valencia
   energy source is the total rate along the path, of which this part is produced by the re-inversion of the drifted state, so it is
   removed from the source applied in the kick. Only where the drift re-inverts the moved cells (GR_DRIFT_PATH_SOURCE): elsewhere the kick
   applies the whole source. dg are the metric derivatives at the cell (gr_source_term).
   At fixed (D, S_j, u), tau = D [h W - (Gamma-1) u/W - 1] depends on the position only through gamma^ij, in W^2 = 1 + gamma^ij s_i s_j/h^2
   (s = S/D, h = 1 + Gamma u), so (d tau/d x^k) w^k = -D (h + (Gamma-1) u/W^2)/(2 h^2 W) w^k (d_k gamma_ab) s^a s^b with s^a = gamma^ab s_b.
   A cell at the limiter stays at it, with constant tau, where the motion would raise W, and leaves it at this rate where the motion lowers W */
KOKKOS_INLINE_FUNCTION double gr_drift_tau_path_rate(int i, const struct gr_conservatives *cons, const struct gr_primitives *prims, const struct gr_sym4 dg[3], struct gas_cell_data *cell)
{
#ifdef HYDRO_MESHLESS_FINITE_VOLUME
    const Vec3<double> vc = cell[i].ParticleVel; /* MFV: the cell is a mesh point, displaced with its own velocity */
#else
    const Vec3<double> vc = Vcoord(prims->v, &(cell[i].gr));
#endif
    if(!(cons->D > 0) || (vc.norm_sq() == 0)) {return 0;}
    const struct gr_metric *gr = &(cell[i].gr);
    const double gamma_eos = GAMMA_DEFAULT, u = prims->u, h = 1. + gamma_eos*u; /* GR: ideal gas with a constant adiabatic index */
    const Vec3<double> s = cons->S/cons->D, s_up = gr->ugamma.matvec(s);
    const double W2 = 1. + dot(s_up, s)/(h*h);
    double dgamma_ss = 0; /* w^k (d_k gamma_ab) s^a s^b */
    for(int k = 0; k < 3; k++) {dgamma_ss += vc[k]*dg[k].ij.quad(s_up);}
    const double W = sqrt(W2), rate = -cons->D*(h + (gamma_eos - 1.)*u/W2)/(2.*h*h*W)*dgamma_ss;
    /* (the margin: W2 of a limited state is the cap W_cap^2 only to rounding errors, which 1 - 1/W_cap^2 amplifies by W_cap^2) */
    const double W2_cap = (double)GR_GAMMA_LIMITER*(double)GR_GAMMA_LIMITER;
    if(gr->limiter && (rate > 0) && (W2*(1. + 1e-14*W2_cap) >= W2_cap)) {return 0;}
    return rate;
}
#endif

/* metric source terms of the momentum and energy equations at cell i, from the stress-energy tensor of the primitives
   (v may be limited in place where required): S_j: (alpha sqrt(gamma)/2) T^{mu nu} d_j g_{mu nu};
   tau: alpha sqrt(gamma) [T^{mu nu} K_ij-terms - (T^00 beta^j + T^0j) d_j alpha], with K_ij = -alpha Gamma^0_ij; also returns the metric
   derivatives at the cell in dg */
KOKKOS_INLINE_FUNCTION void gr_source_term(int i, struct gr_primitives *prims, struct gr_conservatives *Src, struct gr_sym4 dg[3], struct particle_data *pp, struct gas_cell_data *cell)
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
    SymmetricTensor2<double> Kij; Vec3<double> d_alpha; Mat3<double> d_beta;
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
    struct gr_conservatives Src; struct gr_sym4 dg[3];
    gr_source_term(i, prims, &Src, dg, pp, cell);
#ifdef GR_DRIFT_PATH_SOURCE
    Src.tau -= gr_drift_tau_path_rate(i, cons, prims, dg, cell); /* the part of the energy source the drift accounts for */
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
            gr_source_term(i, prims, &Src_corr, dg, pp, cell);
#ifdef GR_DRIFT_PATH_SOURCE
            Src_corr.tau -= gr_drift_tau_path_rate(i, cons, prims, dg, cell);
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
