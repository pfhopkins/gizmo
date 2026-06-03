"""Shared IC-writer + analysis helpers for the CBE Vlasov-moment test
problems (cbe_two_stream, cbe_density_wave, cbe_free_slot_1d,
cbe_free_slot_3d).

These mirror the 1D Python harness at
  /Users/phopkins/Documents/work/papers/numerical_methods/cbe_cosmology/python_harness/
ported to run on GIZMO like the other pytest-bench test problems.

CBE-in-GIZMO contract (RELATIVE-FRAME storage convention; see also
`reference_cbe_method_fix_list.md` and `do_cbe_initialization` /
`do_cbe_postgravity_kernel` in sidm/cbe_integrator{.cc,_functions.h}):

  * CBE basis moments live on Type=1 particles. CBE_INTEGRATOR=N => NBASIS=N.
  * NMOMENTS (no CBE_INTEGRATOR_SECONDMOMENT) = BOX_SPATIAL_DIMENSION + 1.
      1D -> 2  : per basis [m, p_x]
      3D -> 4  : per basis [m, p_x, p_y, p_z]
    (full GIZMO ordering is [m, p_x, p_y, p_z, T_xx, T_yy, T_zz, T_xy, T_xz,
     T_yz] truncated to NMOMENTS.)
  * The IC carries an HDF5 dataset /PartType1/VlasovMoments of shape
    (N, NBASIS*NMOMENTS), basis-major: index = NMOMENTS*basis + moment,
    read straight into P[i].CBE_basis_moments[basis][moment]. Presence of
    the dataset sets CBE_Moments_LoadedFromIC_PType[1] -> the loaded values
    are TRUSTED (validated finite + sum_basis m > 0, else endrun(8889));
    absent -> GIZMO synthesizes a cold single-stream default instead.

  * RELATIVE-FRAME STORAGE CONVENTION (binding):
        basis_p_stored[α]  = m_α * (v_phys[α] − P.Vel)
        v_phys[α]          = basis_p_stored[α] / m_α + P.Vel
    i.e. per-basis momenta are stored relative to the mesh-generating-point
    velocity P.Vel. The absolute-frame bulk momentum is carried entirely by
    P.Mass*P.Vel, and Σ_α basis_p_stored = 0 by construction. The IC writer
    below enforces this: it subtracts the mass-weighted-mean bulk velocity
    from every basis velocity before forming the momentum slot.

  * do_cbe_initialization() runs a momentum closure that enforces
    Σ basis_p_stored = 0 (relative convention; no boost of the
    per-stream velocities). Velocities[] in the IC stores P.Vel = the
    mass-weighted-mean basis velocity per particle. For zero-bulk ICs
    (counter-streaming with equal mass; Velocities = 0) the relative
    convention is byte-identical to absolute storage; for boosted ICs the
    distinction is load-bearing.

Units: raw P.Vel convention (no a/H factors); these are non-cosmological
boxes so all cf_* = 1. v_stream = 1 in code velocity units throughout,
matching the harness.
"""

from __future__ import annotations
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# IC construction
# ---------------------------------------------------------------------------

def n_stress_for_dim(dim: int) -> int:
    """Number of independent symmetric-stress components: dim*(dim+1)/2."""
    return dim * (dim + 1) // 2


def n_moments_for_dim(dim: int, secondmoment: bool = False) -> int:
    """NMOMENTS with or without CBE_INTEGRATOR_SECONDMOMENT.
       Cold: dim + 1 (mass + dim momentum slots).
       SECONDMOMENT: dim + 1 + dim*(dim+1)/2.
         1D -> 3, 2D -> 6, 3D -> 10.
       Matches the GIZMO layout from sidm/cbe_integrator_functions.h
       cbe_T_idx(a,b) = 1 + dim + a (diagonal a==b)
                     OR 1 + 2*dim + off (upper triangle in lex order)."""
    return dim + 1 + (n_stress_for_dim(dim) if secondmoment else 0)


def _stress_slot(a: int, b: int, dim: int) -> int:
    """Stress-slot offset in a basis row matching cbe_T_idx(a,b). a<=b
    expected (caller symmetrizes); returns dim+1+a for diagonal,
    1+2*dim+offset for upper-triangle off-diagonal."""
    if a > b:
        a, b = b, a
    if a == b:
        return 1 + dim + a
    # off-diagonal upper triangle (a,b) with a<b: lex-order linear index
    off = a * (2 * dim - a - 1) // 2 + (b - a - 1)
    return 1 + 2 * dim + off


def basis_moment_row(mass: float, vel_rel_xyz, dim: int,
                     sigma: float = 0.0) -> np.ndarray:
    """One basis's [m, p_x(,p_y,p_z) (,T_xx,...,T_xy,...)] row in the
    storage convention. vel_rel_xyz is the RELATIVE-frame basis velocity
    (already with bulk subtracted by the caller / build_vlasov_moments).

    Cold path (sigma=0): NMOMENTS = dim+1, slots [m, p_x,...,p_dim].

    SECONDMOMENT path (sigma>0): NMOMENTS = dim+1+dim*(dim+1)/2; stress
    slots store the RELATIVE-frame raw second moment
        T_rel_ab = m * (v_rel_a * v_rel_b + sigma² * delta_ab)
    matching the
       p_rel = m * (v - V)
       T_rel = m * <(v - V) (v - V)>
    convention. Isotropic dispersion sigma per basis (the simplest warm
    seed; per-component / anisotropic stress is a future extension)."""
    secondmoment = (sigma > 0.0)
    nm = n_moments_for_dim(dim, secondmoment=secondmoment)
    row = np.zeros(nm)
    row[0] = mass
    for k in range(dim):
        row[1 + k] = mass * vel_rel_xyz[k]
    if secondmoment:
        sigma2 = sigma * sigma
        for a in range(dim):
            row[_stress_slot(a, a, dim)] = mass * (vel_rel_xyz[a] * vel_rel_xyz[a] + sigma2)
            for b in range(a + 1, dim):
                row[_stress_slot(a, b, dim)] = mass * vel_rel_xyz[a] * vel_rel_xyz[b]
    return row


def build_vlasov_moments(per_particle_bases, dim: int,
                         vel_bulk_per_particle=None,
                         sigma: float = 0.0) -> np.ndarray:
    """per_particle_bases: list (len N) of lists of (mass, vel_xyz) tuples,
    one tuple per basis (all particles must have the same NBASIS).

    vel_bulk_per_particle: optional (N,3) per-particle bulk velocity. When
    provided, each basis's momentum slot is written in the RELATIVE frame:
        basis_p_stored = m_basis * (v_basis - v_bulk_i)
    so that Σ_basis basis_p_stored = 0 by construction. When None (legacy),
    the absolute-frame momentum m*v_basis is stored — kept for testing /
    invariant checks; production tests should pass `bulk_velocity(...)`.

    sigma: optional isotropic per-basis dispersion seed (warm IC). 0 = cold
    layout (NMOMENTS=dim+1); >0 = SECONDMOMENT layout (NMOMENTS=dim+1+n_stress).
    Stress slots store T_rel_ab = m*(v_rel_a*v_rel_b + sigma²*delta_ab) per
    basis, matching the relative-frame storage convention.

    Returns (N, NBASIS*NMOMENTS) basis-major flat array."""
    N = len(per_particle_bases)
    nbasis = len(per_particle_bases[0])
    secondmoment = (sigma > 0.0)
    nm = n_moments_for_dim(dim, secondmoment=secondmoment)
    out = np.zeros((N, nbasis * nm))
    use_bulk = vel_bulk_per_particle is not None
    for i, bases in enumerate(per_particle_bases):
        assert len(bases) == nbasis, "all particles need the same NBASIS"
        vbulk_i = np.asarray(vel_bulk_per_particle[i]) if use_bulk else np.zeros(3)
        for b, (m, v) in enumerate(bases):
            v_rel = np.asarray(v, dtype=float) - vbulk_i
            out[i, nm * b:nm * (b + 1)] = basis_moment_row(m, v_rel, dim, sigma=sigma)
    return out


def bulk_velocity(per_particle_bases) -> np.ndarray:
    """Mass-weighted-mean velocity per particle (N,3). This is what the IC's
    Velocities MUST be set to so do_cbe_initialization's momentum-closure
    renorm is a no-op and the prescribed stream velocities survive."""
    N = len(per_particle_bases)
    vbulk = np.zeros((N, 3))
    for i, bases in enumerate(per_particle_bases):
        mtot = sum(m for (m, v) in bases)
        if mtot > 0:
            p = np.zeros(3)
            for (m, v) in bases:
                p += m * np.asarray(v, dtype=float)
            vbulk[i] = p / mtot
    return vbulk


def write_cbe_ic(fname, pos, per_particle_bases, dim, box_size,
                 box_long=(1.0, 1.0, 1.0), sigma: float = 0.0):
    """Write a CBE test IC.

    pos: (N,3) coordinates (for 1D set y=z=0; GIZMO wraps only the active
         axes). box_size: scalar BoxSize. box_long: (lx,ly,lz) BOX_LONG_X/Y/Z
         multipliers for header-consistent extents (informational here; the
         physical extent is BoxSize*box_long per axis).
    per_particle_bases: list of per-basis (mass, vel_xyz) — see
         build_vlasov_moments.
    sigma: optional isotropic per-basis dispersion seed (warm IC). 0 = cold
         layout (NMOMENTS=dim+1); >0 = SECONDMOMENT layout with isotropic
         relative-frame stress T_rel_aa = m*(v_rel_a^2 + sigma^2).
    """
    pos = np.asarray(pos, dtype=float)
    N = pos.shape[0]
    masses = np.array([sum(m for (m, v) in bases) for bases in per_particle_bases])
    vel = bulk_velocity(per_particle_bases)            # (N,3) mass-weighted mean
    # RELATIVE-FRAME STORAGE: pass per-particle bulk so each basis momentum
    # slot is m_basis*(v_basis - v_bulk) — Σ_basis basis_p_stored = 0.
    vmoments = build_vlasov_moments(per_particle_bases, dim,
                                    vel_bulk_per_particle=vel,
                                    sigma=sigma)
    ids = np.arange(1, N + 1, dtype=np.uint32)

    with h5py.File(fname, "w") as F:
        h = F.create_group("Header")
        h.attrs["NumPart_ThisFile"] = [0, N, 0, 0, 0, 0]
        h.attrs["NumPart_Total"]    = [0, N, 0, 0, 0, 0]
        h.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        h.attrs["MassTable"]        = [0.0, 0.0, 0, 0, 0, 0]   # per-particle masses given explicitly
        h.attrs["BoxSize"]          = box_size
        h.attrs["Time"]             = 0.0
        h.attrs["Redshift"]         = 0.0
        h.attrs["NumFilesPerSnapshot"] = 1
        h.attrs["Omega0"]           = 0.0
        h.attrs["OmegaLambda"]      = 0.0
        h.attrs["HubbleParam"]      = 1.0
        h.attrs["Flag_Sfr"]         = 0
        h.attrs["Flag_Cooling"]     = 0
        h.attrs["Flag_StellarAge"]  = 0
        h.attrs["Flag_Metals"]      = 0
        h.attrs["Flag_Feedback"]    = 0
        h.attrs["Flag_DoublePrecision"] = 1

        g = F.create_group("PartType1")
        g.create_dataset("Coordinates",   data=pos.astype(np.float64))
        g.create_dataset("Velocities",    data=vel.astype(np.float64))
        g.create_dataset("Masses",        data=masses.astype(np.float64))
        g.create_dataset("ParticleIDs",   data=ids)
        g.create_dataset("VlasovMoments", data=vmoments.astype(np.float64))

    nbasis = len(per_particle_bases[0])
    secondmoment = (sigma > 0.0)
    nm = n_moments_for_dim(dim, secondmoment=secondmoment)
    print(f"Wrote {fname}: N={N} Type=1, dim={dim}, NBASIS={nbasis}, "
          f"NMOMENTS={nm}{' (SECONDMOMENT, sigma=%g)' % sigma if secondmoment else ''}, "
          f"VlasovMoments shape ({N},{nbasis*nm}), "
          f"BoxSize={box_size}, box_long={box_long}, Mtot={masses.sum():.6g}")
    return fname


# ---------------------------------------------------------------------------
# Analysis (reads a GIZMO snapshot's VlasovMoments back out)
# ---------------------------------------------------------------------------

def read_snapshot(path, dim):
    """Return dict with x, vel(bulk), mass, ids, and per-basis (m, v_xyz)
    decoded from VlasovMoments. Sorted by ParticleID."""
    with h5py.File(path, "r") as f:
        p = f["PartType1"]
        x   = p["Coordinates"][:]
        vel = p["Velocities"][:]
        m   = p["Masses"][:]
        ids = p["ParticleIDs"][:]
        vm  = p["VlasovMoments"][:] if "VlasovMoments" in p else None
        t   = float(f["Header"].attrs.get("Time", -1.0))
    order = np.argsort(ids)
    x, vel, m, ids = x[order], vel[order], m[order], ids[order]
    nm = n_moments_for_dim(dim)
    bases = None
    if vm is not None:
        vm = vm[order]
        N = vm.shape[0]
        nbasis = vm.shape[1] // nm
        vm = vm.reshape(N, nbasis, nm)
        bm = vm[:, :, 0]                                   # (N, nbasis) basis mass
        bv = np.zeros((N, nbasis, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            for k in range(dim):
                bv[:, :, k] = np.where(bm > 0, vm[:, :, 1 + k] / bm, 0.0)
        bases = {"m": bm, "v": bv, "nbasis": nbasis}
    return {"x": x, "vel": vel, "mass": m, "ids": ids, "t": t, "bases": bases}


def conservation(snap):
    """Total mass + total momentum (summed over particles AND bases via the
    bulk fields, which equal the basis sums by the IC closure)."""
    M = float(snap["mass"].sum())
    P = (snap["mass"][:, None] * snap["vel"]).sum(axis=0)
    return {"M": M, "P": P}


def per_stream_mass_density(snap, v_target, dim, tol=0.3):
    """Per-particle basis-mass that sits in the velocity band |v_x - v_target|
    < tol, divided by an estimate of the particle volume (~ mass/<rho>); here
    returned as raw band-mass per particle (the harness plots band density;
    for GIZMO the band-mass-per-particle profile is the directly comparable
    quantity). Returns (x_sorted, band_mass)."""
    b = snap["bases"]
    if b is None:
        return snap["x"][:, 0], np.zeros(snap["x"].shape[0])
    vx = b["v"][:, :, 0]
    inband = np.abs(vx - v_target) < tol
    band_mass = (b["m"] * inband).sum(axis=1)
    return snap["x"][:, 0], band_mass


def total_band_mass(snap, v_target, tol=0.3):
    """Total mass (summed over all particles + bases) in velocity band
    |v_x - v_target| < tol. Used for free-slot per-stream-band comparisons."""
    b = snap["bases"]
    if b is None:
        return 0.0
    vx = b["v"][:, :, 0]
    return float((b["m"] * (np.abs(vx - v_target) < tol)).sum())


def read_cbe_diagnostics(outputdir):
    """Parse output/cbe_diagnostics.txt. Returns list of dict rows with the
    documented column semantics (WITHGRADIENTS-off 9-col schema; these test
    problems do not use WITHGRADIENTS). Columns:
      1 time, 2 face_residual_max, 3 face_residual_sum, 4 bracket_fail,
      5 rho_clamp, 6 S_clamp, 7 repair_dP, 8 repair_dT, 9 free_slot_count."""
    import os
    path = os.path.join(outputdir, "cbe_diagnostics.txt")
    cols = ["time", "face_res_max", "face_res_sum", "bracket_fail",
            "rho_clamp", "S_clamp", "repair_dP", "repair_dT", "free_slot"]
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 9:
                continue
            rows.append({c: float(vals[i]) for i, c in enumerate(cols)})
    return rows


def plot_cbe_streams(test_name, snap_initial, snap_final, dim, v_bands,
                     output_dir=None):
    """Per-stream band-mass profiles vs x: IC overlaid with final snapshot.
    One panel per velocity band. Saves <test_name>_streams.png. Mirrors the
    harness per-stream-density figures."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if output_dir is None:
        output_dir = f"test/{test_name}"
    nb = len(v_bands)
    ncols = min(3, nb)
    nrows = (nb + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                            squeeze=False)
    for i, (v, lab) in enumerate(v_bands):
        ax = axs[i // ncols, i % ncols]
        xi, bi = per_stream_mass_density(snap_initial, v, dim)
        xf, bf = per_stream_mass_density(snap_final, v, dim)
        oi = np.argsort(xi); of = np.argsort(xf)
        ax.plot(xi[oi], bi[oi], "-",  color="C0", lw=0.8, label="IC")
        ax.plot(xf[of], bf[of], "o", color="C3", ms=3,
                label=f"t={snap_final['t']:.4g}")
        ax.set_xlabel("x"); ax.set_ylabel("band mass / particle")
        ax.set_title(f"stream v={lab}"); ax.legend(fontsize=8)
    for j in range(nb, nrows * ncols):
        axs[j // ncols, j % ncols].axis("off")
    fig.suptitle(f"{test_name}: per-stream band mass (IC vs final)")
    fig.tight_layout()
    out = os.path.join(output_dir, f"{test_name}_streams.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out
