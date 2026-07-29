"""Analyze HSE Earth-smoke run: check that the body sits in equilibrium.

For each snapshot, compute:
  - mean and rms |v| / v_dyn  (should stay << 1)
  - radial density profile vs IC
  - center-of-mass drift
  - max excursion of any particle from its IC radius (in units of R_body)

t_dyn(R_surface) ~ sqrt(R^3/GM) ~ 810 s for Earth-mass body.
"""

import os, glob, numpy as np
import h5py

G_CGS = 6.674e-8


def read_snap(path):
    with h5py.File(path, "r") as f:
        x = f["PartType0/Coordinates"][:]
        v = f["PartType0/Velocities"][:]
        m = f["PartType0/Masses"][:]
        u = f["PartType0/InternalEnergy"][:]
        ids = f["PartType0/ParticleIDs"][:]
        t = f["Header"].attrs["Time"]
    # GIZMO domain decomposition reorders particles between snapshots — sort by ID
    # so per-particle comparisons (dr, dv) are meaningful.
    order = np.argsort(ids)
    return {"x": x[order], "v": v[order], "m": m[order], "u": u[order],
            "ids": ids[order], "t": float(t)}


def main():
    here = os.path.dirname(__file__) or "."
    snaps = sorted(glob.glob(os.path.join(here, "output", "snapshot_*.hdf5")))
    if not snaps:
        print("no snapshots found"); return

    ic = read_snap(snaps[0])
    M_total = ic["m"].sum()
    com0 = (ic["m"][:, None] * ic["x"]).sum(axis=0) / M_total
    r0 = np.linalg.norm(ic["x"] - com0, axis=1)
    R_body = r0.max()
    t_dyn = np.sqrt(R_body ** 3 / (G_CGS * M_total))
    v_dyn = np.sqrt(G_CGS * M_total / R_body)
    print(f"IC: M={M_total:.3e} g  R={R_body:.3e} cm  "
          f"t_dyn={t_dyn:.3e} s  v_dyn={v_dyn:.3e} cm/s")
    print(f"# {'t':>10s} {'t/t_dyn':>8s} {'|com|/R':>10s} {'<|v|>/v_dyn':>12s} "
          f"{'rms|v|/v_dyn':>13s} {'max(dr)/R':>10s}  {'rho_c/rho_c0':>12s}")

    # Reference central density: mean density of innermost 5% by IC radius.
    inner_mask = r0 < 0.2 * R_body
    rho_c0 = (M_total * inner_mask.sum() / len(r0)) / (4.0 / 3.0 * np.pi * (0.2 * R_body) ** 3)

    for snap in snaps:
        s = read_snap(snap)
        com = (s["m"][:, None] * s["x"]).sum(axis=0) / M_total
        com_drift = np.linalg.norm(com - com0) / R_body
        r = np.linalg.norm(s["x"] - com, axis=1)
        v = np.linalg.norm(s["v"], axis=1)
        v_mean = v.mean(); v_rms = np.sqrt((v ** 2).mean())
        dr = np.abs(r - r0)
        # Effective central density now (using same inner_mask of IC).
        rho_c = (s["m"][inner_mask].sum()) / (4.0 / 3.0 * np.pi
                * np.maximum(r[inner_mask].max(), 1e-30) ** 3)
        print(f"  {s['t']:10.3e} {s['t']/t_dyn:8.3f} {com_drift:10.3e} "
              f"{v_mean/v_dyn:12.3e} {v_rms/v_dyn:13.3e} "
              f"{dr.max()/R_body:10.3e}  {rho_c/rho_c0:12.4f}")

    # Pass criterion: rms |v| / v_dyn < 0.1 in last snapshot, max dr / R < 0.05.
    last = read_snap(snaps[-1])
    v_last = np.linalg.norm(last["v"], axis=1)
    com_last = (last["m"][:, None] * last["x"]).sum(axis=0) / M_total
    r_last = np.linalg.norm(last["x"] - com_last, axis=1)
    v_rms_last = np.sqrt((v_last ** 2).mean()) / v_dyn
    dr_last = np.abs(r_last - r0).max() / R_body
    print()
    print(f"FINAL: rms|v|/v_dyn = {v_rms_last:.3e}  max(dr)/R = {dr_last:.3e}")
    if v_rms_last < 0.1 and dr_last < 0.05:
        print("PASS: body remains in HSE")
    elif v_rms_last < 0.3 and dr_last < 0.15:
        print("MARGINAL: some drift but no catastrophic failure")
    else:
        print("FAIL: large drift / velocities")


if __name__ == "__main__":
    main()
