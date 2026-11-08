#!/usr/bin/env python3
"""Equation-of-state state census of a GIZMO run's snapshots.

Every tabulated (ANEOS / SESAME) cell of every snapshot is classified with GIZMO's own lookup and inversion (libcensus, built by
build.sh from the tree's src/eos/aneos.cc): density outside the table, energy at the validity floor or the table top, proximity to
each (bands fixed below), fold ranges (energies a density interval's u(T) reaches more than once), several roots, plateaus.

It samples the states at OUTPUT times, not the paths between them: the run's eos_limits.txt is the record of what happened between
snapshots, and is printed beside the census for comparison (no implication between the two: a cell can sit at the floor without a
floor event in that interval, and events between snapshots leave no snapshot state).

Every cell gets the whole-column root scan (~1 s per 1e6 cells, single thread). Build the library first: ./build.sh (once per GIZMO
tree; it compiles that tree's src/eos/aneos.cc). Self-test: known_positives.py <workdir> --tables <dir with the EOS tables>.

    census.py <output_dir> [--run-dir DIR] [--families families.json] [--chunk N] [--out PREFIX]
              [--expect-sha SLOT=SHA256 ...] [--eos-limits FILE] [--no-figures]
"""
import argparse, ctypes, glob, hashlib, json, os, re, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))

FLAGS = ["RHO_BELOW", "RHO_ABOVE", "U_AT_FLOOR", "U_AT_TOP", "NEAR_FLOOR", "NEAR_TOP", "NEAR_RHO_LOW", "NEAR_RHO_HIGH",
         "FOLD_IN", "FOLD_NEAR", "ROOTS_MULTI", "PLATEAU", "FOLD_GATE_MISS", "SCANNED", "FLOOR_IS_BLOCK"]
BIT = {name: 1 << k for k, name in enumerate(FLAGS)}
REPORTED = [f for f in FLAGS if f not in ("SCANNED", "FLOOR_IS_BLOCK")]   # the classification; the other two qualify it
# blockers: any of these at a snapshot fails settling acceptance
BLOCKERS = ["RHO_BELOW", "RHO_ABOVE", "U_AT_FLOOR", "U_AT_TOP", "ROOTS_MULTI", "FOLD_GATE_MISS"]
BANDS = ("NEAR_FLOOR: u < 1.25 u_floor or the interpolation cell holds the floor row; NEAR_TOP: u > 0.8 u_top; "
         "NEAR_RHO_LOW: within log10(1/0.9) = 0.04576 dex of the lowest density; NEAR_RHO_HIGH: within log10(1.1) = 0.04139 dex of the "
         "highest; FOLD_NEAR: within log10(1.25) = 0.0969 dex of the fold range (the perturbations rho x0.9/1.1, u x0.8/1.25)")
N_TILLOTSON_MATERIALS = 7          # src/eos/composition_registry.h
PROTONMASS, BOLTZMANN = 1.67262178e-24, 1.3806485e-16
IDEAL_GAS_GUESS = (5.0 / 3.0 - 1.0) * 0.59 * PROTONMASS / BOLTZMANN   # T(u) seeding the inversion when a snapshot has no Temperature


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


class Lib:
    def __init__(self, path=os.path.join(HERE, "libcensus.so")):
        if not os.path.exists(path):
            sys.exit(f"{path} missing: run build.sh first")
        self.L = ctypes.CDLL(path)
        self.build = open(os.path.join(HERE, "libcensus.build.txt")).read().strip()
        d, i, l = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_long
        u8 = ctypes.POINTER(ctypes.c_ubyte)
        self.L.census_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.L.census_table_info.argtypes = [ctypes.c_int, d]
        self.L.census_rhoT.argtypes = [ctypes.c_int, l, d, d, d, d]
        self.L.census_energy_bounds.argtypes = [ctypes.c_int, l, d, d, d]
        self.L.census_classify.argtypes = [ctypes.c_int, l, d, d, d, u8] + [i, i] + [d] * 9 + [i, d, d]

    @staticmethod
    def _p(a, t=ctypes.c_double):
        return a.ctypes.data_as(ctypes.POINTER(t))

    def load(self, path, slot):
        if self.L.census_load(path.encode(), slot) != 0:
            sys.exit(f"table {path} failed to load into slot {slot}")
        out = np.zeros(8)
        self.L.census_table_info(slot, self._p(out))
        return dict(nrho=int(out[0]), nT=int(out[1]), logrho=(out[2], out[3]), logT=(out[4], out[5]), floor_block=bool(out[6]),
                    fold_intervals=int(out[7]))

    def rhoT(self, slot, rho, T):
        rho, T = np.ascontiguousarray(rho, float), np.ascontiguousarray(T, float)
        u, P = np.zeros_like(rho), np.zeros_like(rho)
        self.L.census_rhoT(slot, rho.size, self._p(rho), self._p(T), self._p(u), self._p(P))
        return u, P

    def energy_bounds(self, slot, rho):
        rho = np.ascontiguousarray(rho, float)
        lo, hi = np.zeros_like(rho), np.zeros_like(rho)
        self.L.census_energy_bounds(slot, rho.size, self._p(rho), self._p(lo), self._p(hi))
        return lo, hi

    def classify(self, slot, rho, u, T_guess, scan_all):
        rho, u, Tg = (np.ascontiguousarray(a, float) for a in (rho, u, T_guess))
        scan = np.ascontiguousarray(scan_all, np.uint8)
        n = rho.size
        r = {k: np.zeros(n, np.int32) for k in ("flags", "runtime_flags", "n_roots")}
        for k in ("T", "P", "floor_rows", "dlnT_floor", "u_floor_ratio", "u_top_ratio", "fold_margin", "dex_lo", "dex_hi",
                  "plateau_width", "plateau_spread"):
            r[k] = np.zeros(n)
        ip = lambda a: self._p(a, ctypes.c_int)
        self.L.census_classify(slot, n, self._p(rho), self._p(u), self._p(Tg), self._p(scan, ctypes.c_ubyte),
                               ip(r["flags"]), ip(r["runtime_flags"]), self._p(r["T"]), self._p(r["P"]), self._p(r["floor_rows"]),
                               self._p(r["dlnT_floor"]), self._p(r["u_floor_ratio"]), self._p(r["u_top_ratio"]),
                               self._p(r["fold_margin"]), self._p(r["dex_lo"]), self._p(r["dex_hi"]), ip(r["n_roots"]),
                               self._p(r["plateau_width"]), self._p(r["plateau_spread"]))
        return r


def slot_mapping(effective_options):
    """CompositionType -> (branch, table slot), exactly as eos_branch_of / aneos_subindex in the build that wrote the snapshot"""
    opts = set(o.split("=")[0] for o in effective_options.split())
    till, aneos, dgas = "EOS_TILLOTSON" in opts, "EOS_ANEOS" in opts, "EOS_TYPES_DEFAULTGAS_AND_SOLIDS" in opts
    if not aneos:
        sys.exit("the build has no EOS_ANEOS: nothing tabulated to census")
    def mapping(ct):
        if dgas and ct == 0:
            return "gas", -1
        if till and ct < N_TILLOTSON_MATERIALS:
            return "tillotson", -1
        return "table", ct - (N_TILLOTSON_MATERIALS if till else (1 if dgas else 0))
    return mapping, ("EOS_ELASTIC" in opts)


def find_snapshots(outdir):
    snaps = {}
    for p in glob.glob(os.path.join(outdir, "snapshot_*.hdf5")):
        m = re.match(r"snapshot_(\d+)\.hdf5$", os.path.basename(p))
        if m:
            snaps[int(m.group(1))] = [p]
    for d in glob.glob(os.path.join(outdir, "snapdir_*")):
        files = sorted(glob.glob(os.path.join(d, "snapshot_*.*.hdf5")), key=lambda q: int(q.split(".")[-2]))
        if files:
            snaps[int(re.search(r"snapdir_(\d+)", d).group(1))] = files
    if not snaps:
        sys.exit(f"no snapshots in {outdir}")
    return [(k, snaps[k]) for k in sorted(snaps)]


def read_usedvalues(path):
    vals = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0]] = parts[1]
    return vals


def cell_keys(ids, child, gen, split_keys):
    """one key per cell: the ID, or, when the run writes the split datasets (split_keys, decided once per run so that every chunk and
    snapshot has the same key type), (ID, child number, generation)"""
    if not split_keys:
        return ids.astype(np.uint64)
    rec = np.empty(ids.size, dtype=[("i", "<u8"), ("c", "<u8"), ("g", "<u8")])
    rec["i"], rec["c"], rec["g"] = ids, child, (gen if gen is not None else 0)
    return rec.view("V24").ravel()


def parse_eos_limits(path):
    """per line: time, startup (0) / run (1), CompositionType, and the interval event columns the census can be set beside"""
    if not path or not os.path.exists(path):
        return None
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        rows.append(dict(time=float(v[0]), run=int(v[2]), ct=int(v[3]), updates=float(v[5]), rho_below=float(v[6]),
                         rho_above=float(v[7]), u_floor=float(v[8]), u_top=float(v[9]), plateau=float(v[10]), recovered=float(v[11]),
                         kick_floor=float(v[13]), drift_floor=float(v[16])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir")
    ap.add_argument("--run-dir", help="directory the run started in (table paths are relative to it); default: the output dir's parent")
    ap.add_argument("--families", help="JSON {family name: [CompositionType, ...]}")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--out", help="output prefix (default <output_dir>/eos_census)")
    ap.add_argument("--expect-sha", action="append", default=[], help="SLOT=SHA256: refuse if that table's hash differs")
    ap.add_argument("--eos-limits", help="default <output_dir>/eos_limits.txt")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()

    outdir = os.path.abspath(a.output_dir)
    rundir = os.path.abspath(a.run_dir or os.path.dirname(outdir))
    prefix = a.out or os.path.join(outdir, "eos_census")
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    snaps = find_snapshots(outdir)
    lib = Lib()

    with h5py.File(snaps[0][1][0], "r") as f:
        H = dict(f["Header"].attrs)
    if int(H.get("ComovingIntegrationOn", 0)):
        sys.exit("comoving run: refused (the census assumes physical units)")
    eff = H["Compile_Effective_Options"].decode() if isinstance(H["Compile_Effective_Options"], bytes) else H["Compile_Effective_Options"]
    mapping, elastic = slot_mapping(eff)
    UM, UL, UV = float(H["UnitMass_In_CGS"]), float(H["UnitLength_In_CGS"]), float(H["UnitVelocity_In_CGS"])
    f_rho, f_u, f_P = UM / UL**3, UV**2, UM * UV**2 / UL**3
    double_precision = int(H.get("Flag_DoublePrecision", 0))
    with h5py.File(snaps[0][1][0], "r") as f:
        split_keys = "PartType0" in f and ("ParticleChildIDsNumber" in f["PartType0"] or "ParticleIDGenerationNumber" in f["PartType0"])

    uv = read_usedvalues(os.path.join(outdir, "parameters-usedvalues"))
    n_tables = int(uv.get("AneosNumMaterials", 0))
    expect = dict(s.split("=") for s in a.expect_sha)
    tables = {}
    for k in range(n_tables):
        path = uv[f"AneosTable{k}"]
        path = path if os.path.isabs(path) else os.path.join(rundir, path)
        sha = sha256_file(path)
        if str(k) in expect and expect[str(k)] != sha:
            sys.exit(f"table slot {k} ({path}) sha256 {sha} != expected {expect[str(k)]}: refused")
        tables[k] = dict(path=path, sha=sha, **lib.load(path, k))

    families = json.load(open(a.families)) if a.families else {}
    times, snap_sha, per_snap = [], [], []        # per_snap[i][key][flag] = [count, mass, rho_min, rho_max, T_min, T_max, u_min, u_max, P_min, P_max, mass on a floor row above the coldest row]
    totals, plateau_max, selfcheck, first = [], [], [], {}
    hist_edges = dict(u_floor_ratio=np.linspace(-1, 3, 201), fold_margin=np.linspace(-1, 2, 151), dex_lo=np.linspace(0, 2, 101),
                      dex_hi=np.linspace(0, 2, 101))
    hists = {}
    seen = {}                                     # (key, flag) -> sorted keys already flagged
    ever_mass = {}
    flagged_points = {}                           # slot -> list of (rho, T, flags) samples for the figure
    rng = np.random.default_rng(12345)
    other_mass = {}

    for isnap, (num, files) in enumerate(snaps):
        with h5py.File(files[0], "r") as f:
            times.append(float(f["Header"].attrs["Time"]))
        snap_sha.append([(os.path.basename(p), sha256_file(p)) for p in files])
        acc, tot, pmax = {}, {}, {}
        sc = dict(n=0, max_dP=0.0, sum_dP=[], max_dT=0.0, tension=0, tension_bad=0, skipped=0)
        all_keys = []
        for path in files:
            with h5py.File(path, "r") as f:
                if "PartType0" not in f:
                    continue
                g = f["PartType0"]
                need = ["Density", "InternalEnergy", "CompositionType", "Masses", "ParticleIDs"]
                missing = [n for n in need if n not in g]
                if missing:
                    sys.exit(f"{path}: missing datasets {missing}: refused")
                N = g["Density"].shape[0]
                for s in range(0, N, a.chunk):
                    sl = slice(s, min(N, s + a.chunk))
                    rho = g["Density"][sl].astype(float) * f_rho
                    u = g["InternalEnergy"][sl].astype(float) * f_u
                    ct = g["CompositionType"][sl].astype(np.int64)
                    m = g["Masses"][sl].astype(float)
                    ids = g["ParticleIDs"][sl]
                    child = g["ParticleChildIDsNumber"][sl] if "ParticleChildIDsNumber" in g else None
                    gen = g["ParticleIDGenerationNumber"][sl] if "ParticleIDGenerationNumber" in g else None
                    keys = cell_keys(ids, child, gen, split_keys)
                    all_keys.append(keys)
                    Psnap = g["Pressure"][sl].astype(float) * f_P if "Pressure" in g else None
                    Tsnap = g["Temperature"][sl].astype(float) if "Temperature" in g else None
                    for c in np.unique(ct):
                        sel = ct == c
                        branch, slot = mapping(int(c))
                        if branch == "table" and slot not in tables:
                            sys.exit(f"{path}: {int(sel.sum())} cells carry CompositionType {int(c)}, a tabulated material (table slot {slot}), but "
                                     f"the run loaded tables 0..{n_tables - 1}: refused (a corrupt snapshot or a layout mismatch)")
                        if branch != "table":
                            other_mass[(branch, int(c))] = other_mass.get((branch, int(c)), 0.0) + m[sel].sum()
                            continue
                        # the inversion is seeded as the run seeds it, from the cell's stored temperature (an ideal-gas guess if the
                        # snapshot has none); the seed decides only where on a plateau the solve lands
                        seed = Tsnap[sel] if Tsnap is not None else IDEAL_GAS_GUESS * u[sel]
                        r = lib.classify(slot, rho[sel], u[sel], seed, np.ones(sel.sum(), np.uint8))
                        fl = r["flags"]
                        groups = [("ct%d" % c)] + [fam for fam, cts in families.items() if int(c) in cts]
                        for key in groups:
                            tot.setdefault(key, [0, 0.0])
                            tot[key][0] += sel.sum(); tot[key][1] += m[sel].sum()
                            pk = pmax.setdefault(key, [0.0, 0.0])
                            pk[0] = max(pk[0], r["plateau_width"].max(initial=0)); pk[1] = max(pk[1], r["plateau_spread"].max(initial=0))
                            for name in FLAGS:
                                hit = (fl & BIT[name]) != 0
                                if not hit.any():
                                    continue
                                e = acc.setdefault(key, {}).setdefault(name, [0, 0.0, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, 0.0])
                                rr, TT, uu, PP = rho[sel][hit], r["T"][hit], u[sel][hit], r["P"][hit]
                                e[0] += hit.sum(); e[1] += m[sel][hit].sum()
                                for j, q in enumerate((rr, TT, uu, PP)):
                                    e[2 + 2 * j] = min(e[2 + 2 * j], q.min()); e[3 + 2 * j] = max(e[3 + 2 * j], q.max())
                                if name == "U_AT_FLOOR":
                                    blk = hit & ((fl & BIT["FLOOR_IS_BLOCK"]) != 0)
                                    e[10] += m[sel][blk].sum()
                                if name in REPORTED:
                                    fk = (key, name)
                                    kh = keys[sel][hit]
                                    if fk not in first:
                                        first[fk] = (times[-1], num, int(ids[sel][hit][0]))
                                    prev = seen.get(fk, np.array([], dtype=kh.dtype))
                                    newmask = ~np.isin(kh, prev)
                                    ever_mass[fk] = ever_mass.get(fk, 0.0) + m[sel][hit][newmask].sum()
                                    seen[fk] = np.union1d(prev, kh[newmask])
                            if key == "ct%d" % c:
                                for hn, edges in hist_edges.items():
                                    q = np.log10(r[hn]) if hn == "u_floor_ratio" else r[hn]
                                    h = hists.setdefault((isnap, key, hn), np.zeros(edges.size - 1))
                                    h += np.histogram(q, bins=edges, weights=m[sel])[0]
                        # the instrument must see: recomputed P and T vs the run's own, where the state determines them
                        ok = (fl & (BIT["PLATEAU"] | BIT["ROOTS_MULTI"])) == 0
                        sc["skipped"] += int((~ok).sum())
                        if Psnap is not None:
                            Ps = Psnap[sel]
                            pos = ok & (Ps >= 0)
                            if pos.any():
                                d = np.abs(Ps[pos] / r["P"][pos] - 1)
                                sc["n"] += int(pos.sum()); sc["max_dP"] = max(sc["max_dP"], d.max()); sc["sum_dP"].append(d)
                            neg = ok & (Ps < 0)
                            sc["tension"] += int(neg.sum())
                            sc["tension_bad"] += int((neg & (np.abs(Ps) > r["P"] * (1 + 1e-6))).sum())
                        if Tsnap is not None and ok.any():
                            sc["max_dT"] = max(sc["max_dT"], np.abs(Tsnap[sel][ok] / r["T"][ok] - 1).max())
                        flagged = (fl & sum(BIT[n] for n in REPORTED)) != 0
                        pts = flagged_points.setdefault(slot, [])
                        idx = np.flatnonzero(flagged)
                        if idx.size > 4000:
                            idx = rng.choice(idx, 4000, replace=False)
                        pts.append((rho[sel][idx], r["T"][idx], fl[idx]))
                        if isnap == len(snaps) - 1:
                            ib = rng.choice(sel.sum(), min(4000, sel.sum()), replace=False)
                            flagged_points.setdefault(("all", slot), []).append((rho[sel][ib], r["T"][ib]))
        allk = np.concatenate(all_keys) if all_keys else np.array([])
        if np.unique(allk).size != allk.size:
            sys.exit(f"snapshot {num}: cell keys not unique ({allk.size - np.unique(allk).size} duplicates): refused")
        per_snap.append(acc); totals.append(tot); plateau_max.append(pmax); selfcheck.append(sc)

    # time weights: trapezoid over the snapshot times (each snapshot stands for half of each neighbouring interval)
    t = np.array(times)
    w = np.ones(t.size) if t.size == 1 or t[-1] == t[0] else np.array([(t[min(i + 1, t.size - 1)] - t[max(i - 1, 0)]) / 2 for i in range(t.size)])
    keys = sorted(set(k for tt in totals for k in tt))

    # CSV: one row per snapshot x group x flag
    with open(prefix + ".csv", "w") as f:
        f.write("snapshot,time,weight,group,flag,count,mass,mass_fraction,rho_min,rho_max,T_min,T_max,u_min,u_max,P_min,P_max\n")
        for i, (num, _) in enumerate(snaps):
            for key in keys:
                M = totals[i].get(key, [0, 0.0])[1]
                for name in FLAGS:
                    e = per_snap[i].get(key, {}).get(name)
                    if e is None:
                        continue
                    f.write(f"{num},{t[i]:.9g},{w[i]:.9g},{key},{name},{e[0]},{e[1]:.9g},{e[1] / M if M else 0:.9g}," +
                            ",".join(f"{x:.6g}" for x in e[2:10]) + "\n")

    limits = parse_eos_limits(a.eos_limits or os.path.join(outdir, "eos_limits.txt"))
    L = []
    L.append("# EOS state census -- samples snapshot states at output times, not the paths between them; eos_limits.txt is the runtime record")
    L.append("# tool: " + lib.build.replace("\n", "; ") + f"; census.py sha256 {sha256_file(__file__)}")
    L.append(f"# run: {outdir}; build options hash {H.get('Compile_Effective_Options_FNV1a64')}; elastic={elastic}; output precision "
             f"{'double' if double_precision else 'single'}; every cell column-scanned")
    for k, tb in tables.items():
        L.append(f"# table slot {k}: {tb['path']} sha256 {tb['sha']}; {tb['nrho']} x {tb['nT']}, log rho {tb['logrho'][0]:.4g}..{tb['logrho'][1]:.4g},"
                 f" log T {tb['logT'][0]:.4g}..{tb['logT'][1]:.4g}, floor block {tb['floor_block']}, fold intervals {tb['fold_intervals']}")
    for i, (num, _) in enumerate(snaps):
        L.append(f"# snapshot {num}: time {t[i]:.6g}, weight {w[i]:.6g}, " + ", ".join(f"{n} sha256 {s}" for n, s in snap_sha[i]))
    L.append("# bands (fixed before any production result): " + BANDS)
    L.append("# ever-flagged = distinct cells (ID, or ID + child number + generation after splits) with the mass each had when FIRST flagged: "
             "not a conserved union mass")
    for (branch, c), mm in sorted(other_mass.items()):
        L.append(f"# not tabulated: CompositionType {c} ({branch}) total mass over snapshots {mm:.6g}")
    L.append("")
    for key in keys:
        L.append(f"== {key}")
        for name in REPORTED:
            fr = np.array([per_snap[i].get(key, {}).get(name, [0, 0.0])[1] / max(totals[i].get(key, [0, 1.0])[1], 1e-300) for i in range(t.size)])
            if not fr.any():
                L.append(f"  {name:15s} 0")
                continue
            e_all = [per_snap[i][key][name] for i in range(t.size) if name in per_snap[i].get(key, {})]
            rng_txt = " ".join(f"{lab} {min(e[2 + 2 * j] for e in e_all):.4g}..{max(e[3 + 2 * j] for e in e_all):.4g}"
                               for j, lab in enumerate(("rho", "T", "u", "P")))
            ft = first[(key, name)]
            if name == "U_AT_FLOOR":
                fb = sum(e[10] for e in e_all) / max(sum(e[1] for e in e_all), 1e-300)
                rng_txt += f"; on a table floor row above the coldest row: {fb:.3g} of this mass"
            L.append(f"  {name:15s} max {fr.max():.4g} (snapshot {snaps[int(fr.argmax())][0]}), time-weighted {np.sum(w * fr) / np.sum(w):.4g}, "
                     f"first t {ft[0]:.6g} (snapshot {ft[1]}, ID {ft[2]}), ever-flagged {seen[(key, name)].size} cells / mass {ever_mass[(key, name)]:.4g} "
                     f"[{rng_txt}]")
        pw = max(p.get(key, [0, 0])[0] for p in plateau_max); ps = max(p.get(key, [0, 0])[1] for p in plateau_max)
        L.append(f"  plateau widest {pw:.4g} dex in T, largest P spread {ps:.4g}")
        blk = [n for n in BLOCKERS if any(per_snap[i].get(key, {}).get(n) for i in range(t.size))]
        L.append(f"  SETTLING VERDICT (blockers at any snapshot: {', '.join(BLOCKERS)}): {'FAIL: ' + ', '.join(blk) if blk else 'clean'}")
    L.append("")
    L.append("== self-check: recomputed vs the run's own P / T (cells whose state determines them; plateau / several-root cells skipped)")
    for i, (num, _) in enumerate(snaps):
        sc = selfcheck[i]
        med = np.median(np.concatenate(sc["sum_dP"])) if sc["sum_dP"] else float("nan")
        L.append(f"  snapshot {num}: P compared {sc['n']} cells, median |dP/P| {med:.3g}, max {sc['max_dP']:.3g}; max |dT/T| {sc['max_dT']:.3g}; "
                 f"tension cells {sc['tension']} (one-sided violations {sc['tension_bad']}); skipped {sc['skipped']}")
    if limits is not None:
        L.append("")
        L.append("== beside eos_limits.txt (events between snapshots, each line placed by its time; after a resume from an earlier restart, drop the")
        L.append("   lines past that restart's time first, as the User Guide says) -- comparison only, no implication")
        L.append("  snapshot  group   census[U_AT_FLOOR RHO_BELOW RHO_ABOVE U_AT_TOP PLATEAU]   eos_limits[u_floor rho_below rho_above u_top plateau "
                 "recovered kick_floor drift_floor updates]")
        for i, (num, _) in enumerate(snaps):
            t0 = t[i - 1] if i > 0 else -np.inf
            for key in [k for k in keys if k.startswith("ct")]:
                c = int(key[2:])
                rows = [r for r in limits if r["ct"] == c and ((i == 0 and r["time"] <= t[0]) or (i > 0 and t0 < r["time"] <= t[i]))]
                ev = [sum(r[k] for r in rows) for k in ("u_floor", "rho_below", "rho_above", "u_top", "plateau", "recovered", "kick_floor", "drift_floor", "updates")]
                cn = [int(per_snap[i].get(key, {}).get(n, [0])[0]) for n in ("U_AT_FLOOR", "RHO_BELOW", "RHO_ABOVE", "U_AT_TOP", "PLATEAU")]
                L.append(f"  {num:8d}  {key:6s}  {cn}   {[int(x) for x in ev]}")
    open(prefix + "_summary.txt", "w").write("\n".join(L) + "\n")
    np.savez(prefix + "_hist.npz", times=t, weights=w, **{f"s{i}_{k}_{h}": v for (i, k, h), v in hists.items()},
             **{f"edges_{h}": e for h, e in hist_edges.items()})
    print("\n".join(L))

    if not a.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ct_keys = [k for k in keys if k.startswith("ct")]
        fig, axs = plt.subplots(len(ct_keys), 1, figsize=(7, 2.4 * len(ct_keys)), squeeze=False, sharex=True)
        for ax, key in zip(axs[:, 0], ct_keys):
            for name in REPORTED:
                fr = np.array([per_snap[i].get(key, {}).get(name, [0, 0.0])[1] / max(totals[i].get(key, [0, 1.0])[1], 1e-300) for i in range(t.size)])
                if fr.any():
                    ax.semilogy(t, np.maximum(fr, 1e-8), "o-", ms=3, label=name)
            ax.set_ylabel(f"{key} mass fraction"); ax.set_yscale("log"); ax.set_ylim(1e-8, 1.5)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6, ncol=3)
            else:
                ax.text(0.5, 0.5, "no flagged mass at any snapshot", transform=ax.transAxes, ha="center", fontsize=8)
        axs[-1, 0].set_xlabel("time [code units]")
        fig.tight_layout(); fig.savefig(prefix + "_timeseries.png", dpi=130); plt.close(fig)
        slots = sorted(s for s in flagged_points if not isinstance(s, tuple))
        if slots:
            fig, axs = plt.subplots(1, len(slots), figsize=(4.2 * len(slots), 3.6), squeeze=False)
            for ax, slot in zip(axs[0], slots):
                tb = tables[slot]
                lr = np.linspace(tb["logrho"][0], tb["logrho"][1], 400)
                ulo, uhi = lib.energy_bounds(slot, 10**lr)
                rf = lib.classify(slot, 10**lr, ulo, np.full(lr.size, 10**tb["logT"][0]), np.zeros(lr.size, np.uint8))
                ax.plot(lr, np.log10(rf["T"]), "k-", lw=1, label="validity floor")
                for rr, TT in flagged_points.get(("all", slot), []):
                    ax.plot(np.log10(rr), np.log10(TT), ".", color="0.8", ms=1)
                col = {"U_AT_FLOOR": "C3", "NEAR_FLOOR": "C1", "FOLD_IN": "C4", "FOLD_NEAR": "C6", "PLATEAU": "C2", "ROOTS_MULTI": "k",
                       "RHO_BELOW": "C0", "RHO_ABOVE": "C0", "NEAR_RHO_LOW": "C9", "NEAR_RHO_HIGH": "C9", "U_AT_TOP": "C5", "NEAR_TOP": "C5"}
                for name, cc in col.items():
                    xs = [np.log10(rr[(ff & BIT[name]) != 0]) for rr, TT, ff in flagged_points[slot]]
                    ys = [np.log10(TT[(ff & BIT[name]) != 0]) for rr, TT, ff in flagged_points[slot]]
                    xs, ys = np.concatenate(xs), np.concatenate(ys)
                    if xs.size:
                        ax.plot(xs, ys, ".", color=cc, ms=2, label=name)
                occupied = [np.log10(rr) for rr, TT in flagged_points.get(("all", slot), [])] + [np.log10(rr) for rr, TT, ff in flagged_points[slot]]
                occupied = np.concatenate(occupied) if occupied else np.array([])
                if occupied.size:
                    ax.set_xlim(occupied.min() - 1, occupied.max() + 1)   # the occupied densities, not the whole table
                ax.set_title(os.path.basename(tb["path"]), fontsize=8)
                ax.set_xlabel("log10 rho [g/cc]"); ax.set_ylabel("log10 T [K]"); ax.legend(fontsize=5)
            fig.tight_layout(); fig.savefig(prefix + "_rhoT.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
