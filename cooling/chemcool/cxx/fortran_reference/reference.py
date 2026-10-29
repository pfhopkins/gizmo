#!/usr/bin/env python3
"""Frozen chemistry reference: freeze a reference_dump output, or compare a new dump against it.

  reference.py freeze  <dump data dir> <ref dir>   # writes manifest.json, tables.npz, cool_func.npz, onezone.csv
  reference.py compare <ref dir> <dump data dir>   # report + exit 1 on failure

Tables up to MAXFULL entries are stored in full; larger ones as a fixed-stride subsample
(stride in manifest, last entry always kept). Every table carries the SHA-256 of its full bytes.
Pass criteria: tables at stored indices within TAB_REL; cool_func ylam/rates within KER_REL;
one-zone final T within ONEZONE_T_REL (Gate 2/3 criterion); the trajectory maximum is reported
(DVODE vs CVODE step timing gives a few % mid-cooling from 1e6 K). Bitwise counts are always reported.
"""
import hashlib, json, os, sys
import numpy as np

MAXFULL = 20000
NSUB = 4096
TAB_REL = 1e-6        # derivative tables reach ~5e-6 (Fortran vs C++, cancellation); values are <= 1e-15
KER_REL = 1e-10
ONEZONE_T_REL = 1e-2
TINY = 1e-300


def read_tables(d):
    raw = np.fromfile(os.path.join(d, "tables.bin"), dtype="<f8")
    out = {}
    for line in open(os.path.join(d, "tables.idx")):
        name, off, n = line.split()
        out[name] = raw[int(off):int(off) + int(n)]
    return out


def rel(a, b):
    s = np.maximum(np.abs(a), np.abs(b))
    r = np.where(s > TINY, np.abs(a - b) / np.where(s > TINY, s, 1.0), 0.0)
    return r


def freeze(src, dst):
    os.makedirs(dst, exist_ok=True)
    tabs = read_tables(src)
    man = {"tables": {}, "maxfull": MAXFULL, "nsub": NSUB}
    store = {}
    for name, a in tabs.items():
        n = len(a)
        stride = 1 if n <= MAXFULL else -(-n // NSUB)
        idx = np.unique(np.append(np.arange(0, n, stride), n - 1))
        store[name] = a[idx]
        man["tables"][name] = {"n": n, "stride": stride, "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    np.savez_compressed(os.path.join(dst, "tables.npz"), **store)
    cf = np.fromfile(os.path.join(src, "cool_func.bin"), dtype="<f8")
    np.savez_compressed(os.path.join(dst, "cool_func.npz"), cf=cf)
    with open(os.path.join(src, "onezone.csv")) as f, open(os.path.join(dst, "onezone.csv"), "w") as g:
        g.write(f.read())
    json.dump(man, open(os.path.join(dst, "manifest.json"), "w"), indent=1)
    print("froze %d tables -> %s" % (len(tabs), dst))


def compare(ref, new):
    ok = True
    man = json.load(open(os.path.join(ref, "manifest.json")))
    store = np.load(os.path.join(ref, "tables.npz"))
    tabs = read_tables(new)
    print("tables (ref vs new):  name  n  full-sha  bitwise@stored/stored  maxrel")
    for name, m in man["tables"].items():
        if name not in tabs:
            print("  %-44s MISSING" % name); ok = False; continue
        a = tabs[name]
        if len(a) != m["n"]:
            print("  %-44s size %d != %d" % (name, len(a), m["n"])); ok = False; continue
        idx = np.unique(np.append(np.arange(0, len(a), m["stride"]), len(a) - 1))
        r, b = store[name], a[idx]
        same = int(np.sum(r.view("<u8") == b.view("<u8")))
        sha = hashlib.sha256(a.tobytes()).hexdigest() == m["sha256"]
        mr = float(rel(r, b).max()) if len(r) else 0.0
        bad = mr > TAB_REL
        ok &= not bad
        print("  %-44s %-8d %-5s %d/%d  %.3e%s" % (name, m["n"], "same" if sha else "diff", same, len(r), mr, "  FAIL" if bad else ""))

    cr = np.load(os.path.join(ref, "cool_func.npz"))["cf"]
    cn = np.fromfile(os.path.join(new, "cool_func.bin"), dtype="<f8")
    w = 34
    cr, cn = cr.reshape(-1, w), cn.reshape(-1, w)
    rk = rel(cr[:, 4:], cn[:, 4:])
    rk[:, 1] = rel(cr[:, 5], cn[:, 5])
    same = int(np.sum(cr.view("<u8") == cn.view("<u8")))
    mk = float(rk.max())
    print("cool_func: %d records, bitwise %d/%d doubles, maxrel ylam/tdust/rates %.3e%s"
          % (len(cr), same, cr.size, mk, "  FAIL" if mk > KER_REL else ""))
    ok &= mk <= KER_REL

    def load(p):
        lines = open(p).read().split("\n")
        return lines[0].split(","), np.array([[float(x) for x in l.split(",")] for l in lines[1:] if l])
    h, zr = load(os.path.join(ref, "onezone.csv"))
    _, zn = load(os.path.join(new, "onezone.csv"))
    iT = h.index("T")
    rT = rel(zr[:, iT], zn[:, iT])
    same = int(np.sum(zr.view("<u8") == zn.view("<u8")))
    nz = int(zr[:, 0].max()) + 1
    last = [np.where(zr[:, 0] == z)[0][-1] for z in range(nz)]
    print("onezone: %d zones, bitwise %d/%d values, max T rel over all steps %.3e, final T rel max %.3e%s"
          % (nz, same, zr.size, rT.max(), rT[last].max(), "  FAIL" if rT[last].max() > ONEZONE_T_REL else ""))
    for z in range(nz):
        s = zr[:, 0] == z
        print("  zone %2d nH=%-6g T0=%-6g Z=%-4g  T_ref=%-12.6g T_new=%-12.6g max T rel %.2e"
              % (z, zr[s][0, 3], zr[s][0, 4], zr[s][0, 5], zr[s][-1, iT], zn[s][-1, iT], rT[s].max()))
    ok &= rT[last].max() <= ONEZONE_T_REL
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("freeze", "compare"):
        print(__doc__); sys.exit(2)
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2], sys.argv[3])
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
