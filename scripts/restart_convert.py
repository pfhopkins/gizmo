#!/usr/bin/env python3
"""Convert GIZMO restartfiles between two builds whose particle (P) and gas-cell (CellP) layouts differ.

    restart_convert.py SRC_BIN DST_BIN IN_DIR OUT_DIR [-j N] [--ranks A-B]
    restart_convert.py --verify SRC_BIN DST_BIN SRC_FILE DST_FILE

SRC_BIN wrote the files in IN_DIR; DST_BIN is the binary that will resume from OUT_DIR. Both must be built from the
same physics configuration and need debug info (GIZMO builds with -g): field offsets are read from their DWARF with
gdb ('ptype /o'). Fields are matched by name (with the enclosing struct's name for nested fields), so reordered
fields are handled. Per field:
  same size            -> copied
  double -> float      -> rounded to nearest (what a PRECISION_DIET=2 build does when it stores the field)
  float -> double      -> widened exactly
Anything else (a field added, removed or changed in another way) aborts, unless --allow-missing is given, in which
case added fields are zero and removed ones are dropped (both listed).

File layout (file_io/restart.cc): [header: magic, layout version, sizeof All/P/CellP][All][int NumPart][P x NumPart]
[int N_gas][CellP x N_gas][rest: copied byte for byte]. All must have the same size in both builds. The output
header carries DST_BIN's sizes and layout version (the global RestartfileLayoutVersion, 2 if absent).
Files are processed in parallel (-j, default 8); each needs about 2x its size in memory.
"""
import argparse, os, re, struct, subprocess, sys
from multiprocessing import Pool
import numpy as np

MAGIC = 0x52535446
HDR = struct.Struct('<IIQQQ')
FLOATY = re.compile(r'Float|Double|double|float|\bT\b|Vec3|Mat3|Tensor')

def gdb(binary, cmd):
    return subprocess.run(['gdb', '-batch', '-ex', cmd, binary], capture_output=True, text=True, timeout=300).stdout

def leaves(binary, sname):
    """{path: (offset, size, type)} of every leaf of struct sname, and the struct's size."""
    out = gdb(binary, 'ptype /o struct ' + sname)
    stack = [[]]; total = None
    for l in out.splitlines():
        if re.match(r'/\*\s+\d+:\s*\d+\s+\|', l): raise SystemExit('bit-field in %s: not supported' % sname)
        m = re.match(r'/\*\s+(\d+)\s+\|\s+(\d+) \*/\s+(.*)', l)
        if m:
            off, size, txt = int(m.group(1)), int(m.group(2)), m.group(3).rstrip()
            if txt.endswith('{'):
                if not txt.startswith('type ='): stack.append([])
                continue
            decl = txt.rstrip(';'); name = decl.split()[-1]; typ = ' '.join(decl.split()[:-1])
            stack[-1].append((name, off, size, typ)); continue
        c = re.match(r'\s+\} (\w+(?:\[\d+\])*);', l)
        if c and len(stack) > 1:
            kids = stack.pop(); stack[-1].extend((c.group(1) + '.' + n, o, s, t) for n, o, s, t in kids); continue
        t = re.search(r'total size \(bytes\):\s+(\d+) \*/', l)
        if t: total = int(t.group(1))
    if 'union' in out: raise SystemExit('union in %s: not supported' % sname)
    lv = {}
    for name, off, size, typ in stack[0]:
        if name in lv: raise SystemExit('duplicate field path %s.%s' % (sname, name))
        lv[name] = (off, size, typ)
    if not lv or total is None: raise SystemExit('no layout for struct %s in %s (built without -g?)' % (sname, binary))
    return lv, total

def layout_version(binary):
    m = re.search(r'=\s*(\d+)', gdb(binary, 'print RestartfileLayoutVersion'))
    return int(m.group(1)) if m else 2

def sizeof_all(binary):
    return int(re.search(r'=\s*(\d+)', gdb(binary, 'print sizeof(struct global_data_all_processes)')).group(1))

def plan(src_bin, dst_bin, sname, allow_missing):
    a, sa = leaves(src_bin, sname); b, sb = leaves(dst_bin, sname)
    ops, notes = [], []
    for name, (oa, za, ta) in a.items():
        if name not in b: notes.append('dropped %s.%s (%d B)' % (sname, name, za)); continue
        ob, zb, tb = b[name]
        if za == zb: ops.append(('copy', oa, ob, za))
        elif za == 2 * zb and za % 8 == 0 and FLOATY.search(ta + " " + tb): ops.append(('d2f', oa, ob, za))
        elif zb == 2 * za and zb % 8 == 0 and FLOATY.search(ta + " " + tb): ops.append(('f2d', oa, ob, za))
        else: raise SystemExit('cannot map %s.%s: %s %d B -> %s %d B' % (sname, name, ta, za, tb, zb))
        if za != zb: notes.append('%s %s.%s %d -> %d B' % (ops[-1][0], sname, name, za, zb))
    for name, (ob, zb, tb) in b.items():
        if name not in a: notes.append('added %s.%s (%d B, zero)' % (sname, name, zb))
    if any(n.startswith(('dropped', 'added')) for n in notes) and not allow_missing:
        raise SystemExit('field set differs (use --allow-missing to accept):\n  ' + '\n  '.join(n for n in notes if n.startswith(('dropped', 'added'))))
    return ops, sa, sb, notes

def convert_block(raw, n, sa, sb, ops):
    src = np.frombuffer(raw, dtype=np.uint8).reshape(n, sa); dst = np.zeros((n, sb), dtype=np.uint8)
    for kind, oa, ob, z in ops:
        x = np.ascontiguousarray(src[:, oa:oa + z])
        if kind == 'copy': dst[:, ob:ob + z] = x
        elif kind == 'd2f': dst[:, ob:ob + z // 2] = x.view('<f8').astype('<f4').view(np.uint8)
        else: dst[:, ob:ob + 2 * z] = x.view('<f4').astype('<f8').view(np.uint8)
    return dst.tobytes()

def read_blocks(path, sp, sc):
    with open(path, 'rb') as f:
        h = HDR.unpack(f.read(HDR.size))
        if h[0] != MAGIC: raise SystemExit('%s: no layout header (pre-sentinel file); not supported' % path)
        if (h[3], h[4]) != (sp, sc): raise SystemExit('%s: header says P/CellP %d/%d B, binary has %d/%d' % (path, h[3], h[4], sp, sc))
        allb = f.read(h[2]); n = struct.unpack('<i', f.read(4))[0]; P = f.read(n * sp)
        ng = struct.unpack('<i', f.read(4))[0]; C = f.read(ng * sc); rest = f.read()
    return h, allb, n, P, ng, C, rest

def convert_file(job):
    src, dst, pp, pc, ver = job
    (h, allb, n, P, ng, C, rest) = read_blocks(src, pp[1], pc[1])
    with open(dst + '.part', 'wb') as fo:
        fo.write(HDR.pack(MAGIC, ver, h[2], pp[2], pc[2])); fo.write(allb)
        fo.write(struct.pack('<i', n)); fo.write(convert_block(P, n, pp[1], pp[2], pp[0]))
        fo.write(struct.pack('<i', ng)); fo.write(convert_block(C, ng, pc[1], pc[2], pc[0])); fo.write(rest)
    os.replace(dst + '.part', dst)
    return '%s N=%d N_gas=%d' % (os.path.basename(src), n, ng)

def verify(a):
    """Field-by-field check of DST_FILE against SRC_FILE: copies exact, conversions equal to the rounding/widening."""
    bad = 0
    pp = plan(a.src_bin, a.dst_bin, 'particle_data', True); pc = plan(a.src_bin, a.dst_bin, 'gas_cell_data', True)
    A = read_blocks(a.in_dir, pp[1], pc[1]); B = read_blocks(a.out_dir, pp[2], pc[2])
    if A[0][2] != B[0][2] or A[1] != B[1]: bad += 1; print('MISMATCH All')
    if B[0][1] != layout_version(a.dst_bin): bad += 1; print('MISMATCH layout version', B[0][1])
    for (ops, sa, sb, _), (n, ra), (m, rb), nm in [(pp, A[2:4], B[2:4], 'P'), (pc, A[4:6], B[4:6], 'CellP')]:
        if n != m: bad += 1; print('MISMATCH count', nm, n, m); continue
        x = np.frombuffer(ra, np.uint8).reshape(n, sa); y = np.frombuffer(rb, np.uint8).reshape(n, sb)
        for kind, oa, ob, z in ops:
            u = np.ascontiguousarray(x[:, oa:oa + z])
            if kind == 'copy': ok = np.array_equal(u, y[:, ob:ob + z])
            elif kind == 'd2f': ok = np.array_equal(u.view('<f8').astype('<f4').view(np.uint8), y[:, ob:ob + z // 2])
            else: ok = np.array_equal(u.view('<f4').astype('<f8').view(np.uint8), y[:, ob:ob + 2 * z])
            if not ok: bad += 1; print('MISMATCH %s field at src offset %d' % (nm, oa))
        print('%s: %d elements, %d fields checked' % (nm, n, len(ops)))
    if A[6] != B[6]: bad += 1; print('MISMATCH trailing data')
    print('mismatches', bad); return bad

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src_bin'); ap.add_argument('dst_bin'); ap.add_argument('in_dir'); ap.add_argument('out_dir')
    ap.add_argument('-j', type=int, default=8); ap.add_argument('--ranks'); ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--verify', action='store_true', help='in_dir/out_dir are one source and one converted file')
    a = ap.parse_args()
    if a.verify: sys.exit(1 if verify(a) else 0)
    if sizeof_all(a.src_bin) != sizeof_all(a.dst_bin): raise SystemExit('sizeof(All) differs between the binaries: not supported')
    pp = plan(a.src_bin, a.dst_bin, 'particle_data', a.allow_missing); pc = plan(a.src_bin, a.dst_bin, 'gas_cell_data', a.allow_missing)
    ver = layout_version(a.dst_bin)
    print('P %d -> %d B, CellP %d -> %d B, layout version -> %d' % (pp[1], pp[2], pc[1], pc[2], ver))
    for note in pp[3] + pc[3]: print('  ' + note)
    files = sorted((f for f in os.listdir(a.in_dir) if re.fullmatch(r'restart\.\d+', f)), key=lambda f: int(f.split('.')[1]))
    if a.ranks: lo, hi = map(int, a.ranks.split('-')); files = [f for f in files if lo <= int(f.split('.')[1]) <= hi]
    if os.path.realpath(a.in_dir) == os.path.realpath(a.out_dir): raise SystemExit('OUT_DIR must differ from IN_DIR')
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(os.path.join(a.in_dir, f), os.path.join(a.out_dir, f), pp, pc, ver) for f in files]
    with Pool(max(1, min(a.j, len(jobs)))) as pool:
        for r in pool.imap_unordered(convert_file, jobs): print(r, flush=True)

if __name__ == '__main__': main()
