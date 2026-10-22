#!/usr/bin/env python3
"""
CHEMCOOL C++/CVODE port -- GATE 1 rate-layer extractor.
(Spec: /orion/ptmp/uli/orion/stella/CHEMCOOL_CXX_CVODE_ASSESSMENT.md section 5)

Machine-extracts the rate layer used by the production chemistry networks
5 (rate_eq_simple.F) and 17 (rate_eq_ulli.F) from the canonical Fortran
sources (READ-ONLY, nothing in the gizmo tree is modified):

    cheminmo.F     -> chtab(k,T) k(T) expressions, crtab/crphot CR rates
    coolinmo.F     -> cltab(k,T) cooling-coefficient expressions
    const_rates.F  -> cst(k) constant rate coefficients
    rate_eq_simple.F / rate_eq_ulli.F / cool_func.F -> RHS-inline rate terms
                                                       (audit only; Gate-2 scope)

Outputs (this directory):
    rates.json         -- structured extraction (expressions, guards, citations)
    rates_gen.h/.cc    -- generated C++, one function per extractable rate
    rates_registry.inc -- registry consumed by golden/test_rates.cc
    rates_registry16.inc -- network-16 (Gong+17) entries: rate16_* functions
                          for rates whose CHEMISTRYNETWORK == 16 branch differs
    RATE_AUDIT.md      -- Tier-2 audit table

NOTHING is hand-transcribed: every expression is parsed into an AST from the
Fortran source and re-emitted.  Constructs that resist parsing (DATA-table
splines, post-loop table overwrites, subroutine calls) are recorded as
kind=TABLE or kind=SPECIAL with their exact location instead of guessed.

Precision note: Fortran default-real literals (no 'd' exponent, e.g. 0.485,
1.38066e-16) are single precision under the production FFLAGS (-O2, no
-fdefault-real-8).  They are emitted as C float literals ('0.485f') so the
generated code reproduces gfortran's constant rounding exactly -- required
for the 1e-12 relative golden tolerance.
"""

import json, os, re
from collections import OrderedDict

OUT   = os.path.dirname(os.path.abspath(__file__))
# Fortran sources of the tree this script sits in (override: CHEMCOOL_FORTRAN_DIR)
CANON = os.environ.get("CHEMCOOL_FORTRAN_DIR", os.path.dirname(OUT))
CFG5  = "/orion/ptmp/uli/orion/stella/tests/onezone/build_net5/GIZMO_config.h"
CFG17 = "/orion/ptmp/uli/orion/stella/tests/onezone/build_net17/GIZMO_config.h"

# ----------------------------------------------------------------------------
# cpp macro environment + conditional evaluator
# ----------------------------------------------------------------------------
def load_config_macros(path):
    macros = {}
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*#define\s+(\w+)(?:\s+(.*?))?\s*$", line)
            if m:
                macros[m.group(1)] = (m.group(2) or "").strip()
    return macros

def cpp_eval(cond, macros):
    s = cond.split("/*")[0]
    s = re.sub(r"defined\s*\(\s*(\w+)\s*\)",
               lambda m: "1" if m.group(1) in macros else "0", s)
    s = re.sub(r"defined\s+(\w+)",
               lambda m: "1" if m.group(1) in macros else "0", s)
    def sub_ident(m):
        name = m.group(0)
        if name in macros:
            v = macros[name]
            return v if re.match(r"^\d+$", v) else "1"
        return "0"
    s = re.sub(r"\b[A-Za-z_]\w*\b", sub_ident, s)
    s = s.replace("&&", " and ").replace("||", " or ")
    s = re.sub(r"!(?!=)", " not ", s)
    try:
        return bool(eval(s, {"__builtins__": {}}))
    except Exception:
        return True

# ----------------------------------------------------------------------------
# Fixed-form Fortran statement reader
# ----------------------------------------------------------------------------
class Stmt:
    __slots__ = ("text", "l0", "l1", "comment", "label")
    def __init__(self, text, l0, l1, comment, label):
        self.text, self.l0, self.l1 = text, l0, l1
        self.comment, self.label = comment, label
    def __repr__(self):
        return "Stmt(%d:%s)" % (self.l0, self.text[:60])

def read_statements(path, macros):
    stmts, comment_buf, cur = [], [], None
    cpp_stack = []
    def active():
        return all(a for a, _ in cpp_stack)
    with open(path) as f:
        lines = f.readlines()
    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        ls = line.strip()
        if ls.startswith("#"):
            m = re.match(r"#\s*(\w+)\s*(.*)", ls)
            if not m: continue
            d, rest = m.group(1), m.group(2)
            if d == "include":
                continue
            elif d == "if":
                v = cpp_eval(rest, macros); cpp_stack.append([v, v])
            elif d == "ifdef":
                v = rest.split()[0] in macros; cpp_stack.append([v, v])
            elif d == "ifndef":
                v = rest.split()[0] not in macros; cpp_stack.append([v, v])
            elif d == "elif" and cpp_stack:
                if cpp_stack[-1][1]:
                    cpp_stack[-1][0] = False
                else:
                    v = cpp_eval(rest, macros)
                    cpp_stack[-1][0] = v; cpp_stack[-1][1] = v
            elif d == "else" and cpp_stack:
                cpp_stack[-1][0] = not cpp_stack[-1][1]
                cpp_stack[-1][1] = True
            elif d == "endif" and cpp_stack:
                cpp_stack.pop()
            continue
        if not active():
            continue
        if ls == "" or line[:1] in ("c", "C", "*", "!"):
            txt = ls.lstrip("cC*!").strip()
            if txt:
                comment_buf.append(txt)
            continue
        iscont = len(line) > 5 and line[5] not in (" ", "0") and \
                 line[:5].strip() == ""
        body = line[6:] if len(line) > 6 else ""
        q, out = False, []
        for ch in body:
            if ch == "'":
                q = not q
            if ch == "!" and not q:
                break
            out.append(ch)
        body = "".join(out)
        if iscont and cur is not None:
            cur[0] += " " + body.strip()
            cur[2] = lineno
            continue
        if cur is not None:
            stmts.append(Stmt(cur[0].strip(), cur[1], cur[2], cur[3], cur[4]))
        label = line[:5].strip()
        cur = [body.strip(), lineno, lineno, "\n".join(comment_buf).strip(), label]
        comment_buf = []
    if cur is not None:
        stmts.append(Stmt(cur[0].strip(), cur[1], cur[2], cur[3], cur[4]))
    return stmts

# ----------------------------------------------------------------------------
# Expression parsing + C++ emission
# ----------------------------------------------------------------------------
TOK_RE = re.compile(r"""
    (?P<num>((\d+\.\d*)|(\.\d+)|(\d+))([edED][+-]?\d+)?)
  | (?P<rel>\.(le|lt|ge|gt|eq|ne|and|or|not)\.)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<pow>\*\*)
  | (?P<op>[-+*/(),])
""", re.X)

class Tok:
    def __init__(self, kind, val): self.kind, self.val = kind, val
    def __repr__(self): return "%s:%s" % (self.kind, self.val)

def tokenize(s):
    toks, pos = [], 0
    s = s.strip()
    while pos < len(s):
        if s[pos].isspace():
            pos += 1; continue
        m = TOK_RE.match(s, pos)
        if not m:
            raise SyntaxError("tokenize fail at %r" % s[pos:pos+30])
        pos = m.end()
        if m.lastgroup == "num":
            toks.append(Tok("num", m.group("num")))
        elif m.lastgroup == "rel":
            toks.append(Tok("rel", m.group("rel").strip(".").lower()))
        elif m.lastgroup == "name":
            toks.append(Tok("name", m.group("name")))
        elif m.lastgroup == "pow":
            toks.append(Tok("pow", "**"))
        else:
            toks.append(Tok("op", m.group("op")))
    return toks

INTRINSICS = {"dexp": "exp", "exp": "exp", "dlog10": "log10", "log10": "log10",
              "dlog": "log", "log": "log", "dsqrt": "sqrt", "sqrt": "sqrt",
              "dabs": "fabs", "abs": "fabs", "int": "(int)",
              "max": "f_max", "min": "f_min", "dble": "(double)"}

class ExprParser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def eat(self, kind=None, val=None):
        tk = self.peek()
        if tk is None or (kind and tk.kind != kind) or (val and tk.val != val):
            raise SyntaxError("expected %s %s got %r" % (kind, val, tk))
        self.i += 1
        return tk
    def parse(self):
        e = self.expr_or()
        if self.peek() is not None:
            raise SyntaxError("trailing tokens %r" % self.t[self.i:])
        return e
    def expr_or(self):
        e = self.expr_and()
        while self.peek() and self.peek().kind == "rel" and self.peek().val == "or":
            self.eat(); e = ("logic", "||", e, self.expr_and())
        return e
    def expr_and(self):
        e = self.expr_not()
        while self.peek() and self.peek().kind == "rel" and self.peek().val == "and":
            self.eat(); e = ("logic", "&&", e, self.expr_not())
        return e
    def expr_not(self):
        if self.peek() and self.peek().kind == "rel" and self.peek().val == "not":
            self.eat(); return ("not", self.expr_not())
        return self.expr_rel()
    def expr_rel(self):
        e = self.expr_add()
        tk = self.peek()
        if tk and tk.kind == "rel" and tk.val in ("le","lt","ge","gt","eq","ne"):
            op = {"le":"<=","lt":"<","ge":">=","gt":">","eq":"==","ne":"!="}[tk.val]
            self.eat(); e = ("rel", op, e, self.expr_add())
        return e
    def expr_add(self):
        tk = self.peek()
        if tk and tk.kind == "op" and tk.val in "+-":
            self.eat()
            e = ("un", "-", self.expr_mul()) if tk.val == "-" else self.expr_mul()
        else:
            e = self.expr_mul()
        while True:
            tk = self.peek()
            if tk and tk.kind == "op" and tk.val in "+-":
                self.eat(); e = ("bin", tk.val, e, self.expr_mul())
            else:
                return e
    def expr_mul(self):
        e = self.expr_pow()
        while True:
            tk = self.peek()
            if tk and tk.kind == "op" and tk.val in "*/":
                self.eat(); e = ("bin", tk.val, e, self.expr_pow())
            else:
                return e
    def expr_pow(self):
        e = self.atom()
        tk = self.peek()
        if tk and tk.kind == "pow":
            self.eat()
            tk2 = self.peek()
            if tk2 and tk2.kind == "op" and tk2.val == "-":
                self.eat(); rhs = ("un", "-", self.expr_pow())
            else:
                rhs = self.expr_pow()
            e = ("pow", e, rhs)
        return e
    def atom(self):
        tk = self.peek()
        if tk is None:
            raise SyntaxError("unexpected end")
        if tk.kind == "num":
            self.eat(); return ("num", tk.val)
        if tk.kind == "op" and tk.val == "(":
            self.eat(); e = self.expr_or(); self.eat("op", ")")
            return e
        if tk.kind == "op" and tk.val == "-":
            self.eat(); return ("un", "-", self.atom())
        if tk.kind == "name":
            self.eat()
            name = tk.val
            if self.peek() and self.peek().kind == "op" and self.peek().val == "(":
                self.eat()
                args = []
                if not (self.peek().kind == "op" and self.peek().val == ")"):
                    args.append(self.expr_or())
                    while self.peek() and self.peek().kind == "op" and \
                          self.peek().val == ",":
                        self.eat(); args.append(self.expr_or())
                self.eat("op", ")")
                if name.lower() in INTRINSICS:
                    return ("call", name.lower(), args)
                return ("aref", name.lower(), args)
            return ("var", name.lower())
        raise SyntaxError("unexpected token %r" % tk)

def parse_expr(s):
    return ExprParser(tokenize(s)).parse()

def num_kind(text):
    if re.search(r"[dD]", text): return "double"
    if "." in text or re.search(r"[eE]", text): return "float"
    return "int"

def num_cxx(text):
    k = num_kind(text)
    if k == "double":
        return re.sub(r"[dD]", "e", text)
    if k == "float":
        t = text
        if "." not in t and not re.search(r"[eE]", t):
            t += ".0"
        return t + "f"
    return text

def expr_refs(ast, refs):
    kind = ast[0]
    if kind == "num":
        return
    if kind == "var":
        refs.add(("var", ast[1]))
    elif kind == "aref":
        for a in ast[2]:
            expr_refs(a, refs)
        refs.add(("arr", ast[1], ast_key(ast)))
    elif kind == "call":
        for a in ast[2]:
            expr_refs(a, refs)
    elif kind in ("bin", "rel", "logic"):
        expr_refs(ast[2], refs); expr_refs(ast[3], refs)
    elif kind == "pow":
        expr_refs(ast[1], refs); expr_refs(ast[2], refs)
    elif kind in ("un", "not"):
        expr_refs(ast[-1], refs)

def ast_key(aref):
    return "%s(%s)" % (aref[1], ",".join(ast_src(a) for a in aref[2]))

def ast_src(ast):
    k = ast[0]
    if k == "num": return ast[1]
    if k == "var": return ast[1]
    if k == "aref": return "%s(%s)" % (ast[1], ",".join(ast_src(a) for a in ast[2]))
    if k == "call": return "%s(%s)" % (ast[1], ",".join(ast_src(a) for a in ast[2]))
    if k == "bin": return "(%s%s%s)" % (ast_src(ast[2]), ast[1], ast_src(ast[3]))
    if k == "pow": return "(%s**%s)" % (ast_src(ast[1]), ast_src(ast[2]))
    if k == "un": return "(-%s)" % ast_src(ast[2])
    if k == "rel": return "(%s %s %s)" % (ast_src(ast[2]), ast[1], ast_src(ast[3]))
    if k == "logic": return "(%s %s %s)" % (ast_src(ast[2]), ast[1], ast_src(ast[3]))
    if k == "not": return "(.not. %s)" % ast_src(ast[1])
    raise ValueError(k)

GLOBALS_MAP = {
    "deff": "g.deff", "dust_to_gas_ratio": "g.dust_to_gas_ratio",
    "phi_pah": "g.phi_pah", "cosmic_ray_ion_rate": "g.cosmic_ray_ion_rate",
    "abhe": "g.abhe",
    "dtlog": "g.dtlog", "tmin": "g.tmin", "tmax": "g.tmax",
    "iflag_mn": "g.iflag_mn", "iflag_ad": "g.iflag_ad",
    "iflag_atom": "g.iflag_atom", "iflag_3bh2a": "g.iflag_3bh2a",
    "iflag_3bh2b": "g.iflag_3bh2b", "iflag_h3pra": "g.iflag_h3pra",
}

class Ctx:
    def __init__(self, consts, localmap):
        self.consts, self.localmap = consts, localmap
        # canonical keys currently known to hold compile-time constants along
        # the emission path.  Used to reproduce gfortran's DUAL semantics for
        # x**n: runtime -> frontend mult-chain (fpowi); constant-propagated ->
        # correctly-rounded fold (observed in cheminmo.o: constant guard
        # branches like tfix=5d0 hold mpfr-correctly-rounded powers).  For
        # constant bases we emit pow(x, n.0) so g++ folds with mpfr exactly
        # like gfortran did.
        self.constset = set()

def ast_is_const(ast, ctx):
    k = ast[0]
    if k == "num":
        return True
    if k == "var":
        n = ast[1]
        if n in ("temp", "itemp") or n in GLOBALS_MAP:
            return False
        if n in ctx.constset:
            return True
        return n in ctx.consts  # named constants (C_*)
    if k == "aref":
        return ast_key(ast) in ctx.constset
    if k == "call":
        return all(ast_is_const(a, ctx) for a in ast[2])
    if k in ("bin", "rel", "logic"):
        return ast_is_const(ast[2], ctx) and ast_is_const(ast[3], ctx)
    if k == "pow":
        return ast_is_const(ast[1], ctx) and ast_is_const(ast[2], ctx)
    if k in ("un", "not"):
        return ast_is_const(ast[-1], ctx)
    return False

def cxx_expr(ast, ctx):
    k = ast[0]
    if k == "num":
        return num_cxx(ast[1])
    if k == "var":
        n = ast[1]
        if n in ctx.localmap: return ctx.localmap[n]
        if n in GLOBALS_MAP:  return GLOBALS_MAP[n]
        if n in ctx.consts:   return ctx.consts[n]
        raise KeyError("unresolved var '%s'" % n)
    if k == "aref":
        key = ast_key(ast)
        if key in ctx.localmap:
            return ctx.localmap[key]
        raise KeyError("unresolved array ref '%s'" % key)
    if k == "call":
        fn = INTRINSICS[ast[1]]
        args = ", ".join(cxx_expr(a, ctx) for a in ast[2])
        return "%s(%s)" % (fn, args)
    if k == "bin":
        return "(%s %s %s)" % (cxx_expr(ast[2], ctx), ast[1], cxx_expr(ast[3], ctx))
    if k == "pow":
        b = cxx_expr(ast[1], ctx)
        e = ast[2]
        if e[0] == "num" and num_kind(e[1]) == "int":
            if ast_is_const(ast[1], ctx):
                # constant base: gfortran folded this correctly-rounded;
                # pow(const, n.0) lets g++ fold identically (mpfr)
                return "pow(%s, %s.0)" % (b, e[1])
            return "fpowi(%s, %s)" % (b, e[1])
        return "pow(%s, %s)" % (b, cxx_expr(e, ctx))
    if k == "un":
        return "(-%s)" % cxx_expr(ast[2], ctx)
    if k in ("rel", "logic"):
        return "(%s %s %s)" % (cxx_expr(ast[2], ctx), ast[1], cxx_expr(ast[3], ctx))
    if k == "not":
        return "(!%s)" % cxx_expr(ast[1], ctx)
    raise ValueError(k)

# ----------------------------------------------------------------------------
# Block-tree
# ----------------------------------------------------------------------------
class Assign:
    kind = "assign"
    def __init__(self, stmt, lhs, rhs):
        self.stmt, self.lhs, self.rhs = stmt, lhs, rhs

class IfBlock:
    kind = "if"
    def __init__(self):
        self.branches = []
        self.stmt = None

class DoBlock:
    kind = "do"
    def __init__(self, stmt):
        self.stmt, self.body = stmt, []

class Opaque:
    kind = "opaque"
    def __init__(self, stmt):
        self.stmt = stmt

ONE_LINE_IF = re.compile(r"if\s*\((.*)\)\s*(\w.*)$", re.I | re.S)

def classify_stmt(st):
    s = st.text.strip()
    if s == "" or re.match(r"(continue|return|end$|end\s|implicit|integer\b|"
                           r"real\b|parameter|save\b|data\b|character|"
                           r"external|subroutine|program|include|common)\b",
                           s, re.I):
        return None
    m = ONE_LINE_IF.match(s)
    if m and not re.search(r"\)\s*then\s*$", s, re.I):
        # balance parens to split condition properly
        depth, splitat = 0, None
        for i, ch in enumerate(s):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    splitat = i; break
        if splitat is not None:
            cond_src = s[s.index("(")+1:splitat]
            rest = s[splitat+1:].strip()
            if rest:
                inner = classify_stmt(Stmt(rest, st.l0, st.l1, st.comment, ""))
                blk = IfBlock(); blk.stmt = st
                try:
                    cond = parse_expr(cond_src)
                except SyntaxError:
                    return Opaque(st)
                blk.branches.append((cond, [inner] if inner else []))
                return blk
        return Opaque(st)
    if re.match(r"(call|goto|go\s*to|print|write|if\b|abort|stop)", s, re.I):
        return Opaque(st)
    m = re.match(r"([A-Za-z_]\w*(\s*\([^=]*?\))?)\s*=\s*(.*)$", s, re.S)
    if m:
        try:
            return Assign(st, parse_expr(m.group(1)), parse_expr(m.group(3)))
        except SyntaxError:
            return Opaque(st)
    return Opaque(st)

def build_tree(stmts):
    root = []
    stack = [("root", root)]
    do_labels = []

    def top_list():
        t = stack[-1]
        if t[0] == "if":
            return t[1].branches[-1][1]
        if t[0] == "do":
            return t[1].body
        return t[1]

    for st in stmts:
        s = st.text
        sl = s.lower().replace(" ", "")
        if st.label and do_labels and st.label == do_labels[-1]:
            if not sl.startswith("continue"):
                node = classify_stmt(st)
                if node is not None:
                    top_list().append(node)
            # close ALL loops waiting on this label (shared-label loops)
            while do_labels and do_labels[-1] == st.label:
                do_labels.pop()
                done = stack.pop()
                top_list().append(done[1])
            continue
        m = re.match(r"do\s+(\d+)\s+", s, re.I)
        m2 = re.match(r"do\s+\w+\s*=", s, re.I)
        if m:
            blk = DoBlock(st); stack.append(("do", blk))
            do_labels.append(m.group(1)); continue
        if m2:
            blk = DoBlock(st); stack.append(("do", blk))
            do_labels.append("<enddo>"); continue
        if sl == "enddo":
            done = stack.pop(); do_labels.pop()
            top_list().append(done[1]); continue
        if re.match(r"if\s*\(.*\)\s*then\s*$", s, re.I | re.S):
            mif = re.match(r"if\s*\((.*)\)\s*then\s*$", s, re.I | re.S)
            blk = IfBlock(); blk.stmt = st
            try:
                cond = parse_expr(mif.group(1))
            except SyntaxError:
                cond = ("var", "__unparsed_cond__")
            blk.branches.append((cond, []))
            stack.append(("if", blk)); continue
        melif = re.match(r"else\s*if\s*\((.*)\)\s*then\s*$", s, re.I | re.S)
        if melif:
            blk = stack[-1][1]
            try:
                cond = parse_expr(melif.group(1))
            except SyntaxError:
                cond = ("var", "__unparsed_cond__")
            blk.branches.append((cond, [])); continue
        if sl == "else":
            stack[-1][1].branches.append((None, [])); continue
        if sl == "endif":
            done = stack.pop()
            top_list().append(done[1]); continue
        node = classify_stmt(st)
        if node is not None:
            top_list().append(node)
    return root

# ----------------------------------------------------------------------------
# Scope construction: inline the main T-loop body
# ----------------------------------------------------------------------------
TABLE_ARRAYS = {"chtab": "ch", "cltab": "cl", "crtab": "cr",
                "crphot": "crphot", "cst": "cst"}

def count_table_assigns(nodes):
    n = 0
    for node in nodes:
        if node is None: continue
        if node.kind == "assign" and node.lhs[0] == "aref" and \
           node.lhs[1] in TABLE_ARRAYS:
            n += 1
        elif node.kind == "if":
            for _, b in node.branches:
                n += count_table_assigns(b)
        elif node.kind == "do":
            n += count_table_assigns(node.body)
    return n

def build_scope(tree):
    best, besti = 0, None
    for i, node in enumerate(tree):
        if node is not None and node.kind == "do":
            c = count_table_assigns(node.body)
            if c > best:
                best, besti = c, i
    if besti is None:
        return list(tree)
    return list(tree[:besti]) + list(tree[besti].body) + list(tree[besti+1:])

# ----------------------------------------------------------------------------
# Slicer
# ----------------------------------------------------------------------------
INPUT_ARRAYS = {"temptab"}

def def_key(lhs):
    if lhs[0] == "var":
        return lhs[1]
    if lhs[0] == "aref":
        return ast_key(lhs)
    return None

def add_refs(ast, needed, flags):
    refs = set()
    expr_refs(ast, refs)
    for r in refs:
        if r[0] == "var":
            needed.add(r[1])
        else:
            if r[1] not in INPUT_ARRAYS:
                needed.add(r[2])
                flags.setdefault("arrrefs", set()).add(r[2])

def loop_defs(node, acc=None):
    if acc is None:
        acc = set()
    if node.kind == "do":
        body = node.body
    elif node.kind == "if":
        body = [n for _, b in node.branches for n in b]
    else:
        body = []
    for n in body:
        if n is None: continue
        if n.kind == "assign":
            dk = def_key(n.lhs)
            if dk: acc.add(dk)
        elif n.kind in ("if", "do"):
            loop_defs(n, acc)
    return acc

def slice_scope(nodes, needed, flags):
    kept = []
    for node in reversed(nodes):
        if node is None:
            continue
        if node.kind == "assign":
            dk = def_key(node.lhs)
            if dk in needed:
                add_refs(node.rhs, needed, flags)
                kept.append(node)
        elif node.kind == "if":
            sub, any_kept = [], False
            for cond, body in node.branches:
                pruned = slice_scope(body, needed, flags)
                sub.append((cond, pruned))
                if pruned:
                    any_kept = True
            if any_kept:
                for cond, _ in node.branches:
                    if cond is not None:
                        add_refs(cond, needed, flags)
                nb = IfBlock(); nb.stmt = node.stmt
                nb.branches = sub
                kept.append(nb)
        elif node.kind == "do":
            assigned = {k for k in loop_defs(node)
                        if k.split("(")[0] not in INPUT_ARRAYS}
            if assigned & needed:
                flags.setdefault("loops", []).append(node.stmt.l0)
        elif node.kind == "opaque":
            txt = node.stmt.text.lower()
            m = re.match(r"call\s+(\w+)", txt)
            if m:
                for key in list(needed):
                    base = key.split("(")[0]
                    if re.search(r"\b%s\b" % re.escape(base), txt):
                        flags.setdefault("calls", []).append(
                            (node.stmt.l0, m.group(1)))
                        break
    kept.reverse()
    return kept

# ----------------------------------------------------------------------------
# Named constants (kind-faithful)
# ----------------------------------------------------------------------------
def parse_named_constants():
    """Return (const_decl_lines, name->CxxName map)."""
    decls, names = [], {}
    macro_lit = {}
    with open(os.path.join(CANON, "chemcool_consts.h")) as f:
        for line in f:
            m = re.match(r"\s*#define\s+(\w+)\s+([\d.eEdD+-]+)\s*$", line)
            if m:
                macro_lit[m.group(1)] = m.group(2)
    # cool.h: REAL kboltz = BOLTZMANN ; eV = ELECTRON_VOLT (default-real macros)
    decls.append("static const double C_kboltz = %s;  /* cool.h <- BOLTZMANN "
                 "(default-real literal) */" % num_cxx(macro_lit["BOLTZMANN"]))
    names["kboltz"] = "C_kboltz"
    decls.append("static const double C_ev = %s;  /* cool.h <- ELECTRON_VOLT */"
                 % num_cxx(macro_lit["ELECTRON_VOLT"]))
    names["ev"] = "C_ev"
    decls.append("static const double C_temp_1ev = C_ev / C_kboltz;"
                 "  /* cheminmo.F parameter */")
    names["temp_1ev"] = "C_temp_1ev"
    # fs_data.h parameters
    fs_order = []
    with open(os.path.join(CANON, "fs_data.h")) as f:
        for line in f:
            m = re.match(r"\s*parameter\s*\(\s*(\w+)\s*=\s*(.*?)\)\s*$",
                         line.strip())
            if m:
                fs_order.append((m.group(1).lower(), m.group(2).strip()))
    fsnames = {n for n, _ in fs_order}
    for name, exprsrc in fs_order:
        ast = parse_expr(exprsrc)
        ctx = Ctx({n: "C_fs_%s" % n for n in fsnames}, {})
        try:
            cxx = cxx_expr(ast, ctx)
        except KeyError:
            continue
        decls.append("static const double C_fs_%s = %s;  /* fs_data.h */"
                     % (name, cxx))
        names[name] = "C_fs_%s" % name
    # kboltz shadow: fs_data.h also defines kb (double literal) -- distinct!
    return decls, names

# ----------------------------------------------------------------------------
# ifix pattern (chtab(K,itemp)=chtab(K,ifix) at T beyond fit validity)
# ----------------------------------------------------------------------------
def rewrite_ifix(kept, K, arrname):
    """Rewrite 'copy row ifix' as recursion marker; return fixlog source."""
    fixlog = [None]
    def scan(ns):
        out = []
        for n in ns:
            if n is None: continue
            if n.kind == "assign":
                dk = def_key(n.lhs)
                if dk == "ifix":
                    m = re.search(r"\(([\d.dDeE]+)/dtlog\)", ast_src(n.rhs))
                    if m:
                        fixlog[0] = m.group(1)
                    continue
                if n.lhs[0] == "aref" and \
                   ast_key(n.lhs) == "%s(%d,itemp)" % (arrname, K) and \
                   "ifix" in ast_src(n.rhs):
                    n = Assign(n.stmt, n.lhs, ("var", "__recurse_tfix__"))
                out.append(n)
            elif n.kind == "if":
                nb = IfBlock(); nb.stmt = n.stmt
                for cond, body in n.branches:
                    nb.branches.append((cond, scan(body)))
                out.append(nb)
            else:
                out.append(n)
        return out
    kept[:] = scan(kept)
    return fixlog[0]

# ----------------------------------------------------------------------------
# Code generation
# ----------------------------------------------------------------------------
def gen_function(rid, pfx, K, kept, const_names, fixlog, fprefix="rate_"):
    fname = "%s%s%d" % (fprefix, pfx, K)
    target_keys = {
        "ch": "chtab(%d,itemp)" % K, "cl": "cltab(%d,itemp)" % K,
        "cr": "crtab(%d)" % K, "crphot": "crphot(%d)" % K,
        "cst": "cst(%d)" % K}
    target_key = target_keys[pfx]
    localmap = OrderedDict()
    localmap["temp"] = "temp"
    localmap["itemp"] = "f_itemp(temp, g)"
    ctx = Ctx(const_names, localmap)
    lines = []
    declared = []

    def name_for(key):
        if key not in localmap:
            v = "v_" + re.sub(r"\W+", "_", key).strip("_")
            localmap[key] = v
            declared.append(v)
        return localmap[key]

    def branch_defs(ns, acc):
        for n in ns:
            if n is None: continue
            if n.kind == "assign":
                acc.add(def_key(n.lhs))
            elif n.kind == "if":
                for _, b in n.branches:
                    branch_defs(b, acc)

    def emit(ns, indent):
        pad = "    " * indent
        for n in ns:
            if n is None: continue
            if n.kind == "assign":
                dk = def_key(n.lhs)
                if dk == "temp" and n.rhs[0] == "aref" and \
                   n.rhs[1] == "temptab":
                    continue  # temp is the function argument
                var = name_for(dk)
                if n.rhs == ("var", "__recurse_tfix__"):
                    lines.append("%s%s = %s(pow(10.0, ((int)(1.0 + (%s / "
                                 "g.dtlog)) - 1) * g.dtlog), g);"
                                 % (pad, var, fname, num_cxx(fixlog)))
                    ctx.constset.discard(dk)
                    continue
                lines.append("%s%s = %s;" % (pad, var, cxx_expr(n.rhs, ctx)))
                if ast_is_const(n.rhs, ctx):
                    ctx.constset.add(dk)
                else:
                    ctx.constset.discard(dk)
            elif n.kind == "if":
                saved = set(ctx.constset)
                first = True
                for cond, body in n.branches:
                    if cond is not None:
                        kw = "if" if first else "} else if"
                        lines.append("%s%s (%s) {"
                                     % (pad, kw, cxx_expr(cond, ctx)))
                    else:
                        lines.append("%s} else {" % pad)
                    first = False
                    ctx.constset = set(saved)
                    emit(body, indent + 1)
                lines.append("%s}" % pad)
                # after the if: anything assigned in any branch is no longer
                # a single known constant
                assigned = set()
                for _, b in n.branches:
                    branch_defs(b, assigned)
                ctx.constset = saved - assigned
            elif n.kind == "do":
                raise RuntimeError("do-loop in slice")

    emit(kept, 1)
    if target_key not in localmap:
        raise RuntimeError("target %s never assigned" % target_key)
    body = []
    if declared:
        body.append("    double " +
                    ", ".join("%s = GUARD_NAN" % v for v in declared) + ";")
    body += lines
    body.append("    return %s;" % localmap[target_key])
    sig = "double %s(double temp, const ChemGlobals& g)" % fname
    return sig + ";", sig + " {\n" + "\n".join(body) + "\n}\n"

# ----------------------------------------------------------------------------
# Usage tagging
# ----------------------------------------------------------------------------
def scan_usage():
    usage = {}
    files = [("rate_eq_simple.F", (5,)), ("rate_eq_ulli.F", (17,)),
             ("cool_func.F", (5, 17)), ("cool_util.F", (5, 17)),
             ("calc_photo.F", (5, 17)), ("evolve_abundances.F", (5, 17))]
    pats = [("chtab", "ch"), ("dtchtab", "ch"), ("cltab", "cl"),
            ("dtcltab", "cl"), ("crtab", "cr"), ("crphot", "crphot"),
            ("cst", "cst")]
    for fn, nets in files:
        try:
            txt = open(os.path.join(CANON, fn)).read()
        except IOError:
            continue
        for arr, pfx in pats:
            for m in re.finditer(r"\b%s\s*\(\s*(\d+)" % arr, txt):
                usage.setdefault((pfx, int(m.group(1))), set()).update(nets)
    return usage

# ----------------------------------------------------------------------------
# RHS-inline audit extraction
# ----------------------------------------------------------------------------
RHS_LHS = re.compile(
    r"^(ydot|rates_chem|rates|h_gr|hgrvar1|hgrvar2|phi|beta|k0|k1|gamma_chx|"
    r"gamma_co|cdex|rdex|fpheat|ncrinv|h2var0|h2var1|h2var2|ykdh|ykdh2|ykdis|"
    r"cdh2_h|cdh2_h_dt|cdh2_h2|cdh2_he|h2_ct_low|ncrit_h2p_h|ncrit_h2p_e|"
    r"ncrit_h2p_inv|h2p_alpha|h2p_dr|h2p_cdh|ph4|ph5|h2p_pd_low|h2p_pd_lte|"
    r"num|denom|abm|f_ex|f_tot|kd1|kd2|kd3)\s*(\(|$)", re.I)

def extract_rhs_inline(fname, nets, macros):
    stmts = read_statements(os.path.join(CANON, fname), macros)
    entries = []
    for st in stmts:
        m = re.match(r"([A-Za-z_]\w*(\s*\([^=]*?\))?)\s*=\s*(.*)$", st.text, re.S)
        if not m:
            continue
        lhs = m.group(1).replace(" ", "")
        if not RHS_LHS.match(lhs):
            continue
        if re.match(r"(call|goto|print|write|if)\b", st.text, re.I):
            continue
        entries.append({
            "rate_id": "%s:%s@%d" % (fname.replace(".F", ""), lhs, st.l0),
            "kind": "RHS_INLINE", "networks": list(nets),
            "fortran_file": fname, "lines": [st.l0, st.l1],
            "expression": [st.text],
            "comment": st.comment[:400]})
    return entries

# ----------------------------------------------------------------------------
# Guard-boundary collection (for the off-grid golden T list)
# ----------------------------------------------------------------------------
def collect_guards(kept, acc, temp_1ev):
    def num_val(n):
        return float(re.sub(r"[dD]", "e", n[1]))
    def from_cond(cond):
        if cond is None or cond[0] != "rel":
            return
        a, b = cond[2], cond[3]
        for x, y in ((a, b), (b, a)):
            if x[0] == "var" and y[0] == "num":
                if x[1] == "temp":
                    acc.add(num_val(y))
                elif x[1] == "te":
                    acc.add(num_val(y) * temp_1ev)
    def walk(ns):
        for n in ns:
            if n is None: continue
            if n.kind == "if":
                for cond, body in n.branches:
                    from_cond(cond)
                    walk(body)
    walk(kept)

def write_custom_grid(guards, nmd, path):
    import math
    eps_list = (1e-9, 1e-12)
    pts = set()
    for b in guards:
        for e in eps_list:
            for s in (1.0 - e, 1.0, 1.0 + e):
                v = b * s
                if 1.0 <= v <= 1e8:
                    pts.add(v)
    pts = sorted(pts)
    nfill = nmd - len(pts)
    fill = [10.0 ** (8.0 * i / (nfill - 1)) for i in range(nfill)]
    allpts = sorted(set(pts) | set(fill))
    # ensure strictly increasing, pad/trim to exactly nmd
    while len(allpts) < nmd:
        allpts.append(allpts[-1] * 1.0001)
    allpts = allpts[:nmd]
    with open(path, "w") as f:
        for v in allpts:
            f.write("%.17e\n" % v)

# ----------------------------------------------------------------------------
# Audit helpers
# ----------------------------------------------------------------------------
def branches_summary(nodes, target_key=None):
    """Human-readable expression summary.  If target_key is given, restrict
    to the assignments of that table element (and their guard structure)."""
    def contains_target(ns):
        for n in ns:
            if n is None: continue
            if n.kind == "assign" and def_key(n.lhs) == target_key:
                return True
            if n.kind == "if" and any(contains_target(b)
                                      for _, b in n.branches):
                return True
        return False
    lines = []
    def walk(ns, indent):
        for n in ns:
            if n is None: continue
            if n.kind == "assign":
                if target_key is None or def_key(n.lhs) == target_key:
                    lines.append("%s%s = %s" % ("  " * indent,
                                                ast_src(n.lhs),
                                                ast_src(n.rhs)))
            elif n.kind == "if":
                if target_key is not None and not contains_target(
                        [x for _, b in n.branches for x in b]):
                    continue
                first = True
                for cond, body in n.branches:
                    if cond is not None:
                        lines.append("%s%s %s:" % ("  " * indent,
                                     "if" if first else "elseif",
                                     ast_src(cond)))
                    else:
                        lines.append("%selse:" % ("  " * indent))
                    first = False
                    walk(body, indent + 1)
    walk(nodes, 0)
    return lines

def find_table_assigns(scope):
    """(pfx,K) -> [(in_residual_loop, node)] for every constant-index table
    assignment in scope."""
    found = OrderedDict()
    def walk(nodes, in_loop):
        for node in nodes:
            if node is None: continue
            if node.kind == "assign" and node.lhs[0] == "aref" and \
               node.lhs[1] in TABLE_ARRAYS:
                idx0 = node.lhs[2][0]
                if idx0[0] == "num" and num_kind(idx0[1]) == "int":
                    key = (TABLE_ARRAYS[node.lhs[1]], int(idx0[1]))
                    found.setdefault(key, []).append((in_loop, node))
            elif node.kind == "if":
                for _, b in node.branches:
                    walk(b, in_loop)
            elif node.kind == "do":
                walk(node.body, True)
    walk(scope, False)
    return found

# ----------------------------------------------------------------------------
# network 16 (Gong+17): rates whose CHEMISTRYNETWORK == 16 code differs from
# the net-5 build, or that exist only in net 16 -> rate16_* functions and
# rates_registry16.inc (applied after the base registry by chem_tables_build).
# ----------------------------------------------------------------------------
def extract_net16(macros5, const_names, base_defn, records, gen_decls,
                  gen_defns, counts):
    macros16 = dict(macros5)
    macros16["CHEMISTRYNETWORK"] = "16"
    registry16 = []
    counts["generated_net16"] = 0
    for fname in ("cheminmo.F", "const_rates.F"):
        scope = build_scope(build_tree(
            read_statements(os.path.join(CANON, fname), macros16)))
        for (pfx, K), nodes in sorted(find_table_assigns(scope).items()):
            rid = "%s%d" % (pfx, K)
            arrname = {"ch": "chtab", "cl": "cltab", "cr": "crtab",
                       "crphot": "crphot", "cst": "cst"}[pfx]
            tkey = "%s(%d,itemp)" % (arrname, K) if pfx in ("ch", "cl") \
                else "%s(%d)" % (arrname, K)
            flags = {}
            kept = slice_scope(scope, {tkey}, flags)
            fixlog = None
            skip = any(in_loop for in_loop, _ in nodes) or \
                flags.get("loops") or flags.get("calls")
            if not skip and any(re.match(r"(chtab|cltab)\(\d+,ifix\)", u)
                                for u in flags.get("arrrefs", set())):
                fixlog = rewrite_ifix(kept, K, arrname)
                skip = fixlog is None
            if skip:
                if rid not in base_defn:
                    print("NOTE net16: %s not extractable (TABLE/SPECIAL)" % rid)
                continue
            try:
                decl, defn = gen_function(rid, pfx, K, kept, const_names,
                                          fixlog, "rate16_")
            except (KeyError, RuntimeError) as e:
                print("NOTE net16: %s codegen failed: %s" % (rid, e))
                continue
            if defn.replace("rate16_", "rate_") == base_defn.get(rid):
                continue
            lines_span = [min(n.stmt.l0 for _, n in nodes),
                          max(n.stmt.l1 for _, n in nodes)]
            gen_decls.append(decl)
            gen_defns.append("/* %s  %s:%d-%d  nets:16 */\n%s"
                             % (rid, fname, lines_span[0], lines_span[1], defn))
            grid_only = 1 if (fixlog is not None or "f_itemp(" in defn) else 0
            registry16.append('    {"%s", "%s", %d, rate16_%s%d, %d},'
                              % (rid, pfx, K, pfx, K, grid_only))
            comment = next((n.stmt.comment for _, n in nodes
                            if n.stmt.comment), "")
            records.append({"rate_id": "%s@net16" % rid, "networks": [16],
                            "fortran_file": fname, "lines": lines_span,
                            "comment": comment[:600], "kind": "GENERATED",
                            "expression": branches_summary(kept, tkey)[:40]})
            counts["generated_net16"] += 1
    return registry16

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    macros5 = load_config_macros(CFG5)
    macros17 = load_config_macros(CFG17)
    macros = dict(macros5)

    const_decls, const_names = parse_named_constants()
    usage = scan_usage()

    records, gen_decls, gen_defns, registry = [], [], [], []
    base_defn = {}   # rid -> net-5 function text, for the net-16 pass
    counts = {"generated": 0, "TABLE": 0, "SPECIAL": 0, "RHS_INLINE": 0}
    guard_temps = set()
    # default-real macro values as gfortran sees them (float-rounded)
    temp_1ev_d = 1.60219e-12 / 1.38066e-16

    for fname in ("cheminmo.F", "coolinmo.F", "const_rates.F"):
        stmts = read_statements(os.path.join(CANON, fname), macros)
        tree = build_tree(stmts)
        scope = build_scope(tree)

        # locate every constant-index table assignment (with loop-nesting info)
        found = find_table_assigns(scope)

        for (pfx, K), nodes in sorted(found.items()):
            rid = "%s%d" % (pfx, K)
            lines_span = [min(n.stmt.l0 for _, n in nodes),
                          max(n.stmt.l1 for _, n in nodes)]
            nets = sorted(usage.get((pfx, K), set()))
            comment = next((n.stmt.comment for _, n in nodes
                            if n.stmt.comment), "")
            rec = {"rate_id": rid, "networks": nets, "fortran_file": fname,
                   "lines": lines_span, "comment": comment[:600]}
            records.append(rec)

            arrname = {"ch": "chtab", "cl": "cltab", "cr": "crtab",
                       "crphot": "crphot", "cst": "cst"}[pfx]
            tkey = "%s(%d,itemp)" % (arrname, K) if pfx in ("ch", "cl") \
                else "%s(%d)" % (arrname, K)
            needed, flags = {tkey}, {}
            kept = slice_scope(scope, needed, flags)
            rec["expression"] = branches_summary(kept, tkey)[:40]

            if any(in_loop for in_loop, _ in nodes):
                rec["kind"] = "TABLE"
                rec["why"] = ("assigned inside a residual do-loop (spline/"
                              "DATA-table construction outside the analytic "
                              "T-loop)")
                counts["TABLE"] += 1
                continue
            if flags.get("loops"):
                rec["kind"] = "TABLE"
                rec["why"] = ("value overwritten/consumed by DATA-driven "
                              "do-loop(s) at line(s) %s" %
                              sorted(set(flags["loops"])))
                counts["TABLE"] += 1
                continue
            if flags.get("calls"):
                rec["kind"] = "SPECIAL"
                rec["why"] = "depends on subroutine call(s): %s" % flags["calls"]
                counts["SPECIAL"] += 1
                continue

            fixlog = None
            if any(re.match(r"(chtab|cltab)\(\d+,ifix\)", u)
                   for u in flags.get("arrrefs", set())):
                fixlog = rewrite_ifix(kept, K, arrname)
                if fixlog is None:
                    rec["kind"] = "SPECIAL"
                    rec["why"] = "unrecognized ifix cross-row pattern"
                    counts["SPECIAL"] += 1
                    continue
                rec["note"] = ("T>fit-range handled by re-evaluating at the "
                               "fixed table temperature 10^%s-ish (Fortran "
                               "row-copy at ifix)" % fixlog)

            try:
                decl, defn = gen_function(rid, pfx, K, kept, const_names,
                                          fixlog)
            except (KeyError, RuntimeError) as e:
                rec["kind"] = "SPECIAL"
                rec["why"] = "codegen: %s" % e
                counts["SPECIAL"] += 1
                continue
            gen_decls.append(decl)
            base_defn[rid] = defn
            gen_defns.append("/* %s  %s:%d-%d  nets:%s */\n%s"
                             % (rid, fname, lines_span[0], lines_span[1],
                                ",".join(map(str, nets)) or "-", defn))
            # grid_only: Fortran semantics tied to the table INDEX (row-copy
            # at ifix, or guards written on itemp) -- exact only on the
            # standard grid, excluded from the off-grid comparison
            grid_only = 1 if (fixlog is not None or "f_itemp(" in defn) else 0
            if grid_only:
                rec["grid_only"] = True
            collect_guards(kept, guard_temps, temp_1ev_d)
            registry.append('    {"%s", "%s", %d, rate_%s%d, %d},'
                            % (rid, pfx, K, pfx, K, grid_only))
            rec["kind"] = "GENERATED"
            counts["generated"] += 1

    registry16 = extract_net16(macros5, const_names, base_defn, records,
                               gen_decls, gen_defns, counts)

    for fname, nets, mac in (("rate_eq_simple.F", (5,), macros5),
                             ("rate_eq_ulli.F", (17,), macros17),
                             ("cool_func.F", (5, 17), macros5)):
        ent = extract_rhs_inline(fname, nets, mac)
        counts["RHS_INLINE"] += len(ent)
        records.extend(ent)

    with open(os.path.join(OUT, "rates.json"), "w") as f:
        json.dump({"counts": counts, "records": records}, f, indent=1)

    with open(os.path.join(OUT, "rates_gen.h"), "w") as f:
        f.write(HEADER_TMPL.format(decls="\n".join(gen_decls)))
    with open(os.path.join(OUT, "rates_gen.cc"), "w") as f:
        f.write(SOURCE_TMPL.format(consts="\n".join(const_decls),
                                   defns="\n".join(gen_defns)))
    with open(os.path.join(OUT, "rates_registry.inc"), "w") as f:
        f.write("\n".join(registry) + "\n")
    with open(os.path.join(OUT, "rates_registry16.inc"), "w") as f:
        f.write("\n".join(registry16) + "\n")

    nmd = 10000
    with open(os.path.join(CANON, "chemcool_consts.h")) as f:
        m = re.search(r"#define\s+NMD\s+(\d+)", f.read())
        if m:
            nmd = int(m.group(1))
    write_custom_grid(guard_temps, nmd,
                      os.path.join(OUT, "golden", "custom_grid.txt"))

    write_audit(records, counts)
    print(json.dumps(counts))
    print("records:", len(records))

def write_audit(records, counts):
    results = {}
    respath = os.path.join(OUT, "golden", "golden_results.txt")
    if os.path.exists(respath):
        for line in open(respath):
            p = line.split()
            if len(p) >= 3 and not p[0].startswith("#"):
                results[p[0]] = (p[1], p[2])
    with open(os.path.join(OUT, "RATE_AUDIT.md"), "w") as f:
        f.write("# CHEMCOOL rate-layer audit -- Gate 1 (Tier-2 deliverable)\n\n")
        f.write("Machine-extracted from canonical Fortran at `%s` "
                "(READ-ONLY).\nGenerated by `extract_rates.py`; do not edit "
                "by hand.\n\n" % CANON)
        f.write("- kind=GENERATED: analytic k(T) extracted and emitted to "
                "rates_gen.cc, golden-tested per rate.\n")
        f.write("- kind=TABLE: DATA-table / spline-driven entry; location "
                "recorded, NOT transcribed (per spec).\n")
        f.write("- kind=SPECIAL: construct resisting extraction (call "
                "dependence etc.); location recorded.\n")
        f.write("- kind=RHS_INLINE: state-dependent rate term inside the "
                "network RHS (rate_eq/cool_func); audited here, ported and "
                "tested at Gate 2.\n\n")
        f.write("Counts: `%s`\n\n" % json.dumps(counts))
        if results:
            f.write("Golden column: comparison of the generated C++ function "
                    "against the Fortran-built table (prebuilt onezone "
                    "build_net5 objects) at all 10000 temptab points "
                    "(T = 1..1e8 K), tolerance 1e-12 relative.\n\n")
        f.write("| rate | kind | nets | source | golden | max relerr | "
                "expression / note | citation comment |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in records:
            rid = r["rate_id"]
            loc = "%s:%d-%d" % (r["fortran_file"], r["lines"][0], r["lines"][1])
            gr = results.get(rid, ("-", "-"))
            expr = "<br>".join(x.strip().replace("|", "\\|")
                               for x in r.get("expression", []))[:500]
            if r.get("why"):
                expr = ("**%s**<br>" % r["why"].replace("|", "\\|")) + expr
            if r.get("note"):
                expr += "<br>_%s_" % r["note"].replace("|", "\\|")
            cite = (r.get("comment", "").split("\n")[0][:120]
                    .replace("|", "\\|"))
            f.write("| %s | %s | %s | %s | %s | %s | %s | %s |\n"
                    % (rid, r.get("kind", "?"),
                       ",".join(str(n) for n in r.get("networks", [])) or "-",
                       loc, gr[0], gr[1], expr, cite))

HEADER_TMPL = """// AUTO-GENERATED by extract_rates.py -- DO NOT EDIT.
// CHEMCOOL rate layer (production networks 5 & 17), machine-extracted from
// the canonical Fortran sources. One function per rate coefficient.
#ifndef CHEMCOOL_RATES_GEN_H
#define CHEMCOOL_RATES_GEN_H

struct ChemGlobals {{
    int iflag_mn, iflag_ad, iflag_atom, iflag_3bh2a, iflag_3bh2b, iflag_h3pra;
    double deff, dust_to_gas_ratio, phi_pah, cosmic_ray_ion_rate;
    double abhe;             /* /chem_comp/ abhe as cheminmo sees it */
    double dtlog, tmin, tmax;
}};

{decls}

#endif
"""

SOURCE_TMPL = """// AUTO-GENERATED by extract_rates.py -- DO NOT EDIT.
#include <cmath>
#include <limits>
#include "rates_gen.h"

using std::pow; using std::exp; using std::log; using std::log10;
using std::sqrt; using std::fabs;

#define GUARD_NAN std::numeric_limits<double>::quiet_NaN()

// Fortran MAX/MIN on doubles (comparison semantics)
static inline double f_max(double a, double b) {{ return a > b ? a : b; }}
static inline double f_min(double a, double b) {{ return a < b ? a : b; }}

// Fortran table index for guards written on itemp (exact at grid points)
static inline int f_itemp(double temp, const ChemGlobals& g) {{
    if (temp <= g.tmin) return 1;
    double x = log10(temp) / g.dtlog;
    return (int)(x + 0.5) + 1;
}}

// Integer powering with EXACTLY the gfortran frontend's expansion of
// REAL**INTEGER (gfc_conv_cst_int_power):  powi(n) = powi(n/2)*powi(n-n/2)
// -- e.g. x**5 = x2*x3, x**7 = x3*x4.  This differs from gcc's C-side
// powi_as_mults (odd n: x*powi(n-1)) for n = 5, 7, 9..., and the last-ulp
// association difference is resolvable by the 1e-12 golden tolerance in
// cancellation-heavy polynomials (verified against the disassembly of the
// production cheminmo.o).
static inline double fpowi(double x, int n) {{
    if (n == 0) return 1.0;
    if (n < 0)  return 1.0 / fpowi(x, -n);
    if (n == 1) return x;
    double a = fpowi(x, n / 2);
    double b = fpowi(x, n - n / 2);
    return a * b;
}}

// named constants (kind-faithful: Fortran default-real literals kept single
// precision so gfortran's constant rounding is reproduced bit-exactly)
{consts}

{defns}
"""

if __name__ == "__main__":
    main()
