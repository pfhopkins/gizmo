#!/usr/bin/env python3
"""Lists every Config flag and parameter in the given files that the source tree does not read.

usage: check_config_reads.py SRC_DIR FILE...   (Config*.sh and *.params; exit 1 if anything is unread)
A flag counts as read when its name appears in src/ outside Template_Config.sh; a parameter when begrun.cc
registers its tag outside DEVELOPER_MODE. A parameter file tag the code does not know is skipped by GIZMO with a warning only.
"""
import re, subprocess, sys
from pathlib import Path

src = Path(sys.argv[1])
files = [Path(f) for f in sys.argv[2:]]


def grep_words(names, exclude=()):
    """names found as whole words in src/ (one pass over the tree)."""
    found, pat = set(), re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix not in (".h", ".cc", ".cpp", ".c", ".hpp", ".F", ".f90", ".py", ".inc") or p.name in exclude:
            continue
        found |= set(pat.findall(p.read_text(errors="ignore"))) & names
    return found


flags, params = {}, {}
for f in files:
    for line in f.read_text().splitlines():
        if f.name.startswith("Template") and re.match(r"#[A-Z]", line):
            line = line[1:]  # a template lists its options as #FLAG lines
        s = re.sub(r"\s*[%#].*$", "", line).strip()
        if not s:
            continue
        if f.suffix == ".params":
            params.setdefault(s.split()[0], []).append(f.name)
        else:
            flags.setdefault(re.split(r"[=\s]", s)[0], []).append(f.name)

tags, dev = set(), []  # tags registered outside #ifdef DEVELOPER_MODE (a production build does not read the others)
for line in (src / "core/begrun.cc").read_text().splitlines():
    t = line.strip()
    if t.startswith("#if"):
        dev.append(bool(re.match(r"#ifdef\s+DEVELOPER_MODE\b|#if\s+defined\(DEVELOPER_MODE\)", t)))
    elif t.startswith("#else") and dev:
        dev[-1] = False
    elif t.startswith("#endif") and dev:
        dev.pop()
    m = re.search(r'strcpy\(\s*(?:alternate_)?tag\[nt\]\s*,\s*"([A-Za-z0-9_]+)"', line)
    if m and not any(dev):
        tags.add(m.group(1))
read_flags = grep_words(set(flags), exclude=("Template_Config.sh",))
bad = [(n, w, "flag") for n, w in flags.items() if n not in read_flags]
bad += [(n, w, "parameter") for n, w in params.items() if n not in tags]
for n, w, k in sorted(bad):
    print(f"UNREAD {k}: {n}  in {', '.join(sorted(set(w)))}")
print(f"{len(flags)} flags, {len(params)} parameters checked, {len(bad)} unread")
sys.exit(1 if bad else 0)
