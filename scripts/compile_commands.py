#!/usr/bin/env python3
"""Write compile_commands.json for the current Config.sh, for clang-tidy and clangd.

Reads the compile lines from a dry run (`make -n -B`), so nothing is built. Keeps only the
preprocessor/language flags (-I -D -U -include -std -fopenmp) so clang does not choke on
gcc-only optimisation flags, and adds the MPI wrapper's include paths.

    python3 scripts/compile_commands.py [extra make args, e.g. HDF5INCL=...]
    clang-tidy -p . core/main.cc           # checks from .clang-tidy; clangd uses both files
"""
import json
import os
import shlex
import subprocess
import sys

KEEP_WITH_ARG = {"-I", "-D", "-U", "-include", "-isystem"}
KEEP_PREFIX = ("-I", "-D", "-U", "-std=", "-isystem")


def mpi_flags(wrapper):
    """Include/define flags the MPI compiler wrapper adds (Open MPI or MPICH style)."""
    for opt in (["--showme:compile"], ["-compile_info"], ["-show"]):
        try:
            out = subprocess.run([wrapper] + opt, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 universal_newlines=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        return [t for t in shlex.split(out) if t.startswith(("-I", "-D", "-pthread"))]
    return []


def keep_flags(tokens):
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in KEEP_WITH_ARG and i + 1 < len(tokens):
            out += [t, tokens[i + 1]]
            i += 2
            continue
        if t.startswith(KEEP_PREFIX) or t == "-fopenmp":
            out.append(t)
        i += 1
    return out


def main():
    root = os.getcwd()
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"] + sys.argv[1:], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, universal_newlines=True, check=True).stdout
    entries, mpi = [], {}
    for line in dry.splitlines():
        try:
            tok = shlex.split(line)
        except ValueError:
            continue
        if "-c" not in tok or "-o" not in tok:
            continue
        src = tok[tok.index("-c") + 1]
        if not src.endswith(".cc") or not os.path.exists(src):
            continue  # skips generated sources such as compile_time_info.cc
        if tok[0] not in mpi:
            mpi[tok[0]] = mpi_flags(tok[0])
        args = ["c++"] + mpi[tok[0]] + keep_flags(tok[1:]) + ["-c", src]
        entries.append({"directory": root, "file": src, "arguments": args})
    if not entries:
        sys.exit("no compile lines found in `make -n -B` output")
    with open("compile_commands.json", "w") as f:
        json.dump(entries, f, indent=1)
    print(f"compile_commands.json: {len(entries)} translation units")


if __name__ == "__main__":
    main()
