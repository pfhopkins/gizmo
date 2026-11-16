"""Mass- and energy-trace closure checks for Stella/GIZMO runs.

Reads <output>/mass_balance_steps.txt (always written) and <output>/energy_step_trace.txt
(OUTPUT_ENERGY_STEP_TRACE) and turns every violation into a finding (kind, stage, count, magnitude).

Bars (fixed, see reports/energy_step_trace.md; do not loosen per test):
  mass, metal ledger : |delta| / value <= 1e-12 at every checkpoint (round-off; nothing in a step may
                       create or destroy baryons or metals)
  exact energy stages: the code writes a STAGE line only when |dE - booked| > 1e-12 (E_abs + |booked|),
                       so every STAGE line is a finding
  time integration   : |int_res| / E_scale <= 1e-2 at every full sync (kick-drift-kick + tree gravity
                       accuracy; int_res excludes all stage changes)

    python -m gizmo.conservation_trace <output_dir>
"""

import math
import os
import sys
from collections import OrderedDict

MASS_BAR = 1.0e-12
INTEGRATION_BAR = 1.0e-2


def _kv(line):
    out = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


class TraceReport:
    def __init__(self):
        self.findings = []
        self.notes = []

    @property
    def ok(self):
        return not self.findings

    def summary(self):
        lines = [f"{len(self.findings)} conservation finding(s)"]
        for f in self.findings:
            lines.append(f"  {f['kind']:<18s} {f['stage']:<18s} n={f['count']:<6d} max|rel|={f['max_rel']:.3e} "
                         f"sum_res={f['sum_res']:+.3e} first_t={f['first_t']:.6e}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _add(groups, kind, stage, rel, res, t):
    g = groups.setdefault((kind, stage), {"kind": kind, "stage": stage, "count": 0, "max_rel": 0.0, "sum_res": 0.0, "first_t": t})
    g["count"] += 1
    g["max_rel"] = max(g["max_rel"], abs(rel))
    g["sum_res"] += res


def _check_mass(path, groups, rep, bar):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    for r in rows[1:]:  # first row is the baseline
        tag, t, M, dM, L, dL = r[1], float(r[2]), float(r[3]), float(r[4]), float(r[-2]), float(r[-1])
        if M > 0 and abs(dM) / M > bar:
            _add(groups, "mass", tag, dM / M, dM, t)
        if L > 0 and abs(dL) / L > bar:
            _add(groups, "metal_ledger", tag, dL / L, dL, t)
    rep.notes.append(f"mass trace: {len(rows)} checkpoint rows")


def _check_energy(path, groups, rep, bar):
    pot_valid = True
    n_sync = 0
    worst = None
    for line in open(path):
        if line.startswith("#"):
            if "pot_valid=0" in line:
                pot_valid = False
            continue
        if line.startswith("STAGE"):
            d = _kv(line)
            _add(groups, "energy_stage", d["stage"], d["res"] / d["E_abs"], d["res"], d["t"])
        elif line.startswith("SYNC"):
            d = _kv(line)
            n_sync += 1
            r, s = d.get("int_res", float("nan")), d.get("E_scale", 0.0)
            if not pot_valid or not isinstance(r, float) or math.isnan(r) or s <= 0:
                continue
            if worst is None or abs(r) / s > worst[0]:
                worst = (abs(r) / s, d)
            if abs(r) / s > bar:
                _add(groups, "energy_integration", "kick_drift_grav", r / s, 0.0, d["t"])
    for g in groups.values():  # integration residual is cumulative: report the latest value, not a sum
        if g["kind"] == "energy_integration":
            g["sum_res"] = float("nan")
    if not pot_valid:
        rep.notes.append("integration not checked: no valid potential (EVALPOTENTIAL off, comoving, or adaptive tree-force)")
    elif worst is not None:
        rep.notes.append(f"integration: worst |int_res|/E_scale = {worst[0]:.3e} at t={worst[1]['t']:.6e} ({n_sync} syncs)")
    if n_sync == 0:
        rep.notes.append("energy trace has no SYNC line")


def check_conservation_traces(output_dir, require_energy=True, mass_bar=MASS_BAR, integration_bar=INTEGRATION_BAR):
    """Return a TraceReport; findings are ranked by largest relative magnitude."""
    rep = TraceReport()
    groups = OrderedDict()
    mpath = os.path.join(output_dir, "mass_balance_steps.txt")
    epath = os.path.join(output_dir, "energy_step_trace.txt")
    if os.path.exists(mpath):
        _check_mass(mpath, groups, rep, mass_bar)
    else:
        groups[("missing", "mass")] = {"kind": "missing", "stage": "mass_trace", "count": 1, "max_rel": float("inf"), "sum_res": 0.0, "first_t": 0.0}
    if os.path.exists(epath):
        _check_energy(epath, groups, rep, integration_bar)
    elif require_energy:
        groups[("missing", "energy")] = {"kind": "missing", "stage": "energy_trace", "count": 1, "max_rel": float("inf"), "sum_res": 0.0, "first_t": 0.0}
    rep.findings = sorted(groups.values(), key=lambda g: -g["max_rel"])
    return rep


TRACE_FILES = ("mass_balance_steps.txt", "energy_step_trace.txt")


def keep_and_check_traces(run_output_dir, dest_dir, **kw):
    """Copy the trace files of a run (before its run directory is deleted) and check them."""
    import shutil
    os.makedirs(dest_dir, exist_ok=True)
    for f in TRACE_FILES:
        if os.path.exists(os.path.join(run_output_dir, f)) and os.path.abspath(run_output_dir) != os.path.abspath(dest_dir):
            shutil.copy(os.path.join(run_output_dir, f), dest_dir)
    rep = check_conservation_traces(dest_dir, **kw)
    with open(os.path.join(dest_dir, "conservation_trace_summary.txt"), "w") as fh:
        fh.write(rep.summary() + "\n")
    print(rep.summary())
    return rep


def assert_conservation_traces(output_dir, **kw):
    """Assert mass- and energy-trace closure; the message lists every finding."""
    rep = check_conservation_traces(output_dir, **kw)
    assert rep.ok, rep.summary()
    return rep


if __name__ == "__main__":
    r = check_conservation_traces(sys.argv[1], require_energy=("--no-energy" not in sys.argv))
    print(r.summary())
    sys.exit(0 if r.ok else 1)
