"""Rebuild convergence plots from a test log, without rerunning anything.

The order sweeps in test/binary and test/triple print each (eta, error) point and the fitted
exponent, and now also save them to summary_<variant>.npz. Logs from before that change -- and
logs from runs whose npz was overwritten by a later run -- still carry the numbers, which is
enough: a sweep costs minutes of compute and this costs milliseconds.

    python test/plot_convergence_from_log.py sinktests_slurm-6938997.out [-o outdir]

Emits <test>_<variant>_convergence.png for every sweep it finds.
"""
import argparse
import re
from os import path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import numpy as np                                    # noqa: E402
from matplotlib import pyplot as plt                  # noqa: E402

# "test/binary/test_binary.py::test_binary[starforge_defaults-1-1]" -- the trailing rank/thread
# fields vary, so keep only the variant id, matching what the tests name their outputs.
CASE = re.compile(r"^(test/([a-z_]+)/test_[a-z_]+\.py)::[a-z_]+\[([a-z_0-9]+)[-\]]")
POINT = re.compile(r"sweep eta=\s*([0-9.eE+-]+)\s+\|dE/E\| envelope\s+([0-9.eE+-]+)")


def parse(logfile):
    """[(test, variant, etas, errs)] in the order they appear."""
    out, cur, pts = [], None, []
    with open(logfile, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            m = CASE.match(line)
            if m:
                if cur and pts:
                    out.append((*cur, np.array([p[0] for p in pts]),
                                np.array([p[1] for p in pts])))
                cur, pts = (m.group(2), m.group(3)), []
                continue
            m = POINT.search(line)
            if m:
                pts.append((float(m.group(1)), float(m.group(2))))
    if cur and pts:
        out.append((*cur, np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    return out


def plot(test, variant, etas, errs, outdir):
    order = np.polyfit(np.log(etas), np.log(errs), 1)[0]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(etas, errs, "o", ms=7, zorder=3)
    xs = np.array([etas.min() / 1.6, etas.max() * 1.6])
    c = np.polyfit(np.log(etas), np.log(errs), 1)
    ax.loglog(xs, np.exp(np.polyval(c, np.log(xs))), "-", lw=1.4, zorder=2,
              label=f"fit: $\\eta^{{{order:.2f}}}$  (dt$^{{{2*order:.1f}}}$)")
    # the reference slopes are the point: a 4th-order scheme fed drifted source positions
    # degrades to 2nd, so which line the points follow is the diagnostic, not straightness
    for p, lab in ((1.0, "2nd order  $\\eta^1$"), (2.0, "4th order  $\\eta^2$")):
        ax.loglog(xs, errs[0] * (xs / etas[0]) ** p, "--", lw=1.0, zorder=1, label=lab)
    ax.set_xlabel("ErrTolIntAccuracy")
    ax.set_ylabel("|E/E$_0$ - 1|  (per-orbit envelope)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fn = path.join(outdir, f"{test}_{variant}_convergence.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn, order


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("logfile")
    p.add_argument("-o", "--outdir", default=".")
    a = p.parse_args()
    found = parse(a.logfile)
    if not found:
        raise SystemExit(f"no sweep data in {a.logfile}")
    for test, variant, etas, errs in found:
        if len(etas) < 2:
            print(f"  {test}[{variant}]: only {len(etas)} point(s), skipping")
            continue
        fn, order = plot(test, variant, etas, errs, a.outdir)
        pts = "  ".join(f"{e:.3g}->{v:.3e}" for e, v in zip(etas, errs))
        print(f"  {test}[{variant}]: eta^{order:.2f} (dt^{2*order:.1f})   {pts}")
        print(f"    -> {fn}")
