"""single_star_imf unit test: the IMF draw of galaxy_sf/single_star_imf.h on the keyed generator of gpu_rng.h.
  1. 10^6 draws (keys 1..10^6) follow the mass-weighted Kroupa IMF on [m_min, 350] Msun (Kolmogorov-Smirnov against
     the analytic CDF), for the full IMF (m_min = 0.08) and a truncated one (m_min = 4);
  2. weighting each draw by 1/M (the star-formation acceptance) gives the number-weighted Kroupa mean mass;
  3. a draw depends only on (key, event): the same arguments give bitwise the same masses, another event other masses;
  4. the draws follow the same distribution as the reference implementation's rejection sampler (two-sample
     Kolmogorov-Smirnov, 10^6 draws each, full and truncated IMF).
No GIZMO build or run is needed."""
import subprocess
from pathlib import Path
import numpy as np
from scipy import stats, integrate

TEST_DIR = Path(__file__).parent
SRC = TEST_DIR.parent.parent / "src"
A, MMAX = 0.126512, 350.0


def massweighted(m):
    return np.where(m < 0.5, 2 * A * m ** -0.3, A * m ** -1.3)


def cdf_factory(m_min):
    def F(x):  # integral of the mass-weighted IMF from m_min to x, normalised
        def G(y):
            y = np.asarray(y, dtype=float)
            lo = 2 * A * (np.minimum(y, 0.5) ** 0.7) / 0.7
            hi = np.where(y > 0.5, A * (0.5 ** -0.3 - y ** -0.3) / 0.3, 0.0)
            return lo + hi
        x = np.clip(x, m_min, MMAX)
        return (G(x) - G(m_min)) / (G(MMAX) - G(m_min))
    return F


def draws(n, m_min, event, tmp_path, reference=False):
    exe, out = tmp_path / "imf_draws", tmp_path / f"d_{m_min}_{event}_{int(reference)}"
    if not exe.exists():
        subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(SRC), str(TEST_DIR / "imf_draws.cc"), "-o", str(exe)], check=True)
    subprocess.run([str(exe), str(n), str(m_min), str(event), str(out)] + (["reference"] if reference else []), check=True)
    return np.fromfile(out, dtype=np.float64)


def test_kroupa_massweighted(tmp_path):
    for m_min in (0.08, 4.0):
        m = draws(10**6, m_min, 12345, tmp_path)
        assert m.min() >= m_min and m.max() <= MMAX
        ks = stats.kstest(m, cdf_factory(m_min))
        assert ks.pvalue > 1e-3, (m_min, ks)


def test_number_weighted_mean(tmp_path):
    m = draws(10**6, 0.08, 777, tmp_path)
    w = 1.0 / m
    mean_drawn = np.sum(w * m) / np.sum(w)
    num = integrate.quad(lambda x: massweighted(np.array(x)), 0.08, 0.5)[0] + integrate.quad(lambda x: massweighted(np.array(x)), 0.5, MMAX)[0]
    den = integrate.quad(lambda x: massweighted(np.array(x)) / x, 0.08, 0.5)[0] + integrate.quad(lambda x: massweighted(np.array(x)) / x, 0.5, MMAX)[0]
    assert abs(mean_drawn / (num / den) - 1) < 0.01


def test_keyed_and_reproducible(tmp_path):
    a, b, c = draws(10**4, 0.08, 99, tmp_path), draws(10**4, 0.08, 99, tmp_path), draws(10**4, 0.08, 100, tmp_path)
    assert a.tobytes() == b.tobytes()
    assert np.mean(a == c) < 1e-3


def test_same_distribution_as_reference_sampler(tmp_path):
    for m_min in (0.08, 4.0):
        ks = stats.ks_2samp(draws(10**6, m_min, 4242, tmp_path), draws(10**6, m_min, 4242, tmp_path, reference=True))
        assert ks.pvalue > 1e-3, (m_min, ks)
