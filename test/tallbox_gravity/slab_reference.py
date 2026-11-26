"""Direct-sum references for the tallbox_gravity slab (numpy, G = 1): the images of each source in x and y, none in z."""
import numpy as np


def direct_sum_with_images(pos, mass, n, lxy=1.0, sheet_tail=False):
    """a_i = sum_j m_j sum_(a,b) d / |d|^3 over the nearest x/y image of j and the (2n+1)^2 - 1 around it.

    sheet_tail adds, per source, the field of the uniform sheet m_j / lxy^2 at the source's height outside the square the
    images cover (half-width W = (n + 1/2) lxy around the nearest image): the sum periodic in x and y. Closed form for the
    square, a = sigma [[F]] over its corners with F_x = -ln(y + r), F_y = -ln(x + r), F_z = arctan(x y / (z r)); the whole
    sheet pulls with 2 pi sigma sign(z). What is left is the lattice's graininess beyond W, which falls with n (the tests
    measure it by comparing two n)."""
    off = np.array([(a, b, 0) for a in range(-n, n + 1) for b in range(-n, n + 1)], float) * lxy
    w = (n + 0.5) * lxy
    acc = np.zeros_like(pos)
    for i in range(len(pos)):
        dr = np.delete(pos, i, 0) - pos[i]
        m = np.delete(mass, i)
        dr[:, :2] -= lxy * np.round(dr[:, :2] / lxy)
        for c in range(0, len(off), 1200):
            d = dr[:, None, :] + off[None, c:c + 1200, :]
            acc[i] += np.einsum("j,jkc->c", m, d / np.linalg.norm(d, axis=2)[:, :, None] ** 3)
        if sheet_tail:
            z = dr[:, 2]
            f = np.zeros((len(m), 3))
            for sx in (1, -1):
                for sy in (1, -1):
                    x, y = dr[:, 0] + sx * w, dr[:, 1] + sy * w
                    r = np.sqrt(x * x + y * y + z * z)
                    f[:, 0] -= sx * sy * np.log(y + r)
                    f[:, 1] -= sx * sy * np.log(x + r)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        f[:, 2] += np.where(z != 0, sx * sy * np.arctan(x * y / (z * r)), 0.0)
            f = -f
            f[:, 2] += 2 * np.pi * np.sign(z)
            acc[i] += ((m / lxy ** 2)[:, None] * f).sum(0)
    return acc


def relerr(a, ref):
    return np.linalg.norm(a - ref, axis=1) / np.median(np.linalg.norm(ref, axis=1))
