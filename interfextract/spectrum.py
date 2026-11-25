"""Per-mode radial spectra and explicitly requested periodic gap reconstruction."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def radial_spectrum(power, qx, qy, edges):
    """Arithmetic mean of actual 2-D modes in each half-open radial band.

    Empty bands are omitted. q_effective = mean(q**-2)**(-1/2), so the
    band average of C/q**2 is exactly C/q_effective**2. The zero mode is excluded.
    Counts include both conjugate modes and are not independent sample counts.
    """
    power = np.asarray(power, dtype=float)
    qx, qy, edges = [np.asarray(x, dtype=float) for x in (qx, qy, edges)]
    if qx.ndim != 1 or qy.ndim != 1 or power.shape != (len(qx), len(qy)):
        raise ValueError("power shape must match the 1-D qx and qy arrays")
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or edges[0] < 0 or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must be finite, nonnegative and strictly increasing")
    if not all(np.isfinite(x).all() for x in (power, qx, qy)) or np.any(power < 0):
        raise ValueError("power and wavevectors must be finite, with nonnegative power")
    q = np.hypot(qx[:, None], qy[None, :])
    bands = np.searchsorted(edges, q, side="right") - 1
    valid = (q > 0) & (bands >= 0) & (bands < len(edges) - 1)
    count = np.bincount(bands[valid], minlength=len(edges) - 1)
    keep = count > 0
    mean = np.bincount(bands[valid], weights=power[valid], minlength=len(count))[keep] / count[keep]
    inverse_q2 = np.bincount(bands[valid], weights=q[valid]**-2, minlength=len(count))[keep] / count[keep]
    return {"q_low": edges[:-1][keep], "q_high": edges[1:][keep],
            "q_effective": inverse_q2**-.5, "power_mean": mean,
            "mode_count": count[keep]}


def harmonic_fill(height):
    """Fill NaN columns by the periodic discrete Laplace equation.

    Observed values remain exact. This reconstruction supplies an assumed
    smooth continuation across holes and can suppress short-wave power there.
    The original missing mask must be retained alongside the returned field.
    """
    h = np.asarray(height, dtype=float)
    if h.ndim != 2 or min(h.shape) < 2 or np.isinf(h).any():
        raise ValueError("height must be 2-D with at least two cells per axis and no infinity")
    missing = np.isnan(h)
    if missing.all():
        raise ValueError("at least one observed height is required")
    if not missing.any():
        return h.copy()
    x, y = np.nonzero(missing)
    n = len(x)
    index = np.full(h.shape, -1, dtype=int)
    index[x, y] = np.arange(n)
    rows, cols, values = [np.arange(n)], [np.arange(n)], [np.full(n, 4.)]
    rhs = np.zeros(n)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        i, j = (x + dx) % h.shape[0], (y + dy) % h.shape[1]
        neighbor = index[i, j]
        unknown = neighbor >= 0
        rows.append(np.flatnonzero(unknown))
        cols.append(neighbor[unknown])
        values.append(np.full(unknown.sum(), -1.))
        rhs[~unknown] += h[i[~unknown], j[~unknown]]
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n, n)).tocsc()
    result = h.copy()
    result[missing] = spsolve(matrix, rhs)
    if (not np.isfinite(result).all()
            or np.max(abs(matrix @ result[missing] - rhs)) > 1e-9 * max(1., np.max(abs(rhs)))):
        raise RuntimeError("harmonic reconstruction failed its residual check")
    return result
