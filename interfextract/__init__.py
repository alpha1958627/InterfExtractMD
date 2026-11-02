"""Three liquid-vapour interface definitions for orthorhombic periodic MD boxes.

Positions are fluid-particle Cartesian coordinates; box is [[xmin,xmax], ...].
All three spatial directions are periodic for density and neighbour queries.
"""
from itertools import zip_longest
import numpy as np
from scipy.spatial import cKDTree
from ._branch import density_roots, choose


def _geometry(positions, box):
    p = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("positions must be a finite (N, 3) array")
    if box.shape != (3, 2) or not np.isfinite(box).all():
        raise ValueError("box must have shape (3, 2)")
    lengths = box[:, 1] - box[:, 0]
    if np.any(lengths <= 0):
        raise ValueError("box lengths must be positive")
    return (p - box[:, 0]) % lengths, box, lengths


def _shape(shape, ndim, max_cells):
    a = np.asarray(shape)
    if a.shape != (ndim,) or not np.isfinite(a).all() or np.any(a < 1) or np.any(a != np.floor(a)):
        raise ValueError(f"shape must contain {ndim} positive integers")
    sh = tuple(int(x) for x in a)
    if np.prod(sh, dtype=np.int64) > max_cells:
        raise MemoryError("grid exceeds max_cells; check available RAM before explicitly raising it")
    return sh


def hard_bin_density(positions, box, shape, *, max_cells=20_000_000):
    """Number density: one fluid bead contributes to exactly one periodic bin."""
    p, _, lengths = _geometry(positions, box)
    sh = _shape(shape, 3, max_cells)
    ind = np.floor(p / lengths * sh).astype(np.int64)
    key = (ind[:, 0] * sh[1] + ind[:, 1]) * sh[2] + ind[:, 2]
    counts = np.bincount(key, minlength=int(np.prod(sh))).reshape(sh)
    return counts / np.prod(lengths / sh)


def gaussian_density(positions, box, shape, *, width=1 / np.sqrt(20), truncate=4., max_cells=20_000_000):
    """Direct particle-to-grid Gaussian density, not Gaussian-filtered hard bins.

    Weights exp(-distance**2/(2*width**2)) are evaluated at bin centres.
    Each truncated periodic particle kernel is normalized to unit total mass.
    This readable reference implementation loops over particles.
    """
    p, _, lengths = _geometry(positions, box)
    sh = _shape(shape, 3, max_cells)
    if not np.isfinite(width) or width <= 0 or not np.isfinite(truncate) or truncate <= 0:
        raise ValueError("width and truncate must be positive and finite")
    spacing = lengths / sh
    density = np.zeros(sh, dtype=float)
    for point in p:
        indices, weights = [], []
        for x, L, n, dx in zip(point, lengths, sh, spacing):
            lo = int(np.ceil((x - truncate * width) / dx - .5))
            hi = int(np.floor((x + truncate * width) / dx - .5))
            raw = np.arange(lo, hi + 1)
            if not len(raw):
                raise ValueError("grid has no sample inside the Gaussian support; refine it")
            w = np.exp(-.5 * (((raw + .5) * dx - x) / width) ** 2)
            if not np.isfinite(w.sum()) or w.sum() <= 0:
                raise ValueError("Gaussian normalization failed")
            indices.append(raw % n)
            weights.append(w / w.sum())
        mass = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
        np.add.at(density, np.ix_(*indices), mass)
    return density / np.prod(spacing)


def average_density(frames, box, shape, *, method="gaussian", weights=None, **kwargs):
    """Average density before interface extraction; process frames one by one.

    weights=None means equal weights. For a closed 201-frame 1tau interval,
    explicit trapezoid weights are [0.5] + [1.0]*199 + [0.5].
    """
    functions = {"hard": hard_bin_density, "gaussian": gaussian_density}
    if method not in functions:
        raise ValueError("method must be 'hard' or 'gaussian'")
    sentinel = object()
    pairs = ((f, 1.) for f in frames) if weights is None else zip_longest(frames, weights, fillvalue=sentinel)
    total, weight_sum = None, 0.
    for frame, weight in pairs:
        if frame is sentinel or weight is sentinel:
            raise ValueError("frames and weights must have the same length")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weights must be finite and nonnegative")
        if weight == 0:
            continue
        rho = functions[method](frame, box, shape, **kwargs)
        total = weight * rho if total is None else total + weight * rho
        weight_sum += weight
    if total is None or weight_sum <= 0:
        raise ValueError("at least one positive-weight frame is required")
    return total / weight_sum


def surface_from_density(density, box, *, level, reference_height, branch="continuous"):
    """Actual downward linear density roots at cell-centred lateral positions.

    continuous: retain the periodic main-liquid isosurface component, then
    select among its actual roots. nearest: select the root nearest the
    declared reference independently in each column. Neither fills holes.
    """
    rho = np.asarray(density, dtype=float)
    _, box, lengths = _geometry(np.empty((0, 3)), box)
    if rho.ndim != 3 or min(rho.shape) < 2 or not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("density must be a finite nonnegative 3D grid with at least two points per axis")
    if not np.isfinite(level) or level <= 0:
        raise ValueError("level must be positive and finite")
    if branch not in ("continuous", "nearest"):
        raise ValueError("branch must be 'continuous' or 'nearest'")
    guide = np.broadcast_to(np.asarray(reference_height, dtype=float), rho.shape[:2]).copy()
    if not np.isfinite(guide).all():
        raise ValueError("reference_height must be finite")
    dz = lengths[2] / rho.shape[2]
    roots, cross = density_roots(rho, dz, level, box[2, 0])
    missing = int((~cross.any(axis=-1)).sum())
    if not cross.any():
        return np.full(rho.shape[:2], np.nan), {"valid": False, "raw_missing": missing, "missing": missing}
    mesh = {}
    if branch == "continuous":
        from ._mesh import surface_components
        roots, _, mesh = surface_components(rho, level, dz, guide, z_origin=box[2, 0])
    h, index, stats = choose(roots, guide, np.inf, 1. if branch == "continuous" else 0.)
    ok = index >= 0
    if ok.any():
        ix = np.maximum(index, 0)
        lo = np.take_along_axis(rho, ix[:, :, None], axis=-1)[:, :, 0]
        hi = np.take_along_axis(rho, (ix + 1)[:, :, None], axis=-1)[:, :, 0]
        zlo = box[2, 0] + (ix + .5) * dz
        residual = float(np.max(np.abs((lo + (hi - lo) * (h - zlo) / dz - level)[ok])))
        if residual > 1e-10 * max(1., level):
            raise RuntimeError("selected height is not an actual density root")
    else:
        residual = None
    periodic = mesh.get("main_periodic_gate", True)
    return h, {**stats, "raw_missing": missing, "branch": branch, "mesh": mesh,
               "root_residual": residual, "valid": bool(stats["finite_FFT"] and periodic)}


def neighbour_density(positions, box, *, radius=3.5):
    """Particle-centred number density; exclude self, include bonded partners."""
    p, _, lengths = _geometry(positions, box)
    if not np.isfinite(radius) or radius <= 0 or radius >= lengths.min() / 2:
        raise ValueError("radius must be positive and less than half the shortest box length")
    count = cKDTree(p, boxsize=lengths).query_ball_point(p, radius, eps=0, workers=1, return_length=True) - 1
    return count / (4 * np.pi * radius ** 3 / 3), count


def neighbour_surface(positions, box, shape_xy, *, radius=3.5, density_threshold=.415):
    """Classify fluid particles, then take the actual highest liquid bead in each bin."""
    p, box, lengths = _geometry(positions, box)
    sh = _shape(shape_xy, 2, 20_000_000)
    if not np.isfinite(density_threshold) or density_threshold < 0:
        raise ValueError("density_threshold must be finite and nonnegative")
    density, count = neighbour_density(positions, box, radius=radius)
    liquid = density > density_threshold
    ij = np.floor(p[:, :2] / lengths[:2] * sh).astype(int)
    key = ij[:, 0] * sh[1] + ij[:, 1]
    top = np.full(int(np.prod(sh)), -np.inf)
    np.maximum.at(top, key[liquid], p[liquid, 2] + box[2, 0])
    height = top.reshape(sh);height[~np.isfinite(height)] = np.nan
    return height, {"valid": bool(np.isfinite(height).all()), "missing": int(np.isnan(height).sum()),
                    "density": density, "count": count, "liquid": liquid}


def power_spectrum(height, box):
    """Two-dimensional per-mode power, normalized FFT; reject every missing height."""
    h = np.asarray(height, dtype=float)
    _, _, lengths = _geometry(np.empty((0, 3)), box)
    if h.ndim != 2 or not h.size or not np.isfinite(h).all():
        raise ValueError("complete finite 2D height required; missing columns cannot be filled or FFTed")
    power = abs(np.fft.fft2(h - h.mean()) / h.size) ** 2
    if not np.isclose(power.sum(), h.var(), rtol=1e-12, atol=1e-15):
        raise RuntimeError("Parseval check failed")
    qx = 2 * np.pi * np.fft.fftfreq(h.shape[0], d=lengths[0] / h.shape[0])
    qy = 2 * np.pi * np.fft.fftfreq(h.shape[1], d=lengths[1] / h.shape[1])
    return power, qx, qy
