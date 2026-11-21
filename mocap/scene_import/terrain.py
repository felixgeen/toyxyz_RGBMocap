"""Terrain height lookups and ray hits on an imported mesh, as a Y-up heightfield. Pure NumPy.

The mesh is densely point-sampled (so low-poly terrain works too) and binned into a regular XZ grid holding the
HIGHEST surface in each cell. Empty cells are filled from the nearest filled cell. Overhangs are not modelled:
for terrain under a walking actor that is the right surface.
"""
import numpy as np


class Heightfield:
    def __init__(self, vertices, faces, cell=0.05, max_samples=12_000_000):
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        lo, hi = v[:, [0, 2]].min(0), v[:, [0, 2]].max(0)
        span = np.maximum(hi - lo, cell)
        # Keep the grid bounded on very large terrains.
        cell = max(cell, float(np.sqrt(span[0] * span[1] / 16e6)))
        self.cell, self.origin = cell, lo
        self.shape = tuple((np.ceil(span / cell).astype(int) + 1).tolist())
        points = self._samples(v, f, cell, max_samples)
        ij = np.floor((points[:, [0, 2]] - lo) / cell).astype(np.int64)
        ij = np.clip(ij, 0, np.array(self.shape) - 1)
        flat = ij[:, 0] * self.shape[1] + ij[:, 1]
        grid = np.full(self.shape[0] * self.shape[1], -np.inf)
        np.maximum.at(grid, flat, points[:, 1])
        filled = np.isfinite(grid)
        self.coverage = float(filled.mean())
        if not filled.all():
            from scipy.spatial import cKDTree
            centres = np.stack(np.unravel_index(np.arange(grid.size), self.shape), 1)
            _, nearest = cKDTree(centres[filled]).query(centres[~filled])
            grid[~filled] = grid[filled][nearest]
        self.grid = grid.reshape(self.shape)
        # Random sampling can miss single cells inside the mesh; close those pinholes (not the outer boundary).
        from scipy.ndimage import binary_closing
        self.inside_mask = binary_closing(filled.reshape(self.shape), iterations=2, border_value=0) | filled.reshape(self.shape)

    @staticmethod
    def _samples(v, f, cell, max_samples):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        # Plan-view area decides how many samples a triangle needs to touch every cell it covers.
        plan = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 2] - a[:, 2]) - (c[:, 0] - a[:, 0]) * (b[:, 2] - a[:, 2]))
        counts = np.ceil(plan / (cell * cell) * 6).astype(np.int64)
        extra = counts.sum()
        if extra > max_samples:
            counts = np.ceil(counts * (max_samples / extra)).astype(np.int64)
        points = [v]  # vertices always count
        many = np.flatnonzero(counts > 0)
        reps = counts[many]
        if reps.sum():
            tri = np.repeat(many, reps)
            rng = np.random.default_rng(0)
            r1, r2 = rng.random(len(tri)), rng.random(len(tri))
            flip = r1 + r2 > 1
            r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
            points.append(a[tri] + r1[:, None] * (b[tri] - a[tri]) + r2[:, None] * (c[tri] - a[tri]))
        return np.concatenate(points)

    def height(self, x, z):
        """Bilinear terrain height at world X/Z (arrays). Outside the mesh: the nearest edge height."""
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        u = np.clip((x - self.origin[0]) / self.cell, 0, self.shape[0] - 1.000001)
        w = np.clip((z - self.origin[1]) / self.cell, 0, self.shape[1] - 1.000001)
        i, j = np.floor(u).astype(np.int64), np.floor(w).astype(np.int64)
        fu, fw = u - i, w - j
        g = self.grid
        return ((1 - fu) * (1 - fw) * g[i, j] + fu * (1 - fw) * g[i + 1, j] +
                (1 - fu) * fw * g[i, j + 1] + fu * fw * g[i + 1, j + 1])

    def inside(self, x, z):
        u = np.floor((np.asarray(x) - self.origin[0]) / self.cell).astype(np.int64)
        w = np.floor((np.asarray(z) - self.origin[1]) / self.cell).astype(np.int64)
        ok = (u >= 0) & (w >= 0) & (u < self.shape[0]) & (w < self.shape[1])
        out = np.zeros(np.shape(u), dtype=bool)
        out[ok] = self.inside_mask[u[ok], w[ok]]
        return out

    def ray_scale(self, origins, directions, lo=0.2, hi=5.0, steps=400):
        """For rays origin + s * direction, the first s in [lo, hi] where the ray meets the terrain (NaN if never).

        Marches from lo to hi and refines the first downward crossing linearly; s = 1 is the input point."""
        o = np.asarray(origins, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        s = np.geomspace(lo, hi, steps)
        p = o[:, None, :] + s[None, :, None] * d[:, None, :]
        gap = p[..., 1] - self.height(p[..., 0], p[..., 2])  # > 0 above the terrain
        below = gap <= 0
        first = np.argmax(below, axis=1)
        hit = below[np.arange(len(o)), first] & (first > 0)
        out = np.full(len(o), np.nan)
        k = np.flatnonzero(hit)
        f = first[k]
        g0, g1 = gap[k, f - 1], gap[k, f]
        t = g0 / (g0 - g1)
        out[k] = s[f - 1] + t * (s[f] - s[f - 1])
        return out


def wireframe(heightfield, spacing=0.5, step=0.25):
    """Line segments (n,2,3) following the terrain: one line every `spacing` metres along X and Z."""
    hf = heightfield
    lo = hf.origin
    hi = lo + (np.array(hf.shape) - 1) * hf.cell
    lines = []
    for axis in (0, 1):
        across = np.arange(lo[1 - axis], hi[1 - axis] + 1e-9, spacing)
        along = np.arange(lo[axis], hi[axis] + 1e-9, step)
        for c in across:
            xz = np.empty((len(along), 2))
            xz[:, axis], xz[:, 1 - axis] = along, c
            inside = hf.inside(xz[:, 0], xz[:, 1])
            y = hf.height(xz[:, 0], xz[:, 1])
            pts = np.stack([xz[:, 0], y, xz[:, 1]], 1)
            keep = inside[:-1] & inside[1:]
            if keep.any():
                lines.append(np.stack([pts[:-1][keep], pts[1:][keep]], 1))
    return np.concatenate(lines) if lines else np.zeros((0, 2, 3))
