"""Which file axis is up? Many exporters declare one axis and write another (SynthEyes USD: "Y-up", Z-up data).

Evidence, in raw file coordinates, scored for each of the six axis directions:
  image_down  the camera is upright: its image-down axis points against up (mean over frames)
  level       the camera is level: its image-right axis is perpendicular to up
  terrain     the terrain is mostly flat: its best-fit plane normal lies along up (axis only, not the sign)
  above       the camera is above the terrain along up (decides the sign when the camera cues are weak)
Simplified from posefit/calibrate/up_axis.py.
"""
import numpy as np

AXES = {'+X': (1, 0, 0), '-X': (-1, 0, 0), '+Y': (0, 1, 0), '-Y': (0, -1, 0), '+Z': (0, 0, 1), '-Z': (0, 0, -1)}


def terrain_normal(vertices, max_points=100000):
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) > max_points:
        v = v[np.random.default_rng(0).choice(len(v), max_points, replace=False)]
    _, s, vt = np.linalg.svd(v - v.mean(0), full_matrices=False)
    flatness = 1.0 - s[2] / max(s[1], 1e-12)  # 1 = a perfect plane, 0 = no preferred normal
    return vt[2], float(flatness)


def above_fraction(cam_centres, vertices, u):
    """Fraction of camera centres that sit above the terrain along u (terrain point nearest in the plane _|_ u)."""
    from scipy.spatial import cKDTree
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) > 100000:
        v = v[np.random.default_rng(0).choice(len(v), 100000, replace=False)]
    u = np.asarray(u, dtype=np.float64)
    basis = np.linalg.svd(np.eye(3) - np.outer(u, u))[0][:, :2]
    _, idx = cKDTree(v @ basis).query(np.asarray(cam_centres) @ basis)
    return float(np.mean((np.asarray(cam_centres) - v[idx]) @ u > 0))


def estimate(cam_to_file_cv=None, vertices=None):
    """(best axis name, {axis: score}, evidence dict). Higher score = more likely up."""
    scores = {name: 0.0 for name in AXES}
    evidence = {}
    normal = None
    if vertices is not None and len(vertices) > 3:
        normal, flatness = terrain_normal(vertices)
        evidence['terrain_flatness'] = flatness
    for name, axis in AXES.items():
        u = np.asarray(axis, dtype=np.float64)
        if cam_to_file_cv is not None and len(cam_to_file_cv):
            down, right = cam_to_file_cv[:, :3, 1], cam_to_file_cv[:, :3, 0]
            scores[name] += 2.0 * float(np.mean(-down @ u))
            scores[name] += 1.0 * float(1.0 - np.mean(np.abs(right @ u)))
        if normal is not None:
            scores[name] += 1.5 * evidence['terrain_flatness'] * abs(float(normal @ u))
    if cam_to_file_cv is not None and len(cam_to_file_cv) and vertices is not None and len(vertices):
        # Sign from the terrain: only for the best axis and its opposite, which share every other cue.
        best = max(scores, key=scores.get)
        opposite = ('-' if best[0] == '+' else '+') + best[1]
        frac = above_fraction(cam_to_file_cv[:, :3, 3], vertices, AXES[best])
        evidence['camera_above_terrain'] = frac
        scores[best] += 1.0 * frac
        scores[opposite] += 1.0 * (1.0 - frac)
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    evidence['margin'] = float(ranked[0] - ranked[1])
    return best, scores, evidence
