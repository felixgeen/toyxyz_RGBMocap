"""Place a camera-frame body into the imported scene and stand it on the terrain.

The model's camera-frame body (smpl_params_incam) is moved into the world with the KNOWN camera. Its depth comes
from the model's guess of the actor's size, so one factor k per actor slides it along the camera rays until the
stance feet meet the terrain. k scales the camera-frame position only: the body keeps its size and stays on the
same pixels (its image size changes by 1/k, which is small when k is near 1).

k is also the post-solve scale check: k far from 1 means the scene, or the assumed actor size, is off.
Ported in spirit from posefit/solvers/placement.py (size lock: weighted, outlier-rejected median).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from . import convert, scale_check

STANCE = 0.5          # contact probability that counts as a stance foot


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def size_factor(scales, weights):
    """Robust k: weighted median after rejecting samples more than 3 MADs (in log space) from the median."""
    ok = np.isfinite(scales) & (scales > 0) & (weights > 0)
    s, w = scales[ok], weights[ok]
    if not len(s):
        return None, dict(samples=0, kept=0)
    logs = np.log(s)
    med = np.median(logs)
    mad = max(1.4826 * np.median(np.abs(logs - med)), 0.01)
    keep = np.abs(logs - med) <= 3 * mad
    k = weighted_median(s[keep], w[keep])
    return k, dict(samples=int(len(s)), kept=int(keep.sum()), spread=float(np.exp(mad) - 1))


def place(orient_c, transl_c, rest_root, probes_c, probabilities, world_to_camera, heightfield,
          model_height=None, actor_height=None):
    """World-frame global_orient/transl for a camera-frame body, plus a report. All arrays per working frame.

    orient_c (n,3) axis-angle and transl_c (n,3): smpl_params_incam (OpenCV camera frame).
    rest_root (3,): the shaped body's pelvis joint at rest (SMPL-X: pelvis = transl + rest_root).
    probes_c (n,4,3): sole probes (left heel, left toe, right heel, right toe) in the camera frame.
    probabilities (n,4): contact probability per probe.
    """
    n = len(orient_c)
    c2w = convert.invert_rigid(world_to_camera)
    R, C = c2w[:, :3, :3], c2w[:, :3, 3]
    stance = probabilities >= STANCE
    t_idx, p_idx = np.nonzero(stance)
    directions = np.einsum('tij,tj->ti', R[t_idx], probes_c[t_idx, p_idx])
    scales = np.full(len(t_idx), np.nan)
    for start in range(0, len(t_idx), 2000):
        part = slice(start, start + 2000)
        scales[part] = heightfield.ray_scale(C[t_idx[part]], directions[part], lo=1e-3, hi=1e3, steps=700)
    hit = C[t_idx] + scales[:, None] * directions
    on_mesh = np.isfinite(scales)
    on_mesh[on_mesh] = heightfield.inside(hit[on_mesh, 0], hit[on_mesh, 2])
    scales[~on_mesh] = np.nan
    k, stats = size_factor(scales, probabilities[t_idx, p_idx])
    warning = ''
    if k is None:
        k, warning = 1.0, 'No stance foot reached the terrain mesh, so the actor keeps the model depth (k = 1).'
    pelvis_c = transl_c + rest_root
    root_w = np.einsum('tij,tj->ti', R, k * pelvis_c) + C
    transl_w = root_w - rest_root
    orient_w = Rotation.from_matrix(R @ Rotation.from_rotvec(orient_c).as_matrix()).as_rotvec()
    # Foot-to-terrain distance in stance frames after placement (the body keeps its size, so this is not zero).
    feet_w = np.einsum('tij,tpj->tpi', R, k * pelvis_c[:, None] + (probes_c - pelvis_c[:, None])) + C[:, None]
    gap = feet_w[..., 1] - heightfield.height(feet_w[..., 0], feet_w[..., 2])
    stance_gap = np.abs(gap[stance])
    lowest = np.where(stance, gap, np.inf).min(1)
    lowest = lowest[np.isfinite(lowest)]
    report = dict(size_factor=k, size_check=scale_check.size_factor_check(k, model_height, actor_height),
                  model_height_m=model_height, actor_height_m=actor_height, stance_frames=int(stance.any(1).sum()),
                  frames=n, rays=stats, warning=warning,
                  stance_gap_median_cm=float(np.median(stance_gap) * 100) if len(stance_gap) else None,
                  stance_gap_p95_cm=float(np.percentile(stance_gap, 95) * 100) if len(stance_gap) else None,
                  lowest_stance_sole_median_cm=float(np.median(lowest) * 100) if len(lowest) else None)
    return orient_w.astype(np.float32), transl_w.astype(np.float32), report
