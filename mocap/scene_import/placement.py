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


def per_frame_depth(n, frames, scales, weights, k, fps, smooth_seconds=0.3, limit=1.2):
    """Per-frame depth factor: the stance rays' weighted median per frame, interpolated through swing frames and
    smoothed. GVHMR's depth wobbles from frame to frame; the size factor k alone can't remove that, so feet would
    sink into or float above the terrain. Clamped to k/limit .. k*limit, so one bad frame can't throw the body."""
    from scipy.ndimage import gaussian_filter1d
    # A ray that grazes a ridge or passes a hole in the scan hits terrain far behind the foot: those samples are
    # outliers, not depth. Only samples within `limit` of the actor's size factor count; the rest interpolate.
    ok = np.isfinite(scales) & (np.abs(np.log(np.maximum(scales, 1e-9) / k)) <= np.log(limit))
    have, values = [], []
    for t in np.unique(frames[ok]):
        sel = ok & (frames == t)
        have.append(t)
        values.append(np.log(weighted_median(scales[sel], weights[sel])))
    if not have:
        return np.full(n, k)
    log_k = np.interp(np.arange(n), have, values)
    log_k = gaussian_filter1d(log_k, max(1.0, smooth_seconds * fps), mode='nearest')
    return np.clip(np.exp(log_k), k / limit, k * limit)


def align_up_yaw(source, target, weights=None):
    """Similarity (scale s, rotation about +Y, translation) taking source points (n,3) onto target (n,3)."""
    w = np.ones(len(source)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ms, mt = w @ source, w @ target
    a, b = source - ms, target - mt
    # Rotation about Y: maximise sum w * (b_xz . R a_xz)
    sxx = np.sum(w * (a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2]))
    sxz = np.sum(w * (a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]))
    theta = np.arctan2(sxz, sxx)
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
    ra = a @ R.T
    scale = float(np.sum(w[:, None] * ra * b) / max(np.sum(w[:, None] * ra * ra), 1e-12))
    return scale, R, mt - scale * ms @ R.T


def steady_root(raw_root, model_root, fps, low_seconds=1.0):
    """Complementary filter for the root trajectory.

    raw_root: camera-consistent placement (correct pixels and long-term position, but the model's per-frame depth
    noise along the camera ray: GVHMR's camera-frame depth jumps several cm from frame to frame).
    model_root: the model's own world trajectory (temporally smooth, but in its own frame and slowly drifting).
    Result: model_root aligned to the scene (scale + yaw + offset) for the fast motion, plus a low-pass of the
    difference for the slow motion. Fast depth noise goes, the steps and the footage-consistent path stay."""
    from scipy.ndimage import gaussian_filter1d
    s, R, t = align_up_yaw(model_root, raw_root)
    aligned = s * model_root @ R.T + t
    drift = gaussian_filter1d(raw_root - aligned, max(1.0, low_seconds * fps), axis=0, mode='nearest')
    return aligned + drift, dict(model_scale=s, model_yaw_deg=float(np.degrees(np.arctan2(R[0, 2], R[0, 0]))), model_yaw=R)


def upright(orient_camera, orient_model, yaw):
    """Body orientation: heading and pose from the camera-based solve, LEAN from the model's gravity-aware world.

    A camera-frame body put into the world with the known camera inherits the model's view-dependent pitch error:
    from a high camera the actor leans away from it (12.6 deg vs 2.3 deg in GVHMR's own world on the demo shot at a
    37 deg down view). Swing the body's up axis onto the model's up axis (minimal rotation); the twist about that
    axis, i.e. where the actor faces, stays as the footage shows it."""
    from scipy.spatial.transform import Rotation
    Ra = Rotation.from_rotvec(orient_camera).as_matrix()
    Rb = yaw @ Rotation.from_rotvec(orient_model).as_matrix()
    ua, ub = Ra[:, :, 1], Rb[:, :, 1]
    axis = np.cross(ua, ub)
    s = np.linalg.norm(axis, axis=1)
    c = np.einsum('ti,ti->t', ua, ub)
    angle = np.arctan2(s, c)
    axis = axis / np.maximum(s, 1e-12)[:, None]
    swing = Rotation.from_rotvec(axis * angle[:, None]).as_matrix()
    lean = lambda M: np.degrees(np.arccos(np.clip(M[:, 1, 1], -1, 1)))
    out = swing @ Ra
    return Rotation.from_matrix(out).as_rotvec(), swing, dict(lean_camera_deg_median=float(np.median(lean(Ra))),
        lean_camera_deg_p95=float(np.percentile(lean(Ra), 95)), lean_deg_median=float(np.median(lean(out))),
        lean_deg_p95=float(np.percentile(lean(out), 95)), swing_deg_p95=float(np.percentile(np.degrees(angle), 95)))


def place(orient_c, transl_c, rest_root, probes_c, probabilities, world_to_camera, heightfield,
          model_height=None, actor_height=None, per_frame=True, fps=30.0, model_root=None, model_orient=None):
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
    depth = per_frame_depth(n, t_idx, scales, probabilities[t_idx, p_idx], k, fps) if per_frame else np.full(n, k)
    pelvis_c = transl_c + rest_root
    root_w = np.einsum('tij,tj->ti', R, depth[:, None] * pelvis_c) + C
    steady = None
    if model_root is not None and len(model_root) == n and n > 3:
        raw = root_w
        smooth, steady = steady_root(raw, np.asarray(model_root, float), fps)
        # Keep the footage: the root stays on its camera ray (exact pixels); only its distance comes from the
        # steady trajectory. Image-plane motion was never the noisy part, depth along the ray was.
        ray = raw - C
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        root_w = C + ray * np.einsum('ti,ti->t', smooth - C, ray)[:, None]
        # How far the steadied root moved from the camera-consistent one, in pixels at its depth.
        off = root_w - raw
        dist = np.linalg.norm(raw - C, axis=1)
        steady.update(root_shift_cm_median=float(np.median(np.linalg.norm(off, axis=1)) * 100),
                      root_shift_cm_p95=float(np.percentile(np.linalg.norm(off, axis=1), 95) * 100),
                      raw_accel_rms=float(np.sqrt(np.mean(np.sum(np.diff(raw, 2, axis=0) ** 2, 1))) * fps * fps),
                      accel_rms=float(np.sqrt(np.mean(np.sum(np.diff(root_w, 2, axis=0) ** 2, 1))) * fps * fps))
        del dist
    transl_w = root_w - rest_root
    orient_w = Rotation.from_matrix(R @ Rotation.from_rotvec(orient_c).as_matrix()).as_rotvec()
    lean = None
    if model_orient is not None and steady is not None:
        orient_w, swing, lean = upright(orient_w, np.asarray(model_orient, float), steady['model_yaw'])
        # Pivot the lean about the feet, not the pelvis: the feet keep their place in the footage and on the
        # terrain; the upper body moves (the part the camera-based lean got wrong).
        feet = root_w[:, None] + np.einsum('tij,tpj->tpi', R, probes_c - pelvis_c[:, None])
        pivot = feet.mean(1)
        root_w = pivot + np.einsum('tij,tj->ti', swing, root_w - pivot)
        transl_w = root_w - rest_root
    if steady is not None:
        steady['model_yaw'] = steady['model_yaw'].tolist()
    # Foot-to-terrain distance in stance frames after placement (the body keeps its size, so this is not zero).
    feet_w = root_w[:, None] + np.einsum('tij,tpj->tpi', R, probes_c - pelvis_c[:, None])
    gap = feet_w[..., 1] - heightfield.height(feet_w[..., 0], feet_w[..., 2])
    stance_gap = np.abs(gap[stance])
    lowest = np.where(stance, gap, np.inf).min(1)
    lowest = lowest[np.isfinite(lowest)]
    report = dict(size_factor=k, size_check=scale_check.size_factor_check(k, model_height, actor_height),
                  per_frame_depth=bool(per_frame), depth_factor_range=[float(depth.min()), float(depth.max())],
                  steady_root=steady, upright=lean,
                  model_height_m=model_height, actor_height_m=actor_height, stance_frames=int(stance.any(1).sum()),
                  frames=n, rays=stats, warning=warning,
                  stance_gap_median_cm=float(np.median(stance_gap) * 100) if len(stance_gap) else None,
                  stance_gap_p95_cm=float(np.percentile(stance_gap, 95) * 100) if len(stance_gap) else None,
                  lowest_stance_sole_median_cm=float(np.median(lowest) * 100) if len(lowest) else None)
    return orient_w.astype(np.float32), transl_w.astype(np.float32), report
