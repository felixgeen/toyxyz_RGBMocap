"""Pre-calculated Mesh correction: each foot contact locks to the terrain or keeps the footage height.

Runs on top of Multi Level (source-relative 3D stances). Multi Level gives every stance an anchor at the solved
foot height; here each stance gets a TARGET:
  terrain  the anchor height becomes the terrain height under the anchor (it follows the rolling anchor path)
  footage  the anchor keeps the solved height (Multi Level unchanged)
Automatic: terrain, unless terrain and footage disagree by more than FOOTAGE_WINS (posefit's "footage wins" rule:
a big disagreement usually means the mesh is wrong there, e.g. a rock the scan missed). The user overrides any
stance on the contact timeline; overrides are frame-based so they survive re-segmentation.
"""
import numpy as np

FOOTAGE_WINS = 0.10  # metres
TARGETS = ('terrain', 'footage')


def normalize_overrides(overrides):
    """[{'foot': 0|1, 'frame': int, 'target': 'terrain'|'footage'}], validated and de-duplicated (last wins)."""
    out = {}
    for row in overrides or ():
        foot, frame, target = int(row['foot']), int(row['frame']), str(row['target'])
        if foot not in (0, 1) or frame < 0 or target not in TARGETS:
            raise ValueError('Invalid terrain contact override')
        out[(foot, frame)] = target
    return [dict(foot=f, frame=t, target=v) for (f, t), v in sorted(out.items())]


def apply(anchors, segments, heightfield, overrides=(), threshold=FOOTAGE_WINS):
    """Set each segment's target and move terrain-target anchors onto the terrain. Edits anchors/segments in place.

    anchors (n,2,3) per-frame stance anchor per foot (world, +Y up); segments from multilevel_contacts.prepare.
    Returns a summary for the correction report."""
    rows = normalize_overrides(overrides)
    counts = dict(terrain=0, footage=0, footage_auto=0, overridden=0, off_mesh=0)
    gaps = []
    for s in segments:
        foot, first, end = s['foot'], s['start'], s['end']
        path = anchors[first:end, foot]
        ground = heightfield.height(path[:, 0], path[:, 2])
        on_mesh = heightfield.inside(path[:, 0], path[:, 2])
        gap = float(np.median(path[:, 1] - ground))
        override = next((r['target'] for r in reversed(rows) if r['foot'] == foot and first <= r['frame'] < end), None)
        if override:
            target, reason = override, 'user'
            counts['overridden'] += 1
        elif not on_mesh.any():
            target, reason = 'footage', 'off_mesh'
            counts['off_mesh'] += 1
        elif abs(gap) > threshold:
            target, reason = 'footage', 'footage_wins'
            counts['footage_auto'] += 1
        else:
            target, reason = 'terrain', 'auto'
        if target == 'terrain':
            path[:, 1] = ground
        counts[target] += 1
        gaps.append(abs(gap))
        s.update(target=target, target_reason=reason, terrain_gap_m=gap)
    return dict(segments=len(segments), threshold_m=threshold, overrides=len(rows), **counts,
                terrain_gap_median_m=float(np.median(gaps)) if gaps else None)


def context(project):
    """The project's terrain heightfield and saved overrides, or raise if the scene has no terrain mesh."""
    import json
    from pathlib import Path
    from .reader import load
    from .terrain import Heightfield
    scene = load(project)
    if scene is None or 'mesh_vertices' not in scene[0]:
        raise ValueError('Pre-calculated Mesh needs a terrain mesh: choose one under Pre-calculated Camera and Reconstruct 3D.')
    arrays, _ = scene
    state = json.loads((Path(project) / 'project.json').read_text(encoding='utf-8'))
    return dict(heightfield=Heightfield(arrays['mesh_vertices'], arrays['mesh_faces']),
                overrides=(state.get('correction') or {}).get('terrain_overrides', []))


def detect_ranges(probes, probabilities, heightfield, fps, near=0.12, slow=0.8, confidence=0.5, minimum=0.1, gap=0.07):
    """Automatic foot-contact timing on terrain: [[(first, end), ...] per foot].

    Multi Level times contacts from world foot speed alone; with a moving camera and model depth wobble the feet
    rarely look still, so it finds almost nothing. With a terrain we can also ask whether the sole is ON the ground:
    a foot is in contact when the model is confident, the sole is within `near` of the terrain and it moves slower
    than `slow` m/s. Short gaps are bridged and very short contacts dropped."""
    from ..correction import intervals
    n = len(probes)
    patches = np.asarray(probes).reshape(n, 2, 2, 3)
    height = patches[..., 1] - heightfield.height(patches[..., 0], patches[..., 2])
    lowest = height.min(2)
    centre = patches.mean(2)
    speed = np.zeros((n, 2))
    if n > 1:
        speed[1:] = np.linalg.norm(np.diff(centre, axis=0), axis=-1) * fps
        speed[0] = speed[1]
    conf = np.asarray(probabilities).reshape(n, 2, 2).max(2)
    active = (conf >= confidence) & (np.abs(lowest) <= near) & (speed <= slow)
    ranges = []
    for foot in (0, 1):
        track = active[:, foot].copy()
        for first, end in intervals(~track):
            if first > 0 and end < n and end - first <= max(1, round(gap * fps)):
                track[first:end] = True
        ranges.append([(int(a), int(b)) for a, b in intervals(track) if b - a >= max(3, round(minimum * fps))])
    return ranges
