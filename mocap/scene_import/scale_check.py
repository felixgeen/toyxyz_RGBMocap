"""Is the imported scene in real-world metres? Two checks, both pure NumPy.

1. import_check: before any solve. Camera height above the terrain and the terrain's size must be plausible.
   Catches gross unit errors (centimetres read as metres, and so on).
2. size_factor_check: after the first solve. The model's body is roughly metric; placing it on the terrain gives a
   size factor. Near 1 = fine; near a power of ten = unit error with a one-click fix; anything else = re-export.

An actor height the user enters overrides both: scale_from_actor_height.
"""
import math
import numpy as np

UNIT_FIXES = (0.001, 0.01, 0.1, 10.0, 100.0, 1000.0)
# Plausible ranges in metres. Wide on purpose: handheld at knee height up to high drone shots.
CAMERA_HEIGHT = (0.2, 500.0)
CAMERA_HEIGHT_TYPICAL = (0.5, 150.0)
TERRAIN_SIZE = (0.5, 50000.0)
SIZE_OK = 1.3          # size factor within 1/1.3 .. 1.3 is fine
UNIT_TOLERANCE = 1.3   # size factor within x1.3 of a power of ten is a unit error


def camera_heights(camera_centres, vertices, max_vertices=200000):
    """Camera height above the terrain vertex nearest in plan (XZ) view, per centre; NaN when off the terrain."""
    from scipy.spatial import cKDTree
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) > max_vertices:
        v = v[np.random.default_rng(0).choice(len(v), max_vertices, replace=False)]
    c = np.asarray(camera_centres, dtype=np.float64)
    lo, hi = v[:, [0, 2]].min(0), v[:, [0, 2]].max(0)
    spacing = math.sqrt(max(np.prod(hi - lo), 1e-12) / len(v))
    dist, idx = cKDTree(v[:, [0, 2]]).query(c[:, [0, 2]])
    h = c[:, 1] - v[idx, 1]
    h[dist > 20 * spacing] = np.nan
    return h


def import_check(camera_centres=None, vertices=None, preferred=()):
    """Status 'ok' | 'suspicious' | 'no_evidence', the measured values and a suggested unit fix (or None).

    preferred: factors tried first, such as the one that ignores the file's declared unit (a common exporter bug:
    values written in metres while the file says centimetres)."""
    report = dict(status='no_evidence', camera_height_m=None, terrain_size_m=None, suggested_factor=None, candidates=[], message='')
    if vertices is None or not len(vertices):
        report['message'] = 'No terrain mesh, so the scene scale can only be checked after the first solve.'
        return report
    v = np.asarray(vertices, dtype=np.float64)
    size = float(np.linalg.norm(np.percentile(v, 99, 0) - np.percentile(v, 1, 0)))
    report['terrain_size_m'] = size
    height = None
    if camera_centres is not None and len(camera_centres):
        h = camera_heights(camera_centres, v)
        if np.isfinite(h).any():
            height = float(np.nanmedian(h))
            report['camera_height_m'] = height

    def plausible(factor):
        ok = TERRAIN_SIZE[0] <= size * factor <= TERRAIN_SIZE[1]
        if height is not None:
            ok = ok and CAMERA_HEIGHT[0] <= abs(height) * factor <= CAMERA_HEIGHT[1]
        return ok

    def typical(factor):
        return height is not None and CAMERA_HEIGHT_TYPICAL[0] <= abs(height) * factor <= CAMERA_HEIGHT_TYPICAL[1]

    if height is not None and height < 0:
        report.update(status='suspicious', message=(
            f'The camera sits {abs(height):.2f} units below the terrain. The up axis may be wrong; check the export settings.'))
        return report
    if plausible(1.0):
        report.update(status='ok', message='Camera height and terrain size look like real-world metres.')
        return report
    best = next((float(f) for f in preferred if f and plausible(f) and (height is None or typical(f))), None)
    if best is None:
        candidates = [f for f in UNIT_FIXES if plausible(f)]
        # Among plausible fixes, the one putting the camera nearest the middle of the typical band (log scale).
        centre = math.sqrt(CAMERA_HEIGHT_TYPICAL[0] * CAMERA_HEIGHT_TYPICAL[1])
        key = (lambda f: abs(math.log(abs(height) * f / centre))) if height else (lambda f: abs(math.log(f)))
        candidates.sort(key=key)
        best = candidates[0] if candidates else None
    else:
        candidates = [best]
    # The import check alone can't always tell centimetres from millimetres: the UI offers every candidate,
    # best first, and the post-solve size check settles it.
    report.update(status='suspicious', suggested_factor=best, candidates=candidates)
    what = f'the camera is {height:.3g} m above the terrain' if height is not None else f'the terrain is {size:.3g} m across'
    if best is not None:
        report['message'] = f'The scene looks wrongly scaled: {what}. Scaling it by {best:g} would fix a unit mismatch.'
    else:
        report['message'] = f'The scene looks wrongly scaled: {what}. Re-export it in real-world units, or enter the actor height.'
    return report


def size_factor_check(factor, model_height=None, actor_height=None):
    """Judge the size factor that placing the solved body on the terrain needed (scene distance / model distance).

    model_height: the solved body's standing height in metres. actor_height: the real actor's height, if known.
    'fix' multiplies the scene scale. height_fix is the fix that makes the actor exactly actor_height (or, when
    unknown, the model's own estimate) tall in the scene."""
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0:
        return dict(status='failed', factor=factor, fix=None, height_fix=None, message='The body could not be placed on the terrain.')
    target = actor_height or model_height
    height_fix = (target / model_height) / factor if model_height and target else 1.0 / factor
    if actor_height:
        return dict(status='actor_height', factor=factor, fix=height_fix, height_fix=height_fix, message=(
            f'With the actor {actor_height:.2f} m tall, the scene needs a scale of {height_fix:.3g} '
            f'(the solved body is {model_height:.2f} m).' if model_height else
            f'With the actor {actor_height:.2f} m tall, the scene needs a scale of {height_fix:.3g}.'))
    if 1 / SIZE_OK <= factor <= SIZE_OK:
        return dict(status='ok', factor=factor, fix=None, height_fix=height_fix, message='The scene scale matches the actor.')
    power = 10.0 ** round(math.log10(factor))
    if power != 1.0 and 1 / UNIT_TOLERANCE <= factor / power <= UNIT_TOLERANCE:
        return dict(status='unit_error', factor=factor, fix=1.0 / power, height_fix=height_fix, message=(
            f'The actor is about {1 / power:g}x the expected size in this scene, which looks like a unit mismatch. '
            f'Scale the scene by {1 / power:g}?'))
    about = f' (the solved body is {model_height:.2f} m tall)' if model_height else ''
    return dict(status='badly_scaled', factor=factor, fix=None, height_fix=height_fix, message=(
        f'The scene is not in real-world units: the actor would be {1 / factor:.2f}x their solved size{about}. '
        f'Re-export the scene in metres, enter the actor height, or use a Scene Scale of {height_fix:.3g}.'))


def scale_from_actor_height(actor_height_m, scene_height_units):
    """Scene scale (metres per scene unit) from a known actor height and the actor's height measured in the scene."""
    if not actor_height_m or not scene_height_units or actor_height_m <= 0 or scene_height_units <= 0:
        raise ValueError('Actor height and measured height must be positive')
    return float(actor_height_m) / float(scene_height_units)
