"""Pre-calculated Camera / Mesh: read a scene file once into the project as scene_import.npz + scene_import.json.

After this, nothing else in RGB MoCap reads FBX, Alembic or USD. Everything is in the application world
(+Y up, metres) with OpenCV cameras; file_to_world is kept so exports can return to the original file frame.
"""
import json
from pathlib import Path

import numpy as np

from . import convert, scale_check
from .blender_job import inspect, read_raw

VERSION = 1
NPZ, JSON = 'scene_import.npz', 'scene_import.json'
CAMERA_HELPER_MESHES = ('Screen', 'Tracker')  # SynthEyes camera screens and tracker markers


def default_choices(info):
    """The animated camera (else the first camera) and the largest mesh that isn't a camera screen or tracker."""
    cams = info.get('cameras') or []
    cam = next((c['name'] for c in cams if c.get('animated')), cams[0]['name'] if cams else None)
    meshes = [m for m in info.get('meshes') or [] if not any(h in m['name'] for h in CAMERA_HELPER_MESHES)]
    meshes = meshes or (info.get('meshes') or [])
    mesh = max(meshes, key=lambda m: m.get('polygons', 0))['name'] if meshes else None
    return cam, mesh


def plate_frames(info, frame_count, first_frame=None):
    """File frame numbers for each plate frame. Default: the plate starts at the file's first animated frame."""
    if first_frame is None:
        rng = info.get('file_frame_range')
        first_frame = int(rng[0]) if rng else 0
    return np.arange(first_frame, first_frame + frame_count, dtype=np.int64)


def nearest_plate_index(working_times, plate_fps, count):
    """Plate frame index ffmpeg's fps filter shows at each working time (seconds from plate frame 0).

    Nearest frame, with exact halves going to the EARLIER frame (measured on the demo shot, 25 -> 30 fps)."""
    t = np.asarray(working_times, dtype=np.float64) * float(plate_fps)
    return np.clip(np.ceil(t - 0.5 - 1e-6).astype(np.int64), 0, count - 1)


def build(raw, meta, *, image_size, plate_fps, frames, working_times, declared_up=None, metres_per_unit=None,
          scale_fix=1.0):
    """Convert raw Blender output into the arrays saved in scene_import.npz. Pure NumPy (tested without Blender).

    declared_up None = auto: estimated from the camera and terrain; the file's own setting is only a fallback."""
    from . import up_axis
    warnings = []
    file_up = meta.get('declared_up') or '+Y'
    if declared_up in (None, 'auto'):
        cv = convert.cv_from_gl(raw['cam_to_file_gl']) if 'cam_to_file_gl' in raw else None
        estimated, scores, evidence = up_axis.estimate(cv, raw.get('mesh_vertices'))
        up = estimated if (cv is not None or 'mesh_vertices' in raw) else file_up
        up_report = dict(mode='auto', file=file_up, estimated=estimated, scores=scores, evidence=evidence)
        if up != file_up:
            warnings.append(f'The file says {file_up} is up, but the camera and terrain say {up}. Using {up}.')
    else:
        up = declared_up
        up_report = dict(mode='known', file=file_up, estimated=None)
    mpu = metres_per_unit if metres_per_unit is not None else (meta.get('metres_per_unit') or 1.0)
    M = convert.file_to_world(up, mpu, scale_fix)
    out = dict(file_to_world=M, plate_frames=np.asarray(frames, dtype=np.int64))
    width, height = image_size
    plate_times = (np.asarray(frames, dtype=np.float64) - frames[0]) / float(plate_fps)
    out['plate_times'] = plate_times
    centres = None
    if 'cam_to_file_gl' in raw:
        c = meta['camera']
        cam_world = convert.camera_to_world(convert.cv_from_gl(raw['cam_to_file_gl']), M)
        K = convert.intrinsics(raw['focal_mm'], c['sensor_width'], c['sensor_height'], width, height,
                               c['sensor_fit'], c['shift'][0], c['shift'][1])
        out['plate_world_to_camera'] = convert.invert_rigid(cam_world)
        out['plate_K'] = K
        if working_times is not None:
            # The 30 fps working clip is made by ffmpeg's fps filter, which shows the NEAREST plate frame at each
            # working time. Use the same plate frame's camera, so camera and pixels always agree.
            index = nearest_plate_index(working_times, plate_fps, len(frames))
            w2c = convert.invert_rigid(cam_world[index])
            out.update(world_to_camera=w2c, K=K[index], cam_angvel=convert.cam_angvel(w2c),
                       working_times=np.asarray(working_times, dtype=np.float64),
                       working_plate_frames=np.asarray(frames, dtype=np.int64)[index])
        centres = cam_world[:, :3, 3]
    if 'mesh_vertices' in raw:
        out['mesh_vertices'] = convert.points_to_world(raw['mesh_vertices'], M).astype(np.float32)
        out['mesh_faces'] = np.asarray(raw['mesh_faces'], dtype=np.int32)
    # A unit flag that disagrees with the data is the most common error, so ignoring it is the first fix tried.
    declared = meta.get('metres_per_unit')
    preferred = (1.0 / declared,) if declared and metres_per_unit is None and abs(declared - 1.0) > 1e-9 else ()
    report = scale_check.import_check(centres, out.get('mesh_vertices'), preferred=preferred)
    if preferred and report.get('suggested_factor') == preferred[0]:
        report['message'] = (f"The file says one unit is {declared:g} m, but the camera and terrain only make sense "
                             f"in metres. Scaling by {preferred[0]:g} ignores the file's unit setting.")
    return out, dict(up_axis=up, up_axis_report=up_report, metres_per_unit=mpu, scale_fix=scale_fix,
                     scale_check=report, conversion_warnings=warnings)


def import_scene(project, path, *, image_size, plate_fps, frame_count, working_times, camera=None, mesh=None,
                 first_frame=None, scale_fix=1.0, up_axis=None, info=None, emit=None):
    """Read `camera` and/or `mesh` from `path` and save them in `project`. Returns the saved metadata."""
    project, path = Path(project), Path(path).resolve()
    if emit:
        emit('progress', current=0, total=0, message='Reading the scene file')
    info = info or inspect(path)
    if camera is None and mesh is None:
        camera, mesh = default_choices(info)
    frames = plate_frames(info, frame_count, first_frame)
    raw, meta = read_raw(path, frames, camera=camera, mesh=mesh, fps=plate_fps)
    arrays, conv = build(raw, meta, image_size=image_size, plate_fps=plate_fps, frames=frames,
                         working_times=working_times, scale_fix=scale_fix, declared_up=up_axis)
    record = dict(version=VERSION, source=str(path), format=meta['format'], camera=meta.get('camera'),
                  mesh=meta.get('mesh'), plate_fps=plate_fps, image_size=list(image_size),
                  file_frame_range=info.get('file_frame_range'), first_frame=int(frames[0]),
                  warnings=meta.get('warnings', []) + conv.pop('conversion_warnings'), **conv)
    tmp = project / (NPZ + '.partial.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(project / NPZ)
    from ..storage import atomic_json
    atomic_json(project / JSON, record)
    return record


def load(project):
    """(arrays, metadata) of the project's imported scene, or None."""
    project = Path(project)
    if not (project / JSON).is_file() or not (project / NPZ).is_file():
        return None
    meta = json.loads((project / JSON).read_text(encoding='utf-8'))
    with np.load(project / NPZ, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, meta


def run_stage(project, emit):
    """Worker stage 'scene_import': read the options saved by Prepare Video and import the scene into the project.

    project.json "scene_import": {"path", "camera", "mesh", "plate_fps", "first_frame", "scale_fix", "up_axis"}.
    Cached: re-imports only when the file, the options or this importer change."""
    import hashlib
    project = Path(project)
    manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    options = manifest.get('scene_import')
    if not options:
        raise ValueError('No pre-calculated scene was chosen for this project.')
    path = Path(options['path'])
    if not path.is_file():
        raise ValueError(f'The scene file is missing: {path}')
    times = manifest.get('original_times')
    if times is None:
        times = [manifest['start'] + i / manifest['fps'] for i in range(manifest['frames'])]
    times = np.asarray(times, dtype=np.float64)
    plate_fps = float(options['plate_fps'])
    stat = path.stat()
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    identity = hashlib.sha256(json.dumps([str(path.resolve()), stat.st_size, stat.st_mtime_ns, options,
                                          times.tolist(), manifest['width'], manifest['height']]).encode() + code).hexdigest()
    old = json.loads((project / JSON).read_text(encoding='utf-8')) if (project / JSON).is_file() else {}
    if old.get('identity') == identity and old.get('options') == options and (project / NPZ).is_file():
        emit('complete', stage='scene_import', output=str(project / NPZ), cached=True)
        return old
    frame_count = int(np.floor(times.max() * plate_fps + 0.5)) + 1
    record = import_scene(project, path, image_size=(manifest['width'], manifest['height']), plate_fps=plate_fps,
                          frame_count=frame_count, working_times=times, camera=options.get('camera'),
                          mesh=options.get('mesh'), first_frame=options.get('first_frame'),
                          scale_fix=float(options.get('scale_fix') or 1.0), up_axis=options.get('up_axis'), emit=emit)
    record['identity'] = identity
    record['options'] = options
    from ..storage import atomic_json
    atomic_json(project / JSON, record)
    check = record['scale_check']
    for warning in record['warnings']:
        emit('info', message='Scene import: ' + warning)
    emit('info', message='Scene scale: ' + check['message'])
    emit('complete', stage='scene_import', output=str(project / NPZ), cached=False, scale_check=check)
    return record


def preview(path, *, image_size, plate_fps=None, camera=None, mesh=None, up_axis=None, scale_fix=1.0, info=None,
            samples=60):
    """What the Scene panel shows before reconstruction: file contents, defaults, detected up axis and scale check.

    Reads about `samples` camera frames only, so it takes seconds even for long shots."""
    path = Path(path).resolve()
    info = info or inspect(path)
    default_cam, default_mesh = default_choices(info)
    camera = camera or default_cam
    mesh = mesh if mesh is not None else default_mesh
    fps = float(plate_fps or info.get('file_fps') or 30.0)
    rng = info.get('file_frame_range') or [0, 0]
    frames = np.unique(np.linspace(rng[0], rng[1], min(samples, rng[1] - rng[0] + 1)).round().astype(np.int64))
    raw, meta = read_raw(path, frames, camera=camera, mesh=mesh or None, fps=fps)
    _, conv = build(raw, meta, image_size=image_size, plate_fps=fps, frames=frames, working_times=None,
                    declared_up=up_axis, scale_fix=scale_fix)
    return dict(info=info, camera=camera, mesh=mesh, plate_fps=fps, file_frame_range=rng,
                warnings=meta.get('warnings', []) + conv.pop('conversion_warnings'), **conv)
