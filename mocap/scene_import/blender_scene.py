"""Runs INSIDE Blender: blender -b --factory-startup -P blender_scene.py -- job.json

Reads a tracked camera and a mesh from an FBX, Alembic or USD file in the RAW FILE FRAME: coordinates exactly as
stored in the file, with no unit scaling and no axis conversion. The caller (mocap/scene_import/reader.py) converts
to the application's Y-up metre world.

job.json:
  {"mode": "inspect", "path", "format", "out"}            -> out = JSON listing cameras, meshes, fps, units, up axis
  {"mode": "read", "path", "format", "out", "fps", "frames": [int], "camera": name|null, "mesh": name|null}
                                                           -> out = NPZ, out + ".json" = metadata

- ABC: Blender's importer rotates file Y-up into Blender Z-up (blender = Rx(+90) @ file); we undo it. Alembic
  stores time in seconds, so the scene fps is set before import.
- FBX: imported with identity orientation and a global_scale that cancels UnitScaleFactor, so Blender world ==
  file frame. anim_offset=0 so file time t lands on frame t*fps. Intrinsics come from the raw FBX camera
  properties, because Blender ignores ApertureMode/FieldOfView.
- USD: read with pxr (bundled with Blender), not Blender's importer, which applies stage up-axis/units and can
  shift time codes by a frame.

Ported from posefit/io/blender/read_scene.py.
"""
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ABC_BLENDER_FROM_FILE = Matrix.Rotation(math.radians(90.0), 4, 'X')
INCH_MM = 25.4
FBX_TIME_MODE_FPS = {1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
                     10: 25.0, 11: 24.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94, 18: 119.88}


def set_fps(scene, fps):
    whole = max(1, round(fps))
    scene.render.fps = whole
    scene.render.fps_base = whole / fps


# ---------------------------------------------------------------- raw FBX helpers (Blender's own parser)
def fbx_props70(elem):
    out = {}
    for e in elem.elems:
        if e.id == b'Properties70':
            for p in e.elems:
                vals = [v.decode(errors='replace') if isinstance(v, bytes) else v for v in p.props[4:]]
                out[p.props[0].decode(errors='replace')] = vals[0] if len(vals) == 1 else vals
    return out


def fbx_name(b):
    return b.split(b'\x00\x01')[0].decode(errors='replace')


def fbx_ktime(root, version):
    from io_scene_fbx.fbx_utils import FBX_KTIME_V7, FBX_KTIME_V8, FBX_TIMECODE_DEFINITION_TO_KTIME_PER_SECOND
    kt = FBX_KTIME_V8 if version >= 8000 else FBX_KTIME_V7
    for e in root.elems:
        if e.id == b'FBXHeaderExtension':
            for f in e.elems:
                if f.id == b'OtherFlags':
                    for g in f.elems:
                        if g.id == b'TCDefinition':
                            kt = FBX_TIMECODE_DEFINITION_TO_KTIME_PER_SECOND.get(g.props[0], FBX_KTIME_V8)
    return kt


def fbx_raw_info(path):
    """Global settings and raw camera properties per camera model name."""
    from io_scene_fbx import parse_fbx
    root, version = parse_fbx.parse(path)
    top = {e.id: e for e in root.elems}
    gs = fbx_props70(top[b'GlobalSettings'])
    info = {'fbx_version': version, 'global_settings': {k: gs.get(k) for k in (
        'UpAxis', 'UpAxisSign', 'FrontAxis', 'FrontAxisSign', 'UnitScaleFactor', 'OriginalUnitScaleFactor',
        'TimeMode', 'CustomFrameRate')}}
    template = {}
    for d in top[b'Definitions'].elems:
        if d.id == b'ObjectType' and d.props[0] == b'NodeAttribute':
            for t in d.elems:
                if t.id == b'PropertyTemplate' and t.props[0] == b'FbxCamera':
                    template = fbx_props70(t)
    models, attrs, curve_nodes, curves = {}, {}, {}, {}
    for o in top[b'Objects'].elems:
        if o.id == b'Model':
            models[o.props[0]] = fbx_name(o.props[1])
        elif o.id == b'NodeAttribute' and o.props[2] == b'Camera':
            attrs[o.props[0]] = fbx_props70(o)
        elif o.id == b'AnimationCurveNode':
            curve_nodes[o.props[0]] = fbx_name(o.props[1])
        elif o.id == b'AnimationCurve':
            for e in o.elems:
                if e.id == b'KeyTime':
                    curves[o.props[0]] = np.asarray(e.props[0], dtype=np.int64)
    cameras, animated = {}, {}
    for c in top[b'Connections'].elems:
        kind, child, parent = c.props[0], c.props[1], c.props[2]
        if kind == b'OO' and child in attrs and parent in models:
            props = dict(template)
            props.update(attrs[child])
            cameras[models[parent]] = {'props': props, 'attr': child}
        if kind == b'OP' and child in curve_nodes and parent in attrs:
            animated.setdefault(parent, []).append(c.props[3].decode(errors='replace'))
    for cam in cameras.values():
        cam['lens_animated_props'] = animated.get(cam.pop('attr'), [])
    ticks = [v for v in curves.values() if len(v)]
    info['key_times'] = [int(min(v.min() for v in ticks)), int(max(v.max() for v in ticks))] if ticks else None
    info['ktime_per_second'] = fbx_ktime(root, version)
    info['cameras'] = cameras
    return info


def fbx_focal_mm(p):
    """Focal length and film size (mm) from raw FBX camera properties, per ApertureMode."""
    mode = int(p['ApertureMode'])
    fw, fh = float(p['FilmWidth']) * INCH_MM, float(p['FilmHeight']) * INCH_MM
    if mode == 3:
        f = float(p['FocalLength'])
    elif mode == 1:
        f = fw / 2.0 / math.tan(math.radians(float(p['FieldOfView'])) / 2.0)
    elif mode == 2:
        f = fh / 2.0 / math.tan(math.radians(float(p['FieldOfView'])) / 2.0)
    elif mode == 0:
        f = fw / 2.0 / math.tan(math.radians(float(p['FieldOfViewX'])) / 2.0)
    else:
        raise RuntimeError(f'FBX ApertureMode {mode} is not supported')
    return f, fw, fh, mode


# ---------------------------------------------------------------- Blender-imported formats (FBX, ABC)
def import_blender(job):
    """Import the file; return (scene, file_from_blender matrix, metadata)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = float(job.get('fps') or 30.0)
    meta = {'format': job['format'], 'path': job['path'], 'warnings': []}
    if job['format'] == 'abc':
        set_fps(scene, fps)
        bpy.ops.wm.alembic_import(filepath=job['path'], set_frame_range=True, as_background_job=False)
        meta.update(declared_up='+Y', metres_per_unit=None, file_fps=None,
                    file_frame_range=[scene.frame_start, scene.frame_end])
        return scene, ABC_BLENDER_FROM_FILE.inverted(), meta
    raw = fbx_raw_info(job['path'])
    meta['fbx'] = {k: v for k, v in raw.items() if k != 'cameras'}
    meta['fbx_cameras'] = raw['cameras']
    g = raw['global_settings']
    unit = float(g['UnitScaleFactor'] or 1.0)  # centimetres per file unit
    mode = g.get('TimeMode')
    file_fps = float(g.get('CustomFrameRate') or 0) if mode == 14 else FBX_TIME_MODE_FPS.get(mode)
    if not job.get('fps') and file_fps:
        fps = file_fps
    set_fps(scene, fps)
    bpy.ops.import_scene.fbx(filepath=job['path'], use_anim=True, anim_offset=0.0, global_scale=100.0 / unit,
                             use_manual_orientation=True, axis_forward='Y', axis_up='Z',
                             use_custom_props=False, use_image_search=False)
    set_fps(scene, fps)
    rng = None
    if raw['key_times']:
        per_s = raw['ktime_per_second']
        rng = [round(t / per_s * fps) for t in raw['key_times']]
    up = ('+' if (g['UpAxisSign'] or 1) >= 0 else '-') + 'XYZ'[int(g['UpAxis'] if g['UpAxis'] is not None else 1)]
    meta.update(declared_up=up, metres_per_unit=unit / 100.0, file_fps=file_fps, file_frame_range=rng)
    return scene, Matrix.Identity(4), meta


def blender_inventory(scene):
    cams, meshes = [], []
    for ob in bpy.data.objects:
        if ob.type == 'CAMERA':
            chain, p = [], ob
            while p is not None:
                chain.append(p)
                p = p.parent
            # FBX animates through actions; Alembic through Transform Cache constraints.
            animated = any((o.animation_data and o.animation_data.action) or
                           any(c.type == 'TRANSFORM_CACHE' for c in o.constraints) for o in chain)
            cams.append({'name': ob.name, 'animated': bool(animated)})
        elif ob.type == 'MESH':
            me = ob.data
            meshes.append({'name': ob.name, 'vertices': len(me.vertices), 'polygons': len(me.polygons)})
    return cams, meshes


def blender_read(job, scene, file_from_blender, meta):
    objs = bpy.data.objects
    frames = [int(f) for f in job['frames']]
    out = {}
    if job.get('camera'):
        if job['camera'] not in objs or objs[job['camera']].type != 'CAMERA':
            raise RuntimeError(f"Camera {job['camera']!r} was not found in the file.")
        cam = objs[job['camera']]
        poses, lens = [], []
        for f in frames:
            scene.frame_set(f)
            loc, rot, _ = (file_from_blender @ cam.matrix_world).decompose()
            poses.append(np.array(Matrix.Translation(loc) @ rot.to_matrix().to_4x4()))
            lens.append(cam.data.lens)
        cd = cam.data
        cmeta = {'name': cam.name, 'sensor_width': cd.sensor_width, 'sensor_height': cd.sensor_height,
                 'sensor_fit': cd.sensor_fit, 'shift': [cd.shift_x, cd.shift_y]}
        focal = np.asarray(lens, dtype=np.float64)
        if meta['format'] == 'fbx':
            raw = meta['fbx_cameras'].get(cam.name)
            if raw is None:
                raise RuntimeError(f'FBX: no camera attribute for {cam.name!r}')
            f_raw, fw, fh, mode = fbx_focal_mm(raw['props'])
            cmeta.update(sensor_width=fw, sensor_height=fh, fbx_aperture_mode=mode)
            if raw['lens_animated_props'] and mode != 3:
                raise RuntimeError(f"FBX animated lens {raw['lens_animated_props']} with ApertureMode {mode} is not supported.")
            if not raw['lens_animated_props']:
                focal = np.full(len(frames), f_raw)
            sq = raw['props'].get('FilmSqueezeRatio')
            if sq is not None and abs(float(sq) - 1.0) > 1e-6:
                meta['warnings'].append(f'FBX FilmSqueezeRatio {sq} (anamorphic) is ignored.')
        out['cam_to_file_gl'] = np.asarray(poses, dtype=np.float64)
        out['focal_mm'] = focal
        meta['camera'] = cmeta
    if job.get('mesh'):
        if job['mesh'] not in objs or objs[job['mesh']].type != 'MESH':
            raise RuntimeError(f"Mesh {job['mesh']!r} was not found in the file.")
        ob = objs[job['mesh']]
        scene.frame_set(frames[0])
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        co = np.empty(nv * 3)
        me.vertices.foreach_get('co', co)
        tris = np.empty(nt * 3, dtype=np.int64)
        me.loop_triangles.foreach_get('vertices', tris)
        ev.to_mesh_clear()
        mw = np.array(file_from_blender @ ob.matrix_world)
        out['mesh_vertices'] = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        out['mesh_faces'] = tris.reshape(-1, 3)
        meta['mesh'] = {'name': ob.name, 'vertices': nv, 'triangles': nt}
    return out


# ---------------------------------------------------------------- USD (raw via pxr)
def usd_m4(m):
    """Gf.Matrix4d (row-vector) -> numpy column-vector 4x4."""
    return np.array([[m[r][c] for r in range(4)] for c in range(4)], dtype=np.float64)


def usd_open(job):
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(job['path'])
    tcps = st.GetTimeCodesPerSecond()
    fps = float(job.get('fps') or st.GetFramesPerSecond() or tcps)
    meta = {'format': 'usd', 'path': job['path'], 'warnings': [],
            'declared_up': '+' + UsdGeom.GetStageUpAxis(st), 'metres_per_unit': UsdGeom.GetStageMetersPerUnit(st),
            'file_fps': st.GetFramesPerSecond() or tcps, 'time_codes_per_second': tcps,
            'file_frame_range': [round(st.GetStartTimeCode() * fps / tcps), round(st.GetEndTimeCode() * fps / tcps)]}
    return st, fps, meta


def usd_inventory(st):
    from pxr import UsdGeom
    cams, meshes = [], []
    for p in st.Traverse():
        if p.IsA(UsdGeom.Camera):
            ops = UsdGeom.Xformable(p).GetOrderedXformOps()
            animated = any(op.GetNumTimeSamples() > 1 for op in ops)
            par = p.GetParent()
            while not animated and par and par.IsA(UsdGeom.Xformable):
                animated = any(op.GetNumTimeSamples() > 1 for op in UsdGeom.Xformable(par).GetOrderedXformOps())
                par = par.GetParent()
            cams.append({'name': p.GetName(), 'path': str(p.GetPath()), 'animated': animated})
        elif p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            counts = mesh.GetFaceVertexCountsAttr().Get() or []
            pts = mesh.GetPointsAttr().Get() or []
            # SynthEyes writes Xform "<name>" holding Mesh "<name>_mesh"; list the friendlier parent name too.
            meshes.append({'name': p.GetName(), 'path': str(p.GetPath()), 'parent': p.GetParent().GetName(),
                           'vertices': len(pts), 'polygons': len(counts)})
    return cams, meshes


def usd_find(st, name, schema):
    for p in st.Traverse():
        if p.IsA(schema) and (p.GetName() == name or str(p.GetPath()) == name or
                              (schema.__name__ == 'Mesh' and p.GetParent().GetName() == name)):
            return p
    return None


def usd_read(job, st, fps, meta):
    from pxr import UsdGeom
    tcps = meta['time_codes_per_second']
    mpu = meta['metres_per_unit']
    frames = [int(f) for f in job['frames']]
    out = {}
    if job.get('camera'):
        prim = usd_find(st, job['camera'], UsdGeom.Camera)
        if prim is None:
            raise RuntimeError(f"Camera {job['camera']!r} was not found in the file.")
        cam, xf = UsdGeom.Camera(prim), UsdGeom.Xformable(prim)
        to_mm = 100.0 * mpu  # USD lens values are tenths of a scene unit
        poses, lens = [], []
        for f in frames:
            t = f * tcps / fps
            m = usd_m4(xf.ComputeLocalToWorldTransform(t))
            m[:3, :3] /= np.linalg.norm(m[:3, :3], axis=0)
            poses.append(m)
            lens.append(cam.GetFocalLengthAttr().Get(t) * to_mm)
        hap, vap = cam.GetHorizontalApertureAttr().Get(0), cam.GetVerticalApertureAttr().Get(0)
        meta['camera'] = {'name': prim.GetName(), 'sensor_width': hap * to_mm, 'sensor_height': vap * to_mm,
                          'sensor_fit': 'HORIZONTAL',
                          'shift': [cam.GetHorizontalApertureOffsetAttr().Get(0) / hap,
                                    cam.GetVerticalApertureOffsetAttr().Get(0) / max(hap, vap)]}
        out['cam_to_file_gl'] = np.asarray(poses, dtype=np.float64)
        out['focal_mm'] = np.asarray(lens, dtype=np.float64)
    if job.get('mesh'):
        prim = usd_find(st, job['mesh'], UsdGeom.Mesh)
        if prim is None:
            raise RuntimeError(f"Mesh {job['mesh']!r} was not found in the file.")
        mesh = UsdGeom.Mesh(prim)
        t0 = frames[0] * tcps / fps
        pts = np.asarray(mesh.GetPointsAttr().Get(t0), dtype=np.float64)
        mw = usd_m4(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t0))
        pts = pts @ mw[:3, :3].T + mw[:3, 3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(t0), dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(t0), dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tri = [np.stack([starts[counts == n], starts[counts == n] + k, starts[counts == n] + k + 1], axis=1)
               for n in np.unique(counts) for k in range(1, n - 1)]
        faces = idx[np.concatenate(tri)] if tri else np.zeros((0, 3), np.int64)
        out['mesh_vertices'] = pts
        out['mesh_faces'] = faces
        meta['mesh'] = {'name': prim.GetName(), 'path': str(prim.GetPath()), 'vertices': len(pts),
                        'triangles': len(faces)}
    return out


# ---------------------------------------------------------------- entry
def main(job):
    if job['format'] == 'usd':
        st, fps, meta = usd_open(job)
        if job['mode'] == 'inspect':
            meta['cameras'], meta['meshes'] = usd_inventory(st)
        else:
            arrays = usd_read(job, st, fps, meta)
    else:
        scene, file_from_blender, meta = import_blender(job)
        if job['mode'] == 'inspect':
            meta['cameras'], meta['meshes'] = blender_inventory(scene)
        else:
            arrays = blender_read(job, scene, file_from_blender, meta)
    meta.pop('fbx_cameras', None)
    out = Path(job['out'])
    if job['mode'] == 'inspect':
        out.write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
        return
    np.savez(out, frames=np.asarray(job['frames'], dtype=np.int64), **arrays)
    Path(str(out) + '.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    job = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
    try:
        main(job)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
