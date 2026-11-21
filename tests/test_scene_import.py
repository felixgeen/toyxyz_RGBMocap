"""Pre-calculated Camera / Mesh import: conversion, up axis, scale check, and a Blender round trip.

Run: runtime\\python.exe -X utf8 -m unittest tests.test_scene_import
The round-trip test needs the managed Blender runtime and is skipped without it.
"""
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mocap.scene_import import convert, reader, scale_check, up_axis


def look_at_cv(centre, target, up=(0, 1, 0)):
    """Camera-to-world (OpenCV axes) at centre looking at target, image-down along -up."""
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1)
    m[:3, 3] = centre
    return m


def terrain_grid(size=20.0, n=60, up=1):
    """A gently rolling square terrain around the origin; up=1 for Y-up, 2 for Z-up."""
    a = np.linspace(-size / 2, size / 2, n)
    u, w = np.meshgrid(a, a)
    h = 0.3 * np.sin(u / 3.0) * np.cos(w / 4.0)
    pts = np.stack([u.ravel(), h.ravel(), w.ravel()], 1) if up == 1 else np.stack([u.ravel(), w.ravel(), h.ravel()], 1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            k = i * n + j
            faces += [[k, k + 1, k + n], [k + 1, k + n + 1, k + n]]
    return pts, np.asarray(faces)


def walk_cameras(frames=50, height=1.6):
    """A camera walking along X at a fixed height, panning slowly, in a Y-up world (OpenCV axes)."""
    out = []
    for i in range(frames):
        c = np.array([-3.0 + 0.1 * i, height, -6.0])
        out.append(look_at_cv(c, c + np.array([math.sin(0.01 * i), -0.15, 1.0])))
    return np.asarray(out)


Z_FROM_Y = np.array([[1.0, 0, 0, 0], [0, 0, -1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])  # Y-up -> Z-up file


class ConversionTests(unittest.TestCase):
    def test_gl_cv_roundtrip_and_inverse(self):
        cams = walk_cameras(5)
        gl = cams @ convert.GL_TO_CV  # GL_TO_CV is its own inverse
        np.testing.assert_allclose(convert.cv_from_gl(gl), cams, atol=1e-12)
        np.testing.assert_allclose(convert.invert_rigid(cams) @ cams, np.tile(np.eye(4), (5, 1, 1)), atol=1e-12)

    def test_intrinsics_match_blender_convention(self):
        K = convert.intrinsics([36.0], 36.0, 24.0, 1920, 1080, 'HORIZONTAL', 0.1, -0.05)[0]
        self.assertAlmostEqual(K[0, 0], 1920.0)
        self.assertAlmostEqual(K[0, 2], 960.0 - 0.1 * 1920)
        self.assertAlmostEqual(K[1, 2], 540.0 - 0.05 * 1920)

    def test_file_to_world_scale_and_up(self):
        M = convert.file_to_world('+Z', 0.01)
        np.testing.assert_allclose(convert.points_to_world([[100.0, 200.0, 300.0]], M), [[1.0, 3.0, -2.0]])
        cams = walk_cameras(3)
        world = convert.camera_to_world(cams, convert.file_to_world('+Y', 2.0))
        np.testing.assert_allclose(world[:, :3, :3], cams[:, :3, :3])  # scale never touches rotation
        np.testing.assert_allclose(world[:, :3, 3], 2.0 * cams[:, :3, 3])

    def test_resample_25_to_30(self):
        cams = walk_cameras(26)
        K = np.tile(np.diag([1000.0, 1000.0, 1.0]), (26, 1, 1))
        src = np.arange(26) / 25.0
        dst = np.arange(31) / 30.0
        out, K2 = convert.resample(cams, K, src, dst)
        np.testing.assert_allclose(out[0], cams[0], atol=1e-9)
        np.testing.assert_allclose(out[-1], cams[-1], atol=1e-9)
        np.testing.assert_allclose(out[15, :3, 3], cams[:, :3, 3].mean(0) * 0 + np.interp(0.5, src, cams[:, 0, 3]) *
                                   np.array([1, 0, 0]) + cams[0, :3, 3] * np.array([0, 1, 1]), atol=1e-9)
        self.assertEqual(K2.shape, (31, 3, 3))

    def test_cam_angvel_matches_gvhmr_definition(self):
        w2c = convert.invert_rigid(walk_cameras(10))
        got = convert.cam_angvel(w2c)
        R = w2c[:, :3, :3]
        rel = R[1:] @ np.swapaxes(R[:-1], 1, 2)
        expected = np.concatenate([rel[:, :2, :].reshape(-1, 6), rel[-1:, :2, :].reshape(-1, 6)])
        np.testing.assert_allclose(got, expected)
        static = convert.cam_angvel(np.tile(np.eye(4), (4, 1, 1)))
        np.testing.assert_allclose(static, np.tile([1, 0, 0, 0, 1, 0], (4, 1)))


class UpAxisTests(unittest.TestCase):
    def test_detects_z_up_data_in_a_y_up_file(self):
        cams = Z_FROM_Y @ walk_cameras(40)
        pts, _ = terrain_grid(up=2)
        best, _, evidence = up_axis.estimate(cams, pts)
        self.assertEqual(best, '+Z')
        self.assertGreater(evidence['camera_above_terrain'], 0.9)

    def test_y_up(self):
        pts, _ = terrain_grid(up=1)
        self.assertEqual(up_axis.estimate(walk_cameras(40), pts)[0], '+Y')


class ScaleCheckTests(unittest.TestCase):
    def setUp(self):
        self.pts, _ = terrain_grid()
        self.centres = walk_cameras(30)[:, :3, 3]

    def test_metric_scene_is_ok(self):
        self.assertEqual(scale_check.import_check(self.centres, self.pts)['status'], 'ok')

    def test_millimetres_read_as_metres_suggests_a_fix(self):
        r = scale_check.import_check(self.centres * 1000, self.pts * 1000)
        self.assertEqual(r['status'], 'suspicious')
        self.assertIn(0.001, r['candidates'])

    def test_centimetres_can_pass_the_import_check(self):
        # A 1.6 m handheld scene in centimetres looks like a 160 m drone shot: only the post-solve check can tell.
        self.assertEqual(scale_check.import_check(self.centres * 100, self.pts * 100)['status'], 'ok')

    def test_preferred_factor_wins(self):
        r = scale_check.import_check(self.centres * 0.01, self.pts * 0.01, preferred=(100.0,))
        self.assertEqual(r['suggested_factor'], 100.0)

    def test_camera_below_terrain(self):
        c = self.centres.copy()
        c[:, 1] = -5
        self.assertIn('below', scale_check.import_check(c, self.pts)['message'])

    def test_size_factor(self):
        self.assertEqual(scale_check.size_factor_check(1.1)['status'], 'ok')
        r = scale_check.size_factor_check(98.0)
        self.assertEqual((r['status'], r['fix']), ('unit_error', 0.01))
        self.assertEqual(scale_check.size_factor_check(0.105)['fix'], 10.0)
        self.assertEqual(scale_check.size_factor_check(3.0)['status'], 'badly_scaled')
        self.assertAlmostEqual(scale_check.scale_from_actor_height(1.8, 180.0), 0.01)


class BuildTests(unittest.TestCase):
    def test_build_from_raw_z_up_millimetre_file(self):
        """A file declaring +Y and metres that really holds Z-up millimetres: up is fixed, scale is flagged."""
        cams_world = walk_cameras(26)
        pts, faces = terrain_grid()
        file_cv = Z_FROM_Y @ cams_world
        file_cv[:, :3, 3] *= 1000
        raw = dict(cam_to_file_gl=file_cv @ convert.GL_TO_CV, focal_mm=np.full(26, 24.0),
                   mesh_vertices=(pts @ Z_FROM_Y[:3, :3].T) * 1000, mesh_faces=faces)
        meta = dict(declared_up='+Y', metres_per_unit=1.0,
                    camera=dict(sensor_width=36.0, sensor_height=24.0, sensor_fit='HORIZONTAL', shift=[0, 0]))
        arrays, conv = reader.build(raw, meta, image_size=(1920, 1080), plate_fps=25, frames=np.arange(26),
                                    working_times=np.arange(31) / 30.0)
        self.assertEqual(conv['up_axis'], '+Z')
        self.assertIn(0.001, conv['scale_check']['candidates'])
        fixed, conv2 = reader.build(raw, meta, image_size=(1920, 1080), plate_fps=25, frames=np.arange(26),
                                    working_times=np.arange(31) / 30.0, scale_fix=0.001)
        self.assertEqual(conv2['scale_check']['status'], 'ok')
        np.testing.assert_allclose(convert.invert_rigid(fixed['plate_world_to_camera']), cams_world, atol=1e-9)
        self.assertEqual(fixed['world_to_camera'].shape, (31, 4, 4))
        self.assertEqual(fixed['cam_angvel'].shape, (31, 6))
        np.testing.assert_allclose(fixed['mesh_vertices'], pts, atol=1e-4)

    def test_working_frames_follow_ffmpeg_nearest_rule(self):
        # 25 fps plate -> 30 fps working clip; exact halves (t*25 = 92.5) go to the earlier plate frame.
        idx = reader.nearest_plate_index(np.arange(1062) / 30.0, 25.0, 885)
        self.assertEqual(idx[111], 92)
        self.assertEqual(idx[37], 31)
        self.assertEqual(idx[-1], 884)

    def test_default_choices_skip_camera_helpers(self):
        info = dict(cameras=[dict(name='persp', animated=False), dict(name='Camera01', animated=True)],
                    meshes=[dict(name='Camera01Screen', polygons=500000), dict(name='Tracker1', polygons=5),
                            dict(name='Ground', polygons=1000)])
        self.assertEqual(reader.default_choices(info), ('Camera01', 'Ground'))


class TerrainTests(unittest.TestCase):
    def test_heightfield_matches_surface_and_low_poly_mesh(self):
        from mocap.scene_import.terrain import Heightfield
        pts, faces = terrain_grid(size=20.0, n=60)
        hf = Heightfield(pts, faces, cell=0.05)
        x, z = np.array([0.3, -4.1, 7.7]), np.array([1.2, 3.3, -6.0])
        np.testing.assert_allclose(hf.height(x, z), 0.3 * np.sin(x / 3.0) * np.cos(z / 4.0), atol=0.01)
        # Two triangles over 20 m: sampling fills the whole grid.
        quad = np.array([[-10.0, 1.0, -10.0], [10.0, 1.0, -10.0], [10.0, 1.0, 10.0], [-10.0, 1.0, 10.0]])
        flat = Heightfield(quad, np.array([[0, 1, 2], [0, 2, 3]]), cell=0.1)
        self.assertGreater(flat.coverage, 0.9)  # the grid's outer row lies past the mesh edge
        np.testing.assert_allclose(flat.height(x, z), 1.0, atol=1e-9)

    def test_ray_scale(self):
        from mocap.scene_import.terrain import Heightfield
        quad = np.array([[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 0.0, 10.0], [-10.0, 0.0, 10.0]])
        hf = Heightfield(quad, np.array([[0, 1, 2], [0, 2, 3]]), cell=0.1)
        origins = np.array([[0.0, 1.6, 0.0], [0.0, 1.6, 0.0]])
        directions = np.array([[0.0, -0.8, 2.0], [0.0, -0.016, 0.04]])  # hits at s=2 and s=100
        np.testing.assert_allclose(hf.ray_scale(origins, directions, lo=1e-3, hi=1e3, steps=700), [2.0, 100.0], rtol=1e-3)


class PlacementTests(unittest.TestCase):
    """A walking body seen by a moving camera; the model's depth is off by k_true; placement recovers it."""

    def scene(self, k_true, n=40):
        from mocap.scene_import.terrain import Heightfield
        pts, faces = terrain_grid(size=30.0, n=120)
        hf = Heightfield(pts, faces, cell=0.05)
        cams = walk_cameras(n)                        # camera-to-world, OpenCV axes
        w2c = convert.invert_rigid(cams)
        rest_root = np.array([0.0, 0.9, 0.0])         # pelvis 0.9 m above the soles at rest
        probes_local = np.array([[-0.1, -0.9, -0.05], [-0.1, -0.9, 0.15], [0.1, -0.9, -0.05], [0.1, -0.9, 0.15]])
        orient_c, transl_c, probes_c = [], [], []
        for t in range(n):
            x, zz = -2.0 + 0.05 * t, 0.0
            pelvis_w = np.array([x, hf.height(x, zz) + 0.9, zz])
            R_wc = w2c[t, :3, :3]
            body_rot_c = R_wc  # body axes aligned with the world
            pelvis_c = R_wc @ pelvis_w + w2c[t, :3, 3]
            feet_c = pelvis_c + probes_local @ body_rot_c.T
            # The model sees the right pixels but misjudges the actor's size: the whole body is scaled by
            # 1/k_true about the camera centre.
            pelvis_m = pelvis_c / k_true
            orient_c.append(Rotation_from_matrix(body_rot_c))
            transl_c.append(pelvis_m - rest_root)
            probes_c.append(feet_c / k_true)
        prob = np.zeros((n, 4))
        prob[:, [0, 1]] = 0.9                          # left foot planted
        return hf, w2c, rest_root, np.array(orient_c), np.array(transl_c), np.array(probes_c), prob

    def test_recovers_size_factor(self):
        from mocap.scene_import.placement import place
        for k_true in (1.0, 1.15, 100.0):
            hf, w2c, rest_root, orient_c, transl_c, probes_c, prob = self.scene(k_true)
            _, transl_w, report = place(orient_c, transl_c, rest_root, probes_c, prob, w2c, hf)
            self.assertAlmostEqual(report['size_factor'], k_true, delta=0.03 * k_true)
            if k_true == 1.0:
                self.assertLess(report['stance_gap_p95_cm'], 3.0)
                self.assertEqual(report['size_check']['status'], 'ok')
            if k_true == 100.0:
                self.assertEqual(report['size_check']['status'], 'unit_error')

    def test_orientation_moves_into_world(self):
        from mocap.scene_import.placement import place
        hf, w2c, rest_root, orient_c, transl_c, probes_c, prob = self.scene(1.0, n=5)
        orient_w, _, _ = place(orient_c, transl_c, rest_root, probes_c, prob, w2c, hf)
        from scipy.spatial.transform import Rotation
        np.testing.assert_allclose(Rotation.from_rotvec(orient_w).as_matrix(), np.tile(np.eye(3), (5, 1, 1)), atol=1e-5)


class SteadyRootTests(unittest.TestCase):
    def test_depth_noise_removed_pixels_kept(self):
        """Depth noise along the camera ray goes; the root stays on its ray (pixels unchanged)."""
        from mocap.scene_import.placement import steady_root
        rng = np.random.default_rng(1)
        n, fps = 200, 25.0
        t = np.arange(n) / fps
        truth = np.stack([0.8 * t, 0.1 * np.sin(t), 5.0 + 0.2 * t], 1)            # smooth walk
        cam = np.array([0.0, 1.6, -2.0])
        ray = truth - cam
        noisy = cam + ray * (1 + 0.03 * rng.standard_normal(n))[:, None]        # 3% depth noise
        yaw = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        model = (truth - truth[0]) @ yaw.T * 1.1 + 7.0                          # same motion, other frame and scale
        smooth, info = steady_root(noisy, model, fps)
        u = ray / np.linalg.norm(ray, axis=1, keepdims=True)
        on_ray = cam + u * np.einsum('ti,ti->t', smooth - cam, u)[:, None]
        acc = lambda x: np.sqrt(np.mean(np.sum(np.diff(x, 2, axis=0) ** 2, 1)))
        self.assertLess(acc(on_ray), 0.1 * acc(noisy))
        error = np.linalg.norm(on_ray - truth, axis=1)
        self.assertLess(np.median(error), 0.04)      # 3% depth noise at ~6 m is ~16 cm before
        self.assertLess(np.percentile(error, 90), 0.10)  # ends of the clip are the weakest (one-sided low-pass)
        self.assertAlmostEqual(info['model_scale'], 1 / 1.1, delta=0.02)


class TerrainContactTests(unittest.TestCase):
    """Pre-calculated Mesh correction targets: terrain by default, footage when they disagree, user overrides win."""

    def setUp(self):
        from mocap.scene_import.terrain import Heightfield
        quad = np.array([[-10.0, 0.5, -10.0], [10.0, 0.5, -10.0], [10.0, 0.5, 10.0], [-10.0, 0.5, 10.0]])
        self.hf = Heightfield(quad, np.array([[0, 1, 2], [0, 2, 3]]), cell=0.1)

    def plan(self, footage_heights):
        n = 10 * len(footage_heights)
        anchors = np.zeros((n, 2, 3))
        segments = []
        for i, h in enumerate(footage_heights):
            anchors[10 * i:10 * i + 10, 0] = [0.2 * i, h, 0.0]
            segments.append(dict(foot=0, start=10 * i, end=10 * i + 10))
        return anchors, segments

    def test_auto_targets(self):
        from mocap.scene_import.contacts import apply
        anchors, segments = self.plan([0.53, 0.9])   # 3 cm and 40 cm above the 0.5 m terrain
        summary = apply(anchors, segments, self.hf)
        self.assertEqual([s['target'] for s in segments], ['terrain', 'footage'])
        np.testing.assert_allclose(anchors[:10, 0, 1], 0.5, atol=1e-9)
        np.testing.assert_allclose(anchors[10:, 0, 1], 0.9)
        self.assertEqual((summary['terrain'], summary['footage_auto']), (1, 1))

    def test_override_and_options(self):
        from mocap.scene_import.contacts import apply
        from mocap.correction_settings import correction_options
        anchors, segments = self.plan([0.53, 0.9])
        apply(anchors, segments, self.hf, [dict(foot=0, frame=3, target='footage'), dict(foot=0, frame=14, target='terrain')])
        self.assertEqual([s['target'] for s in segments], ['footage', 'terrain'])
        np.testing.assert_allclose(anchors[10:, 0, 1], 0.5, atol=1e-9)
        opts = correction_options(dict(mode='terrain', terrain_overrides=[dict(foot=1, frame=7, target='footage')]))
        self.assertEqual(opts['terrain_overrides'], [dict(foot=1, frame=7, target='footage')])
        with self.assertRaises(ValueError):
            correction_options(dict(mode='terrain', terrain_overrides=[dict(foot=2, frame=7, target='footage')]))

    def test_wireframe(self):
        from mocap.scene_import.terrain import wireframe
        lines = wireframe(self.hf, spacing=1.0, step=0.5)
        self.assertGreater(len(lines), 100)
        np.testing.assert_allclose(lines[..., 1], 0.5, atol=1e-9)


def Rotation_from_matrix(m):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(m).as_rotvec()


WRITE_SCENE = r'''
import bpy, json, sys, math
from mathutils import Matrix
job = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 25
scene.frame_start, scene.frame_end = 0, len(job["cams"]) - 1
cam = bpy.data.objects.new("ShotCam", bpy.data.cameras.new("ShotCam"))
scene.collection.objects.link(cam)
cam.data.lens, cam.data.sensor_width, cam.data.sensor_fit = 24.0, 36.0, "HORIZONTAL"
for f, m in enumerate(job["cams"]):
    cam.matrix_world = Matrix(m)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
me = bpy.data.meshes.new("Ground")
me.from_pydata(job["verts"], [], job["faces"])
scene.collection.objects.link(bpy.data.objects.new("Ground", me))
out = job["out"]
if out.endswith(".fbx"):
    bpy.ops.export_scene.fbx(filepath=out, bake_anim=True, bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0.0, add_leaf_bones=False)
elif out.endswith(".abc"):
    bpy.ops.wm.alembic_export(filepath=out, start=0, end=len(job["cams"]) - 1)
else:
    bpy.ops.wm.usd_export(filepath=out, export_animation=True)
'''


def blender_or_none():
    try:
        from mocap.retarget.runtime import blender
        return blender()
    except Exception:
        return None


@unittest.skipIf(blender_or_none() is None, 'Blender runtime not installed')
class BlenderRoundTripTests(unittest.TestCase):
    """Write a known Z-up metric scene with Blender, then read it back through the importer."""

    def roundtrip(self, ext):
        cams_world = walk_cameras(26)
        pts, faces = terrain_grid(n=30)
        blender_cams = Z_FROM_Y @ cams_world @ convert.GL_TO_CV  # Blender is Z-up with GL camera axes
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / f'scene.{ext}'
            job = dict(cams=blender_cams.tolist(), verts=(pts @ Z_FROM_Y[:3, :3].T).tolist(), faces=faces.tolist(),
                       out=str(out))
            (tmp / 'job.json').write_text(json.dumps(job), encoding='utf-8')
            (tmp / 'write.py').write_text(WRITE_SCENE, encoding='utf-8')
            subprocess.run([str(blender_or_none()), '-b', '--factory-startup', '--python-exit-code', '1', '-P',
                            str(tmp / 'write.py'), '--', str(tmp / 'job.json')], check=True, capture_output=True)
            project = tmp / 'project'
            project.mkdir()
            rec = reader.import_scene(project, out, image_size=(1920, 1080), plate_fps=25, frame_count=26,
                                      working_times=np.arange(31) / 30.0, first_frame=0)
            arrays, _ = reader.load(project)
        self.assertEqual(rec['camera']['name'], 'ShotCam')
        self.assertEqual(rec['mesh']['name'].split('/')[-1].replace('_mesh', ''), 'Ground')
        self.assertEqual(rec['scale_check']['status'], 'ok', rec['scale_check'])
        got = convert.invert_rigid(arrays['plate_world_to_camera'])
        np.testing.assert_allclose(got[:, :3, 3], cams_world[:, :3, 3], atol=2e-3)
        np.testing.assert_allclose(got[:, :3, :3], cams_world[:, :3, :3], atol=2e-3)
        self.assertAlmostEqual(float(arrays['plate_K'][0, 0, 0]), 24.0 / 36.0 * 1920, delta=0.5)
        self.assertLess(np.abs(np.sort(arrays['mesh_vertices'][:, 1]) - np.sort(pts[:, 1])).max(), 2e-3)

    def test_fbx(self):
        self.roundtrip('fbx')

    def test_abc(self):
        self.roundtrip('abc')

    def test_usd(self):
        self.roundtrip('usda')


if __name__ == '__main__':
    unittest.main()
