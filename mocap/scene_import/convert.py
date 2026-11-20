"""Pure NumPy conversion from a raw scene file frame to the application world.

Application world: +Y up, metres, right-handed (as everywhere else in RGB MoCap).
Cameras: OpenCV axes (+X right, +Y down, +Z forward). Stored as world_to_camera, like camera_tracks.record.
"""
import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

# Rotation taking the file's declared up axis to +Y while keeping the frame right-handed.
UP_ROTATIONS = {
    '+Y': np.eye(3),
    '-Y': np.diag([1.0, -1.0, -1.0]),
    '+Z': np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
    '-Z': np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
    '+X': np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]),
    '-X': np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]]),
}


def cv_from_gl(cam_to_file_gl):
    """Camera-to-file poses with OpenGL/Blender camera axes (looking down -Z, +Y up) -> OpenCV camera axes."""
    return np.asarray(cam_to_file_gl, dtype=np.float64) @ GL_TO_CV


def intrinsics(focal_mm, sensor_width_mm, sensor_height_mm, width, height, sensor_fit='AUTO', shift_x=0.0, shift_y=0.0):
    """Per-frame pinhole K from a Blender-style camera. Shift is in units of the larger image side."""
    focal_mm = np.atleast_1d(np.asarray(focal_mm, dtype=np.float64))
    if sensor_fit == 'VERTICAL':
        f_px = focal_mm / sensor_height_mm * height
    elif sensor_fit == 'HORIZONTAL':
        f_px = focal_mm / sensor_width_mm * width
    else:
        f_px = focal_mm / sensor_width_mm * max(width, height)
    side = max(width, height)
    K = np.zeros((len(focal_mm), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f_px
    K[:, 0, 2] = width / 2.0 - shift_x * side
    K[:, 1, 2] = height / 2.0 + shift_y * side
    K[:, 2, 2] = 1.0
    return K


def file_to_world(declared_up='+Y', metres_per_unit=1.0, scale_fix=1.0):
    """4x4 similarity mapping raw file coordinates to the application world."""
    if declared_up not in UP_ROTATIONS:
        raise ValueError(f'Unknown up axis {declared_up!r}')
    s = float(metres_per_unit or 1.0) * float(scale_fix)
    if not np.isfinite(s) or s <= 0:
        raise ValueError('Scene scale must be a positive number')
    M = np.eye(4)
    M[:3, :3] = s * UP_ROTATIONS[declared_up]
    return M


def split_similarity(M):
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    return s, M[:3, :3] / s, M[:3, 3]


def camera_to_world(cam_to_file_cv, M):
    """Rigid camera poses in the world: rotation R @ Rc, centre M(t). Scale moves centres, never rotations."""
    s, R, t = split_similarity(M)
    out = np.array(cam_to_file_cv, dtype=np.float64, copy=True)
    out[:, :3, :3] = R @ out[:, :3, :3]
    out[:, :3, 3] = s * out[:, :3, 3] @ R.T + t
    return out


def points_to_world(points, M):
    return np.asarray(points, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]


def invert_rigid(m):
    m = np.asarray(m, dtype=np.float64)
    out = np.zeros_like(m)
    R = m[..., :3, :3]
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum('...ji,...j->...i', R, m[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def resample(cam_to_world, K, source_times, target_times):
    """Camera poses and K at new times: slerp for rotation, linear for centre and K, clamped at the ends."""
    from scipy.spatial.transform import Rotation, Slerp
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.clip(np.asarray(target_times, dtype=np.float64), source_times[0], source_times[-1])
    if len(source_times) == 1:
        n = len(target_times)
        return np.repeat(cam_to_world[:1], n, 0), np.repeat(K[:1], n, 0)
    if np.any(np.diff(source_times) <= 0):
        raise ValueError('Camera sample times must increase')
    rot = Slerp(source_times, Rotation.from_matrix(cam_to_world[:, :3, :3]))(target_times)
    out = np.zeros((len(target_times), 4, 4))
    out[:, :3, :3] = rot.as_matrix()
    for i in range(3):
        out[:, i, 3] = np.interp(target_times, source_times, cam_to_world[:, i, 3])
    out[:, 3, 3] = 1.0
    K_out = np.zeros((len(target_times), 3, 3))
    for i in range(3):
        for j in range(3):
            K_out[:, i, j] = np.interp(target_times, source_times, K[:, i, j])
    return out, K_out


def cam_angvel(world_to_camera):
    """GVHMR's camera angular velocity (L,6), as hmr4d.utils.geo_transform.compute_cam_angvel computes it.

    6D of R_w2c(t+1) @ R_w2c(t)^T (pytorch3d matrix_to_rotation_6d = first two ROWS), with the last row repeated."""
    R = np.asarray(world_to_camera, dtype=np.float64)[:, :3, :3]
    if len(R) < 2:
        return np.tile([1.0, 0, 0, 0, 1.0, 0], (len(R), 1))
    rel = R[1:] @ np.swapaxes(R[:-1], -1, -2)
    six = rel[:, :2, :].reshape(len(rel), 6)
    return np.concatenate([six, six[-1:]], 0)
