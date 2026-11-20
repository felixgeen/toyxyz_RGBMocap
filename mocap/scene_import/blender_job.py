"""Run blender_scene.py in the managed Blender runtime as a subprocess. The app process never imports bpy."""
import json
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name('blender_scene.py')
FORMATS = {'.fbx': 'fbx', '.abc': 'abc', '.usd': 'usd', '.usda': 'usd', '.usdc': 'usd', '.usdz': 'usd'}


class SceneImportError(RuntimeError):
    pass


def file_format(path):
    fmt = FORMATS.get(Path(path).suffix.lower())
    if fmt is None:
        raise SceneImportError(f'Unsupported scene file {Path(path).name}. Use FBX, Alembic (.abc) or USD (.usd/.usda/.usdc/.usdz).')
    return fmt


def run(job, timeout=1800):
    from ..retarget.runtime import blender
    with tempfile.TemporaryDirectory(prefix='rgbmocap_scene_') as tmp:
        job_path = Path(tmp) / 'job.json'
        job_path.write_text(json.dumps(job), encoding='utf-8')
        cmd = [str(blender()), '-b', '--factory-startup', '--python-exit-code', '1', '-P', str(SCRIPT), '--', str(job_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode != 0:
            tail = '\n'.join((proc.stdout + proc.stderr).splitlines()[-25:])
            raise SceneImportError(f'Reading the scene file failed:\n{tail}')


def inspect(path):
    """Cameras, meshes, fps, frame range, declared up axis and units of a scene file."""
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix='rgbmocap_scene_') as tmp:
        out = Path(tmp) / 'inspect.json'
        run({'mode': 'inspect', 'path': str(path), 'format': file_format(path), 'out': str(out)})
        return json.loads(out.read_text(encoding='utf-8'))


def read_raw(path, frames, camera=None, mesh=None, fps=None):
    """Per-frame camera (raw file frame, OpenGL camera axes, focal in mm) and/or a triangulated mesh."""
    import numpy as np
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory(prefix='rgbmocap_scene_') as tmp:
        out = Path(tmp) / 'scene_raw.npz'
        run({'mode': 'read', 'path': str(path), 'format': file_format(path), 'out': str(out), 'fps': fps,
             'frames': [int(f) for f in frames], 'camera': camera, 'mesh': mesh})
        meta = json.loads(Path(str(out) + '.json').read_text(encoding='utf-8'))
        with np.load(out, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, meta
