"""Image sequence import: a numbered plate (EXR, PNG, JPG, TIF, DPX) becomes one video at native resolution.

RGB MoCap's pipeline is built around a video file. Instead of changing every stage, an image sequence is encoded
once into a high-quality MP4 at the plate's own resolution (4K stays 4K) and frame rate, and a small sidecar JSON
records where it came from (first frame number, fps, pattern). The Pre-calculated Camera reads the sidecar so the
plate's frame numbers line up with the scene file.
"""
import json
import re
from pathlib import Path

EXTENSIONS = ('.exr', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.dpx', '.bmp')
FILTER = 'Image sequences (*.exr *.png *.jpg *.jpeg *.tif *.tiff *.dpx *.bmp)'


def detect(path):
    """The numbered sequence a file belongs to: folder, prefix, padding, suffix, sorted frame numbers."""
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f'{path.name} is not a supported image type.')
    match = re.match(r'^(.*?)(\d+)(\D*)$', path.stem)
    if not match:
        raise ValueError(f'{path.name} has no frame number.')
    prefix, digits, tail = match.groups()
    suffix = tail + path.suffix
    pattern = re.compile(re.escape(prefix) + r'(\d+)' + re.escape(suffix) + '$')
    frames = sorted(int(m.group(1)) for f in path.parent.iterdir() if (m := pattern.match(f.name)))
    if len(frames) < 2:
        raise ValueError('Only one frame of this sequence was found.')
    padding = len(digits) if digits.startswith('0') or len(set(len(str(f)) for f in frames)) > 1 else 0
    missing = sorted(set(range(frames[0], frames[-1] + 1)) - set(frames))
    return dict(folder=str(path.parent), prefix=prefix, suffix=suffix, padding=padding, first=frames[0],
                last=frames[-1], count=len(frames), missing=missing[:20], missing_count=len(missing),
                pattern=prefix + (f'%0{padding}d' if padding else '%d') + suffix)


def output_path(info, fps, root):
    import hashlib
    key = hashlib.sha256(json.dumps([info['folder'], info['pattern'], info['first'], info['last'], fps]).encode()).hexdigest()[:12]
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', info['prefix'].strip('._ -')) or 'sequence'
    return Path(root) / f'{name}_{info["first"]}-{info["last"]}_{key}.mp4'


def encode(info, fps, target, progress=lambda done, total: None, cancelled=lambda: False):
    """Encode the sequence to `target` (H.264, CRF 12, native resolution, constant fps). Writes target + '.json'."""
    import subprocess
    from .video import executable
    if info['missing_count']:
        raise ValueError(f"{info['missing_count']} frames are missing from the sequence (first: {info['missing'][0]}).")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.partial.mp4')
    source = str(Path(info['folder']) / info['pattern'])
    exr = info['suffix'].lower().endswith('.exr')
    command = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-progress', 'pipe:1', '-nostats',
               *(['-apply_trc', 'iec61966_2_1'] if exr else []),  # linear EXR -> sRGB for the models
               '-framerate', f'{fps}', '-start_number', str(info['first']), '-i', source,
               '-frames:v', str(info['count']), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
               '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', '-r', f'{fps}', '-y', str(partial)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    for line in process.stdout:
        if line.startswith('frame='):
            progress(int(line.split('=')[1]), info['count'])
        if cancelled():
            process.kill()
            partial.unlink(missing_ok=True)
            raise ValueError('Image sequence import cancelled.')
    error = process.stderr.read()
    if process.wait() != 0:
        partial.unlink(missing_ok=True)
        raise ValueError('Encoding the image sequence failed: ' + error.strip()[-400:])
    partial.replace(target)
    Path(str(target) + '.json').write_text(json.dumps(dict(sequence=info, fps=fps, version=1), indent=1), encoding='utf-8')
    return target


def sidecar(video):
    """The sequence info for a video made by this importer, or None."""
    path = Path(str(video) + '.json')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
