"""GVHMR with a Pre-calculated Camera: known K and camera rotation in, the body placed on the terrain out."""
import json
from pathlib import Path

import numpy as np

from .reader import NPZ, load


def active(project):
    """The project's imported scene arrays when the Pre-calculated Camera is selected, else None."""
    project = Path(project)
    manifest = json.loads((project / 'project.json').read_text(encoding='utf-8'))
    if not manifest.get('scene_import') or not (project / NPZ).is_file():
        return None
    arrays, meta = load(project)
    if 'world_to_camera' not in arrays:
        raise ValueError('The imported scene has no camera. Choose a camera in the scene file.')
    if len(arrays['world_to_camera']) != manifest['frames']:
        raise ValueError('The imported camera does not cover the working clip. Run Reconstruct 3D again.')
    return arrays, meta


def model_inputs(scene, frames):
    """K_fullimg (L,3,3) and cam_angvel (L,6) tensors for GVHMR's predict()."""
    import torch
    arrays, _ = scene
    if len(arrays['K']) != frames:
        raise ValueError('Imported camera and tracking frame counts differ.')
    return (torch.from_numpy(arrays['K'].astype(np.float32)),
            torch.from_numpy(arrays['cam_angvel'].astype(np.float32)))


def place_result(project, result, scene, emit=lambda *a, **k: None):
    """Replace GVHMR's gravity-aligned world (arbitrary heading) with the scene world, standing on the terrain.

    The original world track is kept as smpl_params_global_gvhmr. Writes scene_place.json."""
    import smplx
    import torch
    from scipy.spatial.transform import Rotation
    from ..correction import SoleGeometry
    from ..models import BY_KEY
    from ..storage import atomic_json
    from .placement import place
    from .terrain import Heightfield
    project = Path(project)
    arrays, meta = scene
    incam = result['smpl_params_incam']
    n = len(incam['transl'])
    betas = incam['betas'][0].double().numpy()
    body = smplx.SMPLX(str(BY_KEY['smplx'].path), gender='neutral', num_pca_comps=12, flat_hand_mean=False,
                       batch_size=1).eval()
    geometry = SoleGeometry(body, betas)
    rest_root = geometry.rest[0]
    orient_c = incam['global_orient'].double().numpy()
    transl_c = incam['transl'].double().numpy()
    rv = np.concatenate([orient_c[:, None], incam['body_pose'].double().numpy().reshape(n, 21, 3)], 1)
    quats = Rotation.from_rotvec(rv.reshape(-1, 3)).as_quat().reshape(n, 22, 4)[..., [3, 0, 1, 2]]
    roots = transl_c + rest_root
    probes = np.concatenate([geometry.sample(roots[s:s + 128], quats[s:s + 128])[0] for s in range(0, n, 128)])
    logits = result['net_outputs']['static_conf_logits'][0, :, :4]
    probabilities = torch.as_tensor(logits).float().sigmoid().numpy()
    if 'mesh_vertices' not in arrays:
        raise ValueError('Choose a terrain mesh in the scene file: the actor is placed on it.')
    emit('progress', current=0, total=0, message='Placing the actor on the terrain')
    heightfield = Heightfield(arrays['mesh_vertices'], arrays['mesh_faces'])
    with torch.inference_mode():
        rest = body(betas=torch.from_numpy(betas[None]).float()).vertices[0].numpy()
    model_height = float(rest[:, 1].max() - rest[:, 1].min())
    actor_height = (meta.get('options') or {}).get('actor_height') or None
    orient_w, transl_w, report = place(orient_c, transl_c, rest_root, probes, probabilities,
                                       arrays['world_to_camera'], heightfield, model_height, actor_height)
    original = result['smpl_params_global']
    result['smpl_params_global_gvhmr'] = {k: v.clone() for k, v in original.items()}
    placed = {k: v.clone() for k, v in incam.items()}
    placed['global_orient'] = torch.from_numpy(orient_w)
    placed['transl'] = torch.from_numpy(transl_w)
    result['smpl_params_global'] = placed
    result['scene_import'] = dict(source=meta.get('source'), size_factor=report['size_factor'])
    report.update(source=meta.get('source'), terrain_coverage=heightfield.coverage, heightfield_cell_m=heightfield.cell)
    atomic_json(project / 'scene_place.json', report)
    check = report['size_check']
    emit('info', message=f"Terrain placement: size factor {report['size_factor']:.3f} from {report['rays']['kept']} stance samples. "
                         + check['message'])
    if report['stance_gap_median_cm'] is not None:
        emit('info', message=f"Stance feet to terrain: median {report['stance_gap_median_cm']:.1f} cm, "
                             f"95th percentile {report['stance_gap_p95_cm']:.1f} cm (Motion Correction locks them).")
    if report['warning']:
        emit('info', message='Terrain placement: ' + report['warning'])
    return report
