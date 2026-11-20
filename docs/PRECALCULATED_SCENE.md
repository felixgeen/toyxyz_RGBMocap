# Pre-calculated Camera and Mesh

Use a camera solved in a matchmove tool (for example SynthEyes) together with its terrain mesh, instead of a
static or estimated camera. Work in progress: this stage imports the scene. Using it in GVHMR, the
Pre-calculated Mesh correction mode and export back to the original scene frame come next.

## Use

1. Select **GVHMR**, then **Camera > Pre-calculated Camera**.
2. **Choose Scene File...**: FBX, Alembic (`.abc`) or USD (`.usd`, `.usda`, `.usdc`, `.usdz`).
3. Check the choices. They are filled in automatically and read in a few seconds:
   - **Camera in File**: the animated camera.
   - **Terrain Mesh**: the largest mesh, ignoring camera screens and tracker markers.
   - **Plate Frame Rate** and **First Plate Frame**: the scene-file frame that matches the video's first frame.
   - **Up Axis**: Auto detects up from the camera and terrain. Exporters often declare one axis and write another.
   - **Scene Scale**: suggested when the camera height or terrain size is not plausible in metres.
4. **Reconstruct 3D**. The *Import Scene* stage runs first and saves `scene_import.npz` and `scene_import.json`
   in the project. GVHMR then gets the real lens (K per frame) and camera rotation (`cam_angvel`) and runs in
   moving-camera mode. HTD Refine is skipped, because it assumes a static camera.
5. The actor is placed in the scene world with the known camera and stood on the terrain (`scene_place.json`):
   - One size factor per actor slides the body along the camera rays until the stance feet meet the terrain
     (weighted median over stance rays, outliers rejected). The body keeps its size and its pixels.
   - The size factor is the post-solve scale check: near 1 is fine; near a power of ten is a unit error with a
     suggested fix; anything else asks for a re-export in real-world units.
   - GVHMR's own world track is kept as `smpl_params_global_gvhmr`.
   - The viewer shows the imported camera track.

## How it works

- The bundled Blender (`runtime_character`) reads the file in a subprocess (`mocap/scene_import/blender_scene.py`).
  USD is read with `pxr` directly, because Blender's USD importer applies stage settings and can shift time by a frame.
- Everything is converted once to the application world: +Y up, metres, OpenCV cameras stored as world-to-camera.
  `file_to_world` is kept so exports can return to the original frame.
- The 30 fps working clip shows the nearest plate frame at each working time (exact halves go to the earlier
  frame, as ffmpeg does). Each working frame uses that plate frame's camera, so camera and pixels always agree.
  Checked on a 25 fps shot: all 1062 working frames matched.
- `cam_angvel` is precomputed as GVHMR expects it.

## Scale check

The import check catches gross unit errors, such as millimetres read as metres, or a unit flag that disagrees with
the data. Some errors are invisible before solving: a 1.6 m handheld shot in centimetres looks like a 160 m drone
shot. After the first solve, the size factor needed to stand the actor on the terrain settles it (next milestone).

## Tests

`runtime\python.exe -X utf8 -m unittest discover -s tests -p test_scene_import.py`, including a Blender round trip
for FBX, Alembic and USD.
