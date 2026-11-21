"""Pre-calculated Camera controls (GVHMR page). Reading the scene file runs in a thread; no bpy in the GUI process."""
import json
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from .tooltips import tooltip_text

SCENE_FILTER = 'Scene files (*.fbx *.abc *.usd *.usda *.usdc *.usdz);;All files (*)'
UP_AXES = ('+Y', '+Z', '-Y', '-Z', '+X', '-X')


class ScenePreviewLoader(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, parent, path, kwargs):
        super().__init__(parent)
        self.path, self.kwargs = path, kwargs

    def run(self):
        try:
            from .scene_import.reader import preview
            self.done.emit(preview(self.path, **self.kwargs))
        except Exception as error:  # Shown to the user; the console has the details.
            self.failed.emit(str(error))


class SceneImportWorkspace:
    def build_scene_settings(self, layout):
        """Camera choice for GVHMR: Static, or Pre-calculated from an FBX/Alembic/USD file."""
        from .workspace_ui import WrappedLabel
        layout.addWidget(QLabel('Camera'))
        self.gvhmr_camera = QComboBox()
        self.gvhmr_camera.addItem('Static Camera', 'static')
        self.gvhmr_camera.addItem('Pre-calculated Camera', 'precalculated')
        self.gvhmr_camera.setToolTip(tooltip_text(
            'Static Camera keeps the existing workflow. Pre-calculated Camera uses a tracked camera from an FBX, '
            'Alembic or USD file (for example from SynthEyes), with its terrain mesh. Applies on Reconstruct 3D.'))
        layout.addWidget(self.gvhmr_camera)
        self.precalc_panel = QWidget()
        box = QVBoxLayout(self.precalc_panel)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(5)
        row = QHBoxLayout()
        self.precalc_file_button = QPushButton('Choose Scene File...')
        self.precalc_file_button.clicked.connect(self.choose_scene_file)
        row.addWidget(self.precalc_file_button)
        row.addStretch()
        box.addLayout(row)
        self.precalc_file_label = WrappedLabel('No scene file')
        self.precalc_file_label.setObjectName('muted')
        self.precalc_file_label.setWordWrap(True)
        box.addWidget(self.precalc_file_label)
        self.precalc_camera_choice = QComboBox()
        self.precalc_mesh_choice = QComboBox()
        self.precalc_up_axis = QComboBox()
        self.precalc_scale = QComboBox()
        self.precalc_scale.setEditable(True)
        self.precalc_plate_fps = QDoubleSpinBox()
        self.precalc_plate_fps.setRange(1, 240)
        self.precalc_plate_fps.setDecimals(3)
        self.precalc_first_frame = QSpinBox()
        self.precalc_first_frame.setRange(-1000000, 1000000)
        self.precalc_actor_height = QDoubleSpinBox()
        self.precalc_actor_height.setRange(0, 2.5)
        self.precalc_actor_height.setDecimals(2)
        self.precalc_actor_height.setSingleStep(0.01)
        self.precalc_actor_height.setSuffix(' m')
        self.precalc_actor_height.setSpecialValueText('Unknown')
        for label, widget, tip in (
                ('Camera in File', self.precalc_camera_choice, 'The tracked camera to use.'),
                ('Terrain Mesh', self.precalc_mesh_choice, 'The terrain the feet should stand on. Used by the Pre-calculated Mesh correction mode.'),
                ('Plate Frame Rate', self.precalc_plate_fps, "The plate's frame rate. The camera is matched to the 30 fps working clip using it."),
                ('First Plate Frame', self.precalc_first_frame, 'The scene-file frame that matches the first frame of the video.'),
                ('Up Axis', self.precalc_up_axis, 'Auto detects up from the camera and terrain; many exporters write a different up axis than they declare.'),
                ('Scene Scale', self.precalc_scale, 'Multiplies the scene to real-world metres. A suggested fix appears when the scene looks wrongly scaled, on import and after each Reconstruct 3D.'),
                ('Actor Height', self.precalc_actor_height, "Optional. The actor's real height sets the scene scale after Reconstruct 3D. Unknown trusts the solved body's height.")):
            title = QLabel(label)
            title.setToolTip(tooltip_text(tip))
            widget.setToolTip(tooltip_text(tip))
            box.addWidget(title)
            box.addWidget(widget)
        self.precalc_status = WrappedLabel('')
        self.precalc_status.setObjectName('muted')
        self.precalc_status.setWordWrap(True)
        box.addWidget(self.precalc_status)
        layout.addWidget(self.precalc_panel)
        self.precalc_preview = None
        self.precalc_path = None
        self.precalc_loader = None
        for combo in (self.precalc_camera_choice, self.precalc_mesh_choice, self.precalc_up_axis):
            combo.activated.connect(lambda _: self.refresh_precalc_preview())
        self.precalc_scale.activated.connect(lambda _: self.update_precalc_status())
        self.gvhmr_camera.currentIndexChanged.connect(lambda _: self.precalc_camera_mode_changed(True))
        self.precalc_camera_mode_changed(False)

    def precalc_camera_mode_changed(self, refresh=True):
        self.precalc_panel.setVisible(self.gvhmr_camera.currentData() == 'precalculated')
        if refresh:
            self.update_readiness()

    def precalculated_camera(self):
        return self.model_choice.currentData() == 'gvhmr' and self.gvhmr_camera.currentData() == 'precalculated'

    def choose_scene_file(self):
        start = str(self.precalc_path.parent) if self.precalc_path else (str(Path(self.video_path).parent) if getattr(self, 'video_path', None) else '')
        path, _ = QFileDialog.getOpenFileName(self, 'Choose Scene File', start, SCENE_FILTER)
        if not path:
            return
        self.precalc_path = Path(path)
        self.precalc_preview = None
        self.precalc_file_label.setText(self.precalc_path.name)
        for combo in (self.precalc_camera_choice, self.precalc_mesh_choice, self.precalc_up_axis, self.precalc_scale):
            combo.clear()
        self.refresh_precalc_preview(first=True)

    def refresh_precalc_preview(self, first=False):
        if self.precalc_path is None or (self.precalc_loader is not None and self.precalc_loader.isRunning()):
            return
        size = (int(self.info['width']), int(self.info['height'])) if getattr(self, 'info', None) else (1920, 1080)
        kwargs = dict(image_size=size)
        if not first and self.precalc_preview is not None:
            kwargs.update(info=self.precalc_preview['info'], camera=self.precalc_camera_choice.currentData(),
                          mesh=self.precalc_mesh_choice.currentData() or '', plate_fps=self.precalc_plate_fps.value(),
                          up_axis=self.precalc_up_axis.currentData())
        elif getattr(self, 'info', None) and self.info.get('fps'):
            kwargs['plate_fps'] = float(self.info['fps'])
        self.precalc_status.setText('Reading the scene file...')
        self.precalc_file_button.setEnabled(False)
        self.precalc_loader = ScenePreviewLoader(self, self.precalc_path, kwargs)
        self.precalc_loader.done.connect(self.precalc_preview_ready)
        self.precalc_loader.failed.connect(self.precalc_preview_failed)
        self.precalc_loader.start()

    def precalc_preview_failed(self, message):
        self.precalc_file_button.setEnabled(True)
        self.precalc_status.setText('Cannot read the scene file: ' + message.splitlines()[-1] if message else 'Cannot read the scene file.')
        self.log.append('Scene import: ' + message)

    def precalc_preview_ready(self, preview):
        self.precalc_file_button.setEnabled(True)
        first = self.precalc_preview is None
        self.precalc_preview = preview
        info = preview['info']
        if first:
            for cam in info.get('cameras', []):
                self.precalc_camera_choice.addItem(cam['name'] + ('' if cam.get('animated') else ' (not animated)'), cam['name'])
            self.precalc_mesh_choice.addItem('(none)', '')
            for mesh in sorted(info.get('meshes', []), key=lambda m: -m.get('polygons', 0)):
                self.precalc_mesh_choice.addItem(f"{mesh['name']} · {mesh.get('polygons', 0):,} polygons", mesh['name'])
            self.precalc_camera_choice.setCurrentIndex(max(0, self.precalc_camera_choice.findData(preview['camera'])))
            self.precalc_mesh_choice.setCurrentIndex(max(0, self.precalc_mesh_choice.findData(preview['mesh'] or '')))
            report = preview['up_axis_report']
            detected = report.get('estimated') or preview['up_axis']
            self.precalc_up_axis.addItem(f'Auto ({detected})', None)
            for axis in UP_AXES:
                self.precalc_up_axis.addItem(axis + (' (file setting)' if axis == report.get('file') else ''), axis)
            self.precalc_plate_fps.setValue(preview['plate_fps'])
            rng = preview.get('file_frame_range') or [0, 0]
            self.precalc_first_frame.setValue(int(rng[0]))
        if first and preview.get('mesh'):
            # A terrain makes Pre-calculated Mesh the right correction; Flat Ground would flatten a slope.
            self.correction_mode.setCurrentIndex(self.correction_mode.findData('terrain'))
        check = preview['scale_check']
        self.precalc_scale.clear()
        self.precalc_scale.addItem('1 (as in file)', 1.0)
        for factor in check.get('candidates') or []:
            self.precalc_scale.addItem(f'{factor:g}' + (' (suggested)' if factor == check.get('suggested_factor') else ''), factor)
        if check.get('suggested_factor'):
            self.precalc_scale.setCurrentIndex(self.precalc_scale.findData(check['suggested_factor']))
        self.update_precalc_status()

    def precalc_scale_factor(self):
        data = self.precalc_scale.currentData()
        if data is not None and self.precalc_scale.currentText() == self.precalc_scale.itemText(self.precalc_scale.currentIndex()):
            return float(data)
        try:
            return float(self.precalc_scale.currentText().split()[0])
        except (ValueError, IndexError):
            return 1.0

    def update_precalc_status(self):
        preview = self.precalc_preview
        if preview is None:
            return
        check = preview['scale_check']
        lines = [w for w in preview.get('warnings', [])]
        height = check.get('camera_height_m')
        factor = self.precalc_scale_factor()
        if height is not None:
            lines.append(f'Camera height above the terrain: {height * factor:.2f} m after scaling.')
        lines.append(check['message'])
        self.precalc_status.setText('\n'.join(lines))
        if hasattr(self, 'update_readiness'):
            self.update_readiness()

    def precalc_import_options(self):
        """Saved in project.json by Prepare Video, read by the scene_import stage."""
        if not self.precalculated_camera() or self.precalc_path is None or self.precalc_preview is None:
            return None
        return dict(path=str(self.precalc_path.resolve()), camera=self.precalc_camera_choice.currentData(),
                    mesh=self.precalc_mesh_choice.currentData() or None, plate_fps=self.precalc_plate_fps.value(),
                    first_frame=self.precalc_first_frame.value(), up_axis=self.precalc_up_axis.currentData(),
                    scale_fix=self.precalc_scale_factor(), actor_height=self.precalc_actor_height.value() or None)

    def precalc_after_solve(self, project):
        """After GVHMR: show the terrain placement result and offer its scene-scale fix in Scene Scale."""
        try:
            report = json.loads((Path(project) / 'scene_place.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        check = report.get('size_check') or {}
        lines = [f"Terrain placement: size factor {report['size_factor']:.3f}. " + check.get('message', '')]
        if report.get('lowest_stance_sole_median_cm') is not None:
            lines.append(f"Lowest stance sole above the terrain: median {report['lowest_stance_sole_median_cm']:.1f} cm.")
        fix = check.get('fix') or (check.get('height_fix') if check.get('status') == 'badly_scaled' else None)
        if fix:
            current = self.precalc_scale_factor()
            value = round(current * fix, 4)
            index = self.precalc_scale.findData(value)
            if index < 0:
                self.precalc_scale.addItem(f'{value:g} (after Reconstruct 3D)', value)
                index = self.precalc_scale.count() - 1
            self.precalc_scale.setCurrentIndex(index)
            lines.append(f'Scene Scale is now {value:g}. Run Reconstruct 3D again to apply it.')
        self.precalc_status.setText('\n'.join(lines))
        self.log.append(' '.join(lines))

    def precalc_import_environment(self):
        options = self.precalc_import_options()
        return json.dumps(options) if options else ''

    # ---- Pre-calculated Mesh correction: per-contact Terrain / Footage targets on the timeline
    def _contact_range(self, foot, frame):
        for first, end in (self.slider.contacts[foot] if foot < len(self.slider.contacts) else []):
            if first <= frame < end:
                return first, end
        return frame, frame + 1

    def terrain_target_lookup(self, foot, frame):
        first, end = self._contact_range(foot, frame)
        hits = [r['target'] for r in getattr(self, 'terrain_overrides', []) if r['foot'] == foot and first <= r['frame'] < end]
        return hits[-1] if hits else None

    def set_terrain_target(self, foot, frame, target):
        first, end = self._contact_range(foot, frame)
        rows = [r for r in getattr(self, 'terrain_overrides', []) if not (r['foot'] == foot and first <= r['frame'] < end)]
        if target is not None:
            rows.append(dict(foot=int(foot), frame=int((first + end - 1) // 2), target=target))
        self.terrain_overrides = rows
        self.save_correction_options()
        side = 'Left' if foot == 0 else 'Right'
        what = {'terrain': 'locks to the terrain', 'footage': 'keeps the footage height', None: 'uses the automatic target'}[target]
        self.status.setText(f'{side} foot contact at frames {first + 1}-{end}: {what}. Click Apply Correction to update the motion.')
        self.update_readiness()

    def terrain_targets_mode_changed(self, *_):
        self.slider.terrain_targets = self.terrain_target_lookup if self.correction_mode.currentData() == 'terrain' else None

    def precalc_project_loaded(self, project):
        """Show the imported terrain in the 3D view for projects with a Pre-calculated Camera scene."""
        self.motion_view.terrain_lines = None
        try:
            from .scene_import.reader import load
            from .scene_import.terrain import Heightfield, wireframe
            scene = load(project) if project is not None else None
            if scene is not None and 'mesh_vertices' in scene[0]:
                hf = Heightfield(scene[0]['mesh_vertices'], scene[0]['mesh_faces'], cell=0.1)
                self.motion_view.terrain_lines = wireframe(hf)
        except (OSError, ValueError, KeyError) as error:
            self.log.append('Scene terrain preview unavailable: ' + str(error))
        self.motion_view.update()
