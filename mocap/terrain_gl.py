"""Static, full-resolution terrain mesh on the GPU, depth-tested against the actor (display only)."""
import numpy as np
from shiboken6 import VoidPtr
from PySide6.QtGui import QMatrix4x4, QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLFunctions_4_3_Core, QOpenGLShader, QOpenGLVertexArrayObject

from .mesh_gl import FRAGMENT, VERTEX, MeshRenderer


def vertex_normals(vertices, faces):
    """Area-weighted smooth normals (float32)."""
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    # A terrain is seen from above: flip normals that point down (scan winding varies).
    out[out[:, 1] < 0] *= -1
    return out


class TerrainRenderer:
    def __init__(self):
        self.gl = QOpenGLFunctions_4_3_Core()
        if not self.gl.initializeOpenGLFunctions():
            raise RuntimeError('The terrain mesh needs OpenGL 4.3.')
        self.surface = MeshRenderer.program(((QOpenGLShader.Vertex, VERTEX), (QOpenGLShader.Fragment, FRAGMENT)))
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.buffers = []
        self.key = None
        self.index_count = 0

    def load(self, mesh):
        if mesh['key'] == self.key:
            return
        self.release()
        packed = np.zeros((len(mesh['vertices']), 8), np.float32)  # position + pad, normal + pad (32-byte stride)
        packed[:, :3] = mesh['vertices']
        packed[:, 4:7] = mesh['normals']
        faces = np.ascontiguousarray(mesh['faces'], dtype=np.uint32)
        self.vao.bind()
        vertices = self._buffer(packed, QOpenGLBuffer.VertexBuffer)
        self._buffer(faces, QOpenGLBuffer.IndexBuffer)
        self.surface.bind()
        vertices.bind()
        self.surface.enableAttributeArray(0)
        self.surface.enableAttributeArray(1)
        self.surface.setAttributeBuffer(0, 0x1406, 0, 3, 32)
        self.surface.setAttributeBuffer(1, 0x1406, 16, 3, 32)
        self.vao.release()
        self.surface.release()
        self.index_count = faces.size
        self.key = mesh['key']

    def _buffer(self, data, kind):
        b = QOpenGLBuffer(kind)
        if not b.create() or not b.bind():
            raise RuntimeError('Cannot create a terrain GPU buffer')
        raw = data.tobytes()
        b.allocate(raw, len(raw))
        self.buffers.append(b)
        return b

    def draw(self, eye, basis, width, height, color=(.42, .47, .40)):
        projection = QMatrix4x4()
        projection.perspective(40, width / max(1, height), .03, 2000)
        view = np.eye(4)
        view[:3, :3] = basis * np.array([[1], [1], [-1]])
        view[:3, 3] = -view[:3, :3] @ eye
        self.surface.bind()
        self.surface.setUniformValue('mvp', projection * QMatrix4x4(*view.reshape(-1).tolist()))
        self.surface.setUniformValue('sceneTransform', QMatrix4x4())
        self.surface.setUniformValue('bodyColor', QVector3D(*color))
        self.gl.glEnable(0x0B71)      # depth test against the actor drawn before
        self.gl.glDepthMask(True)
        self.gl.glDepthFunc(0x0203)   # LEQUAL
        self.gl.glDisable(0x0B44)     # no culling: scan winding varies
        self.vao.bind()
        self.gl.glDrawElements(0x4, self.index_count, 0x1405, VoidPtr(0))
        self.vao.release()
        self.gl.glDisable(0x0B71)
        self.surface.release()

    def release(self):
        for b in self.buffers:
            b.destroy()
        self.buffers.clear()
        self.key = None

    def close(self):
        self.release()
        self.vao.destroy()
        self.surface.removeAllShaders()
