import argparse
import os

from OffscreenRenderer.platform_utils import ensure_pyopengl_platform, load_egl_module

ensure_pyopengl_platform()

import numpy as np
egl = load_egl_module()
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image

from OffscreenRenderer.gl_utils import create_opengl_context, init_frame_buffer
from OffscreenRenderer.base_shaders  import VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
from OffscreenRenderer.off_screen_render import VertextAttribType, IndexType, generate_vao, set_shader_params, render, \
    read_texture

VertextAttribType = np.float32
IndexType = np.uint32
OpenglVertexAttrType = GL_FLOAT
OpenglTriangleIndexType = GL_UNSIGNED_INT


class BaseRenderer:
    def __init__(self, height=512, width=512):
        ctx_info = create_opengl_context((width, height))
        self._gl_backend = ctx_info["backend"]
        self.display = ctx_info.get("display")
        self.egl_surf = ctx_info.get("surface")
        self.opengl_context = ctx_info.get("context")
        self._window = ctx_info.get("window")
        self._cleanup = ctx_info.get("cleanup")
        self.height = height
        self.width = width


        self.vao = None
        self.triangle_vertex_properties = None
        self.shader = shaders.compileProgram(
            shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
            shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
        self.rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.render_frame_object = init_frame_buffer(self.rendered_image)

    def __del__(self):
        if self._gl_backend == "egl" and egl is not None:
            egl.eglDestroySurface(self.display, self.egl_surf)
            egl.eglDestroyContext(self.display, self.opengl_context)
        elif self._gl_backend == "pyglet":
            if self._window is not None:
                self._window.close()
        if self._cleanup is not None:
            try:
                self._cleanup()
            except Exception:
                pass

    def render(self, vertex_positions, face_indices, matrix_model=None, matrix_view=None,
               matrix_proj=None):


        if self.vao is None:
            self.generate_vao(vertex_positions, face_indices)
        else:
            self.update_vao(vertex_positions, face_indices)
        #print(self.vao)
        glBindFramebuffer(GL_FRAMEBUFFER, self.render_frame_object)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0, 0, 0.0, 1.0)
        # glEnable(GL_MULTISAMPLE)
        # glDisable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)

        glDisable(GL_CULL_FACE)
        # glCullFace(GL_FRONT)
        glUseProgram(self.shader)
        # Orto projection in range [0; 1]
        set_shader_params(self.shader, matrix_model=matrix_model, matrix_view=matrix_view, matrix_proj=matrix_proj)

        glBindVertexArray(self.vao)
        render(GL_TRIANGLES, len(face_indices))
        glBindVertexArray(0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.render_frame_object)

        glReadBuffer(GL_COLOR_ATTACHMENT0)


        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)


        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # glDeleteFramebuffers(1, render_frame_object)
        glUseProgram(self.shader)

        image = Image.frombuffer("RGB", (self.width, self.height), data)
        return np.array(image)

    def generate_vao(self, vertex_positions: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3

        assert face_indices.ndim == 1

        face_indices = face_indices.astype(IndexType)

        #vertex_attributes = np.hstack(
        # (vertex_positions, texcoord)).astype(VertextAttribType)
        vertex_attributes = np.hstack(
            (vertex_positions,)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]
        #num_texcoord_per_vertex = texcoord.shape[1]

        triangle_vao = glGenVertexArrays(1)

        self.vao = triangle_vao

        glBindVertexArray(triangle_vao)

        num_properties_per_vertex = vertex_attributes.shape[1]

        triangle_vertex_properties = glGenBuffers(1)
        self.triangle_vertex_properties = triangle_vertex_properties

        glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_properties)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes.nbytes,
                     vertex_attributes, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        # Position attribute layout, size,   stride, offset
        glVertexAttribPointer(0, num_pos_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex, ctypes.c_void_p(0))
        #glEnableVertexAttribArray(1)
        #glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
        #                      num_properties_per_vertex,
         #                     ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))


        triangle_indices = glGenBuffers(1)
        self.triangle_indices = triangle_indices

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.nbytes,
                     face_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def update_vao(self, vertex_positions: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3

        assert face_indices.ndim == 1

        #vertex_attributes = np.hstack(
        # (vertex_positions, texcoord)).astype(VertextAttribType)
        vertex_attributes = np.hstack(
            (vertex_positions,)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]

        triangle_vao = self.vao

        glBindVertexArray(triangle_vao)

        num_properties_per_vertex = vertex_attributes.shape[1]

        triangle_vertex_properties = self.triangle_vertex_properties

        glBindBuffer(GL_ARRAY_BUFFER, triangle_vertex_properties)
        glBufferData(GL_ARRAY_BUFFER, vertex_attributes.nbytes,
                     vertex_attributes, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        # Position attribute layout, size,   stride, offset
        glVertexAttribPointer(0, num_pos_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex, ctypes.c_void_p(0))
        #glEnableVertexAttribArray(1)
        #glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
         #                     num_properties_per_vertex,
         #                     ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))


        #glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.triangle_indices)
        #glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.nbytes,
         #            face_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)
