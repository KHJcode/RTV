import argparse
import os

from OffscreenRenderer.platform_utils import ensure_pyopengl_platform, load_egl_module

ensure_pyopengl_platform()

import numpy as np
egl = load_egl_module()
try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders
except Exception:  # pragma: no cover - GL import failure handled via CPU fallback
    shaders = None  # type: ignore
from PIL import Image

from OffscreenRenderer.gl_utils import create_opengl_context, init_frame_buffer

from OffscreenRenderer.off_screen_render import VertextAttribType, IndexType, generate_vao, set_shader_params, render, \
    read_texture

VertextAttribType = np.float32
IndexType = np.uint32
OpenglVertexAttrType = GL_FLOAT if 'GL_FLOAT' in globals() else None  # type: ignore
OpenglTriangleIndexType = GL_UNSIGNED_INT if 'GL_UNSIGNED_INT' in globals() else None  # type: ignore


class FlatRenderer:
    def __init__(self, height=512, width=512, texPath="./assets/measurement.png", back_black=False, phong=False):
        self.height = height
        self.width = width
        self._gl_backend = None
        self.display = None
        self.egl_surf = None
        self.opengl_context = None
        self._window = None
        self._cleanup = None
        self._use_cpu = False
        self._cpu_texture = None
        self._cpu_back_black = back_black

        try:
            ctx_info = create_opengl_context((width, height))
            self._gl_backend = ctx_info["backend"]
            self.display = ctx_info.get("display")
            self.egl_surf = ctx_info.get("surface")
            self.opengl_context = ctx_info.get("context")
            self._window = ctx_info.get("window")
            self._cleanup = ctx_info.get("cleanup")

            self.texID = read_texture(texPath)
            self.vao = None
            self.triangle_vertex_properties = None
            self.back_black = back_black
            if phong:
                from OffscreenRenderer.Shaders.backblack_phong_shaders import VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
                self.shader = shaders.compileProgram(
                    shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                    shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
                    shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
            else:
                if not back_black:
                    from OffscreenRenderer.shaders import VERTEX_SHADER, FRAGMENT_SHADER
                    self.shader = shaders.compileProgram(
                        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                        shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
                else:
                    from OffscreenRenderer.Shaders.backblack_shaders import VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER
                    self.shader = shaders.compileProgram(
                        shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                        shaders.compileShader(GEOMETRY_SHADER, GL_GEOMETRY_SHADER),
                        shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER), )
            self.rendered_image = np.zeros((height, width, 3), dtype=np.uint8)
            self.render_frame_object = init_frame_buffer(self.rendered_image)
        except Exception:
            self._initialize_cpu_renderer(texPath)

    def __del__(self):
        if self._use_cpu:
            return
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

    def render(self, vertex_positions, vertex_texcoord, face_indices, matrix_model=None, matrix_view=None,
               matrix_proj=None):

        if self._use_cpu:
            return self._cpu_render(vertex_positions, vertex_texcoord, face_indices, matrix_model, matrix_view, matrix_proj)

        if self.vao is None:
            self.generate_vao(vertex_positions, vertex_texcoord, face_indices)
        else:
            self.update_vao(vertex_positions, vertex_texcoord, face_indices)

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
        render(GL_TRIANGLES, len(face_indices), self.texID)
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

    def _initialize_cpu_renderer(self, tex_path):
        self._use_cpu = True
        self._gl_backend = "cpu"
        tex_image = Image.open(tex_path).convert("RGB")
        self._cpu_texture = np.array(tex_image, dtype=np.uint8)
        self.rendered_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def _cpu_render(self, vertex_positions, vertex_texcoord, face_indices, matrix_model, matrix_view, matrix_proj):
        if matrix_model is None:
            matrix_model = np.eye(4, dtype=np.float32)
        if matrix_view is None:
            matrix_view = np.eye(4, dtype=np.float32)
        if matrix_proj is None:
            matrix_proj = np.eye(4, dtype=np.float32)

        texture = self._cpu_texture
        tex_h, tex_w = texture.shape[:2]

        mvp = matrix_proj @ matrix_view @ matrix_model
        vertices = np.concatenate(
            [vertex_positions.astype(np.float32), np.ones((vertex_positions.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        clip = (mvp @ vertices.T).T
        w = np.clip(clip[:, 3:4], 1e-6, None)
        ndc = clip[:, :3] / w

        screen_x = (ndc[:, 0] * 0.5 + 0.5) * (self.width - 1)
        screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (self.height - 1)
        screen_z = ndc[:, 2]
        screen = np.stack([screen_x, screen_y, screen_z], axis=1)

        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        z_buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)

        faces = face_indices.reshape(-1, 3)

        for tri in faces:
            pts = screen[tri]
            if np.any(~np.isfinite(pts)):
                continue
            min_x = max(int(np.floor(np.min(pts[:, 0]))), 0)
            max_x = min(int(np.ceil(np.max(pts[:, 0]))), self.width - 1)
            min_y = max(int(np.floor(np.min(pts[:, 1]))), 0)
            max_y = min(int(np.ceil(np.max(pts[:, 1]))), self.height - 1)
            if min_x > max_x or min_y > max_y:
                continue

            p0, p1, p2 = pts
            denom = ((p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1]))
            if abs(denom) < 1e-6:
                continue

            uv = vertex_texcoord[tri]
            z_vals = screen_z[tri]

            xs = np.arange(min_x, max_x + 1)
            ys = np.arange(min_y, max_y + 1)
            xs_grid, ys_grid = np.meshgrid(xs + 0.5, ys + 0.5)

            denom_inv = 1.0 / denom
            w0 = ((p1[1] - p2[1]) * (xs_grid - p2[0]) + (p2[0] - p1[0]) * (ys_grid - p2[1])) * denom_inv
            w1 = ((p2[1] - p0[1]) * (xs_grid - p2[0]) + (p0[0] - p2[0]) * (ys_grid - p2[1])) * denom_inv
            w2 = 1.0 - w0 - w1

            mask = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
            if not np.any(mask):
                continue

            depth = w0 * z_vals[0] + w1 * z_vals[1] + w2 * z_vals[2]

            zb_patch = z_buffer[min_y:max_y + 1, min_x:max_x + 1]
            img_patch = image[min_y:max_y + 1, min_x:max_x + 1]

            depth_mask = depth < zb_patch
            update_mask = mask & depth_mask
            if not np.any(update_mask):
                continue

            tu = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
            tv = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
            tu = np.clip(tu, 0.0, 1.0) * (tex_w - 1)
            tv = np.clip(1.0 - tv, 0.0, 1.0) * (tex_h - 1)

            u_idx = tu.astype(np.int32)
            v_idx = tv.astype(np.int32)
            colors = texture[v_idx, u_idx]

            zb_patch[update_mask] = depth[update_mask]
            img_patch[update_mask] = colors[update_mask]

        return image.astype(np.uint8)

    def generate_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3
        assert vertex_positions.shape[0] == texcoord.shape[0]
        assert texcoord.shape[1] == 2
        assert face_indices.ndim == 1

        face_indices = face_indices.astype(IndexType)

        vertex_attributes = np.hstack(
            (vertex_positions, texcoord)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]
        num_texcoord_per_vertex = texcoord.shape[1]

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

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex,
                              ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))

        triangle_indices = glGenBuffers(1)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_indices.nbytes,
                     face_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def update_vao(self, vertex_positions: np.ndarray, texcoord: np.ndarray, face_indices: np.ndarray):
        assert vertex_positions.ndim == 2
        assert vertex_positions.shape[1] == 3
        assert vertex_positions.shape[0] == texcoord.shape[0]
        assert texcoord.shape[1] == 2
        assert face_indices.ndim == 1

        vertex_attributes = np.hstack(
            (vertex_positions, texcoord)).astype(VertextAttribType)

        num_pos_per_vertex = vertex_positions.shape[1]
        num_texcoord_per_vertex = texcoord.shape[1]

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

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, num_texcoord_per_vertex, OpenglVertexAttrType, GL_FALSE, vertex_attributes.itemsize *
                              num_properties_per_vertex,
                              ctypes.c_void_p(vertex_attributes.itemsize * num_pos_per_vertex))
        glBindVertexArray(0)
