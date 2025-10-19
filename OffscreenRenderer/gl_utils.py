from typing import Tuple, Dict, Any
import ctypes
import os

import numpy as np
try:
    import OpenGL.EGL as egl
    from OpenGL.raw.EGL.EXT.platform_device import EGL_PLATFORM_DEVICE_EXT
    from OpenGL.EGL.EXT.device_base import egl_get_devices
    from OpenGL import error
except Exception:  # pragma: no cover - EGL not available
    egl = None
from OpenGL.GL import *


def create_initialized_headless_egl_display():
    """Creates an initialized EGL display directly on a device."""
    if egl is None:
        return None
    try:
        devices = egl_get_devices()
    except Exception:
        return None
    for device in devices:
        display = egl.eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, device, None)

        if display != egl.EGL_NO_DISPLAY and egl.eglGetError() == egl.EGL_SUCCESS:
            try:
                initialized = egl.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == egl.EGL_TRUE and egl.eglGetError() == egl.EGL_SUCCESS:
                    return display
    return None


def _create_egl_context(surface_size: Tuple[int, int]) -> Dict[str, Any]:
    egl_display = create_initialized_headless_egl_display()

    if egl_display in (None, egl.EGL_NO_DISPLAY):
        raise RuntimeError('Cannot initialize a headless EGL display.')

    major, minor = egl.EGLint(), egl.EGLint()

    egl.eglInitialize(egl_display, ctypes.pointer(
        major), ctypes.pointer(minor))

    config_attribs = [
        egl.EGL_SURFACE_TYPE, egl.EGL_PBUFFER_BIT, egl.EGL_BLUE_SIZE, 8,
        egl.EGL_GREEN_SIZE, 8, egl.EGL_RED_SIZE, 8, egl.EGL_DEPTH_SIZE, 24,
        egl.EGL_RENDERABLE_TYPE, egl.EGL_OPENGL_BIT, egl.EGL_NONE
    ]
    config_attribs = (egl.EGLint * len(config_attribs))(*config_attribs)

    num_configs = egl.EGLint()
    egl_cfg = egl.EGLConfig()

    egl.eglChooseConfig(egl_display, config_attribs, ctypes.pointer(egl_cfg), 1,
                        ctypes.pointer(num_configs))

    width, height = surface_size

    pbuffer_attribs = [
        egl.EGL_WIDTH,
        width,
        egl.EGL_HEIGHT,
        height,
        egl.EGL_NONE,
    ]
    pbuffer_attribs = (egl.EGLint * len(pbuffer_attribs))(*pbuffer_attribs)
    egl_surf = egl.eglCreatePbufferSurface(
        egl_display, egl_cfg, pbuffer_attribs)

    egl.eglBindAPI(egl.EGL_OPENGL_API)

    egl_context = egl.eglCreateContext(
        egl_display, egl_cfg, egl.EGL_NO_CONTEXT, None)

    egl.eglMakeCurrent(egl_display, egl_surf, egl_surf, egl_context)

    return {
        "backend": "egl",
        "display": egl_display,
        "surface": egl_surf,
        "context": egl_context,
        "window": None,
        "cleanup": None,
    }


def _create_pyglet_context(surface_size: Tuple[int, int]) -> Dict[str, Any]:
    width, height = surface_size
    try:
        import pyglet
        from pyglet import gl as pyglet_gl
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("pyglet is required for OpenGL context creation on this platform.") from exc

    config = None
    try:
        config = pyglet.gl.Config(
            double_buffer=False,
            depth_size=24,
            alpha_size=8,
            sample_buffers=0,
            samples=0,
            stencil_size=8,
        )
    except Exception:
        config = None

    window = pyglet.window.Window(
        width=width,
        height=height,
        visible=False,
        config=config,
    )
    window.switch_to()
    pyglet_gl.glViewport(0, 0, width, height)

    def _cleanup():
        try:
            window.close()
        except Exception:
            pass

    return {
        "backend": "pyglet",
        "display": None,
        "surface": None,
        "context": window.context,
        "window": window,
        "cleanup": _cleanup,
    }


def create_opengl_context(surface_size: Tuple[int, int]) -> Dict[str, Any]:
    """Create offscreen OpenGL context and make it current, returning backend info."""
    prefer_egl = os.environ.get("PYOPENGL_PLATFORM", "").lower() == "egl"
    if prefer_egl and egl is not None:
        try:
            return _create_egl_context(surface_size)
        except Exception as exc:
            print(f"[gl_utils] EGL context creation failed ({exc}); falling back to pyglet.")
    return _create_pyglet_context(surface_size)


def init_frame_buffer(frame_image: np.ndarray, num_samples: int = 16):
    """Create an FBO and assign a texture buffer to it for the purpose of offscreen rendering to the texture buffer
    """
    samples = min(num_samples, GL_SAMPLES - 1)

    width = frame_image.shape[1]
    height = frame_image.shape[0]

    if frame_image.shape[-1] == 3:
        texture_type = GL_RGB
        internal_texture_type = GL_RGB8
    elif frame_image.shape[-1] == 4:
        texture_type = GL_RGBA
        internal_texture_type = GL_RGBA8

    render_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, render_texture)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_texture_type, width,
                 height, 0, texture_type, GL_UNSIGNED_BYTE, None)

    glBindTexture(GL_TEXTURE_2D, 0)

    render_frame_object = glGenFramebuffers(1)

    glBindFramebuffer(GL_FRAMEBUFFER, render_frame_object)

    # Create color render buffer
    color_buffer = glGenRenderbuffers(1)

    glBindRenderbuffer(GL_RENDERBUFFER, color_buffer)
    glRenderbufferStorage(
        GL_RENDERBUFFER, internal_texture_type, width, height)
    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer)

    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)

    depth_buffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    glBindRenderbuffer(GL_RENDERBUFFER,0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_buffer)

    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    # Sees if your GPU can handle the FBO configuration defined above
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(
            "Framebuffer binding failed, probably because your GPU does not support this FBO configuration.")

    glBindFramebuffer(GL_FRAMEBUFFER, 0)



    return render_frame_object
