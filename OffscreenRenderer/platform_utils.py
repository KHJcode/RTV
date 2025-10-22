import ctypes
import ctypes.util
import os
from types import ModuleType
from typing import Optional


def _find_egl_library() -> Optional[str]:
    egl_lib = ctypes.util.find_library("EGL")
    if egl_lib:
        return egl_lib
    candidates = []
    ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    candidates.extend(ld_paths)
    candidates.extend(
        [
            "/usr/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib",
            "/lib64",
            "/lib/x86_64-linux-gnu",
            "/usr/local/lib",
        ]
    )
    lib_names = ["libEGL.so.1", "libEGL.so", "libEGL_nvidia.so.0"]
    for directory in candidates:
        for name in lib_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def _can_use_egl() -> bool:
    if os.name == "nt":
        return False
    egl_lib = _find_egl_library()
    if not egl_lib:
        return False
    try:
        ctypes.CDLL(egl_lib)
    except OSError:
        return False
    return True


def ensure_pyopengl_platform() -> str:
    platform = os.environ.get("PYOPENGL_PLATFORM")
    if platform:
        platform = platform.lower()
        if platform == "egl" and not _can_use_egl():
            platform = "pyglet"
    else:
        if os.name == "nt":
            platform = "win32"
        elif _can_use_egl():
            platform = "egl"
        else:
            platform = "pyglet"
    os.environ["PYOPENGL_PLATFORM"] = platform
    return platform


def load_egl_module() -> Optional[ModuleType]:
    if ensure_pyopengl_platform() != "egl":
        return None
    try:
        import OpenGL.EGL as egl  # type: ignore
    except Exception:
        return None
    return egl
