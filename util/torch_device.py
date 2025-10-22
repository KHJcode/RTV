import os
import warnings

import torch

_checked = False
_cached_device = torch.device("cpu")


def _force_cpu():
    """Disable CUDA use when the available GPU is unsupported."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]


def ensure_compatible_cuda():
    """Return a safe torch.device, falling back to CPU if the GPU arch is unsupported."""
    global _checked, _cached_device
    if _checked:
        return _cached_device
    _checked = True

    if not torch.cuda.is_available():
        _cached_device = torch.device("cpu")
        return _cached_device

    try:
        device_index = torch.cuda.current_device() if torch.cuda.device_count() else 0
    except Exception:
        device_index = 0

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
    except Exception:
        warnings.warn(
            "Unable to query CUDA device capability; falling back to CPU.",
            RuntimeWarning,
        )
        _force_cpu()
        _cached_device = torch.device("cpu")
        return _cached_device

    arch = f"sm_{major}{minor}"
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()

    if arch_list and arch not in arch_list:
        warnings.warn(
            f"CUDA capability {arch} is not supported by this PyTorch build; falling back to CPU.",
            RuntimeWarning,
        )
        _force_cpu()
        _cached_device = torch.device("cpu")
        return _cached_device

    _cached_device = torch.device("cuda", device_index)
    return _cached_device

