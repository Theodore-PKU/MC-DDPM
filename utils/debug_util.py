"""
Helpers for debugging code.
"""

import numpy as np
import pynvml


def show_gpu_usage(text=None, idx=0):
    """
    Show the gpu usage.

    Args:
        text (string): something to print to tell what has happened. Default is None.
        idx (int): gpu index.
    """
    if text is not None:
        print(text)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total / 1e9
    used = meminfo.used / 1e9
    free = meminfo.free / 1e9
    # logger.log(f"GPU usage (G). total: {total:.2f}, used: {used:.2f}, free: {free:.2f}")
    print(f"GPU usage (G). total: {total:.2f}, used: {used:.2f}, free: {free:.2f}")
    return None


def count_parameters_in_M(model):
    """
    Compute parameters number.
    """
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6
