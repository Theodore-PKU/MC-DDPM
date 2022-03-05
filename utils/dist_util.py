"""
Helpers for distributed training.
"""

import os
import socket
import torch as th
import torch.distributed as dist


def setup_dist():
    """
    Setup a distributed process group.
    Return (bool, int) which indicates whether to use distrubted training
        and which GPU rank is used for current device.
    """
    if th.cuda.is_available():
        th.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, int(os.environ["RANK"])
    else:
        return False, 0


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


# ytxie: we do not use this function.
def _find_free_port():
    """
    Find free port number.
    Return int.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    free_port = s.getsockname()[1]
    s.close()
    return free_port
