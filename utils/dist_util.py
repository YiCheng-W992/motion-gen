"""
Helpers for distributed training.
"""

import os
import socket

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    """
    global used_device
    if dist.is_initialized():
        if device is not None:
            used_device = device
        return

    if _is_dist_env():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", device if device is not None else 0))
        if world_size > 1:
            used_device = local_rank
            if th.cuda.is_available() and used_device >= 0:
                th.cuda.set_device(used_device)
            backend = "nccl" if th.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
        else:
            used_device = local_rank if local_rank is not None else 0
        return

    used_device = device if device is not None else 0


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    if not is_dist_avail_and_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def _is_dist_env():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        dist.barrier()
