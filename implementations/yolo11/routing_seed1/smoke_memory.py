"""Identical 15GiB allocator budget for joint smoke and both formal workers."""
from contextlib import contextmanager
from common import CAPS_GIB
SMOKE_CAP_GIB = None
GIB = 2**30


def configure(execution):
    global SMOKE_CAP_GIB
    SMOKE_CAP_GIB = CAPS_GIB[execution]


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < .5 * GIB:
        raise RuntimeError('GPU headroom below 0.5GiB; no gate, no automatic batch reduction')
    return free/GIB, total/GIB


@contextmanager
def bounded_smoke(cuda, device):
    if SMOKE_CAP_GIB is None:
        raise RuntimeError('Allocator cap not configured')
    _, total_gib = check_headroom(cuda, device)
    fraction = SMOKE_CAP_GIB/total_gib
    if not 0 < fraction <= 1:
        raise RuntimeError('Invalid byte/GiB allocator budget')
    cuda.set_per_process_memory_fraction(fraction, device)
    yield
