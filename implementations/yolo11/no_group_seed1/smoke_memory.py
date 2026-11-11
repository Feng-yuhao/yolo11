"""Fixed caps: E58=15GiB, E59=13GiB; same in smoke and formal."""
from contextlib import contextmanager
from pair_safety import CAPS
GIB = 2**30
SMOKE_CAP_GIB = None


def configure(variant):
    global SMOKE_CAP_GIB
    SMOKE_CAP_GIB = CAPS[variant]


def check_headroom(cuda, device):
    """Return (free_GiB, total_GiB); CUDA's raw byte values stay local."""
    free, total = cuda.mem_get_info(device)
    if free < 2 * GIB:
        raise RuntimeError('Less than 2GiB GPU headroom; stop without issuing smoke receipt. No batch reduction.')
    return free / GIB, total / GIB


@contextmanager
def bounded_smoke(cuda, device):
    if SMOKE_CAP_GIB is None:
        raise RuntimeError('Memory budget not configured')
    _, total_gib = check_headroom(cuda, device)
    fraction = SMOKE_CAP_GIB / total_gib
    if not 0 < fraction <= 1:
        raise RuntimeError(f'Invalid allocator budget: cap={SMOKE_CAP_GIB}GiB, total={total_gib}GiB')
    cuda.set_per_process_memory_fraction(fraction, device)
    print(f'SMOKE/FORMAL allocator cap={SMOKE_CAP_GIB:g}GiB; batch16 unchanged', flush=True)
    yield
