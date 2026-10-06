"""Fixed 13GiB allocation ceiling for check, real smoke AND formal training.

Busy smoke needs explicit permission and >=16.5GiB free at entry. Free VRAM
is not a future peak guarantee; no batch downgrade or fabricated receipt.
"""
from contextlib import contextmanager
GIB = 2**30
SMOKE_CAP_GIB = 13.
ENTRY_FREE_GIB = 16.5


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < 2 * GIB:
        raise RuntimeError('Less than 2 GiB GPU headroom; stop this smoke, not the other job. No receipt.')
    return free/GIB, total/GIB


def configure_cap(cuda, device):
    _, total = cuda.mem_get_info(device)
    if total < 30*GIB:
        raise RuntimeError('Real server gates require the 32GiB GPU')
    cuda.set_per_process_memory_fraction(SMOKE_CAP_GIB*GIB/total, device)


def require_entry(cuda, device, allow_busy=False):
    free, total = cuda.mem_get_info(device)
    needed = ENTRY_FREE_GIB if allow_busy else 30.
    if free < needed*GIB:
        raise RuntimeError(f'Need >= {needed:g} GiB free (currently {free/GIB:.2f}); '
                           'no auto batch change. For an authorized busy-GPU gate use --allow-busy-gpu.')
    return dict(free_GiB=free/GIB, total_GiB=total/GIB,
                busy_gpu_permission=allow_busy, allocator_cap_GiB=SMOKE_CAP_GIB)


@contextmanager
def bounded_smoke(cuda, device, allow_busy=False):
    entry = require_entry(cuda, device, allow_busy)
    configure_cap(cuda, device)
    print(f'SMOKE/FORMAL cap={SMOKE_CAP_GIB:g}GiB, entry={entry}; batch16 unchanged', flush=True)
    yield entry
