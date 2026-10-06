"""Bounded concurrent CUDA smoke; never applied to formal training."""
from contextlib import contextmanager

GIB = 2**30
SMOKE_CAP_GIB = 12.0
OTHER_JOB_RESERVE_GIB = 4.0
EXTERNAL_ALLOCATION_MARGIN_GIB = 0.5


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < OTHER_JOB_RESERVE_GIB * GIB:
        raise RuntimeError(f'E41 smoke stopped: only {free/GIB:.2f} GiB free; '
                           'protecting 4 GiB headroom for the other job. No smoke gate issued.')
    return free / GIB, total / GIB


@contextmanager
def bounded_smoke(cuda, device):
    free, total = cuda.mem_get_info(device)
    minimum = SMOKE_CAP_GIB + OTHER_JOB_RESERVE_GIB + EXTERNAL_ALLOCATION_MARGIN_GIB
    if free < minimum * GIB:
        raise RuntimeError(f'E41 smoke needs {minimum:.1f} GiB currently free '
                           f'(12 GiB own allocator cap + 4 GiB other-job reserve + 0.5 GiB margin); '
                           f'found {free/GIB:.2f}. Wait and retry; batch16 remains unchanged.')
    getter = getattr(cuda, 'get_per_process_memory_fraction', None)
    previous = getter(device) if getter is not None else 1.0
    fraction = min(previous, SMOKE_CAP_GIB * GIB / total)
    cuda.set_per_process_memory_fraction(fraction, device)
    print(f'E41 concurrent SMOKE memory: free={free/GIB:.2f}/{total/GIB:.2f} GiB; '
          f'own PyTorch cap={fraction*total/GIB:.2f} GiB; other-job reserve=4 GiB. '
          'Formal training is NOT capped.', flush=True)
    try:
        yield
    finally:
        cuda.set_per_process_memory_fraction(previous, device)
