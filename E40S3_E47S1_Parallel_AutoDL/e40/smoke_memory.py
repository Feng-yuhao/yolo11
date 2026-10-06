"""Fixed 13 GiB cap for paired smoke AND formal; no batch fallback."""
from contextlib import contextmanager

GIB = 2**30
SMOKE_CAP_GIB = 13.0
HEADROOM_GIB = 2.0


def apply_pair_memory(cuda, device):
    from parallel_safety import memory_policy
    if memory_policy()['allocator_cap_GiB'] != SMOKE_CAP_GIB:
        raise RuntimeError('E40 paired memory policy drift')
    free, total = cuda.mem_get_info(device)
    if total < 30 * GIB or free < 15 * GIB:
        raise RuntimeError('E40 paired startup needs a 32GB-class GPU and >=15 GiB free; no batch fallback')
    cuda.set_per_process_memory_fraction(SMOKE_CAP_GIB * GIB / total, device)
    print(f'E40 paired memory: fixed 13 GiB allocator cap; free={free/GIB:.2f} GiB; '
          'batch16/640/200 unchanged', flush=True)


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < HEADROOM_GIB * GIB:
        raise RuntimeError('Paired execution has <2 GiB GPU headroom; no smoke gate issued')
    return free/GIB, total/GIB


@contextmanager
def bounded_smoke(cuda, device):
    # The same cap was installed by setup(), before model allocation, and is
    # deliberately not restored to an uncapped policy at context exit.
    check_headroom(cuda, device)
    yield

