"""The SAME cap for real smoke and formal training; never lower batch silently."""
from contextlib import contextmanager
GIB=2**30
SMOKE_CAP_GIB=28.


def check_headroom(cuda,device):
    free,total=cuda.mem_get_info(device)
    if free < 2*GIB:
        raise RuntimeError('Less than 2 GiB GPU headroom; no smoke receipt. Use unchanged-configuration serial runs.')
    return free/GIB,total/GIB


@contextmanager
def bounded_smoke(cuda,device):
    free,total=cuda.mem_get_info(device)
    if free<30*GIB:
        raise RuntimeError('Need >=30 GiB free: E46 is a single-job experiment; stop other GPU jobs normally before smoke.')
    cuda.set_per_process_memory_fraction(28*GIB/total,device)
    print('SMOKE and FORMAL both use 28 GiB allocator cap; no batch reduction',flush=True)
    yield

