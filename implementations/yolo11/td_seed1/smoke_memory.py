"""Identical 28GiB allocator cap for formal and real smoke; no batch fallback."""
GIB = 2 ** 30


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < 2 * GIB:
        raise RuntimeError('Less than2GiB free GPU memory; stop, do not reduce batch')
    return free / GIB, total / GIB


def set_limit(cuda, device):
    _, total_gib = check_headroom(cuda, device)
    fraction = 28. / total_gib
    if not 0 < fraction <= 1:
        raise RuntimeError('28GiB cap requires the original32GiB server GPU')
    cuda.set_per_process_memory_fraction(fraction, device)
