"""Fixed paired resource policy; never changes model or training parameters."""
from contextlib import contextmanager

GIB = 2**30
SMOKE_CAP_GIB = 13.0
MINIMUM_START_GIB = 15.0
HEADROOM_GIB = 2.0


def prepare(torch, args):
    from parallel_safety import validate_worker_context, memory_policy
    context = validate_worker_context(args, 'e47', args.mode)
    policy = memory_policy()
    if policy['allocator_cap_GiB'] != SMOKE_CAP_GIB:
        raise RuntimeError('Paired allocator policy drift')
    free, total = torch.cuda.mem_get_info(args.device)
    if total < 30 * GIB:
        raise RuntimeError('Real paired smoke/run requires the server 32GB-class GPU')
    if free < MINIMUM_START_GIB * GIB:
        raise RuntimeError(f'E47 startup needs >=15 GiB currently free; found {free/GIB:.2f}. '
                           'No batch reduction or training fallback is allowed.')
    torch.cuda.set_per_process_memory_fraction(SMOKE_CAP_GIB * GIB / total, args.device)
    print(f'E47 paired {args.mode}: fixed {SMOKE_CAP_GIB:g} GiB allocator cap, '
          f'{free/GIB:.2f} GiB free; batch16/640/200 unchanged', flush=True)
    return context


def check_headroom(cuda, device):
    free, total = cuda.mem_get_info(device)
    if free < HEADROOM_GIB * GIB:
        raise RuntimeError('Paired smoke has <2 GiB GPU headroom; no gate issued')
    return free / GIB, total / GIB


@contextmanager
def bounded_smoke(cuda, args):
    from parallel_safety import validate_worker_context
    validate_worker_context(args, 'e47', 'smoke')
    check_headroom(cuda, args.device)
    # setup() already applied the SAME 13GiB cap used by formal run.
    yield
