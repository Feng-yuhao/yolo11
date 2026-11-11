"""Shared memory guard for unchanged frozen validation stress."""
CAP_GIB=28.
def configure(cap):
    global CAP_GIB
    CAP_GIB=float(cap)
def check_headroom(cuda,device):
    free,total=cuda.mem_get_info(device)
    if free<.5*2**30:raise RuntimeError('GPU headroom below 0.5 GiB; no automatic batch reduction')
    return free/2**30,total/2**30
