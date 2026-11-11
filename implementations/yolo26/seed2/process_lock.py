"""Linux production flock; Windows private engineering harness only."""
import os


def acquire(path, local=False):
    handle = path.open('a+b')
    try:
        if os.name == 'nt':
            if not local:
                raise RuntimeError('Production controller requires Linux server')
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BaseException:
        handle.close()
        raise
