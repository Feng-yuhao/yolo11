"""Verify delivered ZIP, not the richer local working directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from run_e41 import REQUIRED_FILES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('zip', type=Path)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    folder = 'E41a_ChannelDepthTaskHead_AutoDL'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(args.zip) as archive:
            for entry in archive.namelist():
                if not entry.startswith(folder+'/') or '..' in Path(entry).parts:
                    raise RuntimeError('Unexpected ZIP target: '+entry)
            archive.extractall(root)
        package = root/folder
        manifest = json.loads((package/'manifest.json').read_text(encoding='utf-8'))
        assert set(REQUIRED_FILES) <= set(manifest), 'Runtime dependencies absent from manifest'
        assert all((package/name).is_file() for name in REQUIRED_FILES), 'Missing runtime file in ZIP'
        for name, expected in manifest.items():
            actual = hashlib.sha256((package/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            assert actual == expected, name
        env = os.environ.copy()
        # No import from the original E41 directory is allowed here.
        env['PYTHONPATH'] = os.pathsep.join((str(args.source.resolve()), env.get('PYTHONPATH','')))
        env['YOLO_CONFIG_DIR'] = str(root/'settings')
        env['PYTHONIOENCODING'] = 'utf-8'
        code = ('import pathlib,sys; sys.path.insert(0,sys.argv[1]); '
                'import run_e41; run_e41.verify_package_files(); '
                'import p2_tal_chunked,checks,evaluate,channel_task_head,smoke_memory,queue_after_e40; '
                'p=pathlib.Path(sys.argv[1]).resolve(); '
                'assert pathlib.Path(p2_tal_chunked.__file__).resolve().parent==p; '
                'print("ZIP extraction / dependency imports / manifest passed")')
        result = subprocess.run([sys.executable,'-c',code,str(package)],cwd=root,
                                env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        print(result.stdout.strip())
        # Verify missing dependency is detected before CUDA/model setup.
        (package/'p2_tal_chunked.py').unlink()  # disposable extracted fixture only
        code = ('import sys; sys.path.insert(0,sys.argv[1]); import run_e41; '
                'run_e41.verify_package_files()')
        broken = subprocess.run([sys.executable,'-c',code,str(package)],cwd=root,
                                env=env,capture_output=True,text=True,encoding='utf-8',timeout=30)
        assert broken.returncode != 0 and 'p2_tal_chunked.py' in broken.stderr
        print('Missing dependency fixture detected before CUDA setup')


if __name__ == '__main__':
    main()
