"""Verify the test-only package without importing torch or starting evaluation."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    for i, entry in enumerate(manifest['files'], 1):
        p = root / entry['path']
        assert p.is_file(), f'Missing: {p}'
        assert p.stat().st_size == entry['bytes'], f'Size mismatch: {p}'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], f'Hash mismatch: {p}'
        if i % 3000 == 0:
            print(f'Verify {i}/{len(manifest["files"])}', flush=True)
    images = sorted((root / 'images/test').glob('*.jpg'))
    labels = sorted((root / 'labels/test').glob('*.txt'))
    assert len(images) == len(labels) == 16592
    assert {p.stem for p in images} == {p.stem for p in labels}
    assert len({p.stem.split('_')[0] for p in images}) == 20
    print(json.dumps({'status': 'passed', 'images': len(images), 'sequences': 20,
                      'files_sha256_checked': len(manifest['files']),
                      'training_started': False, 'evaluation_started': False}, indent=2))


if __name__ == '__main__':
    main()
