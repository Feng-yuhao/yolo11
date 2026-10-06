"""Small, read-only diagnostic provenance helpers (no Torch import)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha(path, normalized=False):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        if normalized:
            digest.update(handle.read().replace(b'\r\n', b'\n'))
        else:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def package_signature():
    manifest = read(HERE / 'manifest.json')
    for name, expected in manifest.items():
        if sha(HERE / name, True) != expected:
            raise RuntimeError('Package hash mismatch: ' + name)
    return manifest


def resolve_reports(project, seed, e47_override=None, e40_override=None):
    """Never select a half-written active report or silently choose latest."""
    patterns = {1: ('E47P_RUN_*', 'E40_RUN_*'), 2: ('E47S2_RUN_*', 'E40S2_RUN_*')}
    root = Path(project) / 'comparison_reports'
    answer = []
    for override, pattern in zip((e47_override, e40_override), patterns[seed]):
        candidates = [Path(override)] if override is not None else sorted(root.glob(pattern))
        valid = []
        for path in candidates:
            if not (path / 'comparison.json').is_file() or not (path / 'training.json').is_file():
                continue
            comparison = read(path / 'comparison.json')
            training = read(path / 'training.json')
            if (comparison.get('status') == training.get('status') == 'completed' and
                    comparison.get('seed') == seed and comparison.get('imgsz') == 640 and
                    training.get('completed_epochs') == 200):
                valid.append(path.resolve())
        if len(valid) != 1:
            raise RuntimeError(f'Need one completed seed{seed}/640/200 report for {pattern}, found {len(valid)}. '
                               'Use --e47-report/--e40-report for ambiguity; do not inspect active seed2 files.')
        answer.append(valid[0])
    return tuple(answer)
