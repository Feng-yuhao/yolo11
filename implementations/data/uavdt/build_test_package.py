"""Build only the official test partition, preserving archived bytes and GT."""
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'LEGACY_ARCHIVE_20261013/original_root'
OUT = ROOT / 'PAPER_EXPERIMENTS/datasets'
NAME = 'UAVDT_TestOnly_YOLO_v1'


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / (NAME + '.zip')
    partial = OUT / (NAME + '.zip.building')
    assert not final.exists() and not partial.exists(), 'Refusing overwrite'
    entries = []
    with ZipFile(SOURCE / 'UAVDT_YOLO.zip') as src, ZipFile(SOURCE / 'UAV-benchmark-MOTD_v1.0.zip') as original, ZipFile(partial, 'x', allowZip64=True) as dst:
        def put(path, payload):
            dst.writestr(NAME + '/' + path, payload,
                         compress_type=ZIP_STORED if path.endswith('.jpg') else ZIP_DEFLATED)
            entries.append({'path': path, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})

        splits = json.loads(src.read('UAVDT_YOLO/split_sequences.json'))
        sequences = splits['official_test_sequences']
        assert sequences == splits['test_sequences'] and len(sequences) == 20
        # Cross-check actual test partition against the author's detection evaluator.
        native = original.read('UAV-benchmark-MOTD_v1.0/utils/CalculateDetectionPR_overall.m').decode()
        seqline = re.search(r'seqDirs\s*=\s*\{([^}]+)\}', native).group(1)
        assert re.findall(r"'(M\d+)'", seqline) == sequences
        lens = [int(v) for v in re.search(r'seqLens\s*=\s*\[([^]]+)\]', native).group(1).split(',')]
        assert sum(lens) == 16592
        expected = {f'{seq}_img{frame:06d}' for seq, length in zip(sequences, lens) for frame in range(1, length + 1)}
        images = sorted(n for n in src.namelist() if n.startswith('UAVDT_YOLO/images/test/') and n.endswith('.jpg'))
        labels = sorted(n for n in src.namelist() if n.startswith('UAVDT_YOLO/labels/test/') and n.endswith('.txt'))
        assert {Path(n).stem for n in images} == {Path(n).stem for n in labels} == expected
        gt = defaultdict(list)
        for seq in sequences:
            for line in original.read(f'UAV-benchmark-MOTD_v1.0/GT/{seq}_gt_whole.txt').decode().splitlines():
                if not line.strip():
                    continue
                vals = list(map(float, line.split(',')))
                assert len(vals) == 9 and all(math.isfinite(v) for v in vals)
                frame, _, x, y, w, h, _, _, category = vals
                assert frame == int(frame) and category in (1, 2, 3) and w > 0 and h > 0
                stem = f'{seq}_img{int(frame):06d}'
                assert stem in expected
                gt[stem].append((int(category) - 1, x, y, w, h))
        classes = Counter()
        total_boxes = 0
        max_error = 0.0
        dimensions = Counter()
        image_records = []
        for i, n in enumerate(images, 1):
            payload = src.read(n)  # ZipFile independently checks source CRC.
            with Image.open(io.BytesIO(payload)) as im:
                im.load()
                width, height = im.size
                assert im.format == 'JPEG'
            dimensions[f'{width}x{height}'] += 1
            stem = Path(n).stem
            label = src.read(f'UAVDT_YOLO/labels/test/{stem}.txt')
            actual = []
            for line in label.decode().splitlines():
                if not line.strip():
                    continue
                vals = list(map(float, line.split()))
                assert len(vals) == 5 and all(math.isfinite(v) for v in vals)
                c, cx, cy, w, h = vals
                assert c in (0, 1, 2) and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
                actual.append((int(c), cx * width, cy * height, w * width, h * height))
                classes[str(int(c))] += 1
            oracle = []
            for c, x, y, w, h in gt[stem]:
                x1, y1 = max(0, min(width, x)), max(0, min(height, y))
                x2, y2 = max(0, min(width, x + w)), max(0, min(height, y + h))
                assert x2 > x1 and y2 > y1
                oracle.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            assert len(actual) == len(oracle), (stem, len(actual), len(oracle))
            # The archived converter follows original GT row order.
            for a, b in zip(actual, oracle):
                assert a[0] == b[0], stem
                err = max(abs(v - u) for v, u in zip(a[1:], b[1:]))
                max_error = max(max_error, err)
                assert err < 0.0001, (stem, a, b, err)
            total_boxes += len(actual)
            put(f'images/test/{stem}.jpg', payload)
            put(f'labels/test/{stem}.txt', label)
            image_records.append({'file_name': f'images/test/{stem}.jpg', 'sequence': stem.split('_')[0],
                                  'frame': int(stem.split('img')[1]), 'width': width, 'height': height,
                                  'boxes': len(actual)})
            if i % 1000 == 0 or i == len(images):
                print(f'DECODE / GT / PACK {i}/{len(images)}', flush=True)
        assert total_boxes == 375884
        # Keep original DET GT, ignore regions, MOT GT, and author evaluation source.
        for n in original.namelist():
            if n.endswith('/'):
                continue
            rel = n.split('/', 1)[1]
            if rel.startswith('GT/'):
                if rel.split('/')[1].split('_')[0] not in sequences:
                    continue
            elif not (rel.endswith('.m') or rel.lower().endswith(('readme.md', 'license', 'license.txt', 'copying'))):
                continue
            put('original_toolkit/' + rel, original.read(n))
        for name in ('split_sequences.json', 'conversion_stats.json', 'data.yaml'):
            put('provenance/archived_' + name, src.read('UAVDT_YOLO/' + name))
        put('test.txt', ''.join('./images/test/' + Path(n).name + '\n' for n in images).encode())
        # Test-only descriptor; no train/val alias that might encourage target selection.
        put('data_test.yaml', b'test: images/test\nnames:\n  0: car\n  1: truck\n  2: bus\n')
        put('test_images.json', json.dumps(image_records, indent=2).encode())
        audit = {'status': 'passed', 'images': len(images), 'sequences': dict(zip(sequences, lens)),
                 'boxes_in_archived_yolo_and_original_det_gt': total_boxes, 'class_counts_0car_1truck_2bus': dict(classes),
                 'dimensions': dict(dimensions), 'max_existing_yolo_vs_clipped_original_gt_error_px': max_error,
                 'full_image_decode': True, 'image_and_label_bytes_unchanged': True,
                 'partition_matched_author_detection_evaluator': True,
                 'ignore_annotations_preserved': True, 'scoring_protocol_finalized': False,
                 'training_started': False, 'evaluation_started': False,
                 'sources': {n: {'bytes': (SOURCE / n).stat().st_size, 'sha256': sha_file(SOURCE / n)}
                             for n in ('UAVDT_YOLO.zip', 'UAV-benchmark-MOTD_v1.0.zip')}}
        put('dataset_audit.json', json.dumps(audit, indent=2).encode())
        put('verify_dataset.py', Path(__file__).with_name('verify_dataset.py').read_bytes())
        readme = '''# UAVDT test-only data package

16,592 original JPEGs from the author's 20 DET test sequences; no train/val images.
Existing YOLO labels are already converted: 0=car, 1=truck, 2=bus.
All 375,884 YOLO boxes were checked against *_gt_whole.txt using continuous
xywh -> image-bound clipping -> normalized center/width/height (no minus-one).
All images fully decoded; image and label bytes remain unchanged.

original_toolkit/GT includes the test sequences' original DET/MOT/ignore files.
Do NOT treat YOLO-only scoring as a reproduction of official ignore-aware AP.
The legacy YOLO labels contain the DET boxes, not a representation of ignored
regions. Ignore files must be used explicitly by the eventual evaluator.
No new scoring protocol, COCO conversion, inference, training or shutdown is run.
This package is data, NOT a complete source-only evaluation program.
data_test.yaml describes only test/names; it intentionally has no train/val keys
and is not intended as a standalone Ultralytics training configuration.

After extraction, run: python UAVDT_TestOnly_YOLO_v1/verify_dataset.py
Keep this folder separate from any old full UAVDT_YOLO folder.
All payload SHA256 values are recorded in manifest.json (excluding itself).
Original authors' README, evaluation source and any bundled license notices are
preserved in original_toolkit. Cite Du et al., ECCV 2018; follow original terms.
'''
        put('README.md', readme.encode())
        dst.writestr(NAME + '/manifest.json', json.dumps({'files': entries}, indent=2), compress_type=ZIP_DEFLATED)
    print('Independently rereading entire output archive / CRC / hashes...', flush=True)
    with ZipFile(partial) as z:
        manifest = json.loads(z.read(NAME + '/manifest.json'))
        assert len(z.namelist()) == len(manifest['files']) + 1
        for i, e in enumerate(manifest['files'], 1):
            data = z.read(NAME + '/' + e['path'])
            assert len(data) == e['bytes'] and hashlib.sha256(data).hexdigest() == e['sha256']
            if i % 5000 == 0:
                print(f'OUTPUT VERIFY {i}/{len(manifest["files"])}', flush=True)
    digest = sha_file(partial)
    partial.rename(final)
    (OUT / (NAME + '.zip.sha256')).write_text(digest + '  ' + final.name + '\n', encoding='ascii', newline='\n')
    (OUT / (NAME + '.build_audit.json')).write_text(json.dumps({**audit, 'output': str(final), 'output_bytes': final.stat().st_size,
                                                             'output_sha256': digest, 'output_all_members_verified': True}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(final), 'bytes': final.stat().st_size, 'sha256': digest, 'max_error_px': max_error}, indent=2), flush=True)


if __name__ == '__main__':
    main()
