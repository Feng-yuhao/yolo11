"""Train-only annotation adapter. Never changes the frozen val evaluator.

Zero-extent class-0 ignore rows remain original raster regions: the frozen
official coverage helper already uses max(1, round(extent)) for region queries.
Foreground boxes, negative extents and malformed classes are still rejected.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import d27_official as official


def parse_training_annotation(path):
    path = Path(path)
    rows, degenerate = [], []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(',')]
        if fields[-1] == '':
            fields.pop()
        if len(fields) < 8:
            raise ValueError(f'{path}:{line_no}: expected at least 8 columns')
        row = [float(item) for item in fields[:8]]
        if not all(math.isfinite(item) for item in row):
            raise ValueError(f'{path}:{line_no}: non-finite annotation: {row}')
        cls = row[5]
        if cls != int(cls) or int(cls) not in range(12) or row[2] < 0 or row[3] < 0:
            raise ValueError(f'{path}:{line_no}: invalid box or class: {row}')
        if row[2] == 0 or row[3] == 0:
            if cls != 0:
                raise ValueError(f'{path}:{line_no}: zero-extent non-class0 box: {row}')
            degenerate.append(dict(annotation=path.name, line=line_no, row=row.copy()))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8), degenerate


def load_training_dataset(dataset_root, expected_images=6471):
    dataset_root = Path(dataset_root)
    annotation_dir, image_dir = dataset_root/'annotations', dataset_root/'images'
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f'Expected images/ and annotations/ under {dataset_root}')
    paths = sorted(annotation_dir.glob('*.txt'))
    if len(paths) != expected_images:
        raise ValueError(f'Expected {expected_images} training annotations, found {len(paths)}')
    records, audit = [], []
    digest = hashlib.sha256()
    counts = {str(k): 0 for k in range(12)}
    zeros = {str(k): 0 for k in range(12)}
    dropped = 0
    for image_id, path in enumerate(paths, 1):
        image_path = next((image_dir/(path.stem+s) for s in ('.jpg','.JPG','.jpeg','.png')
                           if (image_dir/(path.stem+s)).is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f'Image for {path} not found')
        with Image.open(image_path) as image:
            width, height = image.size
        raw, events = parse_training_annotation(path)
        audit.extend(events)
        for row in raw:
            counts[str(int(row[5]))] += 1
            if row[4] == 0:
                zeros[str(int(row[5]))] += 1
        ignore = raw[raw[:,5] == 0, :4].copy()
        others = raw[raw[:,5] == 11, :4].copy()
        gt = official.drop_gt_in_class0_regions(raw, ignore, width, height)
        dropped += int(np.count_nonzero(raw[:,5] != 0)-len(gt))
        digest.update(path.name.encode('utf-8')+b'\0'+path.read_bytes()+b'\0')
        digest.update(f'{width}x{height}'.encode('ascii')+b'\0')
        records.append(official.ImageRecord(image_id, path.stem, image_path, path,
                                           width, height, raw, gt, ignore, others))
        if image_id % 1000 == 0:
            print(f'D47 train annotation audit: {image_id}/{len(paths)}; '
                  f'zero-extent class0 regions={len(audit)}', flush=True)
    summary = dict(dataset_root=str(dataset_root), images=len(records),
        annotation_dimension_sha256=digest.hexdigest(), raw_class_counts=counts,
        score_zero_counts=zeros, class0_ignore_regions=counts['0'], class11_others=counts['11'],
        gt_rows_dropped_by_class0_region=dropped,
        evaluated_gt_rows=int(sum(np.count_nonzero((r.eval_gt[:,5]>=1)&(r.eval_gt[:,5]<=10)) for r in records)),
        annotation_policy='train_only_preserve_zero_extent_class0_with_official_raster_query',
        zero_extent_class0_count=len(audit), zero_extent_class0_rows=audit)
    print(f'D47 train annotation audit: {len(records)} images; '
          f'preserved zero-extent class0 regions={len(audit)}', flush=True)
    return records, summary
