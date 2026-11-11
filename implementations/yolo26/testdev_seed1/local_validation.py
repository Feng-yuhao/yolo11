"""PRIVATE synthetic forward/scoring test with actual downloaded checkpoints.

Does NOT evaluate real test-dev, train, modify weights, or issue server gates.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import run_eval as r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source.resolve()))
    import os
    (args.root / 'settings/Ultralytics').mkdir(parents=True)
    os.environ['YOLO_CONFIG_DIR'] = str(args.root / 'settings')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    image_dir = args.root / 'data/images'
    annotation_dir = args.root / 'data/annotations'
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    rng = np.random.default_rng(26160)
    for i in range(8):
        h, w = ((129, 177), (121, 161), (160, 160), (145, 215))[i % 4]
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(image_dir / f'{i:04d}.jpg')
        (annotation_dir / f'{i:04d}.txt').write_text(f'21,24,30,31,1,{i % 10 + 1},0,0\n', encoding='ascii')
    records, meta = r.load_split(args.root / 'data', 8)
    r.package_audit()
    checkpoints = r.checkpoint_audit(args.project)
    import d27_official as official
    official.run_unit_tests()
    local = SimpleNamespace(local_test=True, local_size=160)
    result = {}
    for variant in r.VARIANTS:
        out = args.root / variant
        prediction = r.predict_fixed(local, variant, Path(checkpoints[variant]['path']), out, records)
        metrics, ignore = r.score_fixed(records, out)
        again = args.root / (variant + '_fresh_reload')
        r.predict_fixed(local, variant, Path(checkpoints[variant]['path']), again, records)
        reproduction = r.assert_val_predictions(r.read(again / 'area_predictions.json'),
                                               r.read(out / 'area_predictions.json'))
        result[variant] = dict(prediction=prediction, metrics=metrics, ignore=ignore, reproduction=reproduction)
    r.write(args.root / 'PRIVATE_LOCAL_VALIDATION.json', dict(status='passed', results=result,
            real_testdev_evaluated=False, training_started=False, server_receipt_issued=False,
            note='Actual trained weights; synthetic160px, not real640/1610-image server evaluation',
            torch=torch.__version__, cuda=torch.version.cuda))
    print('PRIVATE LOCAL VALIDATION PASSED; no server result or receipt issued')


if __name__ == '__main__':
    main()
