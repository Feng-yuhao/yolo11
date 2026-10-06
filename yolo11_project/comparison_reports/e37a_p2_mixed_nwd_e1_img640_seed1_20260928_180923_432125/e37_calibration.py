"""Fix NWD's pixel constant using TRAIN labels and image dimensions only."""
import hashlib
import math
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.data.utils import check_det_dataset, img2label_paths


def calibrate(data_yaml, expected_images=6471):
    data = check_det_dataset(str(data_yaml), autodownload=False)
    directory = Path(data['train'])
    if not directory.is_dir():
        raise RuntimeError('Training split must be one image directory')
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = sorted(p.resolve() for p in directory.rglob('*')
                    if p.is_file() and p.suffix.lower() in suffixes)
    if len(images) != expected_images:
        raise RuntimeError(f'Expected {expected_images} train images, found {len(images)}')
    values, total, duplicates = [], 0, 0
    digest = hashlib.sha256()
    for index, (image, label) in enumerate(zip(images, img2label_paths([str(p) for p in images])), 1):
        path = Path(label)
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        with Image.open(image) as im:
            width, height = im.size
            if im.getexif().get(274, 1) not in (None, 1):
                raise RuntimeError(f'Unexpected EXIF rotation: {image}')
        digest.update(image.name.encode() + str((width, height)).encode() + raw)
        rows = [list(map(float, line.split())) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        if rows:
            labels = np.asarray(rows, dtype=np.float32)
            if (labels.ndim != 2 or labels.shape[1] != 5 or not np.isfinite(labels).all()
                    or (labels[:, 1:] < 0).any() or (labels[:, 1:] > 1).any()
                    or (labels[:, 0] < 0).any() or (labels[:, 0] >= 10).any()
                    or (labels[:, 0] != labels[:, 0].astype(int)).any()
                    or (labels[:, 3:] <= 0).any()):
                raise RuntimeError(f'Invalid train labels: {path}')
            unique = np.unique(labels, axis=0)
            duplicates += len(labels) - len(unique)
            total += len(unique)
            sizes = np.sqrt(unique[:, 3].astype(float)*width*unique[:, 4].astype(float)*height)*640/max(width, height)
            values.extend(sizes[(sizes > 0) & (sizes < 16)].tolist())
        if index % 1000 == 0 or index == len(images):
            print(f'E37a train-only calibration: {index}/{len(images)}', flush=True)
    if not values:
        raise RuntimeError('No train targets below 16 input pixels')
    constant = math.fsum(values)/len(values)
    return dict(constant_px=constant, definition='mean sqrt(w*h) for 0<side<16 after static 640 letterbox resize',
                split='train', input_size=640, images=len(images), gt=total, selected_gt=len(values),
                duplicates_removed=duplicates, train_label_dimension_sha256=digest.hexdigest(),
                uses_validation=False, augmentation_note='Constant is fixed; actual loss uses each augmented batch GT size.')
