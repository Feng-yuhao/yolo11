# Songdo Vision v2 — audited YOLO preparation v1

Original: https://zenodo.org/records/21728430 (version-specific DOI).
Author article: Fonod, Cho, Yeo, Geroliminis, Advanced computer vision for
extracting georeferenced vehicle trajectories from drone imagery,
Transportation Research Part C178 (2025),105205, DOI10.1016/j.trc.2025.105205.
License: CC-BY4.0, original license and source documentation preserved.
This is a layout/split derivative, not a new dataset or author-provided validation split.

## Data and changes

5419 original4K RGB JPEGs,274190 objects, classes0car /1bus /2truck /3motorcycle.
Original official train4335, test1084. No images or annotations were deleted,
repaired, resized, re-encoded, or moved between official train and official test.
All original JPEG, YOLO TXT and VOC XML bytes are preserved with SHA256.
Author v2 already supplies YOLO; we do NOT convert rounded COCO boxes back into
lower-precision YOLO labels. COCO/VOC use integer-rounded corners, YOLO keeps
the author's subpixel normalization; this small difference is documented.
Eight portrait images use their own2160x3840 dimensions, not3840x2160.

The author's example YAML aliases val to test. The primary data.yaml here does
NOT do that. Before any experiment,434 frames (10%, rounded half-up) are held
out from the original train by the smallest SHA256 hashes of
`songdo-v2-holdout:20261017:{original_integer_image_id}`.
Result: train3901 / val434 / test1084. No model predictions/metrics influence this.
Original COCO image/annotation IDs and test membership remain unchanged.
Use this SAME frozen split for E1/E60/all seeds; only val selects checkpoints.
Evaluate official test only after training. Do not tune on test or resplit for gains.

This is frame-level holdout, NOT a scene/video-disjoint generalization split.
The author's original train/test also randomly sample video frames. Similar
frames may exist across splits; distinct hashes do not prove scene independence.
The author's README documents further v2 labeling/provenance limitations.
No gain or generalization conclusion is claimed before actual evaluation.

## Layout

    data.yaml
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
    annotations_coco/{train,val,test}.json
    annotations_voc/{train,val,test}/*.xml
    splits/{train,val,test,official_train,official_test}.txt
    provenance/official/                 # original README/license/names/YAML
    provenance/original_coco/            # original JSON bytes
    provenance/samples.jsonl
    provenance/conversion_summary.json
    preparation_code/
    verify_dataset.py
    SHA256SUMS.json
    VERIFICATION.json
    LICENSE.txt

YOLO trains on the original TXT labels. annotations_coco/test.json preserves
the full official test COCO ground truth. Keep its four categories1..4 and map
YOLO class0..3 by+1 when evaluating COCO predictions. COCO boxes are0-based
continuous xywh, area=width*height, iscrowd0; do not add+1 area or use VisDrone's
ignore-region scoring on this dataset. Native YOLO AP and COCO API AP must be
named separately. Select a final fixed scoring protocol before training.
Statistics at input640 use sqrt(original YOLO bbox area)*640/max(W,H), NOT
the COCO original-image AP-small threshold; these are different definitions.

## Server

Do not extract this~16GB dataset while storing its equally-large ZIP on a
nearly-full50GB data disk. Upload the ZIP to /root/autodl-fs (file storage),
then extract to /root/autodl-tmp/project if at least18GiB is free there.
No GPU is needed for verification; do not alter a running training instance.

    cd /root/autodl-fs
    sha256sum -c SongdoVision_v2_YOLO_v1.zip.sha256
    mkdir -p /root/autodl-tmp/project
    test ! -e /root/autodl-tmp/project/SongdoVision_v2_YOLO_v1
    unzip -q SongdoVision_v2_YOLO_v1.zip -d /root/autodl-tmp/project
    /root/miniconda3/envs/yolo11/bin/python -u \
      /root/autodl-tmp/project/SongdoVision_v2_YOLO_v1/verify_dataset.py

Data argument for future training:
`/root/autodl-tmp/project/SongdoVision_v2_YOLO_v1/data.yaml`.
The YAML contains no workstation or server-specific root and needs no edits
when moved. Always supply its absolute path. Do not reuse the author's unsafe
example val=test YAML as the primary experiment configuration.
This package does not install dependencies, train a model, or shut down anything.
