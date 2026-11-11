# Dataset sources and reconstruction

|Dataset|Official source|Paper partition and scoring|
|---|---|---|
|VisDrone2019 DET|https://github.com/VisDrone/VisDrone-Dataset|6,471 train / 548 val / 1,610 test-dev; ten classes; MATLAB v1.0.4 semantic port with original ignore rules and class-image weighting.|
|Songdo Vision v2|https://doi.org/10.5281/zenodo.21728430|Official 1,084-image test and 55,528 boxes; original integer COCO GT; four vehicle classes; standard COCO maxDets100.|
|DIOR|https://gcheng-nwpu.github.io/#Datasets|All 11,738 official test images, including 8,430 vehicle-negative images; derived vehicle-only task, not the full 20-class benchmark.|
|UAVDT|https://sites.google.com/view/grli-uavdt|Official 20 DET test sequences / 16,592 frames; original continuous xywh GT and author ignore definitions.|

`manifests/*.csv` records the exact input population and available fingerprints without mirroring image or bounding-box data. `target_GT_bindings.json` records original prepared GT hashes. Official inputs should be retained separately, and the preparation scripts should write a new destination.

VisDrone conversion preserves the six-decimal normalized xywh labels of the original 8.3.0 conversion, excludes rows marked score zero, and checks the original training filename/label-byte fingerprint. Original VisDrone annotations remain necessary for final scoring; positive-only YOLO labels cannot encode all ignore semantics.

Songdo already supplies the required formats. Preserve official v2 COCO bytes and each image's actual orientation/dimensions. Do not reconstruct the labels from rounded COCO coordinates or use a historical target-training holdout as the official test split.

DIOR conversion subtracts one from each XML corner and uses `width=xmax-xmin`, `height=ymax-ymin`, with no inclusive +1 extent. Retain the single zero-area vehicle record as ignored, leaving 26,639 positive boxes. Do not delete images without vehicle boxes. The normalized positive-label view is for interoperability; COCO GT and the documented protocol govern scoring.

UAVDT preparation retains the author toolkit's sequence lengths and 18 ignore-sequence definitions. Predictions are removed only if all four edges are strictly inside an ignore region; GT is retained. The three-class COCO-style AP and native class-agnostic AP70 are separate metrics. Original clipped YOLO labels are not the authoritative source of scoring coordinates.

Songdo v2 is CC BY 4.0. The DIOR author page states CC BY-NC 4.0. VisDrone/UAVDT are linked to their official providers; this release does not assert unrestricted permission to mirror their images or raw annotations. Preserve each provider's attribution/license and document derived-format changes.

The local research archive and earlier private recovery backup intentionally did not mirror every downloadable image archive. That is not a claim that a complete offline image backup exists. Model code, recorded predictions, checkpoint artifacts and published reconstruction metadata are supplied separately.
