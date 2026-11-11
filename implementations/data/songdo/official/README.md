# **Songdo Vision: Vehicle Annotations from High-Altitude BeV Drone Imagery in a Smart City**

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Creators:** Robert Fonod¹, Haechan Cho², Hwasoo Yeo², Nikolas Geroliminis¹

1. **EPFL** (École Polytechnique Fédérale de Lausanne), Switzerland
2. **KAIST** (Korea Advanced Institute of Science and Technology), South Korea

## **Overview**  
The **Songdo Vision** dataset provides **high-resolution 4K RGB images** annotated with categorized **axis-aligned bounding boxes (BBs)** for vehicle detection from a **high-altitude bird’s-eye view (BeV)** perspective. Captured over **Songdo International Business District, South Korea**, it comprises **5,419 annotated video frames** (**4,335** training / **1,084** test; 80/20 split) containing **274,190 vehicle instances** across **four classes**:

- **Car** (including vans and light-duty vehicles)  
- **Bus**  
- **Truck**  
- **Motorcycle**  

This dataset can serve as a **benchmark for aerial vehicle detection**, supporting research and real-world applications in **intelligent transportation systems, traffic monitoring, and aerial vision-based mobility analytics**. It was developed in the context of a **multi-drone experiment** aimed at enhancing **geo-referenced vehicle trajectory extraction** **[[1]](https://doi.org/10.1016/j.trc.2025.105205)**.

A full dataset card with usage instructions and download guidance is available on **[Hugging Face](https://huggingface.co/datasets/rfonod/songdo-vision)**.

> **📌 Citation:** If you use this dataset in your work, kindly acknowledge it by citing the accompanying article **[[1]](https://doi.org/10.1016/j.trc.2025.105205)**.
>
> **🔗 Related dataset:** For precisely georeferenced vehicle trajectories extracted from the same large-scale multi-drone experiment, see **Songdo Traffic**: [10.5281/zenodo.13828384](https://doi.org/10.5281/zenodo.13828384).

## **Motivation**  
Publicly available datasets for aerial vehicle detection often exhibit limitations such as:  

- **Non-BeV perspectives** with varying angles and distortions  
- **Inconsistent annotation quality**, with loose or missing bounding boxes  
- **Lower-resolution imagery**, reducing detection accuracy, particularly for smaller vehicles  
- **Lack of annotation detail**, especially for **motorcycles** in dense urban scenes with **complex backgrounds**

To address these challenges, **Songdo Vision** provides **high-quality human-annotated bounding boxes**, with machine learning assistance used to enhance efficiency and consistency. This ensures accurate and reliable **ground truth** for training and evaluating detection models.

## **Dataset Composition**  
The dataset is randomly split into **training (80%)** and **test (20%)** subsets:  

| **Subset** | **Images** | **Car** | **Bus** | **Truck** | **Motorcycle** | **Total Vehicles** |
|------------|------------|---------|---------|-----------|----------------|--------------------|
| **Train**  | 4,335      | 196,719 | 7,052   | 11,861    | 3,030          | **218,662**        |
| **Test**   | 1,084      | 49,862  | 1,766   | 3,080     | 820            | **55,528**         |
| **Total**  | **5,419**  | **246,581** | **8,818** | **14,941** | **3,850**  | **274,190**        |

A subset of **5,274 frames** was randomly sampled from drone video sequences, while an additional **145 frames** were carefully selected to represent **challenging cases**, such as motorcycles at pedestrian crossings, in bicycle lanes, near traffic light poles, and around other distinctive road markers where they may blend into the urban environment.

## **Data Collection**  
The dataset was collected as part of a **collaborative multi-drone experiment** conducted by **KAIST and EPFL** in **Songdo, South Korea**, from **October 4–7, 2022**.  

- **A fleet of 10 drones** monitored **20 busy intersections**, executing advanced flight plans to optimize coverage.  
- **4K (3840×2160) RGB video footage** was recorded at **29.97 FPS** from altitudes of **140–150 meters**.  
- Each drone **flew 10 sessions per day**, covering peak morning and afternoon periods.  
- The experiment resulted in **12TB of 4K raw video data**.  

More details on the **experimental setup and data processing pipeline** are available in **[[1]](https://doi.org/10.1016/j.trc.2025.105205)**.

> **Note on frame dimensions:** 5,411 frames are **3840×2160** (landscape 4K). Eight frames — `05275`–`05281` in the training subset and `05282` in the test subset — are **2160×3840** (portrait 4K), as recorded. Their annotations are correct for that orientation, and the per-image dimensions are stated explicitly in the COCO `images` entries and the Pascal VOC `<size>` element. Normalized YOLO coordinates are relative to each frame's own dimensions. Consumers that assume a fixed 3840×2160 frame size must read the per-image dimensions instead.

## **Bounding Box Annotations & Formats**  
Annotations were generated using a **semi-automated object detection annotation process** in **Azure ML Studio**, leveraging **machine learning-assisted bounding box detection** with **human verification** to ensure precision.

Each annotated frame includes **categorized, axis-aligned bounding boxes**, stored in **three widely-used formats**. All three are generated from a **single canonical source**, so they describe exactly the same boxes: the YOLO files carry sub-pixel precision, while the COCO and Pascal VOC files share one common integer rounding of those coordinates.

1. **COCO JSON format**  
   - **Single annotation file per dataset subset** (i.e., one for training, one for testing).  
   - Contains metadata such as **image dimensions**, **bounding box coordinates**, and **class labels**.  
   - Example snippet:  
     ```json
     {
       "info": {"description": "Songdo Vision: …", "version": "2.0", "year": 2026},
       "licenses": [{"id": 1, "name": "Creative Commons Attribution 4.0 International",
                     "url": "https://creativecommons.org/licenses/by/4.0/"}],
       "images": [{"id": 1, "file_name": "00001.jpg", "width": 3840, "height": 2160}],
       "annotations": [
         {"id": 1, "image_id": 1, "category_id": 2, "bbox": [500, 600, 200, 50],
            "area": 10000, "iscrowd": 0
         }
       ],
       "categories": [
         {"id": 1, "name": "car", "supercategory": "vehicle"},
         {"id": 2, "name": "bus", "supercategory": "vehicle"},
         {"id": 3, "name": "truck", "supercategory": "vehicle"},
         {"id": 4, "name": "motorcycle", "supercategory": "vehicle"}
       ]
     }
     ```
   - `bbox` is `[x_min, y_min, width, height]` in absolute pixels; `area` equals `width × height`.
   - Image and annotation `id`s are **globally unique across both subsets**. Annotations carried over
     from v1 keep their original id (all `≤ 272435`), so records can be traced between releases;
     annotations **added in v2** are numbered from `1000001` upwards, which makes the two sets easy to
     tell apart. Ids are **not** contiguous.

2. **YOLO TXT format**  
   - **One annotation file per image**, following the format:  
     ```
     <class_id> <x_center> <y_center> <width> <height>
     ```
   - Bounding box values are **normalized** to **[0,1]** relative to that frame's own dimensions, with the origin at the top-left corner. Every box lies fully inside the frame.
   - Class ids follow `names.txt`: `0 = car`, `1 = bus`, `2 = truck`, `3 = motorcycle`.
   - Example snippet:  
     ```
     0 0.520000 0.630000 0.100000 0.050000
     2 0.250000 0.400000 0.150000 0.080000
     ```

3. **Pascal VOC XML format**  
   - **One annotation file per image**, structured in XML.  
   - Contains **image properties** and **absolute pixel coordinates** for each bounding box.  
   - `<truncated>` is set to `1` for boxes that touch a frame edge, `0` otherwise.  
   - Example snippet:  
     ```xml
     <annotation>
       <folder>images</folder>
       <filename>00001.jpg</filename>
       <path>images/00001.jpg</path>
       <source><database>Songdo Vision</database></source>
       <size><width>3840</width><height>2160</height><depth>3</depth></size>
       <segmented>0</segmented>
       <object>
         <name>car</name>
         <pose>Unspecified</pose>
         <truncated>0</truncated>
         <difficult>0</difficult>
         <bndbox><xmin>500</xmin><ymin>600</ymin><xmax>600</xmax><ymax>650</ymax></bndbox>
       </object>
     </annotation>
     ```

## **File Structure**  
#### **1. Training Data (train.zip, ≈12.9 GB)**  

```
train/
│── coco_annotations.json  # COCO format
│── images/
│   ├── 00001.jpg
│   ├── …
│── labels/
│   ├── 00001.txt  # YOLO format
│   ├── 00001.xml  # Pascal VOC format
│   ├── …
```

#### **2. Testing Data (test.zip, ≈3.2 GB)**  

```
test/
│── coco_annotations.json
│── images/
│   ├── 00027.jpg
│   ├── …
│── labels/
│   ├── 00027.txt
│   ├── 00027.xml
│   ├── …
```

#### Additional Files
   - `README.md` – Dataset documentation (this file)
   - `LICENSE.txt` – Creative Commons Attribution 4.0 License
   - `names.txt` – Class names (one per line)
   - `data.yaml` – Example YOLO configuration file for training/testing

## **Changes in v2**

v2 is an **annotation-quality release**. The **images are byte-identical to v1** — only the annotation files, `data.yaml` and `names.txt` changed. Ids of surviving v1 annotations are preserved, so v1 and v2 records can be matched directly.

In total **196 invalid boxes were removed** and **1,951 missed vehicles were added**, taking the dataset from 272,435 to **274,190** annotations:

| **Class** | **v1** | **Removed** | **Added** | **v2** |
|------------|--------|-------------|-----------|--------|
| Car        | 245,047 | 140 | 1,674 | **246,581** |
| Bus        | 8,789   | 10  | 39    | **8,818**   |
| Truck      | 14,831  | 34  | 144   | **14,941**  |
| Motorcycle | 3,768   | 12  | 94    | **3,850**   |
| **Total**  | **272,435** | **196** | **1,951** | **274,190** |

Split across subsets, 157 removals and 1,508 additions fall in the training set, and 39 removals and 443 additions in the test set.

### Invalid annotations removed

**196 of 272,435 boxes (0.072%)** were removed. Each geometry threshold was fixed only after visually inspecting the boxes it discards, and every `false_positive` removal was inspected individually.

| Reason | Removed | What it catches |
|---|---|---|
| `false_positive` | 124 | No vehicle at all: painted road markings, rooftop equipment, construction machinery, manhole covers |
| `side_too_small` | 43 | Shorter side below 6 px (smallest genuine motorcycle side is ~12 px) |
| `duplicate` | 12 | One vehicle annotated twice (IoU ≥ 0.5 and comparable box areas) |
| `extreme_aspect_ratio` | 9 | Aspect ratio above 6 *and* shorter side below 35 px |
| `implausibly_large` | 5 | Longer side above 500 px (largest genuine vehicle is ~403 px) |
| `curated` | 2 | Single box spanning several diagonally parked cars |
| `area_too_small` | 1 | Area below 150 px² (1st-percentile motorcycle area is ~520 px²) |
| **Total** | **196** | |

The thresholds are deliberately far outside the legitimate distribution. In particular, the aspect-ratio rule also requires a short side below 35 px, because genuine articulated trucks in this dataset reach an aspect ratio of ~6.9 and would otherwise have been discarded.

### Detector-assisted review

v1 was produced by a semi-automated annotation pass with human verification. That process is good at precision but can still miss objects, so v2 re-examined the annotations against an independent YOLOv8s vehicle detector trained on high-altitude BeV drone imagery ([geo-trax](https://github.com/rfonod/geo-trax)). The detector was used **only to rank candidates for human review** — it never edited an annotation on its own, and every change below was confirmed by a human looking at the image.

**False positives removed.** Every annotation was scored against the detector, and the boxes it did not support were inspected individually. Confirmed non-vehicles — road markings and painted text, rooftop plant, construction machinery, manhole covers — were removed.

**Missed vehicles added — 1,951 annotations.** Confident detections (≥ 0.70) that overlapped no existing annotation were reviewed the same way; confirmed vehicles were added. This signal is stronger than it may appear: the detector was trained *on these very annotations*, so training actively pushed it to stay silent wherever the ground truth was empty. A confident detection in an unannotated spot is therefore good evidence of a genuine miss rather than a model artefact.

Added boxes take their geometry and class from the detector, with a human confirming that a vehicle of that class is present. They carry COCO ids from `1000001` upwards, so anyone who wants only the original v1-provenance annotations can filter them out.

Review coverage: all 1,983 addition candidates and all 1,053 false-positive candidates were inspected on contact sheets; every box marked for change was then re-checked zoomed. A random sample of 72 accepted additions was independently re-checked, of which 71 were correct, so a small residual error rate (order 1%) should be assumed for the added boxes.

### Consistency and correctness fixes

- **All three formats now agree exactly.** In v1, COCO truncated the box *width* while Pascal VOC truncated the box *corner*, so the two disagreed by up to 1 px on roughly 75% of boxes. Both are now derived from one shared rounding of the same source coordinates.
- **1,464 boxes** extended microscopically past the frame border (by at most 0.002 px, a rounding artifact). All boxes are now strictly inside their frame, and every normalized YOLO coordinate lies within `[0,1]`.
- **Duplicate annotations removed.** Twelve vehicles carried two overlapping boxes, usually with conflicting class labels. A pair is treated as a duplicate only when the boxes overlap at IoU ≥ 0.5 **and** have comparable areas (ratio ≥ 0.5), so the axis-aligned box of a diagonally oriented bus can never absorb a small vehicle beside it. The tighter box was kept, except for one pickup truck where the larger box correctly covers both cab and bed.
- **COCO** gained the standard `info` and `licenses` blocks and a `supercategory` field on each category.
- **Pascal VOC** gained `<path>`, `<pose>`, `<truncated>` and `<difficult>` per object, which several standard VOC parsers require.
- **`data.yaml`** now points at the **concept DOI** (which always resolves to the latest version) instead of the v1-specific DOI, and its `path` field follows the Ultralytics convention: a relative `path` is resolved against the Ultralytics *datasets* directory, not against the location of `data.yaml`. The v1 value (`../datasets/songdo_vision`) only worked for one particular layout.
- **`names.txt`** now ends with a newline.
- The **8 portrait frames** are documented (see *Data Collection*); they were undocumented in v1, which described every frame as 3840×2160.

### Known limitations

- **Class boundaries.** Car/van/truck assignments on v1 annotations were not revised. The detector disagreed with the annotated class on roughly 0.55% of boxes, but these are largely genuine judgement calls (e.g. large vans, pickups) that the original annotators owned.
- **Remaining missed vehicles.** The additions came from detections at confidence ≥ 0.70; vehicles the detector found with lower confidence, or missed entirely, are still unannotated. Recall is improved, not complete.
- **Test-set provenance.** Because the added test-split boxes were surfaced by a geo-trax detector, benchmarking that same model family on Songdo Vision test carries a small favourable bias. It is bounded — the additions are 0.8% of the test annotations, and every one was human-confirmed — but it is worth stating for strict benchmarking.
- **Loosely overlapping pairs.** 104 box pairs overlap at an IoU between 0.3 and 0.5, and one box fully contains two others. These are almost always correct — the axis-aligned box of a diagonally parked car naturally encloses its neighbours — so they were flagged rather than changed.

## **Acknowledgments**

In addition to the funding sources listed in the metadata, the creators express their gratitude to **Artem Vasilev** for his dedicated efforts in data annotation. We also thank the research teams of **Prof. Simon Oh** (Korea University) and **Prof. Minju Park** (Hannam University) for their assistance during the data collection campaign, including the provision of drone equipment and student support.

## **Citation & Attribution**

### **Preferred Citation:**
If you use **Songdo Vision** for any purpose — whether academic research, commercial applications, open-source projects, or benchmarking efforts — please cite our accompanying article **[1]**:

> **Robert Fonod, Haechan Cho, Hwasoo Yeo, Nikolas Geroliminis** (2025).  
> *Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery*,  
> Transportation Research Part C: Emerging Technologies, vol. 178, 105205.  
> DOI: [10.1016/j.trc.2025.105205](https://doi.org/10.1016/j.trc.2025.105205)  
> *(Preprint: [arXiv:2411.02136](https://arxiv.org/abs/2411.02136))*
>
><details>
>  <summary><b>BibTeX</b></summary>
>
>  ```bibtex
>  @article{fonod2025advanced,
>    title = {Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery},
>    author = {Fonod, Robert and Cho, Haechan and Yeo, Hwasoo and Geroliminis, Nikolas},
>    journal = {Transportation Research Part C: Emerging Technologies},
>    volume = {178},
>    pages = {105205},
>    year = {2025},
>    publisher = {Elsevier},
>    doi = {10.1016/j.trc.2025.105205},
>    url = {https://doi.org/10.1016/j.trc.2025.105205}
>  }
>  ```
> </details>

**Note:** Although Zenodo automatically provides a formal dataset citation (shown below), including citation export in various formats such as BibTeX, we kindly request that you reference the above article as the primary source of this work.

### **Dataset Citation (for archival purposes):**  
> **Robert Fonod, Haechan Cho, Hwasoo Yeo, Nikolas Geroliminis** (2026).
> *Songdo Vision: Vehicle Annotations from High-Altitude BeV Drone Imagery in a Smart City* (v2). Zenodo. DOI: [10.5281/zenodo.13828407](https://doi.org/10.5281/zenodo.13828407).
>
> The DOI above is the **concept DOI**, which always resolves to the latest version. To cite this specific release, use the version DOI shown on the Zenodo record page.
