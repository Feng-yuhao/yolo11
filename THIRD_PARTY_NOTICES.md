# Third-party sources and attribution

- Frozen Ultralytics 8.3.0 research-workspace sources are under `vendor/ultralytics83`; the actual workspace may contain unrelated registrations in `nn/tasks.py`. The paper model topology and original source pins govern reproduction.
- Frozen Ultralytics 8.4.177 sources are under `vendor/ultralytics26`, verified against the original YOLO26 training source manifest. Upstream: https://github.com/ultralytics/ultralytics and https://pypi.org/project/ultralytics/8.4.177/ . Retain upstream AGPL-3.0 notices and license files.
- TOOD task decomposition is adapted from https://github.com/fcjian/TOOD at commit `1bda27c6f290c5b828c566eb95ea50fed75a53cb`. Original author files and Apache-2.0 notices are retained under the TD implementation. Cite Feng et al., TOOD: Task-Aligned One-Stage Object Detection, ICCV 2021.
- Hierarchical group connections draw on Gao et al., Res2Net: A New Multi-scale Backbone Architecture. This is an architectural attribution, not a claim that the complete Res2Net backbone is implemented.
- COCO evaluation uses the separately installed pycocotools implementation. Original-source VisDrone and UAVDT evaluator provenance is retained in the preserved implementations.
- Dataset images and raw benchmark GT are obtained separately under their providers' terms. Data/source links and license distinctions are in `data/README.md`; this code license does not replace dataset licenses.

The AGPL-3.0 root license applies to the project's code subject to separately retained upstream notices. Initialization/AMP auxiliary checkpoints are clearly distinguished from the paper's trained checkpoints in `manifests/pretrained.json`.
