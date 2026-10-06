"""Compare actual E1/E40 saved-checkpoint inference to unchanged Predictor.

Explicit local synthetic-image CPU test, not the server GPU/memory safety gate.
Never modifies weights. Usage: python local_integration.py --project ... --source ...
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import testdev as app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "0"], help="Local fixture only, never publishes a server smoke gate")
    parser.add_argument("--memory-cap-gib", type=float, default=2.)
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Permit local fixture alongside desktop GPU use")
    args = parser.parse_args()
    args.project, args.source = args.project.resolve(), args.source.resolve()
    app.setup_source(args.source, args.project)
    import spatial_task_head
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    torch.set_num_threads(2)
    app.configure_numeric(torch)
    cpu_test = args.device == "cpu"
    if not cpu_test:
        app.require(torch.cuda.is_available(), "Local CUDA unavailable")
        free, total = torch.cuda.mem_get_info(0)
        app.require(free >= (args.memory_cap_gib + 3) * app.GIB, "Not enough LOCAL CUDA fixture memory")
        torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib * app.GIB / total, 0)
    specs = app.resolve_specs(args.project)
    with tempfile.TemporaryDirectory(prefix="testdev_integration_") as tmp:
        root = Path(tmp)
        images = []
        fixture = args.project / "runs/e1_yolo11s_p2add_img640_seed1/val_batch0_labels.jpg"
        app.require(fixture.is_file(), "Need saved val mosaic for a NONEMPTY detection fixture: " + str(fixture))
        original = Image.open(fixture).convert("RGB")
        # Saved mosaic, not benchmark images/results: test coordinate mechanics only.
        tile = original.crop((0, 0, original.width // 2, original.height // 2))
        # Both rectangular and mixed-shape chunks; official batch4 is preserved.
        for i, shape in enumerate([(480,640)] * 4 + [(640,480),(640,640),(480,640),(600,750)]):
            path = root / f"{i:03d}.jpg"
            tile.resize((shape[1], shape[0])).save(path)
            images.append(dict(id=i+1, stem=path.stem, path=str(path), width=shape[1], height=shape[0],
                               sha256=app.official.sha256(path)))
        payload = dict(images=images, resources={"memory_cap_gib":args.memory_cap_gib},
                       dataset={"annotation_dimension_sha256":"synthetic", "image_content_sha256":"synthetic"},
                       protocol=app.inference_protocol(), environment={"source_sha256":"fixture"})
        for spec in specs:
            # Test unpickling/names/strides for ALL six, real forward for seed1 pair.
            model = YOLO(spec["path"], task="detect")
            app.validate_model(model, spec)
            if spec["seed"] != 1:
                del model
                print(spec["key"] + " real checkpoint class/stride mapping passed", flush=True)
                continue
            expected = []
            options = dict(**app.PREDICTION_OPTIONS, device=args.device)
            predictor = DetectionPredictor(overrides={**model.overrides,**options,"mode":"predict"}, _callbacks=model.callbacks)
            for start in range(0,8,4):
                values = model.predict([r["path"] for r in images[start:start+4]], batch=4,
                                       predictor=predictor, stream=False, **options)
                for index, value in enumerate(values, start+1):
                    for x1,y1,x2,y2,score,cls in value.boxes.data.cpu().tolist():
                        expected.append(dict(image_id=index,category_id=int(cls)+1,
                                             bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)], score=score))
            del model, predictor, values
            if not cpu_test:
                torch.cuda.empty_cache()
            out = root / spec["key"]
            app.predict_model(args, payload, spec, out, smoke=True, cpu_test=cpu_test)
            actual = app.read(out / "predictions.json")
            app.require(len(expected) > 0, "Nonempty final-box test was not exercised")
            if actual != expected:
                raise RuntimeError("Final predictions differ from unchanged standard Predictor: " + spec["key"])
            meta = app.read(out / "prediction_manifest.json")
            app.require(len(meta["actual_tensor_shapes"]) >= 2, "Rectangular/mixed chunk geometry not exercised")
            app.require(meta["cuda_smoke"] == (not cpu_test), "Wrong local fixture device label")
            print(spec["key"] + f" exact final-box equivalence / shapes / original-coordinate mapping passed ({len(actual)} boxes)", flush=True)
    print(f"LOCAL {'CPU' if cpu_test else 'CUDA FIXTURE'} INTEGRATION PASSED; server CUDA smoke NOT performed", flush=True)


if __name__ == "__main__":
    main()
