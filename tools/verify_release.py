"""Verify published repository files; optionally verify downloaded trained weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--weights",action="store_true");a=p.parse_args()
    files=json.loads((ROOT/"manifests/repository_files.json").read_text(encoding="utf-8"))
    for row in files:
        file=(ROOT/row["path"]).resolve()
        if not file.is_relative_to(ROOT.resolve()) or not file.is_file():raise RuntimeError("Missing/unsafe repository file: "+row["path"])
        if file.stat().st_size!=row["bytes"] or sha(file)!=row["sha256"]:raise RuntimeError("Repository bytes differ: "+row["path"])
    if a.weights:
        weights=json.loads((ROOT/"manifests/weights.json").read_text(encoding="utf-8"))["checkpoints"]
        for row in weights:
            file=ROOT/row["destination"]
            if not file.is_file() or file.stat().st_size!=row["bytes"] or sha(file)!=row["sha256"]:raise RuntimeError("Missing/changed checkpoint: "+row["asset"])
    print(f"Verified {len(files)} published files"+(" and all 27 trained checkpoints" if a.weights else ""))


if __name__=="__main__":main()
