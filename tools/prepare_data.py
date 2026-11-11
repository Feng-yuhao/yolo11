"""Prepare paper data from official local downloads without re-encoding images."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
NAMES=["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
TRAIN_LABEL_SHA="7ac9962a06c0150d9e4ba052827ba41d2bec7f487ec9ccd2a72cfafa4eccedd5"


def visdrone_label_bytes(annotation,width,height):
    lines=[];dw=1./width;dh=1./height
    for line in annotation.splitlines():
        if not line.strip():continue
        fields=line.split(",")
        if int(fields[4])==0:continue
        category=int(fields[5])-1
        if category not in range(10):raise RuntimeError("Unexpected scored source category")
        x,y,w,h=map(int,fields[:4])
        values=((x+w/2)*dw,(y+h/2)*dh,w*dw,h*dh)
        lines.append(f"{category} "+" ".join(f"{v:.6f}" for v in values)+"\n")
    return "".join(lines).encode("utf-8")


def write(path,value,newline="\n"):
    path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)+"\n"
    path.write_bytes(text.replace("\n",newline).encode("utf-8"))


def place(source,target,mode):
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    if mode=="hardlink":os.link(source,target)
    else:shutil.copy2(source,target)


def prepare_visdrone(a):
    from PIL import Image
    counts={};train_sha=hashlib.sha256()
    for split,expected in [("train",6471),("val",548),("test-dev",1610)]:
        source=a.raw_root/("VisDrone2019-DET-"+split)
        images=sorted((source/"images").glob("*.jpg"))
        if len(images)!=expected:raise RuntimeError(f"Expected {expected} official {split} images; found {len(images)}")
        positives=0
        for image in images:
            with Image.open(image) as im:width,height=im.size
            annotation=source/"annotations"/(image.stem+".txt")
            data=visdrone_label_bytes(annotation.read_text(encoding="utf-8-sig"),width,height);positives+=len(data.splitlines())
            label=a.output/"labels"/split/(image.stem+".txt");label.parent.mkdir(parents=True,exist_ok=True);label.write_bytes(data)
            place(image,a.output/"images"/split/image.name,a.image_mode)
            if split=="train":train_sha.update(image.name.encode()+b"\0"+data+b"\0")
        counts[split]={"images":len(images),"labels":positives}
    matched=train_sha.hexdigest()==TRAIN_LABEL_SHA
    write(a.output/"verification.json",{"dataset":"VisDrone2019 DET","counts":counts,"train_names_and_label_sha256":train_sha.hexdigest(),"expected_original_training_label_sha256":TRAIN_LABEL_SHA,"matches_original_training_labels":matched,"geometry":"original 8.3.0 xywh normalization; six decimals; no clipping/re-encoding","official_evaluation":"use the original downloaded annotations and tools/score_visdrone.py"})
    (a.output/"data.yaml").write_text("path: "+json.dumps(str(a.output.resolve()))+"\ntrain: images/train\nval: images/val\ntest: images/test-dev\nnames:\n"+"".join(f"  {i}: {name}\n" for i,name in enumerate(NAMES)),encoding="utf-8")
    if not matched:raise RuntimeError("Prepared training-label fingerprint differs; verification.json retained; do not start training")
    print("VisDrone conversion matches the original training-label fingerprint")


def prepare_songdo(a):
    # --raw-root is the extracted official v2 test/ folder.
    annotation=a.raw_root/"coco_annotations.json"
    if not annotation.is_file():annotation=a.raw_root/"annotations.json"
    data=json.loads(annotation.read_text(encoding="utf-8"))
    if len(data["images"])!=1084 or len(data["annotations"])!=55528:raise RuntimeError("Use Songdo Vision v2 official test, not v1 or a training subset")
    target=a.output/"annotations_coco/test.json";target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(annotation,target)
    for item in data["images"]:place(a.raw_root/"images"/Path(item["file_name"]).name,a.output/"images/test"/Path(item["file_name"]).name,a.image_mode)
    write(a.output/"verification.json",{"dataset":"Songdo Vision v2 official test","images":1084,"annotations":55528,"source":"https://doi.org/10.5281/zenodo.21728430","original_coco_bytes_preserved":True,"image_bytes_preserved":True,"source_license":"CC BY 4.0"})


def prepare_dior(a):
    ids=a.split_file.read_text(encoding="utf-8-sig").split()
    if len(ids)!=len(set(ids)) or len(ids)!=11738:raise RuntimeError("Use the complete official 11738-image test split")
    gt={"info":{"description":"DIOR official test, vehicle-only continuous HBB"},"licenses":[],"categories":[{"id":1,"name":"vehicle"}],"images":[],"annotations":[]}
    positive=0
    for stem in ids:
        xml=a.xml_root/(stem+".xml");tree=ET.fromstring(xml.read_bytes())
        width,height=int(tree.findtext("size/width")),int(tree.findtext("size/height"))
        if (width,height)!=(800,800):raise RuntimeError("Unexpected DIOR image dimensions")
        gt["images"].append({"id":int(stem),"file_name":"images/test/"+stem+".jpg","width":width,"height":height})
        lines=[]
        for index,obj in enumerate(tree.findall("object")):
            if obj.findtext("name").strip()!="vehicle":continue
            box=obj.find("bndbox");source=[float(box.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
            x1,y1,x2,y2=[v-1 for v in source];w,h=x2-x1,y2-y1
            if w<0 or h<0:raise RuntimeError("Reversed DIOR box")
            difficult=int(obj.findtext("difficult","0"));ignored=bool(difficult or w==0 or h==0)
            gt["annotations"].append({"id":len(gt["annotations"])+1,"image_id":int(stem),"category_id":1,"bbox":[x1,y1,w,h],"area":w*h,"iscrowd":int(ignored),"ignore":ignored,"source_xml_xyxy":source,"source_object_index":index,"source_difficult":difficult})
            if not ignored:
                positive+=1;lines.append(f"0 {(x1+w/2)/width:.12f} {(y1+h/2)/height:.12f} {w/width:.12f} {h/height:.12f}")
        target=a.output/"labels_positive_only/test"/(stem+".txt");target.parent.mkdir(parents=True,exist_ok=True)
        # Preserve the original Windows-produced DIOR bytes on every host OS.
        target.write_bytes(("\r\n".join(lines)+("\r\n" if lines else "")).encode("ascii"))
        place(a.images_root/(stem+".jpg"),a.output/"images/test"/(stem+".jpg"),a.image_mode)
    if len(gt["annotations"])!=26640 or positive!=26639:raise RuntimeError("DIOR vehicle/ignored annotation counts differ")
    write(a.output/"annotations_coco/test_vehicle.json",gt,newline="\r\n")
    write(a.output/"verification.json",{"dataset":"DIOR official test vehicle-only view","images":11738,"positive_boxes":positive,"ignored_entries":1,"coordinate_rule":"XML corners minus1; continuous width/height; no +1 extent","all_negative_images_retained":True,"not_full_20_class_benchmark":True})


def prepare_uavdt(a):
    # Keep the original author toolkit locally, including its test and ignore definitions.
    target=a.output/"original_toolkit";shutil.copytree(a.toolkit,target)
    sys.path.insert(0,str(ROOT/"implementations/transfer/uavdt_seeds23"))
    import uavdt_data
    sequences,lengths,_=uavdt_data.definition(a.output)
    records=[]
    for sequence,length in zip(sequences,lengths):
        for frame in range(1,length+1):
            name=f"{sequence}_img{frame:06d}.jpg"
            choices=[a.images_root/sequence/f"img{frame:06d}.jpg",a.images_root/sequence/f"{frame:06d}.jpg",a.images_root/"images/test"/name,a.images_root/name]
            source=next((p for p in choices if p.is_file()),None)
            if source is None:raise FileNotFoundError(f"Missing UAVDT frame {sequence}/{frame}")
            place(source,a.output/"images/test"/name,a.image_mode)
            records.append({"file_name":"images/test/"+name,"width":1024,"height":540,"sequence":sequence,"frame":frame})
    write(a.output/"test_images.json",records)
    result=uavdt_data.prepare_gt(a.output,a.output/"annotations")
    write(a.output/"verification.json",{"dataset":"UAVDT official DET test","images":len(records),"sequences":20,"GT":result,"original_frame_bytes_preserved":True,"native_class_agnostic_AP70":"tools/score_transfer.py --target uavdt"})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="target",required=True)
    for name in ["visdrone","songdo","dior","uavdt"]:
        s=sub.add_parser(name);s.add_argument("--output",type=Path,required=True);s.add_argument("--image-mode",choices=["copy","hardlink"],default="copy")
        if name in {"visdrone","songdo"}:s.add_argument("--raw-root",type=Path,required=True)
        if name in {"dior","uavdt"}:s.add_argument("--images-root",type=Path,required=True)
        if name=="dior":s.add_argument("--xml-root",type=Path,required=True);s.add_argument("--split-file",type=Path,required=True)
        if name=="uavdt":s.add_argument("--toolkit",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists; use a new directory")
    a.output.mkdir(parents=True)
    globals()["prepare_"+a.target](a)


if __name__=="__main__":main()
