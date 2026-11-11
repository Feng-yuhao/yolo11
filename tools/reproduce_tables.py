"""Rebuild paper tables from unrounded per-seed records; no PyTorch/GPU required."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics as stats

ROOT=Path(__file__).resolve().parents[1]
IDS={"YOLO11s":"yolo11s","YOLO11s-P2":"yolo11s-p2","ST-MDRF":"st-mdrf","Without grouped enhancement":"no-group","uniform":"uniform","shared":"shared","TD":"tood-td","YOLO26 baseline":"yolo26s-p2","YOLO26 e60":"yolo26s-p2-st-mdrf"}


def read_csv(name):
    with (ROOT/"results"/name).open(encoding="utf-8-sig",newline="") as handle:return list(csv.DictReader(handle))


def metric_triples(value):
    if isinstance(value,dict):
        keys=["AP50_95_percent","AP50_percent","AP75_percent"]
        if all(k in value for k in keys):yield [float(value[k]) for k in keys]
        for key,item in value.items():
            if "per_class" not in key:yield from metric_triples(item)
    elif isinstance(value,list):
        for item in value:yield from metric_triples(item)


def summarize(rows,keys):
    return [f"{stats.mean(float(r[k]) for r in rows):.2f} ± {stats.stdev(float(r[k]) for r in rows):.2f}" for k in keys]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"outputs/paper_tables");a=p.parse_args()
    rows=read_csv("visdrone_testdev_per_seed.csv");groups=defaultdict(list)
    for row in rows:
        keys=["AP_percent","AP50_percent","AP75_percent"];expected=[float(row[k]) for k in keys]
        raw=json.loads((ROOT/f"results/raw/visdrone/test-dev/{IDS[row['configuration']]}_seed{row['seed']}.json").read_text(encoding="utf-8"))
        if not any(all(abs(x-y)<1e-10 for x,y in zip(expected,actual)) for actual in metric_triples(raw)):raise RuntimeError("CSV and original metrics disagree: "+str(row))
        groups[row["configuration"]].append(row)
    if len(groups)!=9 or any(len(r)!=3 for r in groups.values()):raise RuntimeError("Expected nine three-seed configurations")
    lines=[];summary={}
    for title,names in [("Table 1",["YOLO11s","YOLO11s-P2","ST-MDRF","YOLO26 baseline","YOLO26 e60"]),("Table 2",["Without grouped enhancement","uniform","shared","ST-MDRF"])]:
        lines += [title,"","| Configuration | AP | AP50 | AP75 |","|---|---:|---:|---:|"]
        for name in names:
            values=summarize(groups[name],keys);lines.append("| "+" | ".join([name]+values)+" |")
            summary[name]={key:{"mean":stats.mean(float(r[key]) for r in groups[name]),"sample_sd":stats.stdev(float(r[key]) for r in groups[name])} for key in keys}
        lines.append("")
    complexity={r["model"]:r for r in read_csv("model_complexity.csv")}
    lines += ["Table 3","","| Configuration | Parameters (M) | Executed dense GMACs | AP | AP50 | AP75 |","|---|---:|---:|---:|---:|---:|"]
    for name,key in [("YOLO11s-P2","e1"),("TD","td"),("ST-MDRF","e60")]:
        record=complexity[key];values=summarize(groups[name],keys)
        lines.append("| "+" | ".join([name,f"{int(record['parameters'])/1e6:.3f}",f"{float(record['executed_dense_GMACs']):.2f}"]+values)+" |")
    lines += ["","Table 4","","| Target | P2 AP | + ST-MDRF AP | Paired difference (pp) |","|---|---:|---:|---:|"]
    transfer=read_csv("frozen_transfer_AP_per_seed.csv");targets=defaultdict(lambda:defaultdict(dict))
    for row in transfer:
        dataset={"Songdo Vision v2":"songdo","DIOR vehicle task":"dior","UAVDT":"uavdt"}[row["target_task"]]
        model="yolo11s-p2" if row["configuration"]=="e1" else "st-mdrf"
        filename="metrics.json" if dataset=="dior" else "coco_metrics.json"
        raw=json.loads((ROOT/f"results/raw/transfer/{dataset}/{model}_seed{row['source_training_seed']}/{filename}").read_text(encoding="utf-8"))
        def AP_values(value):
            if isinstance(value,dict):
                if "AP_percent" in value:yield float(value["AP_percent"])
                for key,item in value.items():
                    if "per_class" not in key:yield from AP_values(item)
            elif isinstance(value,list):
                for item in value:yield from AP_values(item)
        if not any(abs(v-float(row["AP_percent"]))<1e-10 for v in AP_values(raw)):raise RuntimeError("Transfer CSV differs from original metric JSON")
        targets[row["target_task"]][row["configuration"]][int(row["source_training_seed"])]=float(row["AP_percent"])
    summary["transfer"]={}
    for target,models in targets.items():
        if set(models)!={"e1","e60"} or any(set(v)!={1,2,3} for v in models.values()):raise RuntimeError("Incomplete paired transfer results")
        delta=[models["e60"][s]-models["e1"][s] for s in [1,2,3]]
        left,right=[f"{stats.mean(models[k].values()):.2f} ± {stats.stdev(models[k].values()):.2f}" for k in ["e1","e60"]]
        lines.append(f"| {target} | {left} | {right} | {stats.mean(delta):+.2f} |")
        summary["transfer"][target]={"paired_differences_pp":delta,"mean_paired_difference_pp":stats.mean(delta)}
    native={}
    for model in ["yolo11s-p2","st-mdrf"]:
        native[model]=[json.loads((ROOT/f"results/raw/transfer/uavdt/{model}_seed{s}/native_ap70.json").read_text(encoding="utf-8"))["AP70_overall_percent"] for s in [1,2,3]]
    summary["UAVDT_native_AP70_mean_difference_pp"]=stats.mean(b-a for a,b in zip(native["yolo11s-p2"],native["st-mdrf"]))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"tables.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (a.output/"statistics.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print("Verified 27 original test-dev records; rebuilt Tables 1-4 and paired transfer statistics.")
    print("UAVDT native AP70 paired change:",summary["UAVDT_native_AP70_mean_difference_pp"],"pp")


if __name__=="__main__":main()
