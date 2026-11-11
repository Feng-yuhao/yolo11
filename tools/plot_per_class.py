"""Plot the three paired AP differences, mean, and sample SD used in Figure 5."""
import argparse
import csv
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,default=ROOT/"outputs/figure5");a=p.parse_args()
    with (ROOT/"results/figure5_per_class_paired_differences.csv").open(encoding="utf-8-sig",newline="") as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        values=[float(row[f"seed{s}_delta_pp"]) for s in [1,2,3]]
        assert abs(statistics.mean(values)-float(row["mean_delta_pp"]))<1e-10
        assert abs(statistics.stdev(values)-float(row["sample_sd_pp"]))<1e-10
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(6.4,4.4));color="#8b4056";green="#607b69"
    for i,row in enumerate(rows):
        axis.errorbar(float(row["mean_delta_pp"]),i,xerr=float(row["sample_sd_pp"]),fmt="D",color=color,capsize=3,markersize=4)
        for seed,marker,offset in [(1,"o",-.12),(2,"s",0),(3,"^",.12)]:
            axis.scatter(float(row[f"seed{seed}_delta_pp"]),i+offset,marker=marker,facecolors="none",edgecolors=green,s=23,label=f"Seed {seed}" if i==0 else None)
    axis.axvline(0,color="#999999",ls="--",lw=.8);axis.set_yticks(range(len(rows)),[r["class"] for r in rows]);axis.invert_yaxis()
    axis.set_xlabel("ΔAP (percentage points)");axis.set_title("Test-dev AP differences relative to the P2 baseline",fontsize=10)
    axis.legend(loc="lower center",bbox_to_anchor=(.5,-.25),ncol=3,frameon=False)
    axis.spines[["top","right"]].set_visible(False);axis.grid(axis="x",alpha=.15);fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output.with_suffix(".svg"));fig.savefig(a.output.with_suffix(".png"),dpi=1200)
    print("Figure 5 data verified; error bars are sample SD, not confidence intervals.")


if __name__=="__main__":main()
