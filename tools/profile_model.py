"""Count executed unfused Conv/Linear MACs and TD torch.bmm separately."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_mdrf.runtime import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",required=True);p.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    p.add_argument("--weights",type=Path);p.add_argument("--output",type=Path);a=p.parse_args()
    wrapper,_=load_model(a.model,a.seed,a.weights,device="cpu")
    import torch
    torch.set_num_threads(2);counts={"Conv_Linear_MACs":0,"TD_bmm_MACs":0};hooks=[]
    def count(module,inputs,output):
        if isinstance(module,torch.nn.Conv2d):counts["Conv_Linear_MACs"]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,torch.nn.Linear):counts["Conv_Linear_MACs"]+=output.numel()*module.in_features
    for module in wrapper.model.modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(module.register_forward_hook(count))
    original=torch.bmm
    def bmm(left,right,*args,**kwargs):
        counts["TD_bmm_MACs"]+=left.shape[0]*left.shape[1]*left.shape[2]*right.shape[2]
        return original(left,right,*args,**kwargs)
    try:
        torch.bmm=bmm
        with torch.inference_mode():wrapper.model(torch.zeros(1,3,640,640))
    finally:
        torch.bmm=original
        for hook in hooks:hook.remove()
    result={"model":a.model,"seed":a.seed,"parameters":sum(p.numel() for p in wrapper.model.parameters()),**counts,
            "Conv_Linear_GMACs":counts["Conv_Linear_MACs"]/1e9,"TD_bmm_GMACs":counts["TD_bmm_MACs"]/1e9,
            "executed_dense_GMACs":sum(counts.values())/1e9,"input":[1,3,640,640],"unfused":True,
            "excluded":"normalization, activation, element-wise fusion, decoding and other uncounted operators; no latency measurement"}
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
