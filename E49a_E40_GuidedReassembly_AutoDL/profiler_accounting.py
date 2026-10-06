"""Profiler-version-aware accounting; never changes the trained operator.

THOP versions differ in GroupNorm / nearest Upsample coverage. Compare whole
models with isolated old/new modules under the SAME installed hook policy,
and separately check Conv2d arithmetic from observed layer input/output shapes.
"""
import copy
import torch
from torch import nn


def reassembly_profile_delta(new_module, old_module, imgsz=640, stride=32):
    import thop
    from guided_reassembly import added_conv_flops

    if imgsz % stride or stride % 8:
        raise ValueError('Profile sizes must preserve the actual P3/P2 grid ratio')
    p=next(new_module.parameters())
    coarse=torch.zeros(1,128,stride//8,stride//8,device=p.device,dtype=p.dtype)
    fine=torch.zeros(1,128,stride//4,stride//4,device=p.device,dtype=p.dtype)
    scale=(imgsz/stride)**2
    factor=2*scale/1e9
    module=copy.deepcopy(new_module).eval()
    module.capture_stats=False
    module.enabled=True
    convolutions={}
    hooks=[]

    def observe(name):
        def hook(layer,inputs,output):
            x=inputs[0]
            b,cout,ho,wo=output.shape
            multiply_accumulates=b*cout*ho*wo*(x.shape[1]//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1]
            convolutions[name]=dict(input_shape=list(x.shape),output_shape=list(output.shape),
                                    kernel=list(layer.kernel_size),groups=layer.groups,
                                    GFLOPs640=multiply_accumulates*factor)
        return hook

    for name,layer in module.named_modules():
        if isinstance(layer,nn.Conv2d):
            hooks.append(layer.register_forward_hook(observe(name)))
    try:
        with torch.no_grad(): module((coarse,fine))
    finally:
        for hook in hooks: hook.remove()

    independent_conv=sum(item['GFLOPs640'] for item in convolutions.values())
    analytic_conv=added_conv_flops(imgsz)/1e9
    if len(convolutions)!=4 or abs(independent_conv-analytic_conv)>1e-10:
        raise RuntimeError('Independent Conv2d shape/arithmetic budget drift')

    # Profiling only disposable copies avoids persistent THOP buffers/hooks.
    new_ops,_,tree=thop.profile(module,inputs=((coarse,fine),),verbose=False,ret_layer_info=True)
    old_ops,_=thop.profile(copy.deepcopy(old_module).eval(),inputs=(coarse,),verbose=False)
    leaf_profiles={}

    def visit(nodes,prefix=''):
        for name,(ops,params,children) in nodes.items():
            fullname=prefix+name
            if children:
                visit(children,fullname+'.')
            else:
                leaf_profiles[fullname]=dict(GFLOPs640=float(ops)*factor,parameters=int(params))
    visit(tree)
    new_gflops=float(new_ops)*factor
    old_gflops=float(old_ops)*factor
    if abs(sum(item['GFLOPs640'] for item in leaf_profiles.values())-new_gflops)>1e-10:
        raise RuntimeError('Profiler module/leaf reconciliation failed')
    return dict(thop_version=getattr(thop,'__version__','unavailable'),profile_stride=stride,
                independent_conv_GFLOPs640=independent_conv,
                new_operator_profiler_GFLOPs640=new_gflops,
                old_nn_upsample_profiler_GFLOPs640=old_gflops,
                expected_whole_model_delta_GFLOPs640=new_gflops-old_gflops,
                profiler_nonconv_GFLOPs640=new_gflops-independent_conv,
                observed_convolutions=convolutions,profiler_leaves=leaf_profiles,
                boundary='Installed THOP hook policy; functional reassembly/softmax/copies may remain uncounted. Not runtime latency.')
