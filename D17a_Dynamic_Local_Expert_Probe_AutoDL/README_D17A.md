# D17a：动态尺度局部专家30轮探针

## 确定的缺陷

D15a切片使小目标召回明显提高，但原E1只在整图分布上训练，直接处理放大的局部裁剪会产生背景伪目标。D16a/b只使用框、类别和置信度等元数据，不包含视觉特征，因此无法可靠区分背景纹理与真实目标。

D17a不再训练分数校准器，而是复制E1作为独立局部专家，在D15a同一动态路由选出的训练集裁剪上微调。冻结前10层backbone，只训练neck/head 30轮；关闭mosaic并缩小几何增强，保持训练裁剪与推理裁剪一致。

最终结构：

```text
整图 -> 冻结E1 -> 全图框 + D15a动态路由
                         |
选中1.5块/图 -> 局部专家 -> 局部框 -> NMS融合
```

验证标签只用于局部验证和最终评估，不进入训练。

## 执行

```bash
cd /root/autodl-tmp
unzip -o D17a_Dynamic_Local_Expert_Probe_AutoDL.zip

conda activate yolo11
python /root/autodl-tmp/D17a_Dynamic_Local_Expert_Probe_AutoDL/install_d17a_local_expert.py

cd /root/autodl-tmp/yolo11_project
python -u yolo11s_d17a_dynamic_local_expert_probe.py --check-only
python -u yolo11s_d17a_dynamic_local_expert_probe.py --smoke2
```

smoke通过后执行30轮正式探针：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash run_d17a_local_expert_probe.sh > launcher_d17a_local_expert_probe.log 2>&1 &
tail -f launcher_d17a_local_expert_probe.log
```

第一次运行会在 `generated_d17a_local_expert/formal` 生成约1.5块/训练图的裁剪数据集，随后训练独立局部专家。不会修改E1、D15a、D16或Ultralytics，不会自动关机。

通过门槛：相对D15a，mAP50-95至少增加0.2个百分点、AP-small至少增加0.3个百分点、AP-medium下降不超过0.5个百分点。通过后才延长局部专家训练；失败则保留D15a并停止双视图路线。
