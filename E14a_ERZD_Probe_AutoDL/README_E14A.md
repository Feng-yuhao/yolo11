# E14a：ER-ZD 30轮配对探针

E14a 不再更换 Neck、Detect 或上采样模块。它利用已经证实最强的信号：同一个 E1 模型看局部放大区域时，AP-small 明显高于全图推理。

## E14a究竟改变了什么

```text
全图1024 ──→ E1学生 ──→ 原YOLO检测损失
    │
    └─每批最多选2个含微小目标的576×576区域
                     │ 放大至1024
                     ↓
              冻结E1教师（无梯度）
                     │
          只保留教师更可靠的P2/P3类别响应
                     │
                     └──→ 蒸馏回全图学生对应目标位置
```

- 学生和教师都从同一个冻结 E1 `best.pt` 开始；
- 目标区域约放大 `1024/576=1.78` 倍；
- 只处理等效边长不超过32像素的目标；
- 教师只有在正确类别置信度及类别间隔优于学生时才产生蒸馏梯度；
- 不蒸馏坐标，避免裁剪映射误差破坏定位；
- 推理时没有教师、裁剪或额外模块，网络仍是原始27层 E1；
- 输入1024、batch=8、TAL2、数据增强及原检测损失不变；
- E14a和C14均从E1继续30轮，采用完全相同的SGD微调协议；
- check、smoke2、探针及失败路径全部不自动关机。

这是机制探针，不应在配对控制完成前宣称论文创新或精度提升。

## 1. 安装

把压缩包上传到 `/root/autodl-tmp` 后：

```bash
cd /root/autodl-tmp
unzip E14a_ERZD_Probe_AutoDL.zip
conda activate yolo11
python /root/autodl-tmp/E14a_ERZD_Probe_AutoDL/install_e14a_erzd.py
cd /root/autodl-tmp/yolo11_project
```

安装器只新增 ER-ZD 损失文件、E14/C14脚本，并在 `tasks.py` 增加一个由 YAML 显式启用的损失入口；不会删除或改写已有实验结果。

## 2. 先跑E14a门禁

```bash
python -u yolo11s_p2_erzd_probe_img1024_seed1.py --check-only
python -u yolo11s_p2_erzd_probe_img1024_seed1.py --smoke2
```

只有 smoke2 最后显示 `Completed` 才启动30轮：

```bash
nohup bash run_e14a_erzd_probe.sh > launcher_e14a_erzd.log 2>&1 &
tail -f launcher_e14a_erzd.log
```

预计比普通30轮续训慢约15%至30%，vGPU共享负载会造成波动。自动关机始终禁用。

## 3. 是否跑C14

先看 E14a 相对原 E1 是否至少出现正信号。若 E14a 没有任何提升，停止，不浪费时间跑 C14。若有提升，再执行严格配对控制：

```bash
python -u c14_control_e1_continue30_seed1.py --check-only
python -u c14_control_e1_continue30_seed1.py --smoke2
nohup bash run_c14_e1_continue30.sh > launcher_c14_e1_continue30.log 2>&1 &
tail -f launcher_c14_e1_continue30.log
```

C14不是重跑200轮E1，而是从同一个E1 `best.pt` 以同配置普通续训30轮，用于扣除“多训练30轮”的收益。

正式通过门槛按 `E14a-C14` 计算：

- mAP50-95 至少 `+0.20` 个百分点；
- AP-small 至少 `+0.50 pp`；
- AP-medium 不低于 `-0.50 pp`。

只有通过后，才设计从相同原始预训练权重开始的200轮正式版；30轮探针本身不作为最终论文结果。
