# E13b：仅 P4→P3 的 SGCD 配对探针

这是 D13 后的删除式单变量实验，不是新结构叠加：

- 从同一个冻结 E10a `best.pt` 精确初始化；
- 与已有 C13 一样，只微调 P2/P3 分类塔；
- 唯一额外可训练内容是 P4→P3 分类语义适配器；
- 不构造 E13a 新增的 P3→P2 分类适配器；
- E10a 原有 P3/P2 空间门控仍保留；
- regression、nearest、TAL2、batch=8、imgsz=1024 均不变；
- check、smoke2、30 轮训练均不自动关机。

## 安装

```bash
conda activate yolo11
python /root/autodl-tmp/E13b_P4P3_SGCD_Probe_AutoDL/install_e13b_p4p3_sgcd.py
cd /root/autodl-tmp/yolo11_project
```

## 门禁检查

```bash
python -u yolo11s_p2_p4p3_sgcd_probe_img1024_seed1.py --check-only
python -u yolo11s_p2_p4p3_sgcd_probe_img1024_seed1.py --smoke2
```

只有 smoke2 显示 `Completed` 后才启动 30 轮探针：

```bash
nohup bash run_e13b_p4p3_sgcd_probe.sh > launcher_e13b_p4p3_sgcd.log 2>&1 &
tail -f launcher_e13b_p4p3_sgcd.log
```

已有 C13 不重跑。结果报告会自动寻找 C13 并计算 E13b−C13。预计 30 轮约 1.5–2 小时，受 vGPU 共享负载影响。
