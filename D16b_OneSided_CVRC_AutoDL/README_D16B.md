# D16b：单向局部候选抑制

D16a使TP增加1774，但FP增加12474，说明候选校准器存在少量判别力，双向分数更新却把大量低分误检抬高。D16b不重新训练，复用D16a五个折模型及各折内部确定的alpha，只允许负残差生效：

```text
new_logit = base_logit + alpha * min(0, corrected_logit - base_logit)
```

因此任何局部候选都只能保持或降分，绝不升分。E1、D15a路由、全图预测、D16a权重和Ultralytics全部冻结。

## 执行命令

```bash
cd /root/autodl-tmp
unzip -o D16b_OneSided_CVRC_AutoDL.zip

conda activate yolo11
python /root/autodl-tmp/D16b_OneSided_CVRC_AutoDL/install_d16b_onesided_cvrc.py

cd /root/autodl-tmp/yolo11_project
python -u diagnose_d16b_onesided_cvrc.py --check-only
python -u diagnose_d16b_onesided_cvrc.py --smoke2
```

通过后执行完整548张诊断：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash run_d16b_onesided_cvrc.sh > launcher_d16b_onesided_cvrc.log 2>&1 &
tail -f launcher_d16b_onesided_cvrc.log
```

结束后同步整个 `DIAG_D16B_e1_onesided_cvrc_*` 目录。预计约5至10分钟，不训练、不关机。

通过门槛：相对D15a，mAP至少增加0.2个百分点、AP-small至少增加0.3个百分点、FP减少、TP损失不超过1%。失败后终止CVRC路线并保留D15a。
