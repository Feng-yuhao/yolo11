# Public comparison: YOLO11-P2 + TOOD-TD adaptation, seed1

固定公开对照，不是新E60改进。原E1和E60代码、YOLO26环境不变。
只训练这一项，从原始COCO yolo11s.pt初始化，不从E60/E1 best微调。

## 方法边界与出处

作者实现：[fcjian/TOOD](https://github.com/fcjian/TOOD)，ICCV2021 TOOD。
固定提交：1bda27c6f290c5b828c566eb95ea50fed75a53cb。
author_source保留原tood_head.py与Apache2.0许可证；不依赖MMDetection/MMCV安装。
TD数学有独立显式卷积前向/反向oracle，并与提取出的原作者TD类逐项核验。

保留：六层串行3×3/GN32/ReLU共享特征编码；图像平均池化生成sigmoid层权重；
分类/回归各自的dense 1×1重组与GN/ReLU；编码器和两TD在四尺度共享。
不是softmax逐位置深度路由，不包含E60分组残差。

适配：P2/P3/P4/P5输入128/128/256/512经各自1×1/GN32/ReLU映射到64；
六层共享编码后，TD输出经各自零初始化1×1残差投影加回原特征，再进入原Detect双塔。
64而非原论文256的宽度在运行前固定，为小模型计算预算适配；不是搜索出的最优TOOD宽度。
输入/输出适配和零初始化残差壳并非作者完整原检测头的一部分，须在论文注明。
不能称“复现完整TOOD/T-head”、不能把作者完整检测头增益当成本适配版增益。
不移植TOOD预测对齐/可变形框采样/TAL损失；原YOLO11已有的TAL/DFL/loss/NMS均保持。

公平比较：E1 seed1、TD-adapt seed1、E60 seed1；不用单seed公开对照减自己的三seed均值。
原对照训练、官方评分、checkpoint/prediction SHA和test图像清单随包，只读核验复用，不重新训练或推理。
全部正负结果报告，接近的单seed差距不能写成稳定/显著超越。

## 协议与流程

独立环境：/root/miniconda3/envs/yolo11，Ultralytics8.3.0 / torch2.5.1+cu121。
原始YOLO11s预训练，img640 / batch16 / val32 / workers8 / seed1 / SGD / cosine200e。
与E1/E60原增强、warmup、累积、loss与图像级等价TAL执行一致。
单开28GiB allocator，不降batch、不自动resume/覆盖已存在正式runs。
随包真实yolov8n.pt只用于原AMP辅助检查，正式模型始终为YOLO11s-P2。

check → 真实640/b16至少8次有效AMP更新、全部24参数组梯度及变化 →
EMA-half b32方形及完整548验证 → val/test各8图标准Predictor导出 →
verify-gates → 正式200轮 → 官方val548 → 官方test-dev1610 → 汇总保存 → 独立安全关机。
check、smoke不是性能筛选；本地小样本验证不冒充服务器真实门禁。
原生YOLO mAP和官方AP分开；固定val-selected best，不在test上选epoch/阈值/种子。
官方评分为冻结D27 VisDrone2019 Matlab1.0.4 ignore-region语义移植，全部10类。
预测固定FP32/640/b4/conf.001/NMS IoU.7/max500；实际前8图有独立逆letterbox坐标检查。

## 安装与运行

上传ZIP到 /root/autodl-tmp/yolo11_project，先核SHA，再解压到experiment_code。
运行前确认YOLO26及其他训练/评分已结束；不得关闭正在工作的其他任务。

    cd /root/autodl-tmp/yolo11_project
    sha256sum -c TOOD_TD_E1_Seed1_AutoDL.zip.sha256 &&
    mkdir -p experiment_code logs &&
    test ! -e experiment_code/TOOD_TD_E1_Seed1_AutoDL &&
    unzip -q TOOD_TD_E1_Seed1_AutoDL.zip -d experiment_code

一次后台启动，明确启用自动关机：

    cd /root/autodl-tmp/yolo11_project
    nohup bash experiment_code/TOOD_TD_E1_Seed1_AutoDL/start_all_auto_shutdown.sh \
      > logs/launcher_tood_td_seed1.log 2>&1 < /dev/null &
    printf '控制器PID：%s\n' "$!"

只读、不刷屏监控（Ctrl+C不影响后台训练）：

    /root/miniconda3/envs/yolo11/bin/python -u \
      /root/autodl-tmp/yolo11_project/experiment_code/TOOD_TD_E1_Seed1_AutoDL/watch_progress.py --follow

查看详细阶段日志：

    tail -n 40 -F /root/autodl-tmp/yolo11_project/logs/launcher_tood_td_seed1.log

要分开做门禁可用run.sh check / smoke / verify-gates；随后run.sh run --auto-shutdown。
两种启动方式只选一种，不重复正式训练。
评分中断可用run.sh eval --weights /实际runs/tood_td_e1_p2345_w64_d6_img640_seed1_200e/weights/best.pt；
只恢复评分、不重训，必须核验原200轮记录和固定best，默认不自动关机。

## 关机与保存

专用start_all_auto_shutdown.sh显式传all --auto-shutdown，避免遗漏。
成功须两个官方评分与总报告完整、200轮CSV及best/last可校验，才允许成功终态。
失败保留已有日志/权重/失败状态，缺失评分如实记录；不会编造结果或自动重训。
独立helper等父控制器退出、锁释放、保存已有文件哈希、无其他Python/GPU任务，再倒计时60秒。
倒计时中新任务会重新等待；不强杀其他任务，最多等空闲30分钟后记录阻塞原因并保持开机。
平台ASCII无shebang helper经 /bin/bash /usr/bin/shutdown 调用，修复此前Exec format error。
脚本返回0不等于已确认实例关机/停止计费，最终以AutoDL控制台为准。
本地所有关机测试使用fake runner，未运行平台关机、未连接服务器或启动远程训练。

取消关机（不取消训练）：

    touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_TOOD_TD

## 需要下载的结果：3个目录

1. runs/tood_td_e1_p2345_w64_d6_img640_seed1_200e
2. comparison_reports/TOOD_TD_ALL_*（本次完整目录，含check/smoke/train/val/test和总REPORT）
3. logs/tood_td_all_*（本次完整目录，含各阶段日志、终态、关机和PRESERVATION清单）

正式报告另存实际运行的实验代码ZIP与Ultralytics源码ZIP；之后不要仅保留软链接或删除唯一代码实体。
参数/计算检查包含动态TD bmm，不能使用忽略bmm的THOP数值当完整计算量。
Conv/Linear+bmm MACs与额外算子开销分别说明；不声称这是包含GN/激活/NMS的完整FLOPs。
