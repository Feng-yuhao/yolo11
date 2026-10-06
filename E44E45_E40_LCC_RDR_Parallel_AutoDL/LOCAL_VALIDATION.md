# 交付前本地验证（2026-10-01）

v2新增验证与修复明细见`AUDIT_V2.md`。以下保留首次交付的验证记录，不替代v2记录。

未连接服务器、未训练真实VisDrone数据、未签发CUDA或真实并行smoke凭据。
环境：Windows / Python3.9.25 / torch2.6.0 / 工作区固定Ultralytics8.3.0，执行device=cpu。
服务器仍需torch2.5.1实际CUDA检查；本地结果不保证无OOM或精度提升。

## 已执行通过

1. test_release.py：8项测试通过。包括源码/协议不变、独立公式/初值/公共梯度、激活后的无weight-decay更新、弱权重数值与梯度、pure-half值、AMP跳步、同一次并行smoke凭据与实际子进程失败策略。
2. 两项完整CPU check：原始E1 raw/decoded初值、原损失和公共梯度精确一致；全部26/18参数组有梯度与更新；空GT；新进程reload/fuse；标准Predictor；官方ignore与几何单元测试；TAL等价/真实模型loss梯度更新。
3. 两项标准Trainer CPU合成集成：各4张随机图、160分辨率、batch2、2轮，检查optimizer覆盖、EMA、best/last保存重载、融合、四图矩形预测和原官方评分接口。无性能含义，不是20轮探针。
4. 所有Bash入口通过Git Bash `bash -n`语法检查。实际Linux flock、CUDA并行、真实训练和验证显存尚待服务器执行。

完整CPU检查目录：

- yolo11_project/comparison_reports/E44_CHECK_20261001_225345_349086/
- yolo11_project/comparison_reports/E45_CHECK_20261001_225400_839209/

最终FP32模块调用版本的合成Trainer目录：

- yolo11_project/comparison_reports/E44_LOCAL-INTEGRATION_20261001_225656_005827/
- yolo11_project/comparison_reports/E45_LOCAL-INTEGRATION_20261001_225706_042214/

Trainer测试曾因沙箱无权读取现有pandas库而失败；获准在沙箱外读取既有依赖后通过。
没有为了绕过失败安装服务器依赖或修改原训练保存逻辑。

## 计算量核对

普通FP32新Conv使用注册模块调用；half权重通过stateless FP32视图调用，保持Parameter和保存dtype。
不直接裸F.conv2d绕过hooks，从而避免漏记新Conv的FLOPs。
同一工具、同一未融合架构、640输入：

| 模型 | 参数 | GFLOPs640 | 相对E40新增Conv GFLOPs |
|---|---:|---:|---:|
| E40父体 | 9,854,408 | 39.926528 | — |
| E44 | 9,861,320 | 40.141568 | 0.215040 |
| E45 | 9,877,448 | 40.663808 | 0.737280 |

check强制新增Conv统计与设计手算一致。THOP口径不完整计入所有激活/乘法/数据类型转换，也不能预测墙钟速度。
融合后的参数/FLOPs摘要与未融合统计不同，不跨口径比较。

## 并行行为测试范围

使用两个真实可丢弃Python子进程，模拟E44先失败、E45稍后完成，验证控制器不杀同伴、分别写日志/退出码。
该测试模拟GPU查询和Linux锁导入，不能被描述成实际GPU双开已验证。
release生成ZIP manifest并核对内容哈希；服务器正式run还会核对同一次并行真实smoke凭据。
