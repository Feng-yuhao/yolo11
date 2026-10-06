# E47 seed2 本地验证记录（2026-10-03）

本包复制已完成E47 seed1的独立源码。核心读出、原E40块、装配、数学oracle、TAL、模型YAML和官方评分算法不改。
seed2修改范围：全局训练种子、同seed参考/权重校验、独立路径/锁/门禁、自动check-smoke-run串联。

已执行通过：

- `test_seed2.py`四项测试：固定同seed参考、全流程顺序、每个阶段失败后不启动后续步骤、源码语法/manifest。
- 完整CPU check：原E1初始化逐张量核对、激活E40上的零读出恒等、原损失与梯度、26组参数实际更新、任务入口、空GT、保存/新进程重载/fuse/标准Predictor、TAL与官方评估单元。
- seed2协议逐项对照已有E1 seed2；E1/E40 seed2 checkpoint SHA与原报告一致。
- 参数9,928,136；GFLOPs640=42.285824，新增深度读出73,728参数，与seed1相同。
- 原生Trainer四张合成图两轮CPU集成：seed2、optimizer覆盖、EMA、best/last、半精度保存后重载、矩形图坐标和官方评分接口。
- Bash入口`bash -n`语法检查通过；ZIP内shell统一LF。
- 核心读出/装配/原E40/数学oracle/原E1 helper/模型YAML/TAL/官方评估算法/验证smoke与seed1独立包逐字节SHA一致。

CPU check证据：主工程`comparison_reports/E47S2_CHECK_20261003_155839_026262/check.json`。
合成集成证据：`comparison_reports/E47S2_LOCAL-INTEGRATION_20261003_160159_130598/integration.json`。
环境Windows、Python3.9.25、torch2.6.0+cu124、固定Ultralytics8.3.0。

首个合成集成因本地sandbox无法正常读取pandas而在保存阶段停止；允许读取现有依赖后重跑通过，没有改动服务器源码或训练核心。
失败记录保留，不作为通过凭据。

尚未执行：AutoDL torch2.5.1真实CUDA check、batch16/640训练和batch32/548图显存smoke、本轮200轮训练。
本地CPU/合成集成不代表服务器真实CUDA显存smoke通过；ZIP不包含CPU门禁或任何已训练权重。
