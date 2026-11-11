# ST-MDRF：一次补齐7项种子实验（仅YOLO11，v2工程检查修复）

本包不包含YOLO26，不改网络设计、不把已有best作为新训练初始化，也不覆盖旧实验。

|要补的配置|种子|任务数|
|---|---|---:|
|Shared：保留空间路由、共享分类回归深度权重|2、3|2|
|Uniform：移除自适应深度路由，均匀融合|2、3|2|
|E58：不含分组增强读出的对照|3|1|
|公开TOOD TaskDecomposition适配（不是完整TOOD）|2、3|2|

合计7次正式训练。E1、完整E60及上述配置已有种子的结果不重训，附带原始评分记录供汇总核对。

## v2修复范围

v1在TD seed2的128×160固定合成样本检查中，40次有限优化只覆盖23/24组，缺少P5回归出口`out_reg.3.`。小合成图不保证原始TAL给每个尺度分配回归正样本，原检查没有记录这一前提。

v2仅在TD的check阶段使用预先固定的640×640合成图和四种目标大小，逐步记录原始TAL的正样本及正权重。仍要求FP32、AMP各至少8次真实有限更新、24/24组全部有梯度且实际改变，并额外要求四个尺度都得到正权重。没有强制分配正样本、没有更改损失或梯度、没有降低门槛。合成检查的batch 1仅用于工程覆盖，正式训练仍为batch 16。

原冻结架构、检测头、初始化、TAL/损失、训练函数、真实smoke、评分和关机守护均保留。新增模块为`td_coverage_check.py`；原`frozen/td/checks.py`逐字节保留，通过临时上下文替换其中的合成检查函数，退出后恢复。其他原有检查照常执行。新包只用ASCII文件名、SHA256文本为LF。

日志没有记录旧失败任务的逐层TAL掩码，因此不能把“P5无正样本”表述为该服务器任务已直接测得的事实；这是结合无梯度位置、原损失实现及固定小图覆盖缺陷得到的首要解释。v2的明确覆盖检查会在服务器重新验证，不跳过失败门禁。

先Ctrl+C退出旧包监控，再启动v2；旧包与失败日志保留。新包会重新完成全部check/smoke，不能沿用旧包签名的门禁。正式训练尚未开始，无需重跑任何已完成的200轮任务。

## 训练设置和执行顺序

- 原YOLO11独立环境：`/root/miniconda3/envs/yolo11/bin/python`。
- 原本地源码：`/root/autodl-tmp/yolo11_src`，Ultralytics 8.3.0，torch 2.5.1+cu121。
- 原VisDrone2019划分：6471 train / 548 val / 1610 test-dev。
- 原COCO `weights/yolo11s.pt`初始化；640、batch 16、workers 8、AMP、SGD、200轮、相同增强和学习率设置；验证集fitness选best。
- 原有E1 seed1/2/3的`best.pt`需仍在各自`runs/e1_yolo11s_p2add_img640_seedN/weights/`；它们仅用于工程保存重载的数值夹具，不作正式训练初始化。缺失会在预检停止。
- 原分组残差FP32保留规则不变。不同实现分别在独立进程内运行，避免同名自定义检测头反序列化混用。

运行顺序：输入/源码/环境/磁盘/关机命令核验 → **全部7项check** → **全部7项单项真实smoke** → 3组并行真实smoke → 全部门禁通过后才允许正式训练。

单项/并行smoke包括真实640/b16反向传播、至少8次实际有限优化更新、保留分支梯度与参数变化、训练模型/优化器/EMA同时驻留、完整548张val与最坏方形val32压力测试。还会执行一次原生Trainer两轮小比例工程训练，覆盖原生AMP检查、验证、保存best/last、重载、绘图、预测和官方评分入口。这些smoke结果不作为论文结果，smoke权重不进入正式训练。

默认四波：

1. Shared seed2 + E58 seed3；
2. Uniform seed2 + TD seed2；
3. Shared seed3 + TD seed3；
4. Uniform seed3。

每波最多两项训练；同卡并行主要节约空闲资源，不保证总时间减半。每波训练全部退出后，再串行进行该波各项的完整官方val和test-dev评分。

**已按用户同意的策略设置：** 只有并行smoke明确发生CUDA OOM或不足0.5GiB的显存余量时，自动把整批正式训练改为串行，batch 16和所有训练设置不变。其他check/smoke错误停止且不关机。正式长训出现OOM也停止，不擅自改变实验或自动重训。smoke不能保证未来200轮绝对不发生硬件、磁盘或软件错误。

## 上传和启动

仅上传本包ZIP及同名`.zip.sha256`到：

`/root/autodl-tmp/yolo11_project/`

须开GPU模式（原32GB显存实例），不是2GB内存的无卡模式；先停止其他训练/评分。本包要求至少8GiB可用数据盘、12GiB有效可用主机内存，启动时GPU空闲至少28GiB。磁盘不足会停止，不删除任何旧文件。

```bash
cd /root/autodl-tmp/yolo11_project
sha256sum -c ST_MDRF_SevenRuns_Seeds23_AutoDL_v2.zip.sha256 &&
mkdir -p experiment_code &&
test ! -e experiment_code/ST_MDRF_SevenRuns_Seeds23_AutoDL_v2 &&
unzip -q ST_MDRF_SevenRuns_Seeds23_AutoDL_v2.zip -d experiment_code
```

确认上一步成功后，只启动一次：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash experiment_code/ST_MDRF_SevenRuns_Seeds23_AutoDL_v2/start_all_auto_shutdown.sh \
  > launcher_stmdrf_supp7_v2.log 2>&1 < /dev/null &
printf '控制器PID：%s\n' "$!"
```

如果你主动希望从头串行，在脚本后加`--serial`。此时仍先完成全部7项单项check/smoke，不做并行门禁。

启动脚本已经启用`--enable-shutdown`，不需要另启动关机脚本。它不会在预检失败或训练失败时关机。

## 监控

不刷批次，只在状态变化时打印：

```bash
/root/miniconda3/envs/yolo11/bin/python -I -u \
  /root/autodl-tmp/yolo11_project/experiment_code/ST_MDRF_SevenRuns_Seeds23_AutoDL_v2/watch_progress.py --follow
```

看控制器（预检通过数、各阶段日志位置、串并行决策）：

```bash
tail -n 60 -F /root/autodl-tmp/yolo11_project/launcher_stmdrf_supp7_v2.log
```

单独看某项完整日志（例：Shared seed2正式训练；尚未轮到时该文件还不存在）：

```bash
supp7_session=$(/root/miniconda3/envs/yolo11/bin/python -I -c \
'import json; print(json.load(open("/root/autodl-tmp/yolo11_project/logs/latest_stmdrf_supp7.json"))["session"])')
tail -n 60 -F "$supp7_session/shared_s2_train.log"
```

其他正式日志：`e58_s3_train.log`、`uniform_s2_train.log`、`td_s2_train.log`、`shared_s3_train.log`、`td_s3_train.log`、`uniform_s3_train.log`。

check/smoke也各自有日志，位于同一session。退出监控按Ctrl+C，不会停止后台训练。

## 自动关机的明确条件

关机守护从启动时独立挂起并写`shutdown_status.json`。只有以下条件全部满足才执行：

1. 7项正式训练均完整200轮；每份`results.csv`严格包含1—200轮。
2. 7对`best.pt`、`last.pt`均存在，ZIP CRC与SHA256核验通过，训练进程已实际重载检查；14份完整官方val/test-dev评分成功。
3. 每份预测覆盖全部548/1610张图，预测SHA、权重SHA、源码/数据/评估协议绑定一致；总表、逐种子表、配对差值与最终保存清单写盘并再次核验。
4. 控制器实际退出、全局锁释放，GPU及其他Python任务均空闲，平台关机helper与启动时相同。
5. 120秒倒计时结束前没有取消文件，结束时再次核验和同步磁盘。

取消自动关机但不取消训练：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_ST_MDRF_SUPP7
```

若此文件是上次有意保留的，本次也不会关机；本包不会自动删除它。需要恢复自动关机时自行确认后移除此单个文件。

失败、被中断、评分不全、磁盘不足、文件损坏或未知任务占用时均不执行关机。正常失败会保存错误/日志/已有结果，留机排查。平台命令返回0仅表示命令已受理，最终是否停止计费仍以AutoDL控制台为准。不要同时启动其他旧的自动关机watcher。

## 最终评分与汇总

最终评分沿用原实验的VisDrone Matlab1.0.4 ignore-region规则移植实现：十类、原忽略区处理、FP32推理、640、conf 0.001、IoU 0.7、max_det 500、batch 4、不用TTA。它与训练日志中的Ultralytics原生AP不同，不混用。

`summary.json`、`REPORT.md`、`per_seed.csv`、`paired_differences.csv`自动给出三种子均值、样本标准差及同种子差值，保留全部正负结果。

Shared/Uniform原seed1只有已存的官方val，本包会为新增seed2/3各做完整val和test-dev；因此三种子路由消融主表是**val三种子**。不伪造它们缺少的seed1 test-dev，不把两种子的test均值标为三种子。E58和TD则结合已存旧测试结果形成三种子val/test汇总。若另需路由seed1测试，只需之后评估已有权重，无需新增训练。

分组读出、Uniform、Shared是不同且重叠的结构对照，差值不能当作可独立相加的贡献。TD适配没有替换为完整TOOD，参数计算开销仍应一起报告。

## 下载保存

只下载本批，不必重新下载整个老项目：

- `logs/stmdrf_supp7_<时间_PID>/` 整个session（含关机记录）；
- `comparison_reports/ST_MDRF_SUPP7_<时间_PID>/` 整个报告（含所有门禁、训练证据、val/test评分、预测和代码快照）；
- `runs/`下本批7个目录（名称见`common7.py`中的JOBS；监控和JSON也记录具体路径）；
- `logs/latest_stmdrf_supp7.json`索引，以及本训练包和校验文件。

报告中的预测JSON可能较大，是重算评分的原始证据，不是重复训练。不要在下载备份核验前删除。正式运行目录已存在时本包拒绝覆盖或自动重训；失败后不要盲目反复启动，先看日志，保留已完成权重和评分再按原因恢复。

## 本地验证边界

包内`test_safety.py`可做CPU回归测试，模拟命令不会调用真正关机。原有7项CPU初始输出/损失/梯度/保存重载证据对应未修改的冻结架构。v2另核验TD seed2/3的合成覆盖与完整工程检查，并记录设备、Torch版本和逐层正样本；详见`LOCAL_VALIDATION.json`。本地Torch版本不等于服务器版本，因此这些不是AutoDL服务器check/smoke通过凭证。真正门禁在你启动后现场运行，通过才开正式训练。

已知原生FP16验证的BCE求和可溢出为`val/cls_loss=inf`，旧门禁已有精度审计。本包不改变原验证器或fitness：仅把这一已知诊断项存为字符串，AP、其他损失、原始输出必须有限，并额外在AMP下审计原损失。任何NaN或非有限AP仍失败。

本包的独立settings关闭可选外部跟踪平台登录/上传以及自动安装，保留本地结果与可用的TensorBoard；不修改旧项目settings，不改变模型或训练日程。
