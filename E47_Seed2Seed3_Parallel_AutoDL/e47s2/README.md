# E47 seed1：并行包子任务

请以顶层 `../README.md` 的并行入口为准，不直接运行本子目录的 smoke/run。
本包保留独立 E47 的全部模型、初始化和训练配置；只更换资源门禁、报告命名和并行凭据。

## 固定模型

- 保留 E40 的 P2/P3 四深度编码库、逐位置任务路由和原输出投影。
- 每层、每任务新增三个零初始化 48→128 无 bias 的线性读出。
- 固定对称正交、零和的4×3对比基，实现四深度分别使用不同输出矩阵。
- 不添加新 attention、归一化、loss、TAL、Neck、detach 或框细化。
- 先按40001创建两个原 E40 块，再按47001隔离新增读出；不改变全局训练RNG。
- 与独立 E47 相同的9,928,136参数；相对 E40新增73,728参数和约2.359卷积GFLOPs。

## 固定实验

从原始 `yolo11s.pt` 迁移重新训练，seed1 / imgsz640 / batch16 / workers8 / SGD / AMP / 200轮余弦日程。
不从 E40/E46 best 微调，patience0、close_mosaic10等不改变。
官方评分对照已有 E1 seed1 和 E40 seed1，评分器/ignore/坐标处理不改。

并行模式 smoke 和 formal 均固定每进程13GiB PyTorch allocator上限。
顶层控制器要求空闲32GB级GPU，先依次check，再同步真实batch16训练和batch32全部548图验证。
未获得有效的同一轮并行 smoke 凭据，不能启动双开；不自动降低batch。
本地 CPU / 小CUDA验证不能替代AutoDL完整CUDA check和真实显存门禁。

报告独立为 `comparison_reports/E47P_RUN_时间/`，门禁为 `e47p_*_gate.json`。
权重位置仍是 `runs/e47a_p2_depth_readout_img640_seed1_200e/weights/{best,last}.pt`。
已存在该 run 时拒绝覆盖。原独立包与本包均使用 `.e47_execution.lock`，避免同一实验重复启动。

训练后自动官方评分；`completion.json`仅在200轮和评分全部完成后生成。
若评分单独异常，不重训，使用顶层说明的独立 eval 命令。

假设边界：共同输出变换是否是瓶颈尚未证实。零初始化只保证初始输出/原参数一致，不保证之后训练轨迹相同或一定提升。
