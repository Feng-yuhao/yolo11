# V2：FLOPs门禁漏算归一化计数

原错误：actual=40.2599424，expected=40.2558464，差0.004096G。
原公式只有新增卷积减旧nearest；安装的统计器若计入两层GroupNorm，
它们在640输入下的计数为：

`4 * (32*80*80 + 32*160*160) / 1e9 = 0.004096 GFLOPs`。

在本地用THOP公共custom_ops分别启用/关闭GroupNorm、nearest计数，
四个组合均验证；启用GroupNorm时精确复现旧公式差值。

修复：同一安装统计器下，分别统计被替换的nearest和新reassembly模块，
用两者差值核对全模型；额外按真实输入/输出形状独立核对4个卷积，
并核对逐叶计数与模块总计数一致。全模型容差仍为1e-5G。
check会打印THOP版本、卷积、非卷积、旧nearest及实际/期望总值，
并把详细叶子计数写入check.json。未计数的functional操作不宣称零成本。

原发布包中Python文件仅checks.py、test_package.py改变，新增
profiler_accounting.py。前向、梯度路径、初始化、损失、正式训练、
smoke与评分源码保持原字节内容（按换行归一化SHA核对）。
README/验证记录与manifest随修复更新；不会写入本地CUDA服务器门禁。

上传FixV2 ZIP后用unzip -o覆盖原同名目录，再依次执行check、smoke、
verify-gates。不要手工改manifest、降低门槛或跳过门禁。
