# V2：显存门禁单位修复

原始服务器日志：两项 CUDA check 成功，smoke 在进入真实 batch 前失败。
`check_headroom()` 返回 GiB，旧 `bounded_smoke()` 却计算 `cap_GiB * 2**30 / total_GiB`。
这是代码错误，不是 OOM 或模型失效；此前本机工程测试没有覆盖这个真实 smoke wrapper。

修复只涉及显存比例计算与回归测试，不改变模型、初始化、loss、batch、训练配置及固定15/13GiB预算。
正确比例为 `cap_GiB / total_GiB`，并显式拒绝超出 `(0,1]` 的预算。

新增测试直接调用完整 `bounded_smoke`，检查两个预算对应的实际 PyTorch API 参数，
并验证低剩余显存、预算超出设备显存都拒绝调用 allocator。

修复后11项本机回归测试全部通过；另在本机RTX4060上实际进入同一个wrapper，
使用0.25GiB测试预算成功设置PyTorch CUDA allocator并分配CUDA张量。
此项只验证API/单位和wrapper，不等同服务器15/13GiB、batch16并行smoke。

覆盖安装新版代码后需要重新 check / smoke / verify-gates；代码签名变更，旧凭据不可复用。
真实服务器 batch16 双开与 batch32 验证未通过前，不能启动正式训练。
