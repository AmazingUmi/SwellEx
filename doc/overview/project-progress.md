# SwellEx 项目进展

更新日期：2026-10-08

项目已统一为 Python 数据处理、定位网络与结果可视化，依赖由 uv 管理。
本地修改源码和配置，数据生成、数值验证、训练、预测及绘图在 haiqin 进行。
使用方式及环境以 [远端工作流](../workflows/remote-validation.md) 为准。

## 已完成

- Python 读取原始数值 MAT、距离轨迹和 CTD，生成 RBD、ELM、SCM HDF5。
  支持窗口组合、periodic/Range_nearby 划分和 full/mel/deep/shallow/adapt
  选频；adapt 只拟合有效训练窗口。MATLAB 源码仅保留作迁移参考。
- RBD 保存实际 Green 特征，支持最强波束和多径聚合。旧分段入口误存 FFT
  的数据需重新生成。ELM 使用严格上三角 LS 比值，SCM 使用含对角线的
  归一化协方差；字段和几何约束见 [HDF5 契约](../datasets/hdf5-datasets-and-splits.md)。
- 每套定位网络有 complex CNN、real CNN、ResNet18、ResNet50，保持独立
  loader/registry；共享训练、续训、预测与 normalized/km 损失。SCM-GRNN
  独立构建 reference 并做核回归。
- 远端工作流固定源码快照、uv 锁环境与输入身份，支持内容校验后的数据缓存、
  后台运行、失败报告与结果回收；已有远端源码仓库保持不变。
- `./scripts/run.sh METHOD ARCH` 支持三套方法的四类网络及独立 SCM-GRNN，
  共 13 个合法组合；按配置生成或复用三路数据，只执行所选模型并绘图回收。
  支持配置/epochs/seed 覆盖、dry-run、后台运行、status 和 fetch；
  GRNN 拒绝 epochs，未知参数和非法组合在 SSH 前拒绝。
- 可视化校验测试窗口身份、重算 CSV 指标并输出距离、残差、误差分布、
  训练曲线及 sigma 搜索图。日志/报告/history/CSV 回收到
  `output/data/<run_id>/`，图表回收到 `output/figure/<run_id>/`。

## 验证证据与边界

完整模型实验 `20261008T132520Z-23c12726` 的 49 个阶段 PASS：
12 个神经网络分别完成 1-epoch 训练/预测，SCM-GRNN 构建/预测、
RBD 断点续训和结果绘图通过，实际使用 RTX A5000/CUDA。
最终源码快速回归为 `20261008T132746Z-434c901a`。

- 29 项 Python 快速检查通过，包含 61 个工作流单元测试、22 个可视化测试、
  13 个公开方法/模型组合与真实 registry 的一致性检查，以及 12 个模型的
  GPU forward/backward；8 项数据处理数值检查通过。
- 6 个 HDF5 文件通过 12 项数据契约检查；三类数据各有 48 个训练窗口、
  12 个测试窗口和 22 个频点，跨方法标签、时间、划分与频率对齐。
- 单模型 ELM 命令 `20261008T132659Z-3a8f2e2e` 使用 epochs=2、seed=0、
  no-cache，通过生成、训练、预测、绘图及前台自动回收；报告只有 ELM 结果。
- 后台 SCM-GRNN 命令 `20261008T132732Z-0d168753` 通过构建、3-fold sigma
  选择、预测、绘图及 status/fetch；复用完整模型实验的数据缓存，内容哈希
  与数据契约重新校验通过。两套单模型报告均无未执行的依赖阶段。
- 完整模型实验的 18 张汇总图导出 PNG/PDF 共 36 个文件；两套单模型各有
  6 张汇总图、12 个 PNG/PDF 文件。原模型图、日志、表格与 provenance
  一并回收并核对哈希。图表定义见 [可视化指南](../workflows/result-visualization.md)。

这些是 S5 0–59 s 起始窗的功能检查，不能据此宣称正式定位精度、模型优劣
或论文精确复现。正式研究仍需独立时间段/航迹划分、扩大样本范围、重复随机
种子、定义验收指标，并单独设计不同 snapshot 几何的跨方法对齐。
