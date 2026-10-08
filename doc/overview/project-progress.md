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
- 可视化校验测试窗口身份、重算 CSV 指标并输出距离、残差、误差分布、
  训练曲线及 sigma 搜索图。日志/报告/history/CSV 回收到
  `output/data/<run_id>/`，图表回收到 `output/figure/<run_id>/`。

## 验证证据与边界

最近完整链路记录为 `20261008T112624Z-a6cac089`：21 个阶段 PASS，
生成阶段复用 `20261008T112002Z-494a8fc1` 的产物而标记 SKIP；缓存内容哈希
与数据契约重新校验通过。后者执行了全部 22 个阶段。

- 28 项 Python 快速检查通过，包含 35 个工作流单元测试、17 个可视化测试
  和 12 个模型的 GPU forward/backward；8 项数据处理数值检查通过。
- 6 个 HDF5 文件通过 12 项数据契约检查；三类数据各有 48 个训练窗口、
  12 个测试窗口和 22 个频点，跨方法标签、时间、划分与频率对齐。
- 三套 complex CNN 的 2-epoch 训练/预测、SCM-GRNN 的 3-fold sigma
  选择与预测、RBD 断点续训通过，实际使用 RTX A5000/CUDA。
- 9 张新增结果图导出 PNG/PDF 共 18 个文件，连同原模型图、日志与表格完成
  回收及哈希检查。图表定义见 [可视化指南](../workflows/result-visualization.md)。

这些是 S5 0–59 s 起始窗的功能检查，不能据此宣称正式定位精度、模型优劣
或论文精确复现。正式研究仍需独立时间段/航迹划分、扩大样本范围、重复随机
种子、定义验收指标，并单独设计不同 snapshot 几何的跨方法对齐。
