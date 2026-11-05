# SwellEx 项目进展

核对日期：2026-11-05（依据整理后的仓库源码、`output/data/` 活动结果与 `archive/output/` 归档结果）。每项 PASS 仅对应其冻结源码与配置快照；整理前的完整进展文档备份在 [`archive/source/doc/project-progress.pre-cleanup.md`](../archive/source/doc/project-progress.pre-cleanup.md)。

## 当前活动范围

| 范围 | 已实现 |
|---|---|
| 数据生成（`data_processing`） | Python 生成 RBD、SCM HDF5；支持 `full_interleaved` 与 `range_overlap` 划分；窗口内逐阵元去均值**未**启用（整段录音去均值，含测试时段） |
| `base_pack` | RBD、SCM 两路独立特征；每路 `complex_CNN`、`ResNet18`、`ResNet50` 与独立 GRNN（共 8 模型；smoke 4 模型） |
| 调度与交付（`remote_workflow`） | 统一 `run.sh base_pack` 入口、描述性任务 ID、远端冻结 uv/CUDA、断线存活、`status`/`fetch` 与交付文件 SHA256 校验 |

已退役（见 [archive 索引](../archive/README.md)）：ELM 特征、仅幅度 `real_CNN`、`joint_feature_pack`、`data_scale_pack`、`MoE_data_scale`、`MoE_learn`。

## 保留的活动结果

明细表见 [`output/data/README.md`](../output/data/README.md)。两个运行均为单种子 2026、单一 S5 录音，且在清理前按旧计划生成（报告含 ELM / `real_CNN` 行，此处只引用 RBD/SCM 部分）。

| 划分 | 最优 | 其余要点 |
|---|---|---|
| `range_overlap` before→after（600/200 训练/验证，700 测试窗） | RBD `complex_cnn_range`：MAE 72.6、RMSE 98.7 m | SCM `scm_complex_cnn_range` 90.5 / 121.0 m；RBD/SCM GRNN 130.7 / 177.7 m MAE；所有模型偏差均为负（−67 到 −178 m），去偏后 RBD/SCM GRNN RMSE 约 76 / 83 m |
| `full_interleaved` 4:1（900 测试窗） | SCM GRNN：MAE 4.7、RMSE 31.5 m | `complex_CNN` / `ResNet` 的 RMSE 为 317–854 m；测试窗与训练窗只差 1 s，GRNN 近乎邻窗查表 |

## 对特征与网络的已验证观察（来自保留及归档结果）

- 同侧随机验证集（MAE 约 14–72 m）远低于 before→after 测试误差，不能用于选模；`full_interleaved` 的优劣只反映同录音内插。
- before→after 的误差主要是系统性低估距离（域偏移/外推），而非网络容量；加入少量 after 侧窗（归档的 `MoE_learn`）后测试 RMSE 降至 22 m，说明问题在分布而非容量。
- 联合输入（归档的 `joint_feature_pack`）单种子下未优于单路 RBD；残差 MoE（归档）在同为 600 训练窗时未优于联合 complex CNN。因此活动范围只保留基础的 RBD/SCM 与三类神经网络加 GRNN。
- `complex_CNN` 末端的全局平均池化丢弃频点/阵元位置信息，是后续结构改进的首要候选（保留频率轴的逐频点编码 + 聚合），当前代码未改动。

## 未完成工作与解释边界

| 项目 | 当前状态与边界 |
|---|---|
| 多种子与同源配对比较 | 活动结果均为单种子；归档的 MoE 矩阵显示种子间 RMSE 标准差可达约 14 m，RBD/SCM 间小于该幅度的差异不应解读为显著 |
| 验证集与选模 | 需要按时间块（带间隔）划分的验证集；当前随机验证集会泄漏邻窗 |
| 双向与跨航段/跨海域泛化 | 现有结果集中于 S5 单录音（`full_interleaved` 或 `before→after`），不构成未知环境或独立航迹泛化证明；`after→before` 尚无结果 |
| 预处理已知边界 | 整段录音去均值（含测试时段）；`range_overlap` 依赖两侧真值距离求交集 |
| 清理后验证 | haiqin 上 `validate --suite quick` 在目录重构后 PASS（`python_quick` 与 `data_numerical`，见 `archive/output/data/cleanup_validation/`）；`smoke`/`experiment`（GPU 长时训练与数据生成）尚未在整理后运行 |
