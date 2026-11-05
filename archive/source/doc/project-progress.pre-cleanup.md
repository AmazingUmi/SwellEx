# SwellEx 项目进展

核对日期：2026-10-17（依据当前仓库源码及本地 `output/data/` 保留的正式运行报告）。
早期烟测、quick 检查与已被替代的 v1 中间产物已完成归档清理；每项 PASS 仅对应其冻结源码与配置快照。

## 当前能力

| 范围 | 已实现 |
|---|---|
| 数据生成（`data_processing`） | Python 生成 RBD/ELM/SCM 与联合拼接 HDF5；支持 `full_interleaved`、`range_overlap` 与 `MoE_learn` 阶段/距离留出划分 |
| `base_pack` | RBD/ELM/SCM 三路独立特征；每路 4 类神经模型 + 独立 GRNN（共 15 模型） |
| `joint_feature_pack` | 7 种单路/组合特征；支持 `block_rms` 与 `raw_concat`（每种融合下 28 神经 + 7 GRNN） |
| `data_scale_pack` | 三路独立特征下按有效训练窗选择复数 CNN 深度，含 4 档固定规模与原 CNN 对照（共 18 模型）及受控预算矩阵 |
| `MoE_data_scale` | 共享复数前端 + 独立残差专家 + Top-2 路由（RBD/ELM/SCM/RBD_SCM，共 20 模型）及 48 单元数据量 × 专家数矩阵 |
| `MoE_learn` | `RBD_SCM` 分阶段持续学习：固定首阶段前端与缩放、新旧专家分组学习率、候选扩容与历史验证保护、完整距离留出 |
| 调度与交付（`remote_workflow`） | 统一 `run.sh` 入口、描述性任务/study ID、远端冻结 uv/CUDA、断线存活、`status`/`fetch` 与交付文件 SHA256 校验 |

## 保留的正式实验结果

### 1. Base 与 Joint 单航迹基线

本地保留 `base_pack`（`full` 13 模型 `8574007c`、`before` 15 模型 `a7b8f27d`）与 `joint_feature_pack` `block_rms`（`full` `fa1115ac`、`before` `b57d7ba5`，各 35 模型，100 epochs + 5 折 GRNN）。在 S5 `range_overlap` `before→after`（800 训练窗 / 700 测试窗，单一种子 2026）下：

| Pack / 类别 | 代表方法与模型 | 测试 MAE (m) | 测试 RMSE (m) |
|---|---|---:|---:|
| `base_pack` GRNN | RBD / ELM / SCM | 130.68 / 161.33 / 177.65 | 150.64 / 209.59 / 195.77 |
| `base_pack` 最优神经 | RBD `complex_cnn_range` | 72.42 | 98.66 |
| `joint_feature_pack` (`block_rms`) | RBD / `RBD_SCM` / SCM `joint_complex_cnn_range` | 72.28 / 87.07 / 90.52 | 98.43 / 116.31 / 120.99 |
| `joint_feature_pack` (`block_rms`) | `RBD_ELM_SCM` `joint_complex_cnn_range` | 93.53 | 127.13 |

单一种子下联合输入尚未优于单路 RBD 复数 CNN；`raw_concat` 与多随机种子的正式配对比较尚未运行。

### 2. SCM 网络容量校准与等更新预算矩阵（`scm-capacity-budget-study-v2`）

21 个子任务 / 105 次训练全部 PASS（S5 `full_interleaved`，训练窗 337/675/1350/2700，验证/测试各 900 窗，新种子 2029–2031）。对照 **100 次遍历**与**固定 4200 次更新**（N=2700 共用执行）：

- 按验证 RMSE 均值选择固定档位，两种预算均选 `small / small / small / large`；固定更新下对应测试 RMSE 为 `1056.14 / 688.28 / 536.57 / 358.46 m`，说明小样本偏好小模型并非仅由更新步数较少引起。
- 基于前序开发集验证面冻结的校准候选（`reference_samples=900, exponent=0.9`）选 `small / small / medium / large`，相对原规则跨 N 等权平均改善为 `22.31 ± 14.94 m`（遍历）与 `15.03 ± 7.98 m`（等更新，3 种子样本 SD）。
- 全量 N=2700 时 `large`（358.46 ± 5.90 m）与原 `baseline`（360.93 ± 9.29 m）接近；默认规则保持 `uncalibrated`。协议与细表见 [受控矩阵实验文档](remote_workflow/matrix-studies.md#scm-capacity-calibration-and-budget-study)。

### 3. 残差 MoE 数据规模 × 专家数矩阵（`rbd-scm-moe-expert-study-v1`）

48/48 子任务全部 PASS（`RBD_SCM` `train_before` + `block_rms`，训练窗 75/150/300/600 × 专家数 E1/E2/E4/E8 × 种子 2026–2028，每模型 100 epochs，固定 200 验证窗 / 700 测试窗）：

- 按跨种子平均最佳验证 RMSE 在四档样本量下分别选 `E4 / E4 / E8 / E4`。
- 600 训练窗下 E1/E2/E4/E8 测试 MAE 均值为 `115.92 / 116.64 / 117.71 / 128.23 m`（E4 测试 RMSE `165.10 ± 13.74 m`，E1 为 `169.94 ± 1.91 m`）；增加专家数未一致降低回归误差。另有 SCM E4 单模型对照 `3c362a7f`（测试 MAE/RMSE `104.53 / 145.68 m`）。详见 [MoE 模型文档](network_training/moe-data-scale.md) 与 [MoE 矩阵工作流](remote_workflow/matrix-studies.md#rbd-scm-moe-expert-study)。

### 4. 持续学习 MoE 分阶段与完整距离留出（`MoE_learn`）

本地保留 8 组 PASS 运行（含短窗 `demo`、前后侧时间交错 `ab9f102e`、距离分组 `25551a7b` 及近/中/远三折完整距离留出）：

- **同距离覆盖增量**（固定 140 个后侧测试窗，30 epochs/阶段）：时间交错 `ab9f102e` 在第 3 阶段触发 `ADD_EXPERT`（扩至 3 专家），测试 RMSE 由首阶段 `152.14 m` 降至 `22.15 m`；按距离引入后侧样本的 `25551a7b` 依次执行 `ADAPT_EXISTING / REUSE / ADAPT_EXISTING`，最终测试 RMSE `22.62 m`。
- **完整距离留出**（测试距离段在前后侧训练、验证与回放中完全剔除）：中距留出 `[1.32, 1.74) km`（`5a6ca0f9`）测试 RMSE 由 `170.67 m` 降至 `161.01 m`；近距 `[0.90, 1.32) km`（`53ef18e0`）与远距 `[1.74, 2.16] km`（`ee7bbbd5`）最终测试 RMSE 分别为 `502.65 m` 与 `597.97 m`，反映已知距离区间内插明显易与边界外推区分。详见 [持续学习 MoE 文档](network_training/moe-learn.md)。

## 未完成工作与解释边界

| 项目 | 当前状态与边界 |
|---|---|
| 联合特征正式多子对比 | 单种子 `block_rms` 已运行；固定测试集下 `block_rms` vs `raw_concat` 多 seed 对照尚未完成 |
| 容量与专家数自动规则 | 默认规则均为 `uncalibrated`；RBD/ELM 数据规模矩阵及 MoE 等更新步数对照尚未运行 |
| 双向与跨航段/跨海域泛化 | 现有正式结果集中于 S5 单录音（`full_interleaved` 或 `before→after`），不构成未知环境或独立航迹泛化证明 |
| 预处理与划分已知边界 | `MoE_learn` 已采用**逐窗口去均值**；其余 4 个旧 pack 仍对整段录音去均值（含测试时段）。`range_overlap` 依赖两侧真值距离求交集；ELM 部分历史带宽搜索曾触及上界 |
