# 归档结果索引（archive/output）

本目录保存清理时从 `output/` 移出的已运行结果，按实验包分类，目录内容保持原样（移动而非改写，SHA256/交付校验文件未改动）。活动结果见 [`output/data/README.md`](../../output/data/README.md)。

- 分类目录：`data/<类别>/<run_id 或 study_id>/` 与 `figure/<类别>/<run_id 或 study_id>/`，同一运行的数据与图件使用相同 ID。
- 少数报告文件（`data_scale_pack/scm-capacity-budget-study-v2/{report.md,completion.json,analysis-v2/delivery-verification.json}`、`MoE_data_scale/rbd-scm-moe-expert-study-v1/study-report.json`、`MoE_learn/20261017T140800Z-…-ee7bbbd5/distance-holdout-summary/visual-qa.json`、`legacy_cleanup_index/…/保留结果索引.md`）内嵌了迁移前的绝对路径 `…/SwellEx/output/{data,figure}/…`，现应读作 `…/SwellEx/archive/output/{data,figure}/<类别>/…`。为保持完整性这些文件未被改写。
- 对应源码在 [`archive/source/`](../source/)；运行时的源码快照与远端模型清单见 `legacy_cleanup_index/cleanup-20261017-key-results/{remote/sources,packs/<pack>/index.json}`（含 base_pack 的远端模型路径与 SHA256）。
- 误差单位 m；偏差 = 预测 − 真值。所有结论均为单录音（S5）结果，除标明的矩阵外为单种子，不构成跨航段/跨海域泛化证据。

## 分类一览

| 类别 | 内容 | 运行 / study | 关键结果（摘要） |
|---|---|---|---|
| `joint_feature_pack/` | 7 种单路/联合特征 × `block_rms`，4 神经 + GRNN（每个 35 模型） | `20261013T092059Z-joint-full-all-all-block_rms-fa1115ac`（`full_interleaved`）、`20261014T083254Z-joint-before-all-all-block_rms-b57d7ba5`（`before→after`） | `before→after` 下 complex CNN 测试 RMSE：RBD 98.4、RBD_SCM 116.3、SCM 121.0、RBD_ELM_SCM 127.1、ELM_SCM 132.1、RBD_ELM 134.5、ELM 142.2；联合输入未优于单路 RBD。`full` 下 SCM / RBD_SCM GRNN 的 MAE 5.2 / 5.7 m 为邻窗近邻内插，神经网络 MAE 为 218–725 m。 |
| `data_scale_pack/` | SCM 容量校准与等更新预算矩阵（21 子任务、105 次训练，训练窗 337/675/1350/2700，种子 2029–2031） | `scm-capacity-budget-study-v2` | 验证选择 small/small/small/large；固定 4200 次更新下测试 RMSE 1056 / 688 / 537 / 358 m；全量 large（358.46 m）与原 baseline（360.93 m）接近。 |
| `MoE_data_scale/` | 残差 MoE 数据量 × 专家数矩阵（`RBD_SCM`，48 单元）及 SCM E4 单模型 | `rbd-scm-moe-expert-study-v1`、`20261016T140024Z-moe-before-SCM-complex_MoE_e4-3c362a7f` | 600 训练窗下 E1/E2/E4/E8 测试 MAE 115.9 / 116.6 / 117.7 / 128.2 m，增加专家数未一致降低误差；同为 600 训练窗时 MoE E1 RMSE 约 170 m，高于联合 complex CNN 的 116 m（训练协议不完全相同，仅作参照）。SCM E4 单模型 MAE/RMSE 104.5 / 145.7 m。 |
| `MoE_learn/` | `RBD_SCM` 分阶段持续学习（8 组运行：`demo`、前后侧交错/按距离、近/中/远完整距离留出） | `20261017T080254Z-…-ecf6b0a3`、`…084146Z-…-78ff6b07`、`…090451Z-…-75703a08`、`…093401Z-…-ab9f102e`、`…134033Z-…-25551a7b`、`…140310Z-…-53ef18e0`、`…140452Z-…-5a6ca0f9`、`…140800Z-…-ee7bbbd5` | 引入少量 after 侧窗后测试 RMSE 由 152.1 降至 22.2 m（`ab9f102e`）；距离留出：中距 161.0、近距 502.7、远距 598.0 m，说明已知距离区间内插与外推差别明显。 |
| `legacy_cleanup_index/` | 2026-10-17 清理时生成的正式结果索引、清理记录、诊断压缩包、源码归档与远端模型清单 | `cleanup-20261017-key-results` | 仅作溯源；其诊断包不作为正式实验结果。 |
| `cleanup_validation/` | 本次整理后在 haiqin 上执行的 `validate --suite quick`：`…6bfb9022` FAIL（`models.*.registry` 仍断言 4 个模型，已修复）、`…9f29ed44` PASS；目录重构（去 `base_pack/` 层、合并 RBD/SCM 神经网络运行时）之后 `…596d8d5b` PASS（`python_quick` 含新增 `grnn.unit_tests`、`data_numerical` 全部通过） | `20261105T040120Z-base-full-all-all-6bfb9022`、`20261105T040345Z-base-full-all-all-9f29ed44`、`20261105T044546Z-base-full-all-all-596d8d5b` | 仅为工程验证，不含精度结果；未运行 smoke/experiment（需 GPU 长时训练）。 |

`output/data` 活动区保留的 `base_pack` 两个运行（`8574007c`、`a7b8f27d`）不在此处。
