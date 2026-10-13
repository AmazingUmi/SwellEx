# SwellEx 项目进展

核对日期：2026-10-13。以下依据当前源码及本地保存的 `report.json`、
`request.json` 和交付记录。
每项 PASS 只属于对应 run 的冻结源码与配置，不自动覆盖后续修改。

## 当前能力

| 范围 | 已实现 |
|---|---|
| 数据 | Python 生成 RBD/ELM/SCM HDF5；两类划分、五类选频、物理轴与来源检查 |
| base pack | 三路独立特征；12 个神经模型与 3 个 GRNN |
| joint pack | 七种特征组合；每种 fusion 条件下 28 个神经模型与 7 个 GRNN |
| 工作流 | 统一入口和参数筛选、描述性任务 ID、源码快照、冻结 uv/CUDA、断线存活、status/fetch |
| 交付 | CSV/history/报告和 PDF；可视化交付文件 size/SHA256 校验 |

命令与参数见 [远端工作流](../workflows/remote-validation.md)，输入与融合约束见
[数据合同](../datasets/hdf5-datasets-and-splits.md)和 [pack 文档](../packs/README.md)。

## 关键工程证据

下表的模型运行均使用短窗口、神经模型 1 epoch 和 GRNN 3 折，验证执行与合同。
本地证据位于 `output/data/<run_id>/`，不纳入 Git。

| 范围 | 记录结果 | run |
|---|---|---|
| 新入口 base quick | 35 项 quick、8 项数值检查 PASS；含 87 项 workflow、13 项公共入口/监视测试 | `20261013T184158Z-base-full-all-all-8aebeec7` |
| 新入口 joint quick | 37 项 quick、8 项数值检查 PASS | `20261013T184507Z-joint-full-all-all-block_rms-521f5b71` |
| 仅筛选方法 `--method SCM` | 5 模型 PASS；17 个交付文件校验 PASS | `20261013T184329Z-base-full-SCM-all-fedc9c0e` |
| 仅筛选网络 `--network GRNN` | joint 7 方法 PASS；21 个交付文件校验 PASS | `20261013T184332Z-joint-full-all-GRNN-block_rms-191e6dd2` |
| 先前 joint `block_rms` 全模型计划 | 35 模型 PASS；77 个交付文件校验 PASS | `20261013T081928Z-66477916` |
| 先前 joint `raw_concat` smoke | 14 模型 PASS；35 个交付文件校验 PASS | `20261013T082119Z-a5ee5c26` |

新入口验证使用 0–59 s 的短窗口；SCM 神经模型各 1 epoch，GRNN 使用 3 折、
3 个 sigma 候选。上述 joint GRNN 任务在本地监视被 Ctrl-C 中断（退出码 130）后
继续完成，随后以原 ID 查询、取回并通过交付校验。网络异常重试另由故障注入测试覆盖。
`train_before` / `train_after` 的全模型选取与 ID 通过合同测试核对；本次没有重跑
100-epoch 全量实验。首次 joint quick 的旧字段测试夹具已修正，上表列出重跑结果。

先前 joint quick run `20261013T083556Z-b46dc461` 另保存 35 模型重绘结果：
`reports/visualization_35_redraw/` 和对应 figure 目录的 `redraw-35-models/`。
交付记录为 77 文件校验 PASS；重绘不构成重新训练或预测。

## 保留的原方案正式结果

迁移前 run `20261012T092941Z-a7b8f27d` 完成 S5 的
`range_overlap` before→after：12 个神经模型各 100 epochs，加三路 GRNN。
报告为 PASS，37 个交付文件校验 PASS。三路共用 800 行训练 HDF5、700 行测试；
神经训练保留 25% 作验证，GRNN 使用全部训练行作为 reference。

| GRNN 方法 | 测试 MAE (m) | 测试 RMSE (m) |
|---|---:|---:|
| RBD | 130.68 | 150.64 |
| ELM | 161.33 | 209.59 |
| SCM | 177.65 | 195.77 |

这些数值来自该 run 的报告，仅描述同一航迹、单一 seed 和该划分。
它们不是 joint pack 的对照实验，也不验证迁移后的全部源码。
其他历史指标和测试日志留在各自 run 产物中，不在本页重复维护。

## 未完成与解释边界

| 项目 | 状态 / 边界 |
|---|---|
| 联合特征正式精度比较 | **NOT_RUN**；需固定测试身份，比较两种 fusion 和多个 seed |
| 双向完整神经实验 | 上述正式证据仅覆盖 before→after；完整反向对比尚缺 |
| 独立时间段 / 航迹泛化 | 尚未建立；交错划分和随机 CV 不能替代该验证 |
| 定位精度验收、论文精确复现 | 未建立验收标准或逐项匹配证据 |
| ELM 带宽搜索 | 历史部分配置命中 sigma 上界，仍需扩展搜索验证 |

`range_overlap` 用两侧真值距离选共同区间；生成前整段去均值包含 held-out
时间；adapt 选频仅拟合有效训练窗。解释精度时须保留这些边界。
特征可代数相关，联合输入的收益也不等于新增独立物理信息。
