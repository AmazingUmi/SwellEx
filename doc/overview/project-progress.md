# SwellEx 项目进展

核对日期：2026-10-16。以下依据当前源码及本地保存的 `report.json`、
`request.json` 和交付记录。
每项 PASS 只属于对应 run 的冻结源码与配置，不自动覆盖后续修改。
2026-10-16 按保留最新完整记录的要求，旧 data scale 试跑与 v1 训练产物已清理；
下述旧 run 为历史验证记录。当前保留 `scm-capacity-budget-study-v2`、其 21 个
子任务、最终图表及必要溯源资料；其他 pack 产物未纳入此次清理。

## 当前能力

| 范围 | 已实现 |
|---|---|
| 数据 | Python 生成 RBD/ELM/SCM HDF5；两类划分、五类选频、物理轴与来源检查 |
| base pack | 三路独立特征；12 个神经模型与 3 个 GRNN |
| joint pack | 七种特征组合；每种 fusion 条件下 28 个神经模型与 7 个 GRNN |
| data scale pack | RBD/ELM/SCM 各含自动规模、四档固定规模和原 CNN 对照；共 18 个模型 |
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
| data scale 短窗口全模型 | 64 项检查 PASS；18 模型训练/预测、RBD 自动模型恢复检查、43 个交付文件校验 PASS | `20261015T071653Z-scale-full-all-all-87430e64` |
| data scale 收紧 adapt 后 quick | 36 项 quick、8 项数值检查 PASS；含两处拒绝 adapt 的回归检查 | `20261015T072436Z-scale-full-all-all-7956e433` |
| 仅筛选方法 `--method SCM` | 5 模型 PASS；17 个交付文件校验 PASS | `20261013T184329Z-base-full-SCM-all-fedc9c0e` |
| 仅筛选网络 `--network GRNN` | joint 7 方法 PASS；21 个交付文件校验 PASS | `20261013T184332Z-joint-full-all-GRNN-block_rms-191e6dd2` |
| 先前 joint `block_rms` 全模型计划 | 35 模型 PASS；77 个交付文件校验 PASS | `20261013T081928Z-66477916` |
| 先前 joint `raw_concat` smoke | 14 模型 PASS；35 个交付文件校验 PASS | `20261013T082119Z-a5ee5c26` |

新入口验证使用 0–59 s 的短窗口；SCM 神经模型各 1 epoch，GRNN 使用 3 折、
3 个 sigma 候选。上述 joint GRNN 任务在本地监视被 Ctrl-C 中断（退出码 130）后
继续完成，随后以原 ID 查询、取回并通过交付校验。网络异常重试另由故障注入测试覆盖。
`train_before` / `train_after` 的全模型选取与 ID 通过合同测试核对；本次没有重跑
100-epoch 全量实验。首次 joint quick 的旧字段测试夹具已修正，上表列出重跑结果。

data scale 验证使用 0–59 s、每模型 1 epoch、`train_fraction=0.5`：
有效优化训练样本各 18 个，验证 12 个、测试 12 个；自动规则选择 small，
实际可训练参数 51,153。自动模型另恢复 1 epoch；汇总误差、预测散点及
距离时间 PDF 已渲染检查。该短窗口验证不构成规模规则校准或精度比较。
后续 quick 快照仅收紧新 pack 的 adapt 入口并补充回归检查，未改变默认
deep/shallow 路径及模型结构。设计与配置见 [data scale pack](../packs/data-scale-pack.md)。

先前 joint quick run `20261013T083556Z-b46dc461` 另保存 35 模型重绘结果：
`reports/visualization_35_redraw/` 和对应 figure 目录的 `redraw-35-models/`。
交付记录为 77 文件校验 PASS；重绘不构成重新训练或预测。

## SCM 数据规模与网络规模正式比较

`scm-capacity-study-v1` 于 2026-10-15 完成：12 个条件、72 个模型各 100 epochs，
全部 PASS；24 个汇总组均含 3 个种子，7,200 条 epoch，无缺失。
训练样本数为 337/675/1,350/2,700，每档另有 900 个验证窗、固定 900 个测试窗。
采用 S5 `full_interleaved`、SCM deep+shallow；试跑和短窗口烟测不进入正式统计。

四档堆叠网络按平均最佳验证 RMSE 选择 small/small/medium/large；
对应测试 RMSE 为 1158.66/780.18/563.88/372.62 m（均值，完整 SD 见报告）。
1,350 样本时 small 与 medium 的验证表现接近；全量原 CNN baseline 的测试
RMSE 为 346.90 ± 3.14 m，优于新堆叠家族。当前自动规则选择
small/medium/medium/medium，仍为 **uncalibrated**。

证据、完整分析和诊断位于 `output/data/scm-capacity-study-v1/`；最终六组 PDF/PNG
位于 `output/figure/scm-capacity-study-v1/v3/`。19 个交付文件校验 PASS；排版重绘
保留原始证据，五张 CSV 表与原版逐字节相同。协议见
[数据规模实验](../packs/data-scale-study.md)，命令见 [pack 文档](../packs/data-scale-pack.md)。
固定 100 epochs 对应不同更新次数，距离覆盖也随 N 改变；该结果描述同录音插值，
不证明最优缩放定律或未知环境泛化，RBD/ELM 正式规模矩阵尚未运行。

## 阈值校准与等更新次数对照

`scm-capacity-budget-study-v2` 于 2026-10-16 17:44（新加坡时间）完成：
21 个子任务、105 次实际训练全部 PASS，无缺失。前 60 次结果保留，
整组调度迁移到 haiqin 后沿用原 ID 与冻结训练源码完成剩余 45 次。
统一使用 `run.sh data_scale_pack status/fetch --run` 管理整组。

新种子 2029–2031 对照 100 次数据遍历与固定 4200 次优化更新；
全量数据两组共用一次执行，120 条逻辑观察与 48 条策略视图不增加独立重复数。
四档可扩展网络按验证均值选择，两种预算均为 small/small/small/large。
固定更新次数下，对应测试 RMSE 为 1056.14/688.28/536.57/358.46 m；
因此小数据偏好小模型的现象在统一更新预算后仍存在。

校准候选仍选 small/small/medium/large，相对原规则的跨 N 等权配对改善为
22.31 ± 14.94 m（遍历预算）和 15.03 ± 7.98 m（更新预算）。
± 为三个种子的样本 SD，先在每个种子内平均 N，不是置信区间。
1350 样本处仍偏向更小模型，默认规则仍为 uncalibrated。
全量 large 与原 baseline 测试均值为 358.46 和 360.93 m，接近，不能宣称明显胜出。

273 个子任务交付清单文件及 27 个整组导出文件校验 PASS；
最终七组 PDF/PNG 位于 `output/figure/scm-capacity-budget-study-v2/v2/`，
报告、表格、诊断与来源校验位于 `output/data/scm-capacity-budget-study-v2/`。
绘图排版修订保留所有数值表的原始字节，未重训。协议和完整解释见
[阈值与预算实验](../packs/data-scale-budget-study.md#completed-results-2026-10-16)。

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
| 数据规模与网络规模关系 | **PASS（SCM 固定协议及等更新次数）**；候选阈值完成新种子比较，默认仍 uncalibrated；独立环境、RBD/ELM 正式矩阵 **NOT_RUN** |
| 双向完整神经实验 | 上述正式证据仅覆盖 before→after；完整反向对比尚缺 |
| 独立时间段 / 航迹泛化 | 尚未建立；交错划分和随机 CV 不能替代该验证 |
| 定位精度验收、论文精确复现 | 未建立验收标准或逐项匹配证据 |
| ELM 带宽搜索 | 历史部分配置命中 sigma 上界，仍需扩展搜索验证 |

`range_overlap` 用两侧真值距离选共同区间；生成前整段去均值包含 held-out
时间；base/joint 的 adapt 选频拟合外层有效训练窗，未按神经验证划分或
训练比例缩减。data scale 暂时拒绝 adapt；整段去均值仍保留。
解释精度时须保留这些边界。
特征可代数相关，联合输入的收益也不等于新增独立物理信息。
