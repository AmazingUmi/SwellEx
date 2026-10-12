# SwellEx 项目进展

更新日期：2026-10-12

本地修改源码和配置；数据生成、数值验证、训练、预测及绘图在 haiqin 的
冻结 uv 环境中运行。命令见 [远端工作流](../workflows/remote-validation.md)。
用户明确指定的本地 CPU GRNN 使用独立环境和同一 workflow 入口。

## 已完成

| 范围 | 当前能力 |
|---|---|
| 数据生成 | Python 读取数值 MAT、距离轨迹和 CTD，生成 RBD/ELM/SCM HDF5；支持 full_interleaved/range_overlap 划分及五类选频 |
| 特征 | RBD 保存实际 Green 特征并支持最强波束/多径聚合；ELM 使用严格上三角 LS 比值；SCM 使用含对角线的归一化协方差 |
| 神经网络 | 每个方法支持 complex CNN、real CNN、ResNet18、ResNet50；独立 loader/registry，共享 CLI、训练、续训和预测 |
| GRNN | RBD/ELM/SCM 三路独立 build/predict、reference 保存、sigma 搜索和共用核回归，无 epochs 或优化器 |
| Shell 工作流 | `./scripts/run.sh METHOD ARCH` 支持 15 个组合；支持配置覆盖、dry-run、后台运行、status/fetch 和内容校验缓存；`local-grnn --method METHOD` 支持显式本地 CPU 环境 |
| 结果交付 | 日志/报告/history/CSV 回收到 `output/data/<run_id>/`，PDF 图表回收到 `output/figure/<run_id>/`；训练曲线归入对应模型的 `training/` |

数据字段、窗口和选频规则以 [HDF5 契约](../datasets/hdf5-datasets-and-splits.md)
为准。MATLAB 源码保持静态，仅供迁移参考。

## Scripts 整理

- 方法包统一为 `rbd/elm/scm/grnn`，入口统一为 `main.py`；公开 Shell 名称保持不变。
- 共享神经 CLI、训练/预测及方法绑定从重复实现收敛到 `common/`；方法相关布局留在各包。
- validation 分为命令入口、quick、datasets、fixtures 和 reporting；职责见
  [脚本组织](../development/scripts-organization.md)。
- 输入和 artifact 核对频率、阵元/深度、pair 身份；缺合同字段的旧 checkpoint/reference 拒绝使用。
- 续训验证 split indices/mode/seed，恢复 optimizer 状态后启动新的 cosine 学习率阶段；
  无改善时保留真实存在的历史 best。数据文件仍需保持不变。
- 数据目录包含配置指纹，已有目录拒绝覆盖；单个 HDF5 完成并关闭后原子发布。
- 远端命令与嵌套测试进程传播超时/终止并清理进程组；fetch 使用 checksum，
  PASS 模型运行必须有可验证的可视化交付清单。

## 数据划分配置与共同距离区间验证

### 三路 GRNN 计算框架

GRNN 共用核计算，分别绑定 RBD 阵元、ELM 严格上三角 pair 和 SCM 含对角线
pair。独立 CLI 必须传 `--method`；每路独立选 sigma、保存 reference、预测并
交付图表。Shell 支持三路 `METHOD GRNN`，experiment 为 15 个组合，smoke
为 3 个 complex CNN 与 3 个 GRNN。跨方法/物理轴不匹配的 reference 拒绝。
已知旧源码计划的 13/4 模型结果保留严格的历史读取路径。

haiqin CUDA 的最终 before smoke `20261012T083824Z-e1af61ba` 完成 28 阶段：
27 PASS、1 次已核验数据缓存复用 SKIP；34 项 quick 检查与 8 项数值检查通过。
神经模型只训练 2 epochs，用于工程流程验证；GRNN 使用完整 5 折/29 候选，
800 reference、700 test。19 个交付文件校验 PASS，其中 17 个 PDF。

新 GRNN 的完整真实 S5 计算如下，均已 PASS 并取回；同一航迹、单一 seed，
不构成跨航迹验证。ELM 的全程和 before 方向 sigma 命中候选上界，尚需单独
扩展带宽搜索验证，不能把流程 PASS 当作精度调优完成。

| 方法/划分 | sigma | 参考/测试 | MAE / RMSE (m) | run |
|---|---:|---:|---:|---|
| RBD before→after | 0.7 | 800/700 | 130.68 / 150.64 | `20261012T083824Z-e1af61ba` |
| ELM before→after | 2.0 | 800/700 | 161.33 / 209.59 | `20261012T083824Z-e1af61ba` |
| RBD after→before | 0.7 | 700/800 | 122.21 / 146.97 | `20261012T083855Z-51121106` |
| ELM after→before | 1.9 | 700/800 | 164.62 / 222.46 | `20261012T083859Z-7c9a23e4` |
| RBD full_interleaved | 0.09 | 3600/900 | 142.24 / 730.31 | `20261012T084019Z-426c4ff3` |
| ELM full_interleaved | 2.0 | 3600/900 | 726.20 / 1675.12 | `20261012T084022Z-3675aeb4` |

SCM before 的 177.65/195.77 m 与原基线一致。完整流程、参数与产物见
[独立 GRNN 工作流](../workflows/standalone-grnn.md)。

图表修订 quick `20261012T085138Z-8600451a` 的 34 项检查与 8 项数值检查
全部 PASS，包含 88 项 workflow、26 项 GRNN、33 项可视化测试。修复了
近乎平坦 sigma 曲线的顶部 marker 裁切；用该冻结源码重绘 31 个 PDF 到
新 QA 目录，37 个交付文件 size/SHA256 核对 PASS，原计算报告与 CSV 未修改。

完整 before→after 实验 `20261012T092941Z-a7b8f27d` 已完成并取回：
12 个神经模型各 100 epochs，加三路完整 GRNN，55 阶段中 54 PASS、
1 次已核验数据缓存复用 SKIP。三路使用相同的 800 行训练 HDF5 和
700 行测试身份；神经训练内部保留 25% 验证集，GRNN 使用全部 800 行 reference。
37 个交付文件校验 PASS，其中 35 个 PDF；三路 GRNN 指标与上表 before 结果一致。
该结果仍为同一航迹和单一 seed，after→before 的全神经模型正式实验尚未执行。

提交前补齐了生成 metadata 的完整原始输入哈希，以及本地 GRNN 包含方法
loader 的完整 Python 快照；生成和复用统一拒绝缺失或改变的来源记录。
最终 quick `20261012T105220Z-00c30d56` 的 34 项检查、8 项数值检查全部
PASS，包含 92 项 workflow、26 项 GRNN、33 项可视化和 5 项生成身份测试。
新增小样本回归验证真实生成→复用→合并、原始文件内容篡改和 loader 变化拒绝。

### 已有划分验证

三份独立配置为 `haiqin_full_interleaved.toml`（默认）、
`haiqin_range_overlap_train_before.toml`、`haiqin_range_overlap_train_after.toml`。
`range_overlap` 在最近距离时刻前后各取最多 800 s 的候选窗，排除中心附近
15 s 后，只保留两侧共同距离区间内的样本；两侧数量可以不同。
Python 只接受 `full_interleaved` 和 `range_overlap`，旧名称明确拒绝。

完整 S5、1 s 步长的实际生成结果如下，RBD/ELM/SCM 三路一致：

| 配置 | 训练 HDF5 行数 | 测试 HDF5 行数 | 远端验证 run |
|---|---:|---:|---|
| full_interleaved | 3600 | 900 | `20261009T051147Z-45bdce55` |
| range_overlap train_before | 800 | 700 | `20261009T051117Z-07435bec` |
| range_overlap train_after | 700 | 800 | `20261009T051140Z-40c7d400` |

候选窗的共同距离区间为约 0.9071–2.1399 km。before smoke 的 22 个阶段全部
PASS，包括 32 项 quick 检查、8 项数值检查、三路生成及合同校验、三个
complex CNN 各 1 epoch、RBD 续训、SCM-GRNN 和图表交付；after 和
full_interleaved 各完成 SCM-GRNN 所选流程，10 个阶段全部 PASS。
三次均使用 RTX A5000/CUDA，交付文件校验分别为 15、9、9 个文件 PASS。
配置中正式神经训练仍为 100 epochs；本次验证不构成定位精度验收。

### GRNN 本地完整计算与保留

此前按用户要求在 macOS CPU 执行，未连接 haiqin。计算编排已移入
`workflow/local_grnn.py`，经 `./scripts/run.sh local-grnn` 调用，复用两份方向配置，
只生成 SCM；训练侧进行
5 折、29 个 sigma 候选搜索，再保存全部训练 reference 并预测全部对侧样本。
搜索按折/batch 复用距离；已用独立的原算法对照验证候选指标和选择结果。
17 项 GRNN 关键测试全部 PASS，包括双向完整流程、同侧数据复用保持原产物、
反向配置与混合距离轨迹拒绝；4 个最终 PDF 已检查页面、字体嵌入和渲染版面。

| 方向 | 参考/测试样本 | sigma | 测试 MAE / RMSE | 本地 run |
|---|---:|---:|---:|---|
| before → after | 800 / 700 | 0.5 | 177.65 / 195.77 m | `20261011T173300Z-a3a141c9` |
| after → before | 700 / 800 | 0.5 | 156.58 / 178.90 m | `20261011T173329Z-ad6d4173` |

SCM HDF5 保留在 `outputs/Datasets/`；各 run 的 `output/data/` 保留
reference、搜索 JSON、预测 CSV、配置、源码快照、依赖版本和文件哈希，
`output/figure/` 保留预测与搜索 PDF。复用数据须核对完整划分和同一原始
距离轨迹；预测显式绑定当次 reference。入口和环境见
[GRNN 本地计算](../workflows/standalone-grnn.md#local-cpu-range-overlap-computation)。
这些指标描述当前共同距离区间内的跨航段结果；随机 CV、两侧真值距离
选区间及整段去均值的科学边界仍适用。
17 项数值测试和上述计算结果产生于入口迁移前；迁移后仅验证标准库命令
路由，未再次执行数据生成、数值测试、build/predict 或绘图。

## A5000 常规配置与验证

默认配置现覆盖完整 S5 4500 s 录音，使用 100 epochs、神经 batch 64、CNN
基础通道 16、4 个 CPU 线程；GRNN 使用 5 折、29 个 sigma 候选和 batch 1024。
A5000 性能测量为 `20261008T163807Z-4786a709`；测量范围和选择理由见
[默认值及 A5000 配置依据](../workflows/command-defaults.md#a5000-sizing)。

配置流程验证为 `20261008T164326Z-c1d6bd00`，实际使用 RTX A5000/CUDA。
49 个阶段中 48 项 PASS、1 项因复用已核验缓存而 SKIP，无失败或未执行阶段。
三路各 3600 个训练窗口、900 个测试窗口、22 个频点；12 个神经模型均使用
batch 64 完成 2 epochs、预测和绘图，另完成完整 SCM-GRNN 搜索及 RBD 续训。
32 项 quick 和 8 项数值检查通过，31 个 PDF 已取回，清单的 33 个文件校验 PASS。
该运行验证配置和全模型流程；正式 100 epochs 结果另见下一节。

## 既有正式结果与合并报告

既有 `20261008T170344Z-8574007c` 已完成完整 S5 的 12 个神经模型
各 100 epochs 和 SCM-GRNN，run 状态为 PASS。当前从本地保存的 history
逐项确认 1–100 epochs，并从 13 份 CSV 确认相同的 900 个测试身份。
全程 GRNN 的 MAE/RMSE 为 4.73/31.46 m；神经模型中 SCM ResNet18 最低，
为 216.75/317.04 m。神经训练实际使用 2700 行并保留 900 行验证，GRNN
使用全部 3600 行 reference；结果只描述当前配置和单一 seed。

上述合并未启动新计算，仅合并既有 CSV、报告和诊断摘要。全程、双向共同距离
跨侧任务分别呈现，重复 GRNN CSV 保留来源但不当作独立重复实验；此前
远端跨侧 GRNN 摘要与本地结果的 MAE 差异小于 0.00002 m。跨侧神经
结果在该报告中只有 before 方向的 1 epoch 摘要；后续完整 before 实验见
上面的三路 GRNN 计算框架记录，双向 100 epochs 对比仍未完成。
当时的完整数据、来源和解释保存在
`output/data/20261011T174757Z-results-comparison/report.md`；
可复用的标准库入口见
[既有结果合并](../workflows/remote-validation.md#local-grnn-and-existing-result-comparison)。

## 重构短流程验证

重构后的完整实验为 `20261008T145422Z-078313cd`，实际使用 RTX A5000/CUDA。
49 个阶段全部 PASS，无跳过或未执行阶段。

| 检查 | 结果 |
|---|---|
| Python quick | 32 项检查 PASS；最近的 PDF 流程验证包含 70 个工作流、24 个可视化、10 个神经合同、8 个 GRNN、4 个生成身份测试，均无跳过 |
| 数据数学 | 8 项独立数值检查 PASS |
| 生成与数据合同 | 6 个 HDF5 文件通过校验；三路各 48 个训练窗口、12 个测试窗口、22 个频点，标签/时间/划分/频率对齐 |
| 全模型流程 | 12 个神经网络各训练 1 epoch，另完成 SCM-GRNN 构建、3-fold sigma 搜索与预测 |
| 断点续训 | RBD 续训通过；关键单测另验证恢复零 LR 后确实更新权重，以及无改善时沿用历史 best |
| 图表交付 | 已从原始 CSV/history 重绘为 31 个 PDF：5 个汇总、13 个模型预测、12 个训练曲线、1 个 sigma 搜索；清单中 33 个文件的 size/SHA256 核对 PASS |
| 取回复原 | 本地文件被改为同大小、同 mtime 的损坏内容后，校验拒绝；重新 fetch 恢复原始 SHA256 |
| 源码边界 | 已验证快照与当前 scripts 文件一致；9 个神经模型模块改名前后 AST 一致；MATLAB 参考未修改 |

PDF 流程修订另经 `20261008T152825Z-3ee5007c` 的 CUDA 短流程验证：
21 个阶段 PASS；数据生成阶段因核验后复用缓存而 SKIP，无未执行阶段。
指定完整实验的预测 CSV、训练 history 和原始执行记录保留不变，旧 PNG 已清除。

训练曲线位于 `models/<METHOD>/<model>/training/history.pdf`；模型预测图位于
`models/<METHOD>/<model>/test_outputs/prediction.pdf`。全部模型图均纳入上述清单。
其他日志的取回使用 rsync checksum。
报告、命令和逐项证据保存在该 run 的本地交付目录；图表含义见
[可视化指南](../workflows/result-visualization.md)。

## 科学验证边界

重构短流程使用 S5 0–59 s 起始窗；A5000 配置验证覆盖完整 S5，但每个神经
模型只训练 2 epochs。两者均验证工程流程，PASS 不代表正式定位精度、模型
优劣或论文精确复现。
正式 100 epochs 结果仍来自同一录音和一次 seed；交错划分下全程 GRNN
数米级平均误差不等同独立航段的泛化能力。

正式研究仍需独立时间段/航迹划分、扩大样本范围、重复随机种子、定义验收
指标，并单独设计不同 snapshot 几何的跨方法对齐。adapt 只拟合有效训练窗；
生成前整段录音去均值仍包含 held-out 时间，评估时需考虑这一预处理边界。
