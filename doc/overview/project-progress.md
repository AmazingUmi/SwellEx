# SwellEx 项目进展

更新日期：2026-10-08

本地修改源码和配置；数据生成、数值验证、训练、预测及绘图在 haiqin 的
冻结 uv 环境中运行。命令见 [远端工作流](../workflows/remote-validation.md)。

## 已完成

| 范围 | 当前能力 |
|---|---|
| 数据生成 | Python 读取数值 MAT、距离轨迹和 CTD，生成 RBD/ELM/SCM HDF5；支持 full_interleaved/range_overlap 划分及五类选频 |
| 特征 | RBD 保存实际 Green 特征并支持最强波束/多径聚合；ELM 使用严格上三角 LS 比值；SCM 使用含对角线的归一化协方差 |
| 神经网络 | 每个方法支持 complex CNN、real CNN、ResNet18、ResNet50；独立 loader/registry，共享 CLI、训练、续训和预测 |
| SCM-GRNN | 独立 build/predict、reference 保存、sigma 搜索和核回归，无 epochs 或优化器 |
| Shell 工作流 | `./scripts/run.sh METHOD ARCH` 支持 13 个组合；支持配置覆盖、dry-run、后台运行、status/fetch 和内容校验缓存 |
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
该运行验证配置和全模型流程；默认的 100 epochs 正式训练尚未执行。

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

正式研究仍需独立时间段/航迹划分、扩大样本范围、重复随机种子、定义验收
指标，并单独设计不同 snapshot 几何的跨方法对齐。adapt 只拟合有效训练窗；
生成前整段录音去均值仍包含 held-out 时间，评估时需考虑这一预处理边界。
