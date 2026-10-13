# RBD/ELM/SCM 特征与独立 GRNN 回归

本页描述仓库实际实现：读取已生成的 RBD、ELM 或 SCM 复数特征，再用
Gaussian-kernel GRNN 回归距离（km）。GRNN 位于
`scripts/base_pack/localization_networks/grnn/`，通过 `build` 保存参考样本、
通过 `predict` 加权预测，不使用 epochs、反向传播或优化器。

运行命令与产物路径见 [独立 GRNN 工作流](../workflows/standalone-grnn.md)；
数据轴、元数据、窗口及划分合同见 [HDF5 数据集](../datasets/hdf5-datasets-and-splits.md)。

## 三路特征与向量化

三路复用各自 HDF5 loader 的布局、物理轴和有效样本合同。GRNN 保留实部、
虚部两个通道，按 `[real_imag, element|pair, frequency]` 展平，不先取幅值。

| 方法 | 特征 | 轴/复数条目 | 实值维度 |
|---|---|---|---|
| RBD | 波束形成得到的等效 Green 估计 | 阵元 `E` | `2*E*F` |
| ELM | 阵元对最小二乘频域比值 | 严格上三角 `i<j`，`E*(E-1)/2` 对 | `E*(E-1)*F` |
| SCM | 归一化空间协方差 | 含对角线 `i<=j`，`E*(E+1)/2` 对 | `E*(E+1)*F` |

各方法分别保存参考特征、训练标签、sigma 和预处理统计，不共享参考库。
ELM/SCM 的 pair 均为 1-based 列序；RBD 不使用 pair 轴。
生成公式见 [HDF5 合同](../datasets/hdf5-datasets-and-splits.md#feature-routes)。

### SCM 构造

`scripts/base_pack/data_processing/feature_math.py:scm_feature` 对每个 snapshot 做 FFT，
保留选定的单边频点。设频点为 $f$，snapshot 数为 $N_s$，阵元数为 $M$，
阵列声压向量为 $x_s(f)\in\mathbb C^M$：

```math
\tilde x_s(f)=\frac{x_s(f)}{\max(\|x_s(f)\|_2,\epsilon_n)},
\qquad
C_f=\frac1{N_s}\sum_{s=1}^{N_s}\tilde x_s(f)\tilde x_s(f)^H,
\qquad \epsilon_n=10^{-12}.
```

这里平均的是逐 snapshot 归一化后的外积，没有减去样本均值。
当所有 snapshot 的范数均不低于下限时，$C_f$ 为 Hermitian 半正定矩阵且
`trace(C_f)=1`；低功率 snapshot 的贡献更小，零向量贡献零矩阵，因此不能
对任意输入都要求 trace 为 1。阵列向量归一化消除整体幅度尺度，同时保留
阵元间的相对幅度与相位关系。

SCM 保留含对角线的上三角 `C_f[i,j]`（`i<=j`），共 $P=M(M+1)/2$ 个复数。
`pair_indices(diagonal=True)` 按列遍历；以 1-based 写出前几项为
`(1,1), (1,2), (2,2), (1,3), (2,3), (3,3)`。这些是协方差条目，
与 ELM 的严格上三角比值特征不同。

SCM 对角线虚部也保留，不压缩独立自由度。预测输入仍需提供有效距离标签，
供当前 CSV 和误差报告使用；加载与物理轴检查见 HDF5 合同。

## 特征尺度

除了生成阶段的阵列向量归一化，还有两种默认关闭的工程选项：

| 选项 | 操作 | 统计范围 |
|---|---|---|
| `per_sample_channel_norm` | 展平前分别对实部、虚部通道做 z-score，标准差下限为 `1e-6` | 单个样本的所有阵元/pair 和频点 |
| `standardize_input` | 对展平后的每个坐标做 z-score，标准差下限为 `eps`（默认 `1e-12`） | 参考样本；CV 时为该折的训练样本 |

坐标标准化使用总体标准差（`unbiased=False`），均值和标准差随 reference
保存，预测沿用这些统计量。关闭时等价于均值零、标准差一。
两种选项均改变欧氏距离和 sigma 的有效尺度；频点数、pair 数或预处理改变后，
应重新选择 sigma。GRNN 标签直接使用 km，没有 CNN/ResNet 的目标归一化步骤。

## GRNN 核回归与数值稳定性

`model.py:GrnnRegressor` 保存处理后的参考特征 $z_i\in\mathbb R^D$
及距离标签 $r_i$。对同样处理后的查询 $z$：

```math
d_i^2=\|z-z_i\|_2^2,
\qquad w_i=\exp\left(-\frac{d_i^2}{2\sigma^2}\right),
\qquad \hat r(z)=\frac{\sum_i w_i r_i}{\sum_i w_i},
\qquad \sigma>0.
```

这对应 GRNN 的模式层（每个参考样本一个核）、两个求和量以及一个距离输出。
实现直接用张量距离与矩阵乘法完成计算，没有显式创建这些“层”。
`fit_reference` 保存样本而不优化权重；模型内部使用 float32，输出为
CPU tensor，形状 `[query_count,1]`，单位 km。

实际核权重先使用 `d_i² - min_j(d_j²)` 替代 `d_i²`。对有限距离，这是每行
权重乘以相同因子，归一化预测不变；至少一个最近邻权重为 1，避免小 sigma、
高维输入导致所有指数权重下溢为零。分母仍以 `eps` 截断。这一处理不修复
NaN/Inf 输入，也不保证任意规模的距离计算都没有浮点误差。
模型拒绝会使 `2*sigma²` 超出 float32 有限正规数范围的极端 sigma；
`eps` 限定为 `(0,1]`，避免分母下限大于最近邻权重而改变归一化核预测。

预测按查询 batch 计算距离，但每个 batch 仍与全部参考样本比较。
保存参考样本的空间规模为 `O(N_reference*D)`；每个查询的距离计算规模
为 `O(N_reference*D)`，距离矩阵空间随 `batch_size*N_reference` 增长。

## Sigma 选择与评估边界

`cli.py` 的选择顺序是：有显式验证集时使用验证集；否则 `cv_folds>1`
时执行交叉验证；否则使用 `spread`（默认 `0.01`）。默认候选为
`0.01,0.02,...,0.10,0.20,0.30,...,2.00`，也可显式覆盖。

验证集搜索以 MAPE 最小为准；CV 以各折 MAPE 的算术平均最小为准。
二者同时记录 MAE（km）、RMSE（km）、MAPE（%）。MAPE 的距离分母为
`max(abs(true_range_km),1e-12)`，接近零距离的样本会显著影响该指标。
完成选择后使用全部参考样本重新构建 reference，验证样本不自动并入。

搜索对每折的查询 batch 只计算一次距离，再复用于所有 sigma 候选；
显式验证集也复用参考标准化及 batch 距离。开启坐标标准化时，每折仍只在
该折训练样本上拟合统计量。候选顺序、核公式和最小 MAPE 选择规则不变。

当前 CV 用固定 seed（默认 2026）随机打乱样本再分折，没有时间、来源或
重叠窗口分组。相邻/重叠窗口可能跨折，因此随机 CV 分数不能直接证明
对独立时间段或独立航迹的泛化。需要这类评估时，应准备具有相应分离关系
的显式验证集，并保留独立测试集；sigma 搜索本身不检查这些关系。

## Reference 产物合同

reference 保存参考特征/标签、sigma、预处理统计和物理输入签名。预测沿用
这些统计，拒绝跨方法、轴不匹配或缺失字段的 reference；详细字段与路径见
[产物说明](../workflows/standalone-grnn.md#artifacts)。
物理轴签名不等于源文件内容哈希，也不证明样本划分独立。

## 实现范围

当前实现将各方法已有复数特征接入同一 Gaussian-kernel GRNN。
本页依据仓库源码整理，不将此前笔记中的论文样本规模、snapshot 设置、
频率组合或实验结果作为当前工程配置或已验证的精确复现结论。

仓库实际使用多频特征、可选的两种标准化、距离偏移稳定计算及 reference
序列化；这些工程选择是否与论文逐项相同，需要另行依据原文和匹配实验核对。
现有 reference 和 sigma 搜索只能说明执行了当前实现，不能单独证明复现了
论文结构细节或定位精度。
