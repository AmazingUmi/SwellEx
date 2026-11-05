# RBD/SCM 特征与独立 GRNN 回归

`scripts/network_training/grnn/` 读取 RBD、SCM 复数特征并用 Gaussian-kernel GRNN 直接回归距离（km），通过 `build` 保存参考库、`predict` 加权预测，不使用 epochs 或优化器。命令详见 [独立 GRNN CLI 说明](grnn-cli.md)，HDF5 特征生成契约见 [数据集与划分文档](../data_processing/hdf5-datasets-and-splits.md)。

## 两路特征展平与 SCM 归一化性质

GRNN 保留实部和虚部通道，按 `[real_imag, element|pair, frequency]` 展平：RBD 实值维度为 $2EF$，SCM（含对角线上三角 `i<=j`）为 $E(E+1)F$。

生成阶段（`data_processing` 中 `feature_math.py:scm_feature`）对每个频点 $f$ 和 $N_s$ 个快拍的阵列声压向量 $x_s(f)\in\mathbb C^M$ 先做逐快拍范数归一化再求外积平均：

```math
\tilde x_s(f)=\frac{x_s(f)}{\max(\|x_s(f)\|_2,\epsilon_n)},
\qquad
C_f=\frac1{N_s}\sum_{s=1}^{N_s}\tilde x_s(f)\tilde x_s(f)^H,
\qquad \epsilon_n=10^{-12}.
```

当各快拍范数不低于 $\epsilon_n$ 时，$C_f$ 为 Hermitian 半正定且 $\operatorname{trace}(C_f)=1$；低功率或零快拍会使迹小于 1。SCM 按 1-based 列序保存上三角条目（含对角线实虚部）。

## 预处理与核回归稳定计算

除生成阶段归一化外，支持两个默认关闭的可选变换（统计量随 reference 保存并在预测时复用）：
- `per_sample_channel_norm`：展平前在单样本内分别对实、虚通道做 z-score（标准差下限 `1e-6`）。
- `standardize_input`：在参考集（或 CV 折内训练集）上对展平后的每个坐标做总体标准差 z-score（下限 `1e-12`）。

`GrnnRegressor` 保存参考特征 $z_i\in\mathbb R^D$ 与 km 标签 $r_i$（无目标归一化）。对查询 $z$ 采用最小距离平移避免高维或小 $\sigma$ 下指数全下溢：

```math
d_i^2=\|z-z_i\|_2^2,\qquad
\tilde w_i=\exp\left(-\frac{d_i^2-\min_j d_j^2}{2\sigma^2}\right),\qquad
\hat r(z)=\frac{\sum_i \tilde w_i r_i}{\max\left(\sum_i \tilde w_i,\epsilon\right)}.
```

模型拒绝使 `2*sigma^2` 超出 `float32` 有限正规数的极端值，并要求 $\epsilon\in(0,1]$。查询按 `batch_size` 分批与全量参考集计算距离，单批显存随 `batch_size * N_reference` 增长。

## Sigma 选择与评估边界

选择优先级为：(1) 显式 `--val-data` 最小化验证集 MAPE；(2) `cv_folds > 1` 最小化各折平均 MAPE（折内独立拟合坐标标准化，距离矩阵在候选 $\sigma$ 间复用）；(3) 固定 `spread`（默认 `0.01`）。选定后用全部参考样本重建最终 reference（验证集不并入）。

默认 CV 按固定种子随机分折，未按时间或航段隔离相邻窗口，因此随机 CV 分数仅用于带宽选择，不能单独证明跨时段或跨航迹泛化。Reference 产物同时绑定 pack、方法名、预处理与物理轴 `input_signature`。
