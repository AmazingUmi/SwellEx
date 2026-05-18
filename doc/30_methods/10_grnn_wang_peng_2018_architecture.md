# GRNN 网络架构整理：用于水声声源定位

参考文章：Yun Wang and Hua Peng, *Underwater acoustic source localization using generalized regression neural network*, JASA, 2018.

---

## 1. 方法定位

这篇文章将水声声源定位建模为一个**监督学习回归问题**。

核心映射关系为：

```text
归一化样本协方差矩阵 normalized SCM  →  声源位置 source position
```

在文中的实验中，主要输出变量是声源距离 range，因此可以写成：

```text
C(ω)  →  R
```

其中：

- `C(ω)`：由阵列频域声压构造出的归一化样本协方差矩阵特征；
- `R`：声源距离；
- 网络模型：GRNN，Generalized Regression Neural Network；
- 任务类型：回归；
- 训练方式：非迭代式，不使用反向传播；
- 关键超参数：spread factor，即高斯径向基函数宽度，记为 `σ`。

---

## 2. 输入数据：阵列频域声压

对于一个频率 `ω`，垂直接收阵列 VLA 的频域声压向量为：

```math
x(ω) = [x_1(ω), x_2(ω), ..., x_M(ω)]^T
```

其中：

- `M`：阵元数量；
- 文章实验中使用 SWellEx-96 的 21 元 VLA；
- 因此 `M = 21`。

---

## 3. 数据预处理流程

文章的网络输入不是原始声信号，也不是普通频谱，而是由阵列声压构造出的 normalized SCM。

完整预处理流程如下：

```text
原始多阵元时域信号
        ↓
分帧 / snapshot
        ↓
DFT 得到每个阵元的频域声压 x(ω)
        ↓
对阵列声压向量归一化
        ↓
构造归一化样本协方差矩阵 SCM
        ↓
取对角线和上三角元素的实部、虚部
        ↓
向量化为实值特征
        ↓
输入 GRNN
```

---

## 4. 声压归一化

为了削弱源谱幅度的影响，文章先对每个频率处的阵列声压向量进行归一化：

```math
\tilde{x}(ω) = \frac{x(ω)}{\|x(ω)\|}
```

展开为：

```math
\tilde{x}(ω)
=
\frac{x(ω)}
{\sqrt{x^H(ω)x(ω)}}
=
\frac{x(ω)}
{\sqrt{\sum_{m=1}^{M}|x_m(ω)|^2}}
```

其中：

- `x(ω)`：原始频域阵列声压向量；
- `\tilde{x}(ω)`：归一化后的阵列声压向量；
- `H`：共轭转置；
- `M`：阵元数。

---

## 5. 归一化 SCM 构造

对于每个频率 `ω`，在多个 snapshots 上构造样本协方差矩阵：

```math
C(ω) = \frac{1}{N_s}\sum_{s=1}^{N_s}\tilde{x}_s(ω)\tilde{x}_s^H(ω)
```

其中：

- `N_s`：snapshot 数量；
- `\tilde{x}_s(ω)`：第 `s` 个 snapshot 的归一化频域声压向量；
- `C(ω)`：`M × M` 的复数 Hermitian 矩阵。

### 5.1 Snapshot 设置

文章中出现两种设置：

#### 仿真实验

```text
每个 range sample 的 SCM 由 2 个 1-s snapshots 平均得到
```

#### SWellEx-96 实验

```text
每个 range-frequency sample 的 SCM 由 4 个 1-s snapshots 平均得到
其中 3 个 snapshots 有重叠
```

---

## 6. SCM 向量化方式

由于 `C(ω)` 是共轭对称矩阵，文章只使用：

```text
对角线 + 上三角元素
```

并将其中的复数项拆成实部和虚部，形成实值输入向量。

文章给出的输入维度为：

```math
M_{in} = M(M+1)
```

对于 `M = 21`：

```math
M_{in} = 21 × 22 = 462
```

因此单频窄带情况下：

```text
input_dim = 462
```

---

## 7. 宽带输入构造

如果使用 `Q` 个频率，则对每个频率分别构造 SCM 特征，然后直接拼接：

```math
X = [C^T(ω_1), C^T(ω_2), ..., C^T(ω_Q)]^T
```

输入维度变为：

```math
M_{in} = M(M+1)Q
```

对于 21 阵元：

```math
M_{in} = 462Q
```

示例：

| 频率数量 Q | 输入维度 |
|---:|---:|
| 1 | 462 |
| 2 | 924 |
| 3 | 1386 |
| 4 | 1848 |
| 5 | 2310 |

---

## 8. 输出标签设计

GRNN 的输出是连续值回归标签。

文中主要输出为声源距离：

```math
Y = R
```

因此输出维度为：

```text
K = 1
```

如果扩展为二维或三维定位，可以设置为：

```text
K = 2: [range, depth]
K = 3: [range, depth, bearing]
```

但原文实验主要是单输出 range regression。

---

## 9. GRNN 总体结构

GRNN 包含四层：

```text
Input Layer
    ↓
Pattern Layer
    ↓
Summation Layer
    ↓
Output Layer
```

对应为：

```text
输入层 → 模式层 → 求和层 → 输出层
```

---

## 10. 网络结构总览

| 层 | 名称 | 作用 | 神经元数量 |
|---|---|---|---:|
| 1 | Input layer | 输入 SCM 向量 | `M_in = 462Q` |
| 2 | Pattern layer | 每个训练样本对应一个高斯径向基神经元 | `N_train` |
| 3 | Summation layer | 计算加权和 `S` 与非加权和 `D` | `K + 1` |
| 4 | Output layer | 输出回归结果 | `K` |

对于本文的单输出距离估计：

```text
K = 1
```

因此 summation layer 中有：

```text
1 个 S 节点 + 1 个 D 节点
```

---

## 11. 输入层 Input Layer

输入层只负责传递特征，不进行非线性变换。

输入向量为：

```math
X = [x_1, x_2, ..., x_{M_{in}}]^T
```

其中：

```math
M_{in} = 462Q
```

### 窄带情况

```text
Input layer size = 462
```

### 宽带情况

```text
Input layer size = 462 × Q
```

---

## 12. 模式层 Pattern Layer

模式层是 GRNN 的核心。

每个训练样本对应一个 pattern neuron。

如果训练集有 `N` 个样本：

```text
Pattern layer size = N
```

对于测试输入 `X`，第 `i` 个模式神经元计算 `X` 与第 `i` 个训练样本 `X_i` 的距离，并通过高斯径向基函数：

```math
P_i
=
\exp\left(
-\frac{(X-X_i)^T(X-X_i)}{2σ^2}
\right)
```

也可以写成：

```math
P_i
=
\exp\left(
-\frac{\|X-X_i\|_2^2}{2σ^2}
\right)
```

其中：

- `X`：测试样本输入；
- `X_i`：第 `i` 个训练样本输入；
- `σ`：spread factor；
- `P_i`：第 `i` 个训练样本对当前测试样本的响应权重。

### 直观理解

```text
X 越接近 X_i → P_i 越大 → 第 i 个训练样本标签对输出影响越大
X 越远离 X_i → P_i 越小 → 第 i 个训练样本标签对输出影响越小
```

---

## 13. Spread factor σ 的作用

`σ` 是 GRNN 中唯一需要学习或选择的超参数。

| σ 大小 | 网络行为 | 可能问题 |
|---|---|---|
| 很小 | 接近最近邻回归 | 对噪声敏感，容易过拟合 |
| 适中 | 近邻样本权重大，远邻样本权重小 | 通常较优 |
| 很大 | 接近所有训练标签的平均值 | 欠拟合，定位结果被平滑 |

文章使用交叉验证选择最优 `σ`。

---

## 14. 求和层 Summation Layer

求和层包含两类节点：

```text
D 节点：非加权求和
S 节点：标签加权求和
```

### 14.1 D 节点

D 节点计算所有 pattern neuron 输出的总和：

```math
D = \sum_{i=1}^{N}P_i
```

### 14.2 S 节点

对于第 `k` 个输出变量，S 节点计算：

```math
S_k = \sum_{i=1}^{N}y_{ik}P_i
```

其中：

- `y_ik`：第 `i` 个训练样本的第 `k` 个输出标签；
- 对于本文的距离回归，`K = 1`，因此只有一个 `S` 节点。

单输出情况下：

```math
S = \sum_{i=1}^{N}R_iP_i
```

---

## 15. 输出层 Output Layer

输出层将加权和除以权重总和：

```math
\hat{y}_k = \frac{S_k}{D}
```

对于本文的单输出距离估计：

```math
\hat{R}
=
\frac{\sum_{i=1}^{N}R_iP_i}
{\sum_{i=1}^{N}P_i}
```

这本质上是一个基于高斯核权重的加权平均回归。

---

## 16. 仿真实验中的 GRNN 配置

仿真实验采用类似 SWellEx-96 的浅海波导模型。

### 16.1 数据规模

| 项目 | 设置 |
|---|---:|
| 阵元数 | 21 |
| source range | 1 km 到 8.5 km |
| range interval | 5 m |
| 总 range samples | 1500 |
| 训练样本 | 3000 |
| 交叉验证训练集 | 2700 |
| 交叉验证验证集 | 300 |
| 测试样本 | 1500 |
| 输出维度 | 1 |

### 16.2 网络维度

窄带：

```text
Input: 462
Pattern: 2700
Summation: 1 S node + 1 D node
Output: 1
```

可写为：

```text
462 → 2700 → (S, D) → 1
```

宽带：

```text
462Q → 2700 → (S, D) → 1
```

---

## 17. SWellEx-96 真实实验中的 GRNN 配置

### 17.1 数据设置

| 项目 | 设置 |
|---|---:|
| 阵元数 | 21 |
| 数据集 | SWellEx-96 event S5 |
| 训练样本数 | 1040 |
| 测试样本数 | 230 |
| 训练集 range interval | 2.5 m |
| 测试集 range interval | 10 m |
| 输出维度 | 1 |
| spread factor | 0.01 |

### 17.2 网络结构

窄带：

```text
462 → 1040 → (S, D) → 1
```

宽带：

```text
462Q → 1040 → (S, D) → 1
```

---

## 18. Spread factor 选择策略

文章使用 k-fold cross-validation 选择 `σ`。

流程如下：

```text
1. 设定候选 spread factor 集合
2. 将训练样本随机划分为 k folds
3. 每次使用 k-1 folds 构建 GRNN
4. 用剩下 1 fold 验证
5. 计算 MAPE
6. 对每个 σ 重复 k 次
7. 取平均 MAPE 最小的 σ
```

### 18.1 文中候选集合

仿真实验中使用的候选范围为：

```text
[0.01 : 0.01 : 0.1, 0.2 : 0.1 : 2.0]
```

即：

```python
sigma_candidates = [0.01, 0.02, ..., 0.10, 0.20, 0.30, ..., 2.00]
```

### 18.2 验证指标

文章使用 MAPE：

```math
MAPE
=
\frac{100}{N}\sum_{i=1}^{N}
\left|\frac{R_{g,i}-R_{t,i}}{R_{t,i}}\right|
```

其中：

- `R_g`：GRNN 预测距离；
- `R_t`：真实距离；
- `N`：测试样本数量。

---

## 19. 训练与推理逻辑

GRNN 没有传统神经网络中的权重迭代优化。

### 19.1 构建阶段

保存训练集输入和标签：

```text
X_train = [X_1, X_2, ..., X_N]
y_train = [y_1, y_2, ..., y_N]
σ = selected_spread_factor
```

### 19.2 推理阶段

对于测试样本 `X`：

```text
1. 计算 X 与所有 X_i 的欧氏距离
2. 通过高斯核得到 P_i
3. 用 P_i 对训练标签 y_i 加权
4. 除以权重和，得到预测值
```

公式：

```math
\hat{y}(X)
=
\frac{\sum_{i=1}^{N}y_i\exp(-\|X-X_i\|_2^2/(2σ^2))}
{\sum_{i=1}^{N}\exp(-\|X-X_i\|_2^2/(2σ^2))}
```

---

## 20. Python 风格伪代码

### 20.1 GRNN 模型

```python
import numpy as np

class GRNN:
    def __init__(self, sigma: float):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        X_train: shape = [N, input_dim]
        y_train: shape = [N] or [N, K]
        """
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)
        return self

    def predict(self, X_test):
        """
        X_test: shape = [B, input_dim]
        return: shape = [B] or [B, K]
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        preds = []

        for x in X_test:
            diff = self.X_train - x
            dist2 = np.sum(diff ** 2, axis=1)
            weights = np.exp(-dist2 / (2 * self.sigma ** 2))

            denominator = np.sum(weights) + 1e-12

            if self.y_train.ndim == 1:
                numerator = np.sum(weights * self.y_train)
            else:
                numerator = np.sum(weights[:, None] * self.y_train, axis=0)

            preds.append(numerator / denominator)

        return np.asarray(preds)
```

---

## 21. SCM 特征构造伪代码

假设已经得到某个频率处的阵列频域数据：

```text
X_snapshots: shape = [Ns, M]
```

其中：

- `Ns`：snapshot 数量；
- `M`：阵元数；
- 每个元素为复数频域声压。

```python
import numpy as np

def build_normalized_scm_feature(X_snapshots):
    """
    X_snapshots: complex array, shape = [Ns, M]
    return: real-valued SCM feature, shape = [M * (M + 1)]
    """
    X_snapshots = np.asarray(X_snapshots, dtype=np.complex128)
    Ns, M = X_snapshots.shape

    # 1. snapshot-wise normalization
    norms = np.linalg.norm(X_snapshots, axis=1, keepdims=True) + 1e-12
    X_norm = X_snapshots / norms

    # 2. SCM averaging
    C = np.zeros((M, M), dtype=np.complex128)
    for s in range(Ns):
        x = X_norm[s][:, None]          # [M, 1]
        C += x @ x.conj().T            # [M, M]
    C = C / Ns

    # 3. use diagonal and upper triangular part
    real_parts = []
    imag_parts = []

    for i in range(M):
        for j in range(i, M):
            real_parts.append(C[i, j].real)
            imag_parts.append(C[i, j].imag)

    # diagonal imaginary parts are theoretically zero, but this form keeps article's M(M+1) dimension
    feature = np.concatenate([real_parts, imag_parts], axis=0)

    return feature
```

注意：

```text
对 M = 21，输出维度应为 462
```

因为：

```math
2 × \frac{M(M+1)}{2} = M(M+1) = 462
```

---

## 22. 宽带 SCM 特征构造伪代码

如果使用多个频率，则每个频率分别构造 SCM 特征，然后拼接：

```python
def build_broadband_feature(freq_snapshot_dict):
    """
    freq_snapshot_dict:
        key: frequency
        value: complex snapshots, shape = [Ns, M]

    return:
        broadband feature, shape = [M * (M + 1) * Q]
    """
    features = []

    for freq in sorted(freq_snapshot_dict.keys()):
        X_snapshots = freq_snapshot_dict[freq]
        feat = build_normalized_scm_feature(X_snapshots)
        features.append(feat)

    return np.concatenate(features, axis=0)
```

---

## 23. σ 交叉验证伪代码

```python
from sklearn.model_selection import KFold
import numpy as np

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.mean(np.abs((y_pred - y_true) / (y_true + 1e-12)))


def select_sigma_by_cv(X_train, y_train, sigma_candidates, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    results = []

    for sigma in sigma_candidates:
        fold_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = GRNN(sigma=sigma)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            score = mape(y_val, y_pred)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        results.append((sigma, avg_score))

    best_sigma, best_score = min(results, key=lambda x: x[1])

    return best_sigma, best_score, results
```

---

## 24. 推荐的工程实现流程

### 24.1 数据准备

```text
1. 读取多阵元时域信号
2. 按时间或 range 切分 snapshot
3. 对每个 snapshot 做 DFT
4. 提取目标频率点的复数频域声压
5. 形成 shape = [Ns, M] 的复数数组
6. 构造 normalized SCM feature
7. 保存 feature 和对应 range label
```

### 24.2 数据格式建议

建议保存为：

```text
X_train.npy: shape = [N_train, 462Q]
y_train.npy: shape = [N_train]
X_val.npy:   shape = [N_val, 462Q]
y_val.npy:   shape = [N_val]
X_test.npy:  shape = [N_test, 462Q]
y_test.npy:  shape = [N_test]
```

如果做多输出定位：

```text
y_train.npy: shape = [N_train, K]
```

---

## 25. 与深度学习网络的区别

GRNN 和常规 FNN/CNN 的主要区别：

| 项目 | GRNN | FNN/CNN |
|---|---|---|
| 学习方式 | 核回归 / 样本记忆 | 参数优化 |
| 是否反向传播 | 否 | 是 |
| 是否需要初始化权重 | 否 | 是 |
| 主要参数 | σ | 层数、神经元数、学习率、激活函数等 |
| 模式层规模 | 等于训练样本数 | 固定网络结构 |
| 推理复杂度 | 随训练样本数增加 | 与网络规模相关 |
| 输出形式 | 连续回归 | 分类或回归均可 |

---

## 26. 实现时需要注意的问题

### 26.1 输入维度较高

宽带情况下输入维度为：

```text
462Q
```

如果使用 5 个频率：

```text
input_dim = 2310
```

这会导致距离计算开销增加。

### 26.2 模式层大小等于训练集大小

GRNN 的 pattern layer 神经元数量等于训练样本数。

如果训练样本很多，推理时需要对每个测试样本计算其与所有训练样本的距离。

复杂度约为：

```text
O(N_train × input_dim)
```

### 26.3 特征尺度需要一致

GRNN 使用欧氏距离，因此特征尺度对结果影响很大。

建议：

```text
训练集、验证集、测试集使用同一套标准化参数
```

可以考虑：

```python
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-12

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std
```

但需要注意：原文主要依赖声压归一化和 SCM 归一化，是否额外做 z-score 需要在实验中验证。

### 26.4 σ 需要重新调参

不同数据集、不同频率组合、不同环境下，最佳 `σ` 可能不同。

建议至少搜索：

```python
sigma_candidates = list(np.arange(0.01, 0.11, 0.01)) + list(np.arange(0.2, 2.1, 0.1))
```

### 26.5 距离标签单位保持一致

文章中 range 常以 km 表示。

如果标签单位使用 m，MAPE 不变，但 `σ` 的最佳值可能因输入标准化方式不同而变化。

建议统一使用：

```text
range label: km
```

---

## 27. 适合复现的最小网络规格

如果按文章 SWellEx-96 实验复现，可以从以下设置开始：

```text
阵元数 M = 21
频率数 Q = 1 或多个
输入维度 input_dim = 462Q
输出维度 K = 1
训练样本数 N_train = 1040
模式层神经元数 = 1040
求和层 = 1 个 S 节点 + 1 个 D 节点
输出层 = 1 个距离值
spread factor σ = 0.01 起步
```

网络结构：

```text
462Q → N_train → (S, D) → 1
```

---

## 28. 可直接参考的类 PyTorch 实现思路

虽然 GRNN 不需要反向传播，但可以用 PyTorch 做矩阵化推理，方便 GPU 加速。

```python
import torch

class TorchGRNN:
    def __init__(self, sigma=0.01, device="cuda"):
        self.sigma = sigma
        self.device = device
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        return self

    @torch.no_grad()
    def predict(self, X_test, batch_size=256):
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=self.device)
        outputs = []

        for start in range(0, X_test.shape[0], batch_size):
            x = X_test[start:start + batch_size]

            # dist2: [B, N_train]
            dist2 = torch.cdist(x, self.X_train, p=2) ** 2
            weights = torch.exp(-dist2 / (2 * self.sigma ** 2))

            denominator = weights.sum(dim=1, keepdim=True) + 1e-12

            if self.y_train.ndim == 1:
                numerator = weights @ self.y_train[:, None]
                pred = numerator / denominator
                pred = pred.squeeze(1)
            else:
                numerator = weights @ self.y_train
                pred = numerator / denominator

            outputs.append(pred.detach().cpu())

        return torch.cat(outputs, dim=0).numpy()
```

---

## 29. 文章方法的核心复现要点

最关键的是以下四点：

```text
1. 不输入原始信号，而是输入 normalized SCM。
2. SCM 只取对角线和上三角元素，并拆成实部和虚部。
3. GRNN 的 pattern neuron 数量等于训练样本数量。
4. 输出不是分类类别，而是连续距离回归值。
```

---

## 30. 总结

该文的 GRNN 网络可以简化为：

```text
Normalized SCM feature
        ↓
Input layer: 462Q nodes
        ↓
Pattern layer: N_train Gaussian RBF nodes
        ↓
Summation layer: weighted sum S and unweighted sum D
        ↓
Output layer: S / D
        ↓
Predicted source range
```

数学形式为：

```math
\hat{R}(X)
=
\frac{\sum_{i=1}^{N}R_i\exp(-\|X-X_i\|_2^2/(2σ^2))}
{\sum_{i=1}^{N}\exp(-\|X-X_i\|_2^2/(2σ^2))}
```

工程上可以理解为：

```text
基于 normalized SCM 特征的高斯核加权最近邻回归器。
```
