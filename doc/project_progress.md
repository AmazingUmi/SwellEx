# SwellEx Project Progress

更新日期：2026-05-12

本文档记录当前仓库结构、已完成流程和下一步建议。当前项目已经从单一
RBD 路线扩展为 RBD 与 ELM pairwise-ratio 两条并行路线。

## 1. 数据目录

原始数据已经集中到：

```text
Origindata/
  events/
  positions/
  environments/
  source/
```

关键 S5 输入：

```text
Origindata/events/S5/vla_matfiles/
Origindata/events/S5/CTD_i9605.mat
Origindata/events/range/RangeEventS5/SproulToVLA.S5.txt
Origindata/positions/positions_vla.txt
```

生成的数据集仍写入：

```text
outputs/Datasets/
```

## 2. MATLAB 当前结构

入口脚本按方法分开：

```text
scripts_matlab/RBD_method/
  Signals_Segmentation.m
  RBD_main.m
  Signals_Analysis.m

scripts_matlab/ELM_method/
  Signals_Segmentation.m
```

公共函数按职责分组：

```text
scripts_matlab/function/dataset_function/   DS_*
scripts_matlab/function/split_function/     SPL_*
scripts_matlab/function/rbd_function/       RBD_* and RBD core functions
scripts_matlab/function/elm_function/       ELM_*
```

### RBD Dataset

生成脚本：

```text
scripts_matlab/RBD_method/Signals_Segmentation.m
```

HDF5 输入：

```text
/X: [sample, element, frequency, real_imag]
```

含义：

```text
X(:,:,:,1) = real(green_freq)
X(:,:,:,2) = imag(green_freq)
```

### ELM Dataset

生成脚本：

```text
scripts_matlab/ELM_method/Signals_Segmentation.m
```

HDF5 输入：

```text
/X: [sample, pair, frequency, real_imag]
/pair/numerator_element_idx: [pair, 1]
/pair/denominator_element_idx: [pair, 1]
```

含义：

```text
ratio_freq(i,j,f) = FFT(element_i,f) / FFT(element_j,f)
pair 为严格上三角阵元对，i < j
X(:,:,:,1) = real(ratio_freq)
X(:,:,:,2) = imag(ratio_freq)
```

## 3. Python 当前结构

Python 网络也按方法分开：

```text
scripts_py/common/
scripts_py/RBD_method/
scripts_py/ELM_method/
```

`scripts_py/common/` 目前承载路径、HDF5 路径解析和 split、训练 epoch、
checkpoint resume、预测 CSV/plot 等公共逻辑。RBD 与 ELM 各自保留 HDF5
layout loader 和 model registry。

RBD 入口：

```bash
python3 scripts_py/RBD_method/Network_main.py train \
  --model complex_cnn_range \
  --data <rbd_dataset_name>
```

ELM 入口：

```bash
python3 scripts_py/ELM_method/Network_main.py train \
  --model elm_complex_cnn_range \
  --data <elm_dataset_name>
```

输出路径：

```text
outputs/networks_results/RBD_method/
outputs/networks_results/ELM_method/
```

RBD PyTorch 输入：

```text
[batch, 2, element, frequency]
```

ELM 当前 PyTorch 输入：

```text
[batch, 2, pair, frequency]
```

其中 `pair = element_count * (element_count - 1) / 2`，旧版 full matrix
ELM 数据仍由 Python loader 兼容展平。

训练损失空间已统一为两套方法都支持：

```text
--loss-space normalized
--loss-space km
```

其中 `km` 直接用实际距离误差计算 SmoothL1 loss，`--huber-beta` 单位为 km。

## 4. 当前判断

项目现在具备两套可比较的输入特征路线：

- RBD：带物理先验的 Green 函数特征。
- ELM：不做 RBD，直接学习阵元间频域 pairwise ratio。
- ELM Mel 频率选择：可按 Mel 关系选取最近 FFT bin，保留复数 ratio，
  用于降低 F 维度。

这两条路线应先保持分离，分别训练和评估，避免 loader、checkpoint 和
实验记录混淆。

## 5. 下一步建议

1. 分别生成一版 RBD 与 ELM 数据集，确认 HDF5 字段和样本数量。
2. 用同一 split 策略训练 RBD 与 ELM baseline。
3. 记录 MAE/RMSE、训练曲线、最大误差样本和对应 `window_center_s`。
4. 若 ELM flat-pair 表现有潜力，再考虑新增 3D CNN，保留
   `[batch, 2, N, N, F]` 结构。
