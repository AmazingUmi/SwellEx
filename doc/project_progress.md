# SwellEx Project Progress

更新日期：2026-05-11

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
/X: [sample, numerator_element, denominator_element, frequency, real_imag]
```

含义：

```text
ratio_freq(i,j,f) = FFT(element_i,f) / FFT(element_j,f)
X(:,:,:,:,1) = real(ratio_freq)
X(:,:,:,:,2) = imag(ratio_freq)
```

## 3. Python 当前结构

Python 网络也按方法分开：

```text
scripts_py/RBD_method/
scripts_py/ELM_method/
```

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
[batch, 2, element_pair, frequency]
```

其中 `element_pair = numerator_element * denominator_element`。

## 4. 当前判断

项目现在具备两套可比较的输入特征路线：

- RBD：带物理先验的 Green 函数特征。
- ELM：不做 RBD，直接学习阵元间频域 pairwise ratio。

这两条路线应先保持分离，分别训练和评估，避免 loader、checkpoint 和
实验记录混淆。

## 5. 下一步建议

1. 分别生成一版 RBD 与 ELM 数据集，确认 HDF5 字段和样本数量。
2. 用同一 split 策略训练 RBD 与 ELM baseline。
3. 记录 MAE/RMSE、训练曲线、最大误差样本和对应 `window_center_s`。
4. 若 ELM flat-pair 表现有潜力，再考虑新增 3D CNN，保留
   `[batch, 2, N, N, F]` 结构。
