# SwellEx Project Progress

更新日期：2026-08-26

本文档记录当前仓库结构、已完成流程和下一步建议。当前项目已经从单一
RBD 路线扩展为 RBD、ELM least-squares ratio 与 SCM 三条并行可训练路线，
另有一条不参与训练的 standalone SCM-GRNN 参考库回归路线。

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

scripts_matlab/SCM_method/
  Signals_Segmentation.m
```

公共函数按职责分组：

```text
scripts_matlab/function/dataset_function/   DS_*
scripts_matlab/function/split_function/     SPL_*
scripts_matlab/function/rbd_function/       RBD_*
scripts_matlab/function/elm_function/       ELM_*
scripts_matlab/function/scm_function/       SCM_*
```

RBD 函数命名规则已经统一为公开函数使用 `RBD_` 前缀，例如
`RBD_decompose`、`RBD_compute_tau`、`RBD_bartlett_beamformer`。旧的小写
函数名已移除，脚本应统一使用 `RBD_*` 命名。

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
ratio_freq(i,j,f) = sum_s FFT_i,s(f) * conj(FFT_j,s(f)) /
                    (sum_s abs(FFT_j,s(f))^2 + floor)
pair 为严格上三角阵元对，i < j
X(:,:,:,1) = real(ratio_freq)
X(:,:,:,2) = imag(ratio_freq)
```

ELM 数据使用脚本中的 `segment_duration_s`、`num_snapshots_per_segment` 和
`snapshot_overlap_count` 控制快拍长度、`Ns` 和重叠数。

### SCM Dataset

生成脚本：

```text
scripts_matlab/SCM_method/Signals_Segmentation.m
```

HDF5 输入：

```text
/X: [sample, pair, frequency, real_imag]
/pair/numerator_element_idx: [pair, 1]
/pair/denominator_element_idx: [pair, 1]
```

含义：

```text
C_q = mean_s((x_s / ||x_s||)(x_s / ||x_s||)^H)
pair 为含对角线的上三角阵元对，i <= j
X(:,:,:,1) = real(C_q pair-vector)
X(:,:,:,2) = imag(C_q pair-vector)
```

## 3. Python 当前结构

### Shared Frequency Selection

RBD、ELM 与 SCM 的差异主要是特征不同，频率选择已经统一到：

```text
scripts_matlab/function/dataset_function/DS_select_frequency_bins.m
```

三者都支持：

```matlab
frequency_selection_modes = "full";
frequency_selection_modes = "mel";
frequency_selection_modes = "deep";
frequency_selection_modes = "shallow";
frequency_selection_modes = "adapt";
frequency_selection_modes = ["deep", "shallow"];
frequency_selection_modes = ["mel", "deep", "adapt"];
```

`full` 表示保留完整 one-sided FFT 频率轴，不能和其它选频模式组合。
默认的命名频率和 bin 数量已经内置到 `DS_select_frequency_bins`，脚本通常只需设置
`frequency_selection_modes`，并让 `frequency_selection_config` 为空。默认
`deep` 频率为 `[49 64 79 94 112 130 148 166 201 235 283 338 388]` Hz，
`shallow` 频率为 `[109 127 145 163 198 232 280 335 385]` Hz，`mel`
默认 64 个 bin，`adapt` 默认 16 个 bin。`adapt` 根据有效候选窗的平均
频谱能量选取最强频点，仍生成固定的 dataset-level 频率轴，保证 CNN 的
frequency 维语义一致。

RBD 通过 `rbd_frequency_estimation` 绑定物理分解频率轴与神经网络输出
频率轴：`"full"` 使用完整 one-sided FFT 频率轴；`"selected"` 使用
`rbd_selected_frequency_modes` 指定的目标频点。tag 中用 `estfull` 或
`estsel` 区分。

RBD 波束选择通过 `rbd_beam_selection` 统一控制：`"best"` 只使用最强
beam angle；`"multipath"` 才启用 `rbd_multipath_options` 中的多径峰值、
角度间隔、峰数和旁瓣剔除参数。

RBD、ELM 与 SCM 脚本都会自动生成 dataset variant tag。`manual_dataset_variant_tag`
非空时会追加在自动 tag 后面，而不是替换自动 tag。数据集生成完成后，
MATLAB Command Window 会打印对应的训练和预测命令。

Python 网络也按方法分开：

```text
scripts_py/common/
scripts_py/RBD_method/
scripts_py/ELM_method/
scripts_py/SCM_method/
scripts_py/0_GRNN_related/
```

`scripts_py/common/` 目前承载路径、HDF5 路径解析和 split、训练 epoch、
checkpoint resume、预测 CSV/plot 等公共逻辑。RBD、ELM 与 SCM 各自保留
HDF5 layout loader 和 model registry。

`scripts_py/0_GRNN_related/` 是独立的 SCM-GRNN 工作流：不走 epoch /
optimizer / checkpoint，而是把 SCM 特征与距离标签存成参考库 artifact，
预测时用高斯核加权平均。入口：

```bash
python3 scripts_py/0_GRNN_related/GRNN_main.py build \
  --data <scm_dataset_name>

python3 scripts_py/0_GRNN_related/GRNN_main.py predict \
  --data <scm_dataset_name>
```

详见 `doc/20_python_workflows/20_standalone_scm_grnn.md`。

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

SCM 入口：

```bash
python3 scripts_py/SCM_method/Network_main.py train \
  --model scm_complex_cnn_range \
  --data <scm_dataset_name>
```

输出路径：

```text
outputs/networks_results/RBD_method/
outputs/networks_results/ELM_method/
outputs/networks_results/SCM_method/
outputs/networks_results/0_GRNN_related/scm_grnn_range/
```

RBD PyTorch 输入：

```text
[batch, 2, element, frequency]
```

ELM 当前 PyTorch 输入：

```text
[batch, 2, pair, frequency]
```

其中 ELM 的 `pair = element_count * (element_count - 1) / 2`，SCM 的
`pair = element_count * (element_count + 1) / 2`。

训练损失空间已统一为三套方法都支持：

```text
--loss-space normalized
--loss-space km
```

其中 `km` 直接用实际距离误差计算 SmoothL1 loss，`--huber-beta` 单位为 km。

## 4. 当前判断

项目现在具备三套可比较的输入特征路线：

- RBD：带物理先验的 Green 函数特征。
- ELM：不做 RBD，学习阵元间频域 least-squares ratio。
- SCM：学习归一化阵列向量的空间协方差矩阵。
- RBD、ELM 与 SCM 共用频率选择：`full`、`mel`、`deep`、`shallow`、`adapt`
  及其组合，用于控制 F 维度和目标频点。
- SCM 数据集同时可喂给 standalone GRNN（`scripts_py/0_GRNN_related/`），
  作为不需要训练的非参数 baseline。

这些路线应先保持分离，分别训练和评估，避免 loader、checkpoint 和
实验记录混淆。GRNN 不加入 RBD/ELM/SCM 的 model registry。

## 5. 下一步建议

1. 分别生成 RBD、ELM 与 SCM 数据集，确认 HDF5 字段和样本数量。
2. 用同一 split 策略训练 RBD、ELM 与 SCM baseline。
3. 记录 MAE/RMSE、训练曲线、最大误差样本和对应 `window_center_s`。
4. 对比 ELM `Ns=4` LS-ratio 与 SCM `Ns=1/4` 设置下的稳健性差异。
5. 用同一 SCM 数据集构建 GRNN 参考库（`build --cv-folds 5` 选 sigma），
   与 SCM CNN/ResNet baseline 对比 MAE/RMSE，作为非参数基准。
