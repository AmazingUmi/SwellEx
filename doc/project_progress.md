# SwellEx 项目进度记录

更新日期：2026-04-25

本文档基于当前仓库中的 `scripts_matlab`、`scripts_py`、`outputs` 目录梳理项目进展，重点记录已完成内容、当前产物、可运行入口和后续待推进事项。

## 1. `scripts_matlab` 当前进度

### 1.1 已完成的主流程

MATLAB 部分已经形成从原始 VLA 阵列信号到 RBD 特征数据集的处理链路：

- `RBD_main.m`：完成单个时间片段的 Ray-Based Deconvolution 验证流程，包括 VLA 阵列几何读取、CTD 声速剖面加载、全通道信号加载、Bartlett 波束形成、最佳入射角估计、Green 函数估计和结果绘图。
- `Signals_Analysis.m`：完成连续滑窗 RBD 分析流程，可输出随时间变化的最佳波束角、波束功率、选定阵元的时域 Green 函数、峰值幅度和峰值时延。
- `Signals_Segmentation.m`：完成面向神经网络训练的数据集生成流程。脚本按固定时间窗切分 S5 VLA 信号，对每个窗口执行 RBD，并将频域 Green 函数保存为 HDF5 输入特征。

### 1.2 数据集生成状态

`Signals_Segmentation.m` 当前参数状态：

- 采样率：`fs = 1500 Hz`
- 窗长：`segment_duration_s = 1.0 s`
- 步长：`segment_step_s = 1.0 s`
- 处理范围：默认覆盖完整记录
- 角度搜索：`-90 deg` 到 `90 deg`，共 181 个角度
- 训练/测试划分：`train_test_ratio = [4 1]`
- 标签来源：`events/range/RangeEventS5/SproulToVLA.S5.txt`
- 特征内容：`green_freq` 的实部和虚部

当前已生成 HDF5 数据：

| 数据集 | 路径                                                                                   | 样本数 | 文件大小 |
| ------ | -------------------------------------------------------------------------------------- | -----: | -------: |
| train  | `outputs/RBD_results/train/RBD_green_freq_nn_S5_start_s0_end_s4499_step_s1_train.h5` |   3600 | 426.5 MB |
| test   | `outputs/RBD_results/test/RBD_green_freq_nn_S5_start_s0_end_s4499_step_s1_test.h5`   |    900 | 106.6 MB |

HDF5 设计已经包含神经网络训练所需的核心字段：

- `/X`：`[window, element, frequency, real_imag]`
- `/y_range_km`：窗口中心时刻对应的声源距离标签
- `/valid_sample`：有效标签掩码
- `/time/window_start_s`、`/time/window_center_s`、`/time/window_stop_s`
- `/frequency/freq_hz`
- `/array/depth_m`
- `/split/source_segment_idx`
- `/rbd/theta_best_rad`、`/rbd/beam_power`、`/rbd/signal_freq_scale`

### 1.3 辅助函数和绘图脚本

当前 MATLAB 辅助代码已经覆盖：

- 数据读取与预处理：`extract_sigs_from_sio.m`、`gunzip_sigs.m`、`sioread.m`、`read_ctd_i9605.m`
- 数据和元信息提取：`Source_info.m`、`extract_interferer_positions_S59.m`
- 地图和航迹绘图：`plotgeomap.m`、`plot_4arrays_on_etopo.m`、`plot_interferer_positions_S59_etopo.m`、`plot_sproul_to_arrays_range.m`
- RBD 核心计算：`compute_tau.m`、`bartlett_beamformer.m`、`rbd_decompose.m`
- RBD 结果展示：`plot_results.m`、`plot_results_series.m`、`plot_sound_speed_profile.m`、`pic_generate.m`
- 声速剖面补全：`extend_sound_speed_profile.m`

### 1.4 当前阶段判断

MATLAB 部分已经完成“数据准备和特征导出”的主要目标，当前已经能稳定为 Python 网络提供训练集和测试集。现阶段 MATLAB 的工作重点不再是搭建主流程，而是进一步提高数据质量和可追溯性。

建议后续推进：

- 增加一次正式的 HDF5 结构校验脚本，检查字段、维度、有效样本数和标签范围。
- 将关键参数集中成配置区或配置文件，减少修改脚本正文带来的误操作。
- 保存每次数据生成的运行日志和参数快照，方便后续模型结果回溯。
- 若继续扩展到其他事件或阵列，需要把 S5 相关路径和文件名参数化。

## 2. `scripts_py` 当前进度

### 2.1 已完成的主流程

Python 部分已经从单脚本训练入口拆分为较清晰的包结构，当前主要目标是基于 MATLAB 导出的 RBD HDF5 特征训练复值 CNN 距离回归模型。

当前入口：

- `train_complex_cnn_range.py`：兼容入口，调用 `range_cnn.cli.main()`。

当前模块划分：

- `range_cnn/cli.py`：命令行入口，支持 `train` 和 `predict` 两个子命令。
- `range_cnn/data.py`：HDF5 路径解析、布局识别、有效样本筛选、懒加载数据集。
- `range_cnn/model.py`：复值卷积层、复值 BatchNorm、modReLU 和 `ComplexRangeCNN`。
- `range_cnn/training.py`：随机训练/验证划分、标签归一化、训练循环、指标记录、checkpoint 保存。
- `range_cnn/prediction.py`：加载 checkpoint、测试集推理、CSV 输出和预测图绘制。
- `range_cnn/paths.py`：默认输入输出路径。
- `README.md`：记录安装、训练、预测、HDF5 格式和网络结构说明。

### 2.2 模型与训练状态

当前模型为复值 CNN 距离回归模型：

- 输入：`[batch, 2, element, frequency]`
- 通道 0：`real(green_freq)`
- 通道 1：`imag(green_freq)`
- 复值卷积：用两组实值 `Conv2d` 实现复数卷积
- 激活：`ComplexModReLU`
- 输出：归一化后的距离回归值
- 标签归一化：使用训练集均值和标准差，报告指标还原为 km

当前训练产物：

| 文件                                                         | 说明                        |
| ------------------------------------------------------------ | --------------------------- |
| `outputs/complex_cnn_range/complex_cnn_range_best.pt`      | 验证集 RMSE 最优 checkpoint |
| `outputs/complex_cnn_range/complex_cnn_range_last.pt`      | 最后一个 epoch checkpoint   |
| `outputs/complex_cnn_range/complex_cnn_range_history.json` | 训练历史                    |

当前训练记录：

- 总训练轮数：100 epoch
- 最优验证结果：第 90 epoch
- 最优验证 MAE：0.256 km
- 最优验证 RMSE：0.361 km
- 最后一轮验证 MAE：0.258 km
- 最后一轮验证 RMSE：0.364 km

### 2.3 测试集预测状态

当前已经使用训练好的 checkpoint 对测试 HDF5 数据进行了预测，并生成：

| 文件                                         | 说明                     |
| -------------------------------------------- | ------------------------ |
| `outputs/Net_results/predictions.csv`      | 测试集逐样本预测结果     |
| `outputs/Net_results/range_prediction.png` | 真实距离曲线与预测散点图 |

预测结果概况：

- 测试样本数：900
- 测试集 MAE：0.243 km
- 测试集 RMSE：0.346 km
- 最大绝对误差：2.121 km
- 最小绝对误差：0.0003 km

`predictions.csv` 已包含可回溯字段：

- `sample`
- `source_segment_idx`
- `window_center_s`
- `pred_range_km`
- `true_range_km`
- `abs_error_km`

### 2.4 环境状态

`scripts_py/requirements.txt` 当前列出：

- `torch`
- `numpy`
- `h5py`
- `matplotlib`

### 2.5 当前阶段判断

Python 部分已经完成“模型训练、验证、测试预测和结果落盘”的闭环。现阶段已经具备初步模型实验能力，下一步重点应转向模型评估可靠性、数据划分策略和可复现实验管理。

建议后续推进：

- 增加独立测试评估命令，统一输出 MAE、RMSE、最大误差、分时间段误差统计。
- 增加训练曲线绘图脚本，直接从 `complex_cnn_range_history.json` 输出 loss/MAE/RMSE 曲线。
- 固定 Python 环境，例如增加 `venv` 或 `conda` 使用说明，并确认 `h5py`、`torch` 版本。
- 将当前 train/val 随机划分与 MATLAB train/test 划分的关系写入实验记录，避免评估口径混淆。
- 增加 baseline，例如简单 MLP、实值 CNN 或传统 RBD 距离估计对比。
- 对最大误差样本进行定位分析，检查是否集中在特定时间段、距离范围或 RBD 质量较差的窗口。

## 3. 总体进度结论

当前项目已经完成从 SwellEx S5 阵列信号到 RBD 特征，再到 Complex CNN 距离回归模型训练和测试预测的第一版端到端流程。

阶段性成果：

- MATLAB 已完成 RBD 特征生成，产出 3600 个训练样本和 900 个测试样本。
- Python 已完成复值 CNN 训练与预测闭环。
- 当前模型在测试集上达到约 0.243 km MAE、0.346 km RMSE。
- 输出结果已经包含可回溯的 `source_segment_idx` 和 `window_center_s`，便于和 MATLAB 分段数据对齐。

当前主要风险：

- 数据集划分、训练/验证划分、测试集评估需要形成更严格的实验记录。
- 当前评估主要是整体误差指标，还需要按时间、距离区间和异常样本做细分分析。

建议下一阶段目标：

1. 固化运行环境和实验命令。
2. 增加自动化评估与绘图脚本。
3. 对误差最大的样本做回溯分析。
4. 在当前 Complex CNN 基线之上开展结构和输入特征对比实验。
