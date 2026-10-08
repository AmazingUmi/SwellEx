# Python 和生成数据验证

从仓库根目录使用锁定的 uv 环境运行：

```bash
uv run python scripts/localization_networks/validation/validate.py quick --device cuda --report reports/quick.json
uv run python scripts/localization_networks/validation/validate.py datasets --root outputs/Datasets --report reports/datasets.json
```

`quick` 检查 Python 语法和导入、四个 CLI、工作流标准库单元测试、共享路径与划分契约、合成 HDF5 加载、全部 12 个注册模型的前向/反向和 GRNN 数值不变量。CUDA 不可用时直接失败，不会回退 CPU。模型使用实际类与默认规模；临时样本验证两种存储轴方向、有效样本筛选和坐标值。CPU 线程数取 `OMP_NUM_THREADS`，最多 4。

`datasets` 递归检查根目录内所有 `.h5`：完整有限特征和标签、标签在原始范围轨迹上的线性插值、频率与时间/索引、列优先（`j` 外层、`i` 内层）的 ELM `i<j` / SCM `i<=j` 对顺序，以及训练/测试原始采样区间不重叠。`/X` 必须有逻辑轴属性 `axis_order`：RBD 为 `sample,element,frequency,real_imag`，ELM/SCM 为 `sample,pair,frequency,real_imag`。新生成器使用原生轴并写入 `storage_axis_order`。加载器优先使用轴声明，仍能读取历史 MATLAB 反序和无声明输入。

数据验收要求三个方法具有相同划分、源窗口和频率，并比较标签、掩码、时间和阵元深度。用于严格冒烟数据；RBD 单窗口与 ELM/SCM 多快照窗口不同的运行会明确失败。

`--expected-config file.json` 接受完整 Python 生成配置（`schema_version`、`dataset`、`rbd`、`result_path`），配置规则直接采用 `scripts/data_processing/generation_config.py`；或使用独立的平面约束 `num_elements`、`freq_hz`、`min_samples`。未知字段直接失败。完整配置核对窗口起止范围、源编号对应的时间网格、`fs` 与窗口长度对应的采样端点、FFT 频率网格、非自适应频率选择、周期或 Range_nearby 划分的完整源成员，以及生成元数据。自适应频率仅核对固定频率并集、个数/范围约束，不重新计算原始信号的谱功率排序。

RBD 依据配置或生成元数据检查每个频率列的阵元复向量 L2 范数。`best` 非零列必须为 1（绝对容差 `2e-6`）；`multipath` 必须不超过 `1+2e-6`。范数不超过 `1e-8` 的零列允许，`full` 模式的 DC 列不要求单位范数。该检查能拒绝误存未归一化 FFT 振幅；极弱非 DC 列若受到核心 `norm+eps` 归一化影响，仍可能失败，报告会给出范数与检查块。

JSON 含 `status: PASS|FAIL`、逐项 `checks`（状态、耗时、详情或错误堆栈）、`dataset_directories`（方法到绝对目录列表）。每个方法只有一个目录时，`datasets` 另给方法到目录的映射。失败返回非零；报告逐项落盘，保留此前证据。工作流的训练/预测及产物验收由远程 worker 执行；这些检查不构成模型精度或物理模型的科学认证。
