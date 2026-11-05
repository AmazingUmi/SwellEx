# Archive

2026-11-05 项目整理：活动范围收缩为 `base_pack`（**RBD、SCM** 两种特征；`complex_CNN`、`ResNet18`、`ResNet50`、`GRNN` 四种网络）。其余实验的代码、文档、Notebook 与已运行结果移入本目录，不删除。

整理前基线：Git `HEAD` = `a9fd89f`（工作区另有未提交的 Notebook 改动，已原样备份在 `source/notebooks/`）。如需完整还原旧状态，可 `git show a9fd89f:<path>` 取出共享文件的整理前版本。

## 目录

| 路径 | 内容 | Git |
|---|---|---|
| `source/scripts/{data_processing,network_training,remote_workflow}/{joint_feature_pack,data_scale_pack,MoE_data_scale,MoE_learn}/` | 四个已退役实验包的完整代码（含各自测试、TOML 预设、matrix plan） | 跟踪 |
| `source/scripts/network_training/base_pack/elm/` | ELM 神经网络包（`main.py`、`network/…`） | 跟踪 |
| `source/scripts/network_training/base_pack/{rbd,scm}/network/models/real_cnn.py` | 仅幅度输入的 `real_CNN`（在 base/joint 保留运行中均差于 `complex_CNN`） | 跟踪 |
| `source/scripts/data_processing/common/{joint_builder,joint_contract,elm_feature}.py` | 联合拼接构造/契约；`elm_feature` 与其数值检查 `test_elm`（自 `feature_math.py` / `numerical_validation.py` 摘出） | 跟踪 |
| `source/scripts/remote_workflow/tests/test_{data_scale_pack,moe_data_scale_pack,moe_learn_pack}.py` | 退役实验包的调度测试 | 跟踪 |
| `source/doc/` | 退役包文档：`joint-feature-pack.md`、`data-scale-pack.md`、`moe-data-scale.md`、`moe-learn.md`、`joint-features.md`、`matrix-studies.md` | 跟踪 |
| `source/notebooks/` | 整理前的两个 Colab Notebook | 跟踪 |
| `source/AGENTS.pre-cleanup.md` | 整理前的 AGENTS.md（含 data-scale / MoE / MoE_learn 的架构契约） | 跟踪 |
| `output/` | 已运行结果，见 [`output/README.md`](output/README.md) | 仅该 README 跟踪，大文件不入 Git |

## 活动代码的变化

- **第一轮（实验包退役）**：`remote_workflow/{packs,pack_cli,run,run_identity}.py` 只注册 `base_pack`（方法 `RBD|SCM`，网络四种，去掉 `--fusion`/`study`/`budget-study`）；共享的 `data_processing`、`network_training`、`remote_workflow/base_pack` 模块里去掉 ELM 与 `real_CNN`。
- **第二轮（目录与命名整理）**：去掉三层代码根下的 `base_pack/` 目录层与 `data_processing/common/`（`base_pack` 只保留为实验包/运行 ID 标识；`network_training/common/` 保留为 neural 与 grnn 的共享层）；RBD、SCM 两个神经网络包合并为 `network_training/neural/`（入口 `neural/main.py {train,predict} --method RBD|SCM`）；`tasks.py` → `model_plan.py` 并复用 `catalog.py`；TOML 预设改名为 `remote_workflow/presets/{full,train_before,train_after}.toml`；`remote_workflow/validation/` 只保留 quick 检查，数据集校验统一到 `data_processing/validate.py datasets`；删去为已退役包保留的注入参数（`forbid_adapt_pack`、`window_section_fn`、`generator_version` 等）。
- 生成器版本号 `python-features-v2`、RBD/SCM 特征数学与数据集命名/指纹均未改变；但 haiqin 的数据集缓存键包含生成/校验源码哈希，整理后的首次运行会重新生成一次 HDF5。
- **`archive/source/` 保持整理前的目录布局（`.../base_pack/...`、`common/`），其导入名与现在的活动布局不兼容**，仅作参考与还原素材。

## 还原某个实验包

归档源码保持整理前的仓库相对路径与导入名，不能直接与现在的活动 `scripts/` 并存。还原步骤：

1. 以 Git `a9fd89f`（整理前基线）为准取回旧布局，或把 `source/scripts/<layer>/<pack>/` 复制回 `scripts/<layer>/<pack>/`，再把其中的 `scripts.*.base_pack.*`、`...common.*` 导入改为现在的平铺模块（`scripts.data_processing.*`、`scripts.network_training.{common,neural,grnn}.*`、`scripts.remote_workflow.*`）。
2. 把包重新加入 `remote_workflow/packs.py`；如需 ELM，则从 `source/scripts/network_training/base_pack/elm/` 与 `source/scripts/data_processing/common/elm_feature.py` 还原，并恢复 `data_processing/input_contract.py` 中的 `METHODS`/`FEATURE_AXIS`。
3. 把 `source/doc/` 中对应文档移回 `doc/`，必要时把结果从 `output/` 移回 `../output/`。
4. 在 haiqin 上运行 `./scripts/run.sh <pack> validate --suite quick` 验证。
