# Development and Runtime Environment

Code editing happens locally on macOS. Dataset generation, numerical tests,
and model execution happen on SSH host `haiqin` (`HQserver`). Use the
[remote workflow](../20_python_workflows/30_remote_validation.md) from the local
checkout.

## Local Dispatcher

Requirements: uv 0.12.19, Python 3.11+, Git, SSH, and rsync. The dispatcher uses
only the standard library. Invoke it with `uv run --no-project --python python3`
to avoid syncing the Linux numerical profile onto the local machine.

## haiqin

```text
Original repository: /mnt/SDD/project/SwellEx
Validation workspace: /mnt/SDD/project/SwellEx_validation
Base Python: /home/yiyang-lu/miniforge3/envs/pytorch/bin/python
uv: /mnt/SDD/project/SwellEx_validation/runtime_profile/tools/uv-0.12.19/uv-x86_64-unknown-linux-gnu/uv
GPU: NVIDIA RTX A5000
OS: Ubuntu 26.04.1 LTS / glibc 2.43
```

Dependencies are pinned in the root `pyproject.toml` and `uv.lock`, including
Python 3.14, PyTorch 2.11.0, torchvision 0.26.0, NumPy, SciPy, h5py, and
Matplotlib. Linux torch/torchvision wheels come from the explicit official
CUDA 12.8 index. The dispatcher uses frozen uv sync and a separate environment
for each lock hash; installed packages and CUDA availability are recorded in
each run's `environment.json`.

All feature processing uses Python. The original MAT files are data inputs;
MATLAB runtime and licenses are not part of execution.

If CUDA appears unavailable only inside a sandboxed command, recheck outside
that sandbox before changing GPU drivers or packages. The validation profile
requires the configured device and rejects silent CPU fallback.
