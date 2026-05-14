# Local Toolchain and Environment

## Install

Use a Python environment with PyTorch, NumPy, h5py, and Matplotlib.

On Linux/macOS:

```bash
python -m pip install -r scripts_py/requirements.txt
```

On Windows PowerShell:

```powershell
pip install -r ".\scripts_py\requirements.txt"
```

Pick the PyTorch install command that matches your CUDA version if you want GPU
training.

## TouJI Local Toolchain

On the local machine identified as `TouJI`, use these explicit paths for
Codex-run debugging and smoke tests.

Python/PyTorch environment:

```text
C:\Users\user\.conda\envs\pytorch
```

Python executable:

```text
C:\Users\user\.conda\envs\pytorch\python.exe
```

MATLAB:

```text
D:\Matlab\bin\matlab.exe
```

Quick Python environment check:

```powershell
C:\Users\user\.conda\envs\pytorch\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

When Codex runs local Python tests on `TouJI`, prefer this interpreter instead
of the default `python`, for example:

```powershell
C:\Users\user\.conda\envs\pytorch\python.exe -m py_compile scripts_py\RBD_method\Network_main.py scripts_py\ELM_method\Network_main.py scripts_py\SCM_method\Network_main.py
```

After shared Python utilities were introduced, include `scripts_py\common` in
broader compile checks:

```powershell
C:\Users\user\.conda\envs\pytorch\python.exe -m py_compile scripts_py\common\*.py scripts_py\RBD_method\Network_main.py scripts_py\ELM_method\Network_main.py scripts_py\SCM_method\Network_main.py
```

When Codex runs local MATLAB checks on `TouJI`, prefer the MATLAB executable
above instead of the default `matlab`, for example:

```powershell
& "D:\Matlab\bin\matlab.exe" -batch "checkcode('scripts_matlab/RBD_method/Signals_Segmentation.m'); checkcode('scripts_matlab/ELM_method/Signals_Segmentation.m'); checkcode('scripts_matlab/SCM_method/Signals_Segmentation.m');"
```

## HaiQin1 Local Toolchain

On the local machine identified as `HaiQin1`, use these explicit executables for
Codex-run debugging and smoke tests. `HaiQin1` is now a Linux machine; do not
use the old Windows `G:\...` or `C:\Program Files\...` paths for this host.

Project root:

```text
/home/yiyang-lu/project/SwellEx
```

Python/PyTorch environment:

```text
/home/yiyang-lu/miniforge3/envs/pytorch
```

Python executable:

```text
/home/yiyang-lu/miniforge3/envs/pytorch/bin/python
```

Interactive shell setup:

```bash
conda activate pytorch
```

MATLAB:

```text
/usr/local/bin/matlab
```

MATLAB target:

```text
/usr/local/MATLAB/R2025b/bin/matlab
```

Quick Python environment check:

```bash
conda activate pytorch
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

The activated `pytorch` shell on `HaiQin1` currently reports Python `3.14.4`,
PyTorch `2.11.0+cu128`, and `torch.cuda.is_available()` as `True`. If a
non-interactive `conda run -n pytorch ...` or direct absolute-path Python call
reports CUDA as unavailable, re-check inside an activated interactive shell
before treating CUDA as broken.

Codex sandbox note: GPU device access may be hidden inside the default sandbox.
If `nvidia-smi` fails with a driver communication error or PyTorch reports
`torch.cuda.is_available()` as `False` only inside a sandboxed command, rerun
the GPU check outside the sandbox before changing CUDA or PyTorch settings.

Confirmed host GPU:

```text
NVIDIA RTX A5000, driver 595.58.03
```

When Codex runs local Python tests on `HaiQin1`, prefer this interpreter instead
of the default `python`, for example:

```bash
conda activate pytorch
python -m py_compile scripts_py/RBD_method/Network_main.py scripts_py/ELM_method/Network_main.py scripts_py/SCM_method/Network_main.py
```

After shared Python utilities were introduced, include `scripts_py/common` in
broader compile checks:

```bash
conda activate pytorch
python -m py_compile scripts_py/common/*.py scripts_py/RBD_method/Network_main.py scripts_py/ELM_method/Network_main.py scripts_py/SCM_method/Network_main.py
```

When Codex runs local MATLAB checks on `HaiQin1`, prefer the MATLAB executable
above instead of the default `matlab`, for example:

```bash
/usr/local/bin/matlab -batch "checkcode('scripts_matlab/RBD_method/Signals_Segmentation.m'); checkcode('scripts_matlab/ELM_method/Signals_Segmentation.m'); checkcode('scripts_matlab/SCM_method/Signals_Segmentation.m');"
```
