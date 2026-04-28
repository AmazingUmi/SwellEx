# Local Toolchain and Environment

## Install

Use a Python environment with PyTorch, NumPy, h5py, and Matplotlib:

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
C:\Users\user\.conda\envs\pytorch\python.exe -m py_compile scripts_py\Network_main.py scripts_py\network\training.py
```

When Codex runs local MATLAB checks on `TouJI`, prefer the MATLAB executable
above instead of the default `matlab`, for example:

```powershell
& "D:\Matlab\bin\matlab.exe" -batch "disp('Signals_Segmentation'); checkcode('scripts_matlab/Signals_Segmentation.m');"
```

## HaiQin1 Local Toolchain

On the local machine identified as `HaiQin1`, use these explicit executables for
Codex-run debugging and smoke tests.

Python/PyTorch:

```text
G:\software\Anaconda\envs\pytorch\python.exe
```

MATLAB:

```text
C:\Program Files\MATLAB\R2025b\bin\matlab.exe
```

Quick Python environment check:

```powershell
G:\software\Anaconda\envs\pytorch\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

When Codex runs local Python tests on `HaiQin1`, prefer this interpreter instead
of the default `python`, for example:

```powershell
G:\software\Anaconda\envs\pytorch\python.exe -m py_compile scripts_py\Network_main.py scripts_py\network\training.py
```

When Codex runs local MATLAB checks on `HaiQin1`, prefer the MATLAB executable
above instead of the default `matlab`, for example:

```powershell
& "C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "disp('Signals_Segmentation'); checkcode('scripts_matlab/Signals_Segmentation.m');"
```
