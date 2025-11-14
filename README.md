# Transitional Grid Map for adverse weather conditions (TGMw)

Probabilistic multi‑layer occupancy mapping for static, dynamic, and weather (snow) layers from LiDAR. Includes baseline filters (ROR/SOR/DROR/DSOR), SLAM integration, and visualization utilities.

## Quickstart

**Supported OS**: Windows 11 (tested), compatible with Linux/Ubuntu with appropriate CUDA setup. Python: 3.12.

**Note**: The following instructions are Windows-specific. Linux users should adapt commands accordingly (e.g., use `source .venv/bin/activate` instead of PowerShell activation).

### Clone
```powershell
git clone https://github.com/josemnl/TGMw.git
cd TGMp
```

### Create venv and install deps
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### GPU (recommended) vs CPU
- **GPU mode** requires NVIDIA driver with CUDA 12.x support and CUDA Toolkit 12.8 installed (Math Libraries), plus CuPy.
- **CPU mode** works without CUDA/CuPy (slower). Set `isGPU: false` in your config and skip GPU Setup steps.

## GPU Setup (Windows)

1) Verify driver
```powershell
nvidia-smi
```

2) Install CUDA Toolkit 12.8 (GUI)
- Download NVIDIA CUDA Toolkit 12.8 for Windows.
- Choose Custom → include Math Libraries: cuBLAS, cuSOLVER, cuFFT, cuSPARSE (cuRAND is typically included).

3) Persist environment (then open a NEW PowerShell)
```powershell
[Environment]::SetEnvironmentVariable('CUDA_PATH','C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8','User')
$old = [Environment]::GetEnvironmentVariable('Path','User')
$bin = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin'
if ($old -notmatch [regex]::Escape($bin)) { [Environment]::SetEnvironmentVariable('Path',"$bin;$old",'User') }
```

4) Install CuPy matching CUDA
```powershell
pip install -U cupy-cuda12x
```

5) Sanity checks
```powershell
python -c "from cupy_backends.cuda.libs import cublas; print('cublas OK')"
python -c "import cupy, cupy.cuda.runtime as rt; print('devices', rt.getDeviceCount())"
python -c "import matplotlib; matplotlib.use('Qt5Agg'); import matplotlib.pyplot as plt; print(plt.get_backend())"
```

## Dataset & Config

- Download and extract the SnowyKITTI dataset to the repository root (the first point cloud is expected at `./snowyKITTI/dataset/sequences/00/snow_velodyne/000000.bin`).
- Configure via `config/config.yaml` (and experiment-specific YAML like `config/snowyKitti.yaml`):
	- `isGPU: true|false` — GPU acceleration (requires CuPy) or CPU mode
	- LiDAR paths (`lidarPath`, `labelPath`, `posePath`)
	- Video/output settings (`saveVideo`, `saveSvg`, `videoSection`)
	- Snow filter parameters (ROR/SOR/DROR/DSOR: `k`, `r`, `s`, `rho`)

## Run

Run the main processing loop (generates TGM maps and applies snow filters):
```powershell
python .\src\snowRunLoop.py
```

Generate analysis plots and metrics from results:
```powershell
python .\src\snowPlotResults.py
```
This produces detailed plots, sensitivity analysis, and summary tables from the data in `./results/`.

**Note**: If config has `isGPU: true` but CuPy is not installed, the code will automatically fall back to CPU mode with a warning.

## Troubleshooting

- ImportError: DLL load failed (cublas)
	- Ensure CUDA 12.8 is installed with cuBLAS/cuSOLVER/cuFFT/cuSPARSE.
	- Check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\cublas64_12.dll` exists.
	- Confirm `CUDA_PATH` and the `...\v12.8\bin` folder are in PATH. Open a new shell after setting env vars.
	- Install CuPy wheel matching your CUDA major: `cupy-cuda12x` for CUDA 12.x.

- cudaErrorInsufficientDriver
	- Driver too old for runtime. Update NVIDIA driver or install matching CuPy (e.g., `cupy-cuda11x` if you only have CUDA 11.x).

- Matplotlib Qt error (PyQt5 missing)
	- `pip install "PyQt5>=5.15.11"` (GUI plots). For headless batch runs use Agg backend or disable plotting in config.

- OpenCV vs NumPy pinning
	- If `opencv-python` complains, stick to `numpy<2.3.0` (2.2.x is known good).

## Known‑Good Setup (tested)

- Windows 11, Python 3.12.10
- NVIDIA driver supporting CUDA 12.8
- CUDA Toolkit 12.8 (Math Libraries installed)
- CuPy 13.6.0 (`cupy-cuda12x`)
- NumPy 2.2.6, Matplotlib 3.10.x, PyQt5 ≥ 5.15.11, SciPy 1.16.x

## Notes

- **CuPy is not in `requirements.txt`** — users must install the correct wheel for their CUDA version (e.g., `cupy-cuda12x` for CUDA 12.x).
- The code automatically falls back to CPU mode if CuPy is unavailable, even when `isGPU: true` is set.
- For CI or headless servers, use CPU mode (`isGPU: false`) or ensure CUDA DLLs are on PATH before imports.
