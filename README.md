# Transitional Grid Map (TGM)

Probabilistic multi‑layer occupancy mapping for static, dynamic, and noise layers from LiDAR. Includes baseline filters (ROR/SOR/DROR/DSOR), SLAM integration, and visualization utilities.

## Quickstart

**Supported OS**: Windows 11 (tested), compatible with Linux/Ubuntu with appropriate CUDA setup. Python: 3.12.

**Note**: The following instructions are Windows-specific. Linux users should adapt commands accordingly (e.g., use `source .venv/bin/activate` instead of PowerShell activation).

### Clone
```powershell
git clone https://github.com/josemnl/tgm.git
cd TGMp
```

### Create venv and install package
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
python -m pip install -e .
```

Optional extras:

```powershell
# CUDA 12.x
python -m pip install -e ".[gpu-cuda12]"

# CUDA 11.x
python -m pip install -e ".[gpu-cuda11]"
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

## Config

- Configure via `config/config.yaml` (and experiment-specific YAML like `config/snowyKitti.yaml`):
	- `isGPU: true|false` — GPU acceleration (requires CuPy) or CPU mode
	- LiDAR paths (`lidarPath`, `labelPath`, `posePath`)
	- Video/output settings (`saveVideo`, `saveSvg`, `videoSection`)

## Repository Layout

- `src/tgm/`: core library package (`import tgm`)
- `examples/`: runnable examples showing how to use the package
- `scripts/`: utility scripts (plotting, conversion, ad-hoc tooling)
- `config/`: experiment and dataset config files
- `data/`: input datasets (typically not versioned)
- `results/`: generated outputs

## Run

Run the main processing loop example (generates TGM maps and applies snow filters):
```powershell
python .\examples\snowRunLoop.py
```

Generate analysis plots and metrics from results:
```powershell
python .\scripts\snowPlotResults.py
```
This produces detailed plots, sensitivity analysis, and summary tables from the data in `./results/`.

You can also run other examples directly:

```powershell
python .\examples\run.py
python .\examples\run3D.py
python .\examples\run-test-cardinality.py
```

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

- **CuPy is optional** and exposed via extras in `pyproject.toml` (`gpu-cuda12`, `gpu-cuda11`), or can be installed manually with a matching wheel.
- The code automatically falls back to CPU mode if CuPy is unavailable, even when `isGPU: true` is set.
- For CI or headless servers, use CPU mode (`isGPU: false`) or ensure CUDA DLLs are on PATH before imports.

## How to Cite

If you use this code in academic work, please cite:

J. M. Gaspar Sanchez, L. Bruns, J. Tumova, P. Jensfelt and M. Torngren, "Transitional Grid Maps: Joint Modeling of Static and Dynamic Occupancy," in IEEE Open Journal of Intelligent Transportation Systems, vol. 6, pp. 1-10, 2025, doi: 10.1109/OJITS.2024.3521449.

BibTeX:

```bibtex
@ARTICLE{10813430,
	author={Gaspar Sanchez, Jose Manuel and Bruns, Leonard and Tumova, Jana and Jensfelt, Patric and Torngren, Martin},
	journal={IEEE Open Journal of Intelligent Transportation Systems},
	title={Transitional Grid Maps: Joint Modeling of Static and Dynamic Occupancy},
	year={2025},
	volume={6},
	number={},
	pages={1-10},
	keywords={Vehicle dynamics;Bayes methods;Hidden Markov models;Time measurement;Simultaneous localization and mapping;Radar tracking;Heuristic algorithms;Current measurement;Computational modeling;Noise measurement;Grid map;Bayesian inference;SLAM},
	doi={10.1109/OJITS.2024.3521449}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
