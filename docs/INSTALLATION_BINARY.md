# Windows x64 distribution

## Run the application

1. Extract `DeepCreamPy-Windows-x64.7z` into a writable folder. Keep `DeepCreamPy.exe` and its `_internal` directory together.
2. Put your green-marked PNGs in `decensor_input`. For mosaic mode, put the matching unmarked images in `decensor_input_original`.
3. Open `DeepCreamPy.exe`, choose the censor type, and click **Decensor Your Images**. Results appear in `decensor_output` beside the EXE.

The bundle includes Python, PySide6, PyTorch, and both converted PEPSI models (`bar` and `mosaic`). It uses CUDA on a compatible NVIDIA GPU with a suitable driver and starts on the CPU when CUDA is unavailable.

### Optional LaMa model (bar mode only)

The Windows distribution does not bundle a LaMa checkpoint. To enable **LaMa** in the GUI, place a converted PyTorch `state_dict` checkpoint at `models/lama/inpaint.pt` beside `DeepCreamPy.exe`, then restart the application. When the file is absent, the **LaMa** option is disabled in the GUI.

## Build the distribution

On Windows x64, prepare the two converted PEPSI model files (`models/bar/inpaint.pt` and `models/mosaic/inpaint.pt`) as described in the [source installation guide](INSTALLATION.md#prepare-the-models), then run this command from the repository root.

```powershell
uv run --locked --group windows-build python src/build_windows.py
```

The build creates `dist/DeepCreamPy/DeepCreamPy.exe` (with an empty `models/lama/` folder and without bundling `models/lama/inpaint.pt`). If NanaZip (`NanaZipC`) is available on `PATH` or in `%LOCALAPPDATA%\Microsoft\WindowsApps`, it also creates `dist/DeepCreamPy-Windows-x64.7z` at compression level 7 (`-mx=7`). If `NanaZipC` is not found, the build prints a warning and outputs only `dist/DeepCreamPy/`. Move an existing distribution out of `dist` before rebuilding.

## Legacy releases

The original binaries remain in the [upstream release archive](https://github.com/Deepshift/DeepCreamPy/releases).
