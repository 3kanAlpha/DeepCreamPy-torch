# Windows x64 distribution

## Run the application

1. Extract `DeepCreamPy-Windows-x64.zip` into a writable folder. Keep `DeepCreamPy.exe` and its `_internal` directory together.
2. Put your green-marked PNGs in `decensor_input`. For mosaic mode, put the matching unmarked images in `decensor_input_original`.
3. Open `DeepCreamPy.exe`, choose the censor type, and click **Decensor Your Images**. Results appear in `decensor_output` beside the EXE.

The bundle includes Python, PySide6, PyTorch, and both converted models. It uses CUDA on a compatible NVIDIA GPU with a suitable driver and starts on the CPU when CUDA is unavailable.

## Build the distribution

On Windows x64, prepare the two converted model files as described in the [source installation guide](INSTALLATION.md#prepare-the-models), then run this command from the repository root.

```powershell
uv run --locked --group windows-build python src/build_windows.py
```

The build creates `dist/DeepCreamPy/DeepCreamPy.exe` and `dist/DeepCreamPy-Windows-x64.zip`. Move an existing distribution out of `dist` before rebuilding.

## Legacy releases

The original binaries remain in the [upstream release archive](https://github.com/Deepshift/DeepCreamPy/releases).
