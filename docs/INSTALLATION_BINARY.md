# Windows x64 distribution

## Run the application

1. Extract `DeepCreamPy-Windows-x64.7z` into a writable folder. Keep `DeepCreamPy.exe` and its `_internal` directory together.
2. Put your green-marked PNGs in `decensor_input`. For mosaic mode, put the matching unmarked images in `decensor_input_original`.
3. Open `DeepCreamPy.exe`, choose the censor type, and click **Decensor Your Images**. Results appear in `decensor_output` beside the EXE.

The bundle includes Python, PySide6, PyTorch, and both converted PEPSI models (`bar` and `mosaic`). It uses CUDA on a compatible NVIDIA GPU with a suitable driver and starts on the CPU when CUDA is unavailable.

### Optional LaMa model (bar mode only)

The Windows distribution does not bundle a LaMa checkpoint. To enable **LaMa** in the GUI, place a converted PyTorch `state_dict` checkpoint at `models/lama/inpaint.pt` beside `DeepCreamPy.exe`, then restart the application. When the file is absent, the **LaMa** option is disabled in the GUI.

### Optional Anima + ControlNet-LLLite model (bar mode, GPU only)

Anima weights are never bundled in Windows distributions due to their non-commercial license. When using an Anima-enabled build (`DeepCreamPy-Anima-Windows-x64.7z`) on a CUDA-capable NVIDIA GPU, place standalone ComfyUI-compatible `.safetensors` files into these folders beside `DeepCreamPy.exe`, then restart the application:

* `models/anima/diffusion/` (e.g. `anima-base-v1.0.safetensors`)
* `models/anima/text_encoder/` (e.g. `qwen_3_06b_base.safetensors`)
* `models/anima/vae/` (e.g. `qwen_image_vae.safetensors`)
* `models/anima/controlnet/` (e.g. `anima-lllite-inpainting-v2.safetensors`)

## Build the distribution

On Windows x64, prepare the two converted PEPSI model files (`models/bar/inpaint.pt` and `models/mosaic/inpaint.pt`) as described in the [source installation guide](INSTALLATION.md#prepare-the-models), then run one of these commands from the repository root.

Standard lightweight build (excludes Anima dependencies):

```powershell
uv run --locked --group windows-build python src/build_windows.py
```

Anima-enabled build (bundles `diffusers`/`transformers`/`safetensors` and offline Qwen3/T5 tokenizers, and creates empty `models/anima/*` folders without bundling any Anima weights):

```powershell
uv run --locked --group windows-build --group anima python src/build_windows.py --with-anima
```

The standard build creates `dist/DeepCreamPy/DeepCreamPy.exe` and `dist/DeepCreamPy-Windows-x64.7z`. The `--with-anima` build emits `dist/DeepCreamPy-Anima-Windows-x64.7z`. If NanaZip (`NanaZipC`) is available on `PATH` or in `%LOCALAPPDATA%\Microsoft\WindowsApps`, the 7z archive is compressed at level 7 (`-mx=7`). If `NanaZipC` is not found, the build prints a warning and outputs only `dist/DeepCreamPy/`. Move an existing distribution out of `dist` before rebuilding.

## Legacy releases

The original binaries remain in the [upstream release archive](https://github.com/Deepshift/DeepCreamPy/releases).
