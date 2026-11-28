# Installation

## Run from source

This project targets Windows x64 with Python 3.13. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run these commands from the repository root.

```powershell
uv sync --locked
```

To enable the optional GPU-only **Anima (+ ControlNet-LLLite)** model, include the `anima` dependency group:

```powershell
uv sync --locked --group anima
```

The lockfile uses the official PyTorch CUDA 13.2 build. With a compatible NVIDIA GPU and driver, the application uses CUDA automatically. The application reports its device and starts on the CPU if CUDA is unavailable or fails to initialize. An error during inference stops the current job.

## Prepare the models

Download the [original TensorFlow model archive](https://drive.google.com/open?id=1IMwzqZUuRnTv5jcuKdvZx-RZweknww5x) if you do not already have both checkpoints. Extract it so each of `models/bar/` and `models/mosaic/` contains its `checkpoint`, `.index`, and `.data-*` files.

Run the conversion once after placing or replacing the original models.

```powershell
uv run --locked --with tensorflow==2.21.0 python src/convert_models.py
```

This writes `models/bar/inpaint.pt` and `models/mosaic/inpaint.pt` and preserves the original checkpoints. TensorFlow is installed only in the temporary environment for this command.

## Start DeepCreamPy

Follow the [image preparation guide](USAGE.md), then use one of these commands.

```powershell
uv run --locked python src/main.py
uv run --locked python src/decensor.py
uv run --locked python src/decensor.py --is_mosaic=true
```

The first command opens the GUI. The second runs the CLI in bar mode. The third runs the CLI in mosaic mode. Use `uv run --locked python src/decensor.py --help` to see the available CLI options.

## Windows distribution

See the [Windows distribution guide](INSTALLATION_BINARY.md) to run the bundled GUI or build a distribution 7z archive from source.
