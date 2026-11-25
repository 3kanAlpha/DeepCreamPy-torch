# Troubleshooting Install

## Running from source

### The model file is missing

Place the original TensorFlow checkpoints in `models/bar/` and `models/mosaic/`, then follow the [model conversion instructions](INSTALLATION.md#prepare-the-models).

### DeepCreamPy starts on the CPU

The application displays the device and reason when CUDA is unavailable or fails to initialize. To use the GPU, check that Windows detects the NVIDIA GPU and that the installed driver supports the CUDA 13.2 build in the [installation instructions](INSTALLATION.md).

### A processing job stops with an error

Check the message in the progress pane or terminal. Fix the reported issue and start a new job.

## Running a legacy binary

Use the [legacy binary installation guide](INSTALLATION_BINARY.md) for previously published Windows x64 releases.
