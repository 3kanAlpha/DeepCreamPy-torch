# DeepCreamPy inpainting models

English | [日本語](./MODELS.ja.md)

DeepCreamPy uses a PEPSI-derived convolutional encoder–decoder to fill masked regions in an RGB image. A contextual attention module at the bottleneck matches and combines feature patches from the same crop to help synthesize the missing content. The PyTorch port runs the attention-based inpainting path in one forward pass. [PyTorch model](../src/model.py) · [PEPSI paper (Sagong et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.html)

## Bar and mosaic checkpoints

Both modes instantiate the same `InpaintModel` and load different pretrained weights. Mask detection starts from exact mask-color pixels and absorbs up to two rings of connected anti-aliased fringe pixels near the mask color. The application crops each connected marked region with its surrounding context and fits the crop into a 256 × 256 input without stretching it. RGB uses mask-aware resampling (`BOX` when downscaling, valid-pixel normalized `BILINEAR` when upscaling) so unmasked lineart stays smooth without bleeding mask colors. When the binary mask shrinks, a sample is marked for filling if it contains any marked source pixels. The application reflects the image and mask together to fill padding and to give marks at the image edge extra context. The encoder receives six channels formed by concatenating RGB normalized to `[-1, 1]` with three copies of the mask, where 1 marks pixels to keep and 0 marks pixels to fill. [Crop preparation](../src/libs/utils.py) · [Model loading](../src/model.py)

| Mode | Checkpoint | RGB values inside the mask |
| --- | --- | --- |
| Bar | `models/bar/inpaint.pt` | Replaced with black before normalization, yielding `-1`. |
| Mosaic | `models/mosaic/inpaint.pt` | Taken from the matching image that still contains the mosaic. The color-marked copy supplies the mask. |

## Network architecture

The generator has 20 convolutions and 3,656,947 parameters, including biases. Shapes below use channels × height × width for one input crop. [Layer definitions](../src/model.py)

| Stage | Operations | Output shape |
| --- | --- | --- |
| Encoder | One 5 × 5 convolution followed by five 3 × 3 convolutions. Output channels are 32, 64, 64, 128, 128, and 256. Convolutions 2, 4, and 6 use stride 2. | 256 × 32 × 32 |
| Dilated encoder | Four 3 × 3 convolutions with dilation rates 2, 4, 8, and 16 expand the receptive field at the same resolution. | 256 × 32 × 32 |
| Contextual attention | Reconstructs features inside the mask and concatenates the resulting feature map with the encoder output. A 1 × 1 convolution projects the 512 concatenated channels to 256. | 256 × 32 × 32 |
| Decoder | Four pairs of 3 × 3 convolutions with 128, 64, 32, and 16 output channels per pair. The first three pairs each end with 2× nearest-neighbor upsampling. | 16 × 256 × 256 |
| RGB output | One zero-padded 3 × 3 convolution followed by clipping to `[-1, 1]`. | 3 × 256 × 256 |

The encoder and hidden decoder convolutions use reflection padding and ELU activations. The 1 × 1 context projection also uses ELU. [Forward pass](../src/model.py)

### Contextual attention and compositing

The attention module downsamples the first mask channel to 32 × 32 using nearest-neighbor sampling. It zeros out masked locations in a copy of the encoder features before extracting 3 × 3 candidate patches. It compares these with query patches from the encoder feature map using squared Euclidean distance. For each query, it standardizes the distances across candidates, applies `-50 * tanh(standardized_distance)`, and takes a softmax over candidates. It then folds the weighted patches back into a feature map and uses them to fill masked feature locations before the 1 × 1 projection. [Contextual attention implementation](../src/model.py)

The forward pass combines the decoded RGB with the source crop using `prediction * (1 - mask) + source * mask`. By default during inference, PEPSI runs 4-view test-time augmentation (original, horizontal flip, and a `(2, 2)` pixel reflection-shift pair) and averages the aligned raw predictions (`--no-tta` or the GUI checkbox disables this). The application then resizes the raw decoded RGB with bicubic sampling, applies a screened harmonic boundary residual correction to reduce seam steps against the surrounding pixels, and projects monochrome contexts back onto the luminance axis to suppress stray color tints before converting to 8-bit RGB. It pastes only the originally marked pixels of the current region. Across a one-pixel gap, the application uses the average of the two intact neighbors when their colors agree closely, so thin gaps follow nearby lines and fills. The filled pixels are estimates generated from the input and surrounding context. [Model compositing](../src/model.py) · [Image compositing](../src/decensor.py)

## Training graph and weight conversion

The original TensorFlow graph uses a shared encoder and two passes through a shared decoder. The coarse pass decodes the encoder features directly and supplies an auxiliary L1 reconstruction loss. The inpainting pass decodes the features refined by contextual attention and uses both L1 reconstruction and an adversarial loss from the Region Ensemble Discriminator (RED). [DeepCreamPy 2.2.0 Beta TensorFlow graph](https://github.com/Deepshift/DeepCreamPy/blob/2.2.0-beta/model.py) · [PEPSI paper, Section 3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sagong_PEPSI__Fast_Image_Inpainting_With_Parallel_Decoding_Network_CVPR_2019_paper.pdf)

The conversion script reads the pretrained TensorFlow encoder (`G_en`), context projection (`CB1/ML`), and decoder (`G_de`) weights and biases. It rearranges convolution kernels from TensorFlow's height × width × input channels × output channels layout to PyTorch's output channels × input channels × height × width layout and saves one state dictionary per mode. [Conversion script](../src/convert_models.py) · [Model setup](INSTALLATION.md#prepare-the-models)

## LaMa model (bar mode only)

In bar mode, you can also select a pure PyTorch implementation of **LaMa (`FFCResNetGenerator`)** built with Fast Fourier Convolutions (FFCs). [LaMa implementation](../src/lama.py) · [LaMa paper (Suvorov et al., WACV 2022)](https://arxiv.org/abs/2109.07161)

* **Checkpoint**: Loads `models/lama/inpaint.pt` by default (or a custom file via `--bar-weights`). Both Big-LaMa (18 FFC blocks, ~51M parameters) and standard LaMa (9 blocks, ~27M parameters) are inferred automatically from the state dictionary.
* **Preprocessing**: Crops larger than 512 px on the long side are downscaled with `LANCZOS` (crops at or below 512 px are not upscaled) and reflection-padded to multiples of 8 alongside the mask. The generator receives 4 channels: `[0, 1]` RGB with masked pixels zeroed out, concatenated with a 1-channel mask where `1` marks pixels to fill and `0` marks pixels to keep. [LaMa crop preparation](../src/libs/utils.py) · [LaMa runner](../src/model.py)
* **Weight conversion**: Existing pretrained checkpoints (TorchScript `.pt`, PyTorch/Lightning `.ckpt`, or `.safetensors`) can be converted to `models/lama/inpaint.pt` with the conversion helper. [LaMa converter](../src/convert_lama.py)

```powershell
uv run --locked python src/convert_lama.py path/to/big-lama.pt
```

For `.safetensors`, pass `--with safetensors`. Select LaMa from the CLI with `--model lama` or in the GUI under the **Model** group (disabled when mosaic mode is selected).

## Anima model + 4-channel Inpaint ControlNet-LLLite (bar mode, GPU only)

When running on a CUDA-capable NVIDIA GPU with the optional `anima` dependency group installed, you can select **Anima (`Anima-Base` + 4ch Inpaint ControlNet-LLLite)** in bar mode. [Anima engine](../src/anima.py) · [ControlNet-LLLite adapter](../src/anima_lllite.py)

* **Dependencies**: Install optional packages with `uv sync --locked --group anima` (not required for PEPSI/LaMa).
* **Offline weights layout**: Place standalone ComfyUI-compatible `.safetensors` files under `models/anima/` (model weights are **not** bundled due to Anima's non-commercial license):
  * `models/anima/diffusion/` — `Anima-Base` diffusion checkpoint (e.g. `anima-base-v1.0.safetensors`)
  * `models/anima/text_encoder/` — Qwen3-0.6B base text encoder (e.g. `qwen_3_06b_base.safetensors`)
  * `models/anima/vae/` — Qwen-Image VAE (e.g. `qwen_image_vae.safetensors`)
  * `models/anima/controlnet/` — 4-channel inpaint ControlNet-LLLite weights (e.g. `anima-lllite-inpainting-v2.safetensors`)
* **Offline tokenizers**: Lightweight Qwen3 and T5 tokenizer definitions are bundled under `assets/anima_tokenizers/` so inference runs completely offline with no network access.
* **VRAM management (8 GB target)**: Runs in `bfloat16` (or `float16` on older GPUs) with sequential component offloading (Qwen3 -> `llm_adapter` -> DiT+LLLite -> VAE), sequential CFG (`batch_size=1`), VAE tiling/slicing, and automatic block-level CPU offloading when total GPU VRAM is 10.5 GiB or less.
* **Inpainting & crop grouping**: Nearby masked regions are iteratively merged into broader context crops (or a single whole-image pass for medium images) aligned to multiples of 16. Denoising starts from pure noise (`denoise=1.0`) guided by 4-channel LLLite conditioning plus step-wise known-region latent preservation, then pastes only masked pixels back with boundary and monochrome correction. Variations use distinct RNG seeds rather than image flipping.
* **Prompts & tuning**: Leave the positive prompt blank to preserve the original art style, or add optional hints (e.g. body part or `@artist`). The negative prompt defaults to `censored, bar censor, mosaic censoring` and can be freely edited or cleared. Select via GUI (**Model -> Anima**) or CLI (`--model anima --anima-prompt "..." --anima-steps 28 --anima-cfg 4.0`).
