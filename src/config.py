import argparse
import sys
from pathlib import Path

from anima import (
  ANIMA_DEFAULT_CACHE_PROMPT,
  ANIMA_DEFAULT_CFG,
  ANIMA_DEFAULT_CFG_SKIP_TAIL,
  ANIMA_DEFAULT_LLLITE_STRENGTH,
  ANIMA_DEFAULT_MAX_SIZE,
  ANIMA_DEFAULT_NEGATIVE_PROMPT,
  ANIMA_DEFAULT_SEED,
  ANIMA_DEFAULT_STEPS,
  ANIMA_MAX_CFG_SKIP_TAIL,
  anima_cuda_available,
  anima_dependencies_available,
)


def str2floatarr(v):
  if type(v) == str:
    try:
      return [float(v) for v in v.split(",")]
    except:
      raise argparse.ArgumentTypeError("Integers separated by commas expected.")
  else:
    raise argparse.ArgumentTypeError("Integers separated by commas expected.")


def str2bool(v):
  if v.lower() in ("yes", "true", "t", "y", "1", True):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0", False):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(*, cli=False):
  parser = argparse.ArgumentParser(description="")
  data_directory = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(".")
  )

  # Input output folders settings
  parser.add_argument(
    "--decensor_input_path",
    default=str(data_directory / "decensor_input"),
    help="directory of PNG images with green masks",
  )
  parser.add_argument(
    "--decensor_input_original_path",
    default=str(data_directory / "decensor_input_original"),
    help="directory of original images for mosaic mode",
  )
  parser.add_argument(
    "--decensor_output_path",
    default=str(data_directory / "decensor_output"),
    help="directory for the generated images",
  )
  parser.add_argument(
    "--clean-up-input-dirs",
    dest="clean_up_input_dirs",
    action="store_true",
    default=False,
    help="whether to delete all image files in the input directories when decensoring is done",
  )

  # Decensor settings
  parser.add_argument(
    "--mask_color_red", default=0, help="red channel of mask color in decensoring"
  )
  parser.add_argument(
    "--mask_color_green", default=255, help="green channel of mask color in decensoring"
  )
  parser.add_argument(
    "--mask_color_blue", default=0, help="blue channel of mask color in decensoring"
  )
  parser.add_argument(
    "--model",
    choices=("pepsi", "lama", "anima"),
    default="pepsi",
    help="inpainting model architecture to use (pepsi, lama, or anima)",
  )
  parser.add_argument(
    "--is_mosaic",
    type=str2bool,
    default="False",
    help="true if image has mosaic censoring, false otherwise",
  )
  parser.add_argument(
    "--variations",
    type=int,
    choices=[1, 2, 4],
    default=1,
    help="number of decensor variations to be generated",
  )
  parser.add_argument(
    "--tta",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="enable horizontal-flip and small-shift TTA for PEPSI (default: enabled)",
  )
  parser.add_argument(
    "--anima-prompt",
    default="",
    help="optional auxiliary positive prompt for Anima (e.g. body part or @artist)",
  )
  parser.add_argument(
    "--anima-negative-prompt",
    default=ANIMA_DEFAULT_NEGATIVE_PROMPT,
    help="negative prompt for Anima (pass empty string to leave blank)",
  )
  parser.add_argument(
    "--anima-steps",
    type=int,
    default=ANIMA_DEFAULT_STEPS,
    help="number of denoising steps for Anima",
  )
  parser.add_argument(
    "--anima-cfg",
    type=float,
    default=ANIMA_DEFAULT_CFG,
    help="CFG guidance scale for Anima",
  )
  parser.add_argument(
    "--anima-strength",
    type=float,
    default=ANIMA_DEFAULT_LLLITE_STRENGTH,
    help="ControlNet-LLLite strength multiplier for Anima",
  )
  parser.add_argument(
    "--anima-seed",
    type=int,
    default=ANIMA_DEFAULT_SEED,
    help="base RNG seed for Anima (-1 for random)",
  )
  parser.add_argument(
    "--anima-max-size",
    type=int,
    default=ANIMA_DEFAULT_MAX_SIZE,
    help="maximum crop/image side length for Anima (512..1536, multiple of 16)",
  )
  parser.add_argument(
    "--anima-fp8",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
      "store Anima DiT weights in FP8 (float8_e4m3fn) for lower VRAM and faster "
      "inference on 8GB GPUs (default: auto-enabled on <=10.5 GiB GPUs)"
    ),
  )
  parser.add_argument(
    "--anima-cache-prompt",
    action=argparse.BooleanOptionalAction,
    default=ANIMA_DEFAULT_CACHE_PROMPT,
    help="cache encoded Anima prompt embeddings in CPU RAM across runs (default: disabled)",
  )
  parser.add_argument(
    "--anima-cfg-skip-tail",
    type=float,
    default=ANIMA_DEFAULT_CFG_SKIP_TAIL,
    help=(
      "fraction of final denoising steps to run with CFG=1.0 "
      f"(0.0..{ANIMA_MAX_CFG_SKIP_TAIL:.1f}, default: 0.0; e.g. 0.2 skips final 20%%)"
    ),
  )
  parser.add_argument(
    "--anima-diffusion",
    default="",
    help="filename or path to Anima diffusion .safetensors checkpoint",
  )
  parser.add_argument(
    "--anima-controlnet",
    default="",
    help="filename or path to Anima 4ch Inpaint ControlNet-LLLite .safetensors checkpoint",
  )
  parser.set_defaults(bar_weights=None)
  if cli:
    parser.add_argument(
      "--bar-weights",
      type=Path,
      help="path to a custom inpaint.pt checkpoint for bar mode",
    )

  # Other settings
  parser.add_argument(
    "--ui_mode",
    default=False,
    help="true if you want ui mode, false if you want command line interface",
  )
  parser.add_argument(
    "--lang",
    default=None,
    help="override GUI display language code (e.g. en, ja)",
  )

  args = parser.parse_args()
  if args.model == "lama" and args.is_mosaic:
    parser.error("--model lama can only be used in bar mode")
  if args.model == "anima" and args.is_mosaic:
    parser.error("--model anima can only be used in bar mode")
  if cli and args.model == "anima":
    if not anima_dependencies_available():
      parser.error(
        "--model anima requires optional dependencies (`uv sync --locked --group anima`)"
      )
    if not anima_cuda_available():
      parser.error("--model anima requires a CUDA-capable NVIDIA GPU")
  if not (0.0 <= args.anima_cfg_skip_tail <= ANIMA_MAX_CFG_SKIP_TAIL):
    parser.error(
      f"--anima-cfg-skip-tail must be between 0.0 and {ANIMA_MAX_CFG_SKIP_TAIL:.1f}"
    )
  if args.bar_weights is not None and args.is_mosaic:
    parser.error("--bar-weights can only be used in bar mode")
  return args


if __name__ == "__main__":
  get_args()
