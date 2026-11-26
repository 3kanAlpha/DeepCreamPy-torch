import argparse
import sys
from pathlib import Path


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
    choices=("pepsi", "lama"),
    default="pepsi",
    help="inpainting model architecture to use (pepsi or lama)",
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

  args = parser.parse_args()
  if args.model == "lama" and args.is_mosaic:
    parser.error("--model lama can only be used in bar mode")
  if args.bar_weights is not None and args.is_mosaic:
    parser.error("--bar-weights can only be used in bar mode")
  return args


if __name__ == "__main__":
  get_args()
