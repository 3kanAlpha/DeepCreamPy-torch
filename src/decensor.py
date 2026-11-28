import gc
import os
import random
import sys
import threading

import numpy as np
import torch
from PIL import Image
from PySide6 import QtCore

import config
import file
from libs.utils import (
  bridge_thin_gaps,
  connected_regions,
  expand_bounding,
  group_anima_regions,
  mask_pixels,
  prepare_crop,
  resize_prediction,
)
from model import AnimaCancelledError, AnimaSettings, create_inpainter

"""
to allow decensoring in CLI, ignore all methods targeting Qt signals,
when they aren't initialized (i.e. when the class isn't created by the GUI)
"""


class ProcessingCancelledError(RuntimeError):
  """Raised when decensoring is cancelled mid-run by the user."""


class IgnoreAll(object):
  def __getattr__(self, attr):
    return IgnoreAll()

  def __call__(self, *args, **kwargs):
    return IgnoreAll()


class Decensor(QtCore.QThread):
  single_image_finished = QtCore.Signal(list, list)
  single_image_failed = QtCore.Signal(str)
  single_image_cancelled = QtCore.Signal()

  def __init__(
    self,
    parentThread=None,
    text_edit=None,
    text_cursor=None,
    ui_mode=None,
    *,
    cli=False,
  ):
    super().__init__(parentThread)
    args = config.get_args(cli=cli)
    self.model_name = args.model
    self.is_mosaic = args.is_mosaic
    self.bar_weights = args.bar_weights
    self.variations = args.variations
    self.use_tta = args.tta
    self.anima_settings = AnimaSettings(
      positive_prompt=args.anima_prompt,
      negative_prompt=args.anima_negative_prompt,
      steps=args.anima_steps,
      cfg_scale=args.anima_cfg,
      lllite_strength=args.anima_strength,
      seed=args.anima_seed,
      max_size=args.anima_max_size,
      use_fp8_storage=args.anima_fp8,
      cache_prompt_embeds=args.anima_cache_prompt,
      cfg_skip_tail=args.anima_cfg_skip_tail,
      diffusion_file=args.anima_diffusion,
      controlnet_file=args.anima_controlnet,
    ).normalized()
    self.mask_color = [
      args.mask_color_red / 255.0,
      args.mask_color_green / 255.0,
      args.mask_color_blue / 255.0,
    ]
    self.decensor_input_path = args.decensor_input_path
    self.decensor_input_original_path = args.decensor_input_original_path
    self.decensor_output_path = args.decensor_output_path
    self.clean_up_input_dirs = args.clean_up_input_dirs

    self.signals = IgnoreAll()  # Signals class will be given by progressWindow

    self.model = None
    self.loaded_config = None
    self.load_only = False
    self.single_job = None
    self._cancel_event = threading.Event()
    self.files_removed = None

    if not os.path.exists(self.decensor_output_path):
      os.makedirs(self.decensor_output_path)

  def request_cancel(self):
    self._cancel_event.set()

  def reset_cancel(self):
    self._cancel_event.clear()

  def is_cancelled(self) -> bool:
    return self._cancel_event.is_set()

  def _check_cancelled(self):
    if self._cancel_event.is_set():
      raise ProcessingCancelledError("Processing cancelled by user.")

  def run(self):
    is_single_job = self.single_job is not None
    try:
      if self.load_only:
        self.load_model()
      elif is_single_job:
        ori_img, colored_img, output_stem = self.single_job
        results, saved_paths = self.decensor_single_image(
          ori_img, colored_img, output_stem
        )
        self.single_image_finished.emit(results, saved_paths)
      else:
        self.decensor_all_images_in_folder()
        self.do_post_jobs()
    except (ProcessingCancelledError, AnimaCancelledError):
      message = "Processing cancelled by user."
      print(message, flush=True)
      self.signals.insertText_progressCursor.emit(message)
      self.signals.update_statusLabel_Text.emit(message)
      if is_single_job:
        self.single_image_cancelled.emit()
    except Exception as error:
      message = f"Processing stopped: {error}"
      print(message, file=sys.stderr)
      self.signals.insertText_progressCursor.emit(message)
      self.signals.update_statusLabel_Text.emit(message)
      if is_single_job:
        self.single_image_failed.emit(str(error))
    finally:
      self.load_only = False
      self.single_job = None
      self._cancel_event.clear()

  def set_single_image_job(self, ori_img, colored_img, output_stem):
    self.single_job = (ori_img.copy(), colored_img.copy(), output_stem)

  def _resolve_image_base_seed(self) -> int | None:
    if self.model_name != "anima":
      return None
    if self.anima_settings.seed >= 0:
      return int(self.anima_settings.seed)
    return random.randint(0, 2**31 - 1)

  def decensor_single_image(self, ori_img, colored_img, output_stem):
    self._check_cancelled()
    self.load_model()
    self._check_cancelled()
    os.makedirs(self.decensor_output_path, exist_ok=True)
    self.signals.insertText_progressCursor.emit(f"Decensoring image: {output_stem}\n")
    results = []
    saved_paths = []
    source_img = ori_img if self.is_mosaic else colored_img
    base_seed = self._resolve_image_base_seed()
    for variant_number in range(self.variations):
      self._check_cancelled()
      var_prefix = (
        f"Var {variant_number + 1}/{self.variations} | " if self.variations > 1 else ""
      )
      seed = (base_seed + variant_number) if base_seed is not None else None
      output_img = self.decensor_image_variation(
        source_img,
        colored_img,
        variant_number,
        None,
        seed=seed,
        progress_prefix=var_prefix,
      )
      if output_img is None:
        raise ValueError(
          "No green mask regions detected! Please mark at least one region before decensoring."
        )
      save_name = f"{output_stem} {variant_number}.png"
      save_path = os.path.join(self.decensor_output_path, save_name)
      output_img.save(save_path)
      print(f"Decensored image saved to {save_path}!")
      self.signals.insertText_progressCursor.emit(
        f"Decensored image saved to {save_path}!"
      )
      self.signals.insertText_progressCursor.emit("=" * 30)
      results.append(output_img)
      saved_paths.append(save_path)
    self.signals.insertText_progressCursor.emit("\nDecensoring complete!")
    self.signals.update_statusLabel_Text.emit("Decensoring complete!")
    return results, saved_paths

  def set_clean_up_input_dirs(self, value):
    self.clean_up_input_dirs = value

  def _current_model_config_key(self):
    if self.model_name == "anima":
      normalized = self.anima_settings.normalized()
      return (
        "anima",
        False,
        normalized.diffusion_file,
        normalized.controlnet_file,
        bool(normalized.use_fp8_storage),
      )
    return (self.model_name, self.is_mosaic, self.bar_weights)

  def load_model(self):
    config_key = self._current_model_config_key()
    if self.model is not None and self.loaded_config == config_key:
      if hasattr(self.model, "use_tta"):
        self.model.use_tta = self.use_tta
      if hasattr(self.model, "update_settings"):
        self.model.update_settings(self.anima_settings)
      return

    if self.model is not None:
      if hasattr(self.model, "unload"):
        self.model.unload()
      self.model = None
      self.loaded_config = None
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    self.signals.insertText_progressCursor.emit(
      f"Loading {self.model_name} model ... please wait ...\n"
    )
    self.signals.update_statusLabel_Text.emit(f"Loading {self.model_name} model...")
    self.model = create_inpainter(
      self.model_name,
      is_mosaic=self.is_mosaic,
      weights_path=self.bar_weights,
      use_tta=self.use_tta,
      anima_settings=self.anima_settings if self.model_name == "anima" else None,
    )
    self.loaded_config = config_key
    mode = "mosaic" if self.is_mosaic else "bar"
    message = f"Loaded {self.model_name} ({mode}) model on {self.model.device}"
    if self.bar_weights is not None:
      message += f" from {self.bar_weights}"
    flags = []
    if getattr(self.model, "use_fp8_storage", False):
      flags.append("FP8")
    if getattr(self.model, "use_block_offload", False):
      flags.append("block offload")
    if flags:
      message += f" [{', '.join(flags)}]"
    if self.model.fallback_reason:
      message += f" ({self.model.fallback_reason})"
    print(message, flush=True)
    self.signals.insertText_progressCursor.emit(message)
    self.signals.update_statusLabel_Text.emit(message)

  def decensor_all_images_in_folder(self):
    self._check_cancelled()
    self.load_model()
    self._check_cancelled()

    input_color_dir = self.decensor_input_path

    input_dir = self.decensor_input_path
    output_dir = self.decensor_output_path

    self.signals.insertText_progressCursor.emit(
      "Checking image files and directory...\n"
    )

    file_names, self.files_removed = file.check_file(input_dir, output_dir, False)

    self.signals.insertText_progressCursor.emit(
      "Decensoring {} image files\n".format(len(file_names))
    )

    total_files = len(file_names)
    for n, file_name in enumerate(file_names, start=1):
      self._check_cancelled()
      img_prefix = f"[{n}/{total_files}] {file_name} | "
      self.signals.insertText_progressCursor.emit(
        "Decensoring image file : {}\n".format(file_name)
      )
      self.signals.update_statusLabel_Text.emit(f"[{n}/{total_files}] {file_name}")

      color_file_path = os.path.join(input_color_dir, file_name)
      color_basename, color_ext = os.path.splitext(file_name)
      if os.path.isfile(color_file_path) and color_ext.casefold() == ".png":
        print(
          "--------------------------------------------------------------------------"
        )
        print("Decensoring the image {}\n".format(color_file_path))
        colored_img = Image.open(color_file_path)

        # if we are doing a mosaic decensor
        if self.is_mosaic:
          ori_dir = self.decensor_input_original_path
          test_file_names = os.listdir(ori_dir)
          valid_formats = {".png", ".jpg", ".jpeg"}
          for test_file_name in test_file_names:
            test_basename, test_ext = os.path.splitext(test_file_name)
            if (test_basename == color_basename) and (
              test_ext.casefold() in valid_formats
            ):
              ori_file_path = os.path.join(ori_dir, test_file_name)
              ori_img = Image.open(ori_file_path)
              self.decensor_image_variations(
                ori_img, colored_img, file_name, progress_prefix=img_prefix
              )
              break
          else:
            print(
              "Corresponding original, uncolored image not found in {}".format(
                color_file_path
              )
            )
            print("Check if it exists and is in the PNG or JPG format.")
            self.signals.insertText_progressCursor.emit(
              "Corresponding original, uncolored image not found in {}\n".format(
                color_file_path
              )
            )
            self.signals.insertText_progressCursor.emit(
              "Check if it exists and is in the PNG or JPG format.\n"
            )
        # if we are doing a bar decensor
        else:
          self.decensor_image_variations(
            colored_img, colored_img, file_name, progress_prefix=img_prefix
          )
      else:
        print(
          "--------------------------------------------------------------------------"
        )
        print("Image can't be found: " + str(color_file_path))
        self.signals.insertText_progressCursor.emit(
          "Image can't be found: " + str(color_file_path) + "\n"
        )

    print("--------------------------------------------------------------------------")
    if self.files_removed is not None:
      file.error_messages(None, self.files_removed)
    print("\nDecensoring complete!")

    self.signals.insertText_progressCursor.emit("\nDecensoring complete!")
    self.signals.update_statusLabel_Text.emit("Decensoring complete!")

  def decensor_image_variations(
    self, ori, colored, file_name=None, *, progress_prefix=""
  ):
    base_seed = self._resolve_image_base_seed()
    for i in range(self.variations):
      self._check_cancelled()
      var_prefix = (
        f"{progress_prefix}Var {i + 1}/{self.variations} | "
        if self.variations > 1
        else progress_prefix
      )
      seed = (base_seed + i) if base_seed is not None else None
      self.decensor_image_variation(
        ori, colored, i, file_name, seed=seed, progress_prefix=var_prefix
      )

  # create different decensors of the same image by flipping the input image (PEPSI/LaMa)
  def apply_variant(self, image, variant_number):
    if variant_number == 0:
      return image
    elif variant_number == 1:
      return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif variant_number == 2:
      return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
      return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

  def _decensor_anima_regions(
    self, ori, marked, regions, output_img_array, *, seed=None, progress_prefix=""
  ):
    clusters = group_anima_regions(
      ori.size, regions, max_size=self.anima_settings.max_size
    )
    boxes = [box for box, _ in clusters]
    cluster_regions = [pix for _, pix in clusters]
    self.signals.insertText_progressCursor.emit(
      f"Anima grouped {len(regions)} region(s) into {len(boxes)} crop(s) "
      f"(seed={seed if seed is not None else 'auto'})."
    )

    def _on_progress(current: int, total: int, stage: str):
      self.signals.update_ProgressBar_MAX_VALUE.emit(max(1, total))
      self.signals.update_ProgressBar_SET_VALUE.emit(min(current, max(1, total)))
      self.signals.update_statusLabel_Text.emit(f"{progress_prefix}{stage}")

    pred_arrays = self.model.inpaint_crops(
      ori,
      marked,
      boxes,
      is_mosaic=False,
      regions=cluster_regions,
      seed=seed,
      cancel_check=self.is_cancelled,
      progress_callback=_on_progress,
    )

    for region, box, pred_array in zip(
      cluster_regions, boxes, pred_arrays, strict=True
    ):
      left, top, _, _ = box
      x, y = np.asarray(list(region)).T
      output_img_array[y, x] = pred_array[y - top, x - left]

  def decensor_image_variation(
    self, ori, colored, variant_number, file_name, *, seed=None, progress_prefix=""
  ):
    self._check_cancelled()
    if ori.size != colored.size:
      raise ValueError("The source image and mask must have the same dimensions.")

    use_flip_variant = self.model_name != "anima"
    same_image = ori is colored
    if use_flip_variant:
      ori = self.apply_variant(ori, variant_number)
      colored = ori if same_image else self.apply_variant(colored, variant_number)

    alpha_channel = None
    if "A" in ori.getbands() or "transparency" in ori.info:
      alpha_channel = ori.convert("RGBA").getchannel("A")
    ori = ori.convert("RGB")

    mask_color = [value * 255 for value in self.mask_color]
    marked = mask_pixels(colored.convert("RGB"), mask_color)
    regions = connected_regions(marked)
    print(
      "Found {region_count} censored regions in this image!".format(
        region_count=len(regions)
      )
    )
    self.signals.insertText_progressCursor.emit(
      "Found {region_count} censored regions in this image!".format(
        region_count=len(regions)
      )
    )

    if len(regions) == 0 and not self.is_mosaic:
      print(
        "No green (0,255,0) regions detected! Make sure you're using exactly the right color."
      )
      self.signals.insertText_progressCursor.emit(
        "No green (0,255,0) regions detected! Make sure you're using exactly the right color.\n"
      )
      return

    print("Found {} masked regions".format(len(regions)))
    self.signals.update_ProgressBar_MIN_VALUE.emit(0)
    self.signals.update_ProgressBar_MAX_VALUE.emit(max(1, len(regions)))
    self.signals.update_ProgressBar_SET_VALUE.emit(0)

    source_pixels = np.asarray(ori)
    output_img_array = source_pixels.copy()

    if self.model_name == "anima":
      self._decensor_anima_regions(
        ori,
        marked,
        regions,
        output_img_array,
        seed=seed,
        progress_prefix=progress_prefix,
      )
    else:
      completed = 0
      while completed < len(regions):
        self._check_cancelled()
        batch = regions[completed : completed + self.model.batch_size]
        boxes = []
        for region_counter, region in enumerate(batch, completed + 1):
          self.signals.insertText_progressCursor.emit(
            f"Decensoring regions in image, Decensoring censor {region_counter}/{len(regions)}"
          )
          boxes.append(expand_bounding(ori, region))

        if hasattr(self.model, "inpaint_crops"):
          pred_arrays = self.model.inpaint_crops(
            ori, marked, boxes, is_mosaic=self.is_mosaic, regions=batch
          )
        else:
          prediction_slices = []
          crops = []
          masks = []
          for box in boxes:
            crop, missing, prediction_slice = prepare_crop(ori, marked, box)
            prediction_slices.append(prediction_slice)
            if not self.is_mosaic:
              crop[missing] = 0
            crops.append(crop)
            masks.append(np.repeat((~missing)[..., None], 3, axis=-1))

          images = np.stack(crops) * 2.0 - 1
          predictions = self.model.predict(images, images, np.stack(masks))
          pred_arrays = [
            resize_prediction(
              prediction[prediction_slice], (box[2] - box[0], box[3] - box[1])
            )
            for box, prediction_slice, prediction in zip(
              boxes, prediction_slices, predictions, strict=True
            )
          ]

        for region, box, pred_array in zip(batch, boxes, pred_arrays, strict=True):
          left, top, _, _ = box
          x, y = np.asarray(list(region)).T
          output_img_array[y, x] = pred_array[y - top, x - left]

          completed += 1
          self.signals.update_ProgressBar_SET_VALUE.emit(completed)
          self.signals.update_statusLabel_Text.emit(
            f"{progress_prefix}Region {completed}/{len(regions)}"
          )
          message = f"{completed} out of {len(regions)} regions decensored."
          self.signals.insertText_progressCursor.emit(message + "\n")
          print(message)

    bridge_thin_gaps(source_pixels, marked, output_img_array)
    output_img = Image.fromarray(output_img_array)
    if alpha_channel is not None:
      output_img.putalpha(alpha_channel)
    if use_flip_variant:
      output_img = self.apply_variant(output_img, variant_number)

    self.signals.insertText_progressCursor.emit(
      "Decensoring of current image finished. Saving image..."
    )
    print("current image finished")

    if file_name is not None:
      base_name, ext = os.path.splitext(file_name)
      file_name = base_name + " " + str(variant_number) + ext
      save_path = os.path.join(self.decensor_output_path, file_name)
      output_img.save(save_path)
      print("Decensored image saved to {save_path}!".format(save_path=save_path))
      self.signals.insertText_progressCursor.emit(
        "Decensored image saved to {save_path}!".format(save_path=save_path)
      )
      self.signals.insertText_progressCursor.emit("=" * 30)
    else:
      print("Decensored image. Returning it.")
      return output_img

  def clean_input_directories(self):
    """Removes .png, .jpg, and .jpeg files from input directories."""

    allowed_extensions = {".png", ".jpg", ".jpeg"}

    self.signals.insertText_progressCursor.emit(
      "Cleaning {}...".format(self.decensor_input_path)
    )

    for file_name in os.listdir(self.decensor_input_path):
      file_path = os.path.join(self.decensor_input_path, file_name)
      if (
        os.path.isfile(file_path)
        and os.path.splitext(file_name)[1].lower() in allowed_extensions
      ):
        os.remove(file_path)

    if self.is_mosaic:
      self.signals.insertText_progressCursor.emit(
        "Cleaning {}...".format(self.decensor_input_original_path)
      )
      for file_name in os.listdir(self.decensor_input_original_path):
        file_path = os.path.join(self.decensor_input_original_path, file_name)
        if (
          os.path.isfile(file_path)
          and os.path.splitext(file_name)[1].lower() in allowed_extensions
        ):
          os.remove(file_path)

    self.signals.insertText_progressCursor.emit("Done!")

  def do_post_jobs(self):
    if self.clean_up_input_dirs:
      self.clean_input_directories()


if __name__ == "__main__":
  try:
    Decensor(cli=True).decensor_all_images_in_folder()
  except (ProcessingCancelledError, AnimaCancelledError):
    print("Processing cancelled.", file=sys.stderr)
    sys.exit(130)
  except Exception as error:
    print(f"Processing stopped: {error}", file=sys.stderr)
    sys.exit(1)
