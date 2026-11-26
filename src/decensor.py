import os
import sys

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
  mask_pixels,
  prepare_crop,
  resize_prediction,
)
from model import create_inpainter

"""
to allow decensoring in CLI, ignore all methods targeting Qt signals,
when they aren't initialized (i.e. when the class isn't created by the GUI)
"""


class IgnoreAll(object):
  def __getattr__(self, attr):
    return IgnoreAll()

  def __call__(self, *args, **kwargs):
    return IgnoreAll()


class Decensor(QtCore.QThread):
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

    # if ui_mode is not None:
    #     self.ui_mode = ui_mode
    # else:
    #     self.ui_mode = args.ui_mode
    #
    # if self.ui_mode:
    #     self.text_edit = text_edit
    #     self.text_cursor = text_cursor
    #     self.ui_mode = True

    if not os.path.exists(self.decensor_output_path):
      os.makedirs(self.decensor_output_path)

  def run(self):
    try:
      if self.load_only:
        self.load_model()
      else:
        self.decensor_all_images_in_folder()
        self.do_post_jobs()
    except Exception as error:
      message = f"Processing stopped: {error}"
      print(message, file=sys.stderr)
      self.signals.insertText_progressCursor.emit(message)
    finally:
      self.load_only = False

  def set_clean_up_input_dirs(self, value):
    self.clean_up_input_dirs = value

  def load_model(self):
    config_key = (self.model_name, self.is_mosaic, self.bar_weights)
    if self.model is not None and self.loaded_config == config_key:
      return

    if self.model is not None:
      self.model = None
      self.loaded_config = None
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    self.signals.insertText_progressCursor.emit("Loading model ... please wait ...\n")
    self.model = create_inpainter(
      self.model_name, is_mosaic=self.is_mosaic, weights_path=self.bar_weights
    )
    self.loaded_config = config_key
    mode = "mosaic" if self.is_mosaic else "bar"
    message = f"Loaded {self.model_name} ({mode}) model on {self.model.device}"
    if self.bar_weights is not None:
      message += f" from {self.bar_weights}"
    if self.model.fallback_reason:
      message += f" ({self.model.fallback_reason})"
    print(message, flush=True)
    self.signals.insertText_progressCursor.emit(message)
    self.signals.update_statusLabel_Text.emit(message)

  def decensor_all_images_in_folder(self):
    self.load_model()

    input_color_dir = self.decensor_input_path

    input_dir = self.decensor_input_path
    output_dir = self.decensor_output_path

    # Change False to True before release --> file.check_file(input_dir, output_dir, True)
    # self.signals.update_progress_LABEL.emit("file.check_file()", "Checking image files and directory...")
    self.signals.insertText_progressCursor.emit(
      "Checking image files and directory...\n"
    )

    file_names, self.files_removed = file.check_file(input_dir, output_dir, False)

    # self.signals.total_ProgressBar_update_MAX_VALUE.emit("set total progress bar MaxValue : "+str(len(file_names)),len(file_names))
    """
        print("set total progress bar MaxValue : "+str(len(file_names)))
        self.signals.update_ProgressBar_MAX_VALUE.emit(len(file_names))
        """
    self.signals.insertText_progressCursor.emit(
      "Decensoring {} image files\n".format(len(file_names))
    )

    # convert all images into np arrays and put them in a list
    for n, file_name in enumerate(file_names, start=1):
      # self.signals.total_ProgressBar_update_VALUE.emit("Decensoring {} / {}".format(n, len(file_names)), n)
      """
            self.update_ProgressBar_SET_VALUE.emit(n)
            print("Decensoring {} / {}".format(n, len(file_names)))
            """
      self.signals.insertText_progressCursor.emit(
        "Decensoring image file : {}\n".format(file_name)
      )

      # signal progress bar value == masks decensored on image ,
      # e.g) sample image : 17
      # self.signals.signal_ProgressBar_update_VALUE.emit("reset value", 0) # set to 0 for every image at start
      # self.signals.update_progress_LABEL.emit("for-loop, \"for file_name in file_names:\"","Decensoring : "+str(file_name))

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
          # get the original file that hasn't been colored
          ori_dir = self.decensor_input_original_path
          test_file_names = os.listdir(ori_dir)
          # since the original image might not be a png, test multiple file formats
          valid_formats = {".png", ".jpg", ".jpeg"}
          for test_file_name in test_file_names:
            test_basename, test_ext = os.path.splitext(test_file_name)
            if (test_basename == color_basename) and (
              test_ext.casefold() in valid_formats
            ):
              ori_file_path = os.path.join(ori_dir, test_file_name)
              ori_img = Image.open(ori_file_path)
              # colored_img.show()
              self.decensor_image_variations(ori_img, colored_img, file_name)
              break
          else:  # for...else, i.e if the loop finished without encountering break
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
          self.decensor_image_variations(colored_img, colored_img, file_name)
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

  def decensor_image_variations(self, ori, colored, file_name=None):
    for i in range(self.variations):
      self.decensor_image_variation(ori, colored, i, file_name)

  # create different decensors of the same image by flipping the input image
  def apply_variant(self, image, variant_number):
    if variant_number == 0:
      return image
    elif variant_number == 1:
      return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif variant_number == 2:
      return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
      return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

  def decensor_image_variation(self, ori, colored, variant_number, file_name):
    if ori.size != colored.size:
      raise ValueError("The source image and mask must have the same dimensions.")
    same_image = ori is colored
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

    # self.signals.signal_ProgressBar_update_MAX_VALUE.emit("Found {} masked regions".format(len(regions)), len(regions))
    print("Found {} masked regions".format(len(regions)))

    # self.signals.insertText_progressCursor.emit("Found {} masked regions\n".format(len(regions)))
    self.signals.update_ProgressBar_MAX_VALUE.emit(len(regions))
    self.signals.update_ProgressBar_SET_VALUE.emit(0)

    source_pixels = np.asarray(ori)
    output_img_array = source_pixels.copy()
    completed = 0
    while completed < len(regions):
      batch = regions[completed : completed + self.model.batch_size]
      boxes = []
      for region_counter, region in enumerate(batch, completed + 1):
        self.signals.insertText_progressCursor.emit(
          f"Decensoring regions in image, Decensoring censor {region_counter}/{len(regions)}"
        )
        boxes.append(expand_bounding(ori, region))

      if hasattr(self.model, "inpaint_crops"):
        pred_arrays = self.model.inpaint_crops(
          ori, marked, boxes, is_mosaic=self.is_mosaic
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
        message = f"{completed} out of {len(regions)} regions decensored."
        self.signals.insertText_progressCursor.emit(message + "\n")
        print(message)

    bridge_thin_gaps(source_pixels, marked, output_img_array)
    output_img = Image.fromarray(output_img_array)
    if alpha_channel is not None:
      output_img.putalpha(alpha_channel)
    output_img = self.apply_variant(output_img, variant_number)

    # self.signals.update_progress_LABEL.emit("current image finished", "Decensoring of current image finished. Saving image...")
    self.signals.insertText_progressCursor.emit(
      "Decensoring of current image finished. Saving image..."
    )
    print("current image finished")

    if file_name != None:
      # save the decensored image
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
      # Legacy Code piece ↓, used when DCPv1 had ui with Painting
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
  except Exception as error:
    print(f"Processing stopped: {error}", file=sys.stderr)
    sys.exit(1)
