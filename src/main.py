#!/usr/bin/python3

# tooltips
# Please read this tutorial on how to prepare your images for use with DeepCreamPy.
# The greater the number of variations, the longer decensoring process will be.

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
  QApplication,
  QCheckBox,
  QGridLayout,
  QGroupBox,
  QHBoxLayout,
  QLabel,
  QProgressBar,
  QPushButton,
  QRadioButton,
  QSizePolicy,
  QStatusBar,
  QTextEdit,
  QVBoxLayout,
  QWidget,
)

from decensor import Decensor
from mask_editor import MaskEditorWindow
from signals import Signals

# from decensor import Decensor

# from progressWindow import ProgressWindow


class MainWindow(QWidget):
  def __init__(self):
    super().__init__()
    self.signals = Signals()
    self._syncing_settings = False
    self._busy_reason = ""
    self.maskEditorWindow = None
    self.initUI()
    self.decensor = Decensor(self)
    self.decensor.signals = self.signals
    self.setSignals()
    self.decensor.finished.connect(self.processing_finished)
    self.decensor.single_image_finished.connect(self._on_single_image_finished)
    self.decensor.single_image_failed.connect(self._on_single_image_failed)
    self.load_model()

  def initUI(self):

    grid_layout = QGridLayout()
    grid_layout.setSpacing(10)
    self.setLayout(grid_layout)

    # Tutorial
    self.tutorialLabel = QLabel()
    self.tutorialLabel.setText(
      "Welcome to DeepCreamPy!\n\nIf you're new to DCP, please read the README.\nThis program does nothing without the proper setup of your images.\n\nReport any bugs you encounter to me on Github or Twitter @deeppomf."
    )
    self.tutorialLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.tutorialLabel.setFont(QFont("Sans Serif", 13))

    # Censor type group
    self.censorTypeGroupBox = QGroupBox("Censor Type")

    self.barButton = QRadioButton("Bar censor")
    self.mosaicButton = QRadioButton("Mosaic censor")
    self.barButton.setChecked(True)
    self.mosaicButton.toggled.connect(self.updateModelAvailability)

    censorLayout = QVBoxLayout()
    censorLayout.addWidget(self.barButton)
    censorLayout.addWidget(self.mosaicButton)
    self.censorTypeGroupBox.setLayout(censorLayout)

    # Model group
    self.modelGroupBox = QGroupBox("Model")

    self.pepsiModelButton = QRadioButton("PEPSI (default)")
    self.lamaModelButton = QRadioButton("LaMa (Bar only)")
    self.lamaModelButton.setToolTip("LaMa is available in Bar censor mode only.")
    self.pepsiModelButton.setChecked(True)
    self.ttaCheckbox = QCheckBox("Use TTA (flip + shift)")
    self.ttaCheckbox.setChecked(True)
    self.ttaCheckbox.setToolTip(
      "Average 4 horizontal-flip and small-shift views for higher PEPSI accuracy."
    )
    self.pepsiModelButton.toggled.connect(self.ttaCheckbox.setEnabled)
    self.pepsiModelButton.toggled.connect(self._emit_settings_to_editor)
    self.ttaCheckbox.toggled.connect(self._emit_settings_to_editor)

    modelLayout = QVBoxLayout()
    modelLayout.addWidget(self.pepsiModelButton)
    modelLayout.addWidget(self.lamaModelButton)
    modelLayout.addWidget(self.ttaCheckbox)
    self.modelGroupBox.setLayout(modelLayout)

    # Variation count group
    self.variationsGroupBox = QGroupBox("Number of Decensor Variations")

    var1Button = QRadioButton("1")
    var2Button = QRadioButton("2")
    var3Button = QRadioButton("4")
    var1Button.setChecked(True)
    self.variationButtons = [var1Button, var2Button, var3Button]
    for vb in self.variationButtons:
      vb.toggled.connect(self._emit_settings_to_editor)

    varLayout = QVBoxLayout()
    varLayout.addWidget(var1Button)
    varLayout.addWidget(var2Button)
    varLayout.addWidget(var3Button)
    # varLayout.addStretch(1)
    self.variationsGroupBox.setLayout(varLayout)

    # Action buttons
    self.decensorButton = QPushButton("Decensor Your Images")
    self.decensorButton.clicked.connect(self.decensorClicked)
    self.decensorButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    self.maskEditorButton = QPushButton("Mask and Decensor")
    self.maskEditorButton.setToolTip(
      "Open interactive editor to mask and decensor an image directly."
    )
    self.maskEditorButton.clicked.connect(self.openMaskEditor)
    self.maskEditorButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    actionButtonsLayout = QHBoxLayout()
    actionButtonsLayout.addWidget(self.decensorButton, 2)
    actionButtonsLayout.addWidget(self.maskEditorButton, 1)

    # Progress message
    # self.progressGroupBox = QGroupBox('Progress')

    self.progressMessage = QTextEdit()
    self.progressCursor = QTextCursor(self.progressMessage.document())
    self.progressMessage.setTextCursor(self.progressCursor)
    self.progressMessage.setReadOnly(True)
    self.progressCursor.insertText(
      "After you prepared your images, click on the decensor button once to begin decensoring.\nPlease be patient.\nDecensoring will take time.\n"
    )

    # Progress Bar
    self.statusBar = QStatusBar(self)
    self.progressBar = QProgressBar()
    self.progressBar.setMinimum(0)
    self.progressBar.setMaximum(100)
    self.progressBar.setValue(0)
    self.statusLabel = QLabel("Showing Progress")

    self.statusBar.addWidget(self.statusLabel, 1)
    self.statusBar.addWidget(self.progressBar, 2)

    self.cleanInputDirectoryCheckbox = QCheckBox(
      "Clean up input directories after decensoring"
    )

    # put all groups into grid
    # addWidget(row, column, rowSpan, columnSpan)
    grid_layout.addWidget(self.tutorialLabel, 0, 0, 1, 3)
    grid_layout.addWidget(self.censorTypeGroupBox, 1, 0, 1, 1)
    grid_layout.addWidget(self.modelGroupBox, 1, 1, 1, 1)
    grid_layout.addWidget(self.variationsGroupBox, 1, 2, 1, 1)
    grid_layout.addWidget(self.cleanInputDirectoryCheckbox, 2, 0, 1, 3)
    grid_layout.addLayout(actionButtonsLayout, 3, 0, 1, 3)
    grid_layout.addWidget(self.progressMessage, 4, 0, 1, 3)
    grid_layout.addWidget(self.statusBar, 5, 0, 1, 3)

    # window size settings
    self.resize(900, 600)
    self.center()
    self.setWindowTitle("DeepCreamPy v3.1.0-Unofficial")
    self.show()

  def load_model(self):
    self._busy_reason = "Loading Model..."
    self.decensorButton.setEnabled(False)
    self.decensorButton.setText("Loading Machine Learning Model (Please Wait...)")
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(True, self._busy_reason)
    self.decensor.load_only = True
    self.decensor.start()

  def processing_finished(self):
    self._busy_reason = ""
    self.decensorButton.setText("Decensor Your Images")
    self.decensorButton.setEnabled(True)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(False)

  def setSignals(self):
    self.signals.update_statusLabel_Text.connect(self.statusLabel.setText)
    self.signals.update_ProgressBar_SET_VALUE.connect(self._on_progress_value)
    self.signals.update_ProgressBar_MAX_VALUE.connect(self._on_progress_max)
    self.signals.update_ProgressBar_MIN_VALUE.connect(self._on_progress_min)
    # self.signals.insertText_progressCursor.connect(self.progressCursor.insertText)
    self.signals.insertText_progressCursor.connect(self.progressMessage.append)
    self.signals.clear_progressMessage.connect(self.progressMessage.clear)
    self.signals.appendText_progressMessage.connect(self.progressMessage.append)
    self.signals.update_clean_up_input_dirs_flag.connect(
      self.decensor.set_clean_up_input_dirs
    )

  def _on_progress_value(self, value):
    self.progressBar.setValue(value)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_progress_value(value)

  def _on_progress_max(self, maximum):
    self.progressBar.setMaximum(maximum)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_progress_range(self.progressBar.minimum(), maximum)

  def _on_progress_min(self, minimum):
    self.progressBar.setMinimum(minimum)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_progress_range(minimum, self.progressBar.maximum())

  def updateModelAvailability(self, is_mosaic):
    self.lamaModelButton.setEnabled(not is_mosaic)
    if is_mosaic and self.lamaModelButton.isChecked():
      self.pepsiModelButton.setChecked(True)
    self._emit_settings_to_editor()

  def get_decensor_settings(self):
    is_mosaic = self.mosaicButton.isChecked()
    model_name = (
      "lama" if self.lamaModelButton.isChecked() and not is_mosaic else "pepsi"
    )
    variations = 1
    for vb in self.variationButtons:
      if vb.isChecked():
        variations = int(vb.text())
        break
    return {
      "is_mosaic": is_mosaic,
      "model_name": model_name,
      "use_tta": self.ttaCheckbox.isChecked(),
      "variations": variations,
    }

  def _emit_settings_to_editor(self, *_args):
    if self._syncing_settings or self.maskEditorWindow is None:
      return
    self.maskEditorWindow.apply_synced_settings(self.get_decensor_settings())

  def apply_editor_settings(self, settings):
    self._syncing_settings = True
    try:
      is_mosaic = bool(settings.get("is_mosaic", False))
      model_name = settings.get("model_name", "pepsi")
      use_tta = bool(settings.get("use_tta", True))
      variations = int(settings.get("variations", 1))

      self.mosaicButton.setChecked(is_mosaic)
      self.barButton.setChecked(not is_mosaic)
      self.lamaModelButton.setEnabled(not is_mosaic)
      if is_mosaic and model_name == "lama":
        model_name = "pepsi"
      self.lamaModelButton.setChecked(model_name == "lama")
      self.pepsiModelButton.setChecked(model_name != "lama")
      self.ttaCheckbox.setChecked(use_tta)
      self.ttaCheckbox.setEnabled(model_name != "lama")
      for vb in self.variationButtons:
        vb.setChecked(int(vb.text()) == variations)
    finally:
      self._syncing_settings = False

  def _apply_current_settings_to_decensor(self):
    settings = self.get_decensor_settings()
    self.decensor.is_mosaic = settings["is_mosaic"]
    self.decensor.model_name = settings["model_name"]
    self.decensor.use_tta = settings["use_tta"]
    self.decensor.variations = settings["variations"]

  def openMaskEditor(self):
    if self.maskEditorWindow is None:
      mask_rgb = tuple(round(v * 255) for v in self.decensor.mask_color)
      self.maskEditorWindow = MaskEditorWindow(
        self.get_decensor_settings(), mask_color=mask_rgb
      )
      self.maskEditorWindow.settings_changed.connect(self.apply_editor_settings)
      self.maskEditorWindow.decensor_requested.connect(self.start_editor_decensor)
      if self.decensor.isRunning():
        self.maskEditorWindow.set_busy(True, self._busy_reason or "Processing...")
    else:
      self.maskEditorWindow.apply_synced_settings(
        self.get_decensor_settings(), auto_switch_tool=False
      )
    self.maskEditorWindow.show()
    self.maskEditorWindow.raise_()
    self.maskEditorWindow.activateWindow()

  def start_editor_decensor(self, ori_img, colored_img, output_stem):
    if self.decensor.isRunning():
      return
    self._busy_reason = "Decensoring..."
    self.decensorButton.setEnabled(False)
    self.decensorButton.setText("Decensoring (Mask Editor)...")
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(True, self._busy_reason)

    self.progressMessage.clear()
    self.progressCursor.insertText("Decensoring from Mask Editor has begun!\n")
    self._apply_current_settings_to_decensor()
    self.decensor.set_single_image_job(ori_img, colored_img, output_stem)
    self.decensor.start()

  def _on_single_image_finished(self, results, saved_paths):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.on_decensor_succeeded(results, saved_paths)

  def _on_single_image_failed(self, error_message):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.on_decensor_failed(error_message)

  def decensorClicked(self):
    if self.decensor.isRunning():
      return
    self._busy_reason = "Decensoring..."
    self.decensorButton.setEnabled(False)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(True, self._busy_reason)
    self.progressMessage.clear()
    self.progressCursor.insertText("Decensoring has begun!\n")

    self._apply_current_settings_to_decensor()

    self.signals.update_clean_up_input_dirs_flag.emit(
      self.cleanInputDirectoryCheckbox.isChecked()
    )

    self.decensor.start()
    # decensor.decensor_all_images_in_folder()

  # #centers the main window
  def center(self):
    qr = self.frameGeometry()
    cp = self.screen().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def closeEvent(self, event):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.close()
    self.decensor.wait()
    super().closeEvent(event)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  ex = MainWindow()
  sys.exit(app.exec())
