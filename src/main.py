#!/usr/bin/python3

# tooltips
# Please read this tutorial on how to prepare your images for use with DeepCreamPy.
# The greater the number of variations, the longer decensoring process will be.

import argparse
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
  QApplication,
  QCheckBox,
  QComboBox,
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

from anima_ui import AnimaSettingsPanel
from decensor import Decensor
from i18n import i18n, t
from mask_editor import MaskEditorWindow
from model import (
  AnimaSettings,
  anima_availability,
  anima_radio_tooltip,
  lama_radio_tooltip,
  lama_weights_available,
)
from signals import Signals

APP_VERSION = "3.4.0-Unofficial"


class MainWindow(QWidget):
  def __init__(self):
    super().__init__()
    self.signals = Signals()
    self._syncing_settings = False
    self._syncing_language = False
    self._busy_reason = ""
    self._busy_reason_key = ""
    self._busy_source = ""
    self._can_cancel = False
    self._cancelling = False
    self._is_initial_progress = True
    self._status_is_default = True
    self.maskEditorWindow = None
    self.initUI()
    i18n.add_listener(self._on_language_changed)
    self.decensor = Decensor(self)
    self.decensor.signals = self.signals
    self.setSignals()
    self.decensor.finished.connect(self.processing_finished)
    self.decensor.single_image_finished.connect(self._on_single_image_finished)
    self.decensor.single_image_failed.connect(self._on_single_image_failed)
    self.decensor.single_image_cancelled.connect(self._on_single_image_cancelled)
    self.load_model()

  def initUI(self):
    grid_layout = QGridLayout()
    grid_layout.setSpacing(10)
    self.setLayout(grid_layout)

    # Top language selector bar
    lang_bar = QHBoxLayout()
    lang_bar.addStretch(1)
    self.languageLabel = QLabel()
    self.languageCombo = QComboBox()
    self.languageCombo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    self._populate_language_combo()
    self.languageCombo.currentIndexChanged.connect(self._on_language_combo_changed)
    lang_bar.addWidget(self.languageLabel)
    lang_bar.addWidget(self.languageCombo)

    # Top language selector bar
    lang_bar = QHBoxLayout()
    lang_bar.addStretch(1)
    self.languageLabel = QLabel()
    self.languageCombo = QComboBox()
    self.languageCombo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    self._populate_language_combo()
    self.languageCombo.currentIndexChanged.connect(self._on_language_combo_changed)
    lang_bar.addWidget(self.languageLabel)
    lang_bar.addWidget(self.languageCombo)

    # Tutorial
    self.tutorialLabel = QLabel()
    self.tutorialLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.tutorialLabel.setFont(QFont("Sans Serif", 13))

    # Censor type group
    self.censorTypeGroupBox = QGroupBox()

    self.barButton = QRadioButton()
    self.mosaicButton = QRadioButton()
    self.barButton.setChecked(True)
    self.mosaicButton.toggled.connect(self.updateModelAvailability)

    censorLayout = QVBoxLayout()
    censorLayout.addWidget(self.barButton)
    censorLayout.addWidget(self.mosaicButton)
    self.censorTypeGroupBox.setLayout(censorLayout)

    # Model group
    self.modelGroupBox = QGroupBox()

    self.pepsiModelButton = QRadioButton()
    self.lamaModelButton = QRadioButton()
    self.animaModelButton = QRadioButton()
    self.pepsiModelButton.setChecked(True)
    self.ttaCheckbox = QCheckBox()
    self.ttaCheckbox.setChecked(True)
    self.animaSettingsPanel = AnimaSettingsPanel(compact=False)
    self.animaSettingsPanel.setVisible(False)
    self.animaSettingsPanel.settings_changed.connect(self._emit_settings_to_editor)

    self._refresh_model_availability(is_mosaic=False)
    self.pepsiModelButton.toggled.connect(self._on_model_selection_changed)
    self.lamaModelButton.toggled.connect(self._on_model_selection_changed)
    self.animaModelButton.toggled.connect(self._on_model_selection_changed)
    self.ttaCheckbox.toggled.connect(self._emit_settings_to_editor)

    modelLayout = QVBoxLayout()
    modelLayout.addWidget(self.pepsiModelButton)
    modelLayout.addWidget(self.lamaModelButton)
    modelLayout.addWidget(self.animaModelButton)
    modelLayout.addWidget(self.ttaCheckbox)
    self.modelGroupBox.setLayout(modelLayout)

    # Variation count group
    self.variationsGroupBox = QGroupBox()

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
    self.variationsGroupBox.setLayout(varLayout)

    # Action buttons
    self.decensorButton = QPushButton()
    self.decensorButton.clicked.connect(self.decensorClicked)
    self.decensorButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    self.maskEditorButton = QPushButton()
    self.maskEditorButton.clicked.connect(self.openMaskEditor)
    self.maskEditorButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    self.cancelButton = QPushButton()
    self.cancelButton.setEnabled(False)
    self.cancelButton.clicked.connect(self.cancel_processing)
    self.cancelButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

    actionButtonsLayout = QHBoxLayout()
    actionButtonsLayout.addWidget(self.decensorButton, 2)
    actionButtonsLayout.addWidget(self.cancelButton, 1)
    actionButtonsLayout.addWidget(self.maskEditorButton, 1)

    # Progress message
    self.progressMessage = QTextEdit()
    self.progressCursor = QTextCursor(self.progressMessage.document())
    self.progressMessage.setTextCursor(self.progressCursor)
    self.progressMessage.setReadOnly(True)

    # Progress Bar
    self.statusBar = QStatusBar(self)
    self.progressBar = QProgressBar()
    self.progressBar.setMinimum(0)
    self.progressBar.setMaximum(100)
    self.progressBar.setValue(0)
    self.statusLabel = QLabel()

    self.statusBar.addWidget(self.statusLabel, 1)
    self.statusBar.addWidget(self.progressBar, 2)

    self.cleanInputDirectoryCheckbox = QCheckBox()

    # put all groups into grid
    # addWidget(row, column, rowSpan, columnSpan)
    grid_layout.addLayout(lang_bar, 0, 0, 1, 3)
    grid_layout.addWidget(self.tutorialLabel, 1, 0, 1, 3)
    grid_layout.addWidget(self.censorTypeGroupBox, 2, 0, 1, 1)
    grid_layout.addWidget(self.modelGroupBox, 2, 1, 1, 1)
    grid_layout.addWidget(self.variationsGroupBox, 2, 2, 1, 1)
    grid_layout.addWidget(self.animaSettingsPanel, 3, 0, 1, 3)
    grid_layout.addWidget(self.cleanInputDirectoryCheckbox, 4, 0, 1, 3)
    grid_layout.addLayout(actionButtonsLayout, 5, 0, 1, 3)
    grid_layout.addWidget(self.progressMessage, 6, 0, 1, 3)
    grid_layout.addWidget(self.statusBar, 7, 0, 1, 3)

    self.retranslate_ui()

    # window size settings
    self.resize(900, 620)
    self.center()
    self.show()

  def _populate_language_combo(self):
    self._syncing_language = True
    try:
      self.languageCombo.clear()
      languages = i18n.available_languages()
      if not languages:
        languages = [(i18n.current_code, i18n.current_code)]
      for code, display_name in languages:
        self.languageCombo.addItem(display_name, code)
      idx = self.languageCombo.findData(i18n.current_code)
      if idx >= 0:
        self.languageCombo.setCurrentIndex(idx)
    finally:
      self._syncing_language = False

  def _on_language_combo_changed(self, index: int):
    if self._syncing_language or index < 0:
      return
    code = self.languageCombo.itemData(index)
    if code:
      i18n.set_language(str(code), persist=True)

  def _on_language_changed(self, code: str):
    self._syncing_language = True
    try:
      idx = self.languageCombo.findData(code)
      if idx < 0:
        self._populate_language_combo()
        idx = self.languageCombo.findData(code)
      if idx >= 0 and self.languageCombo.currentIndex() != idx:
        self.languageCombo.setCurrentIndex(idx)
    finally:
      self._syncing_language = False
    self.retranslate_ui()
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.retranslate_ui()

  def _refresh_action_button_texts(self):
    if self._busy_source == "loading":
      self.decensorButton.setText(t("main.decensor_button_loading"))
    elif self._busy_source == "editor":
      self.decensorButton.setText(t("main.decensor_button_editor_busy"))
    else:
      self.decensorButton.setText(t("main.decensor_button"))

    self.cancelButton.setText(
      t("main.cancel_button_cancelling")
      if self._cancelling
      else t("main.cancel_button")
    )

  def retranslate_ui(self):
    self.setWindowTitle(t("main.window_title", version=APP_VERSION))
    self.languageLabel.setText(t("main.language_label"))
    self.languageCombo.setToolTip(t("main.language_tooltip"))

    self.tutorialLabel.setText(t("main.tutorial"))
    self.censorTypeGroupBox.setTitle(t("main.censor_type_group"))
    self.barButton.setText(t("main.bar_censor"))
    self.mosaicButton.setText(t("main.mosaic_censor"))

    self.modelGroupBox.setTitle(t("main.model_group"))
    self.pepsiModelButton.setText(t("main.model_pepsi"))
    self.lamaModelButton.setText(t("main.model_lama"))
    self.animaModelButton.setText(t("main.model_anima"))
    self.ttaCheckbox.setText(t("main.use_tta"))
    self.ttaCheckbox.setToolTip(t("main.use_tta_tooltip"))
    self.animaSettingsPanel.retranslate_ui()
    self._refresh_model_availability()

    self.variationsGroupBox.setTitle(t("main.variations_group"))
    self.cleanInputDirectoryCheckbox.setText(t("main.clean_input_dirs"))

    self.maskEditorButton.setText(t("main.mask_editor_button"))
    self.maskEditorButton.setToolTip(t("main.mask_editor_tooltip"))
    self.cancelButton.setToolTip(t("main.cancel_tooltip"))
    if self._busy_reason_key:
      self._busy_reason = t(self._busy_reason_key)
    self._refresh_action_button_texts()

    if self._is_initial_progress:
      self.progressMessage.setPlainText(t("main.initial_progress_message"))
    if self._status_is_default:
      self.statusLabel.setText(t("main.status_initial"))
    elif self._cancelling:
      self.statusLabel.setText(t("main.status_cancelling"))

  def load_model(self):
    self._busy_reason_key = "main.status_loading_model"
    self._busy_reason = t(self._busy_reason_key)
    self._busy_source = "loading"
    self._can_cancel = False
    self._cancelling = False
    self.decensorButton.setEnabled(False)
    self.cancelButton.setEnabled(False)
    self._refresh_action_button_texts()
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(
        True,
        self._busy_reason,
        can_cancel=False,
        reason_key=self._busy_reason_key,
      )
    self.decensor.reset_cancel()
    self.decensor.load_only = True
    self.decensor.start()

  def processing_finished(self):
    self._busy_reason_key = ""
    self._busy_reason = ""
    self._busy_source = ""
    self._can_cancel = False
    self._cancelling = False
    self._refresh_action_button_texts()
    self.decensorButton.setEnabled(True)
    self.cancelButton.setEnabled(False)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(False)

  def cancel_processing(self):
    if not self.decensor.isRunning() or not self._can_cancel:
      return
    self._cancelling = True
    self._status_is_default = False
    self.decensor.request_cancel()
    self.cancelButton.setEnabled(False)
    self._refresh_action_button_texts()
    self.statusLabel.setText(t("main.status_cancelling"))
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_cancelling()

  def setSignals(self):
    self.signals.update_statusLabel_Text.connect(self._on_status_text)
    self.signals.update_ProgressBar_SET_VALUE.connect(self._on_progress_value)
    self.signals.update_ProgressBar_MAX_VALUE.connect(self._on_progress_max)
    self.signals.update_ProgressBar_MIN_VALUE.connect(self._on_progress_min)
    self.signals.insertText_progressCursor.connect(self._on_progress_log)
    self.signals.clear_progressMessage.connect(self._on_clear_progress_log)
    self.signals.appendText_progressMessage.connect(self._on_progress_log)
    self.signals.update_clean_up_input_dirs_flag.connect(
      self.decensor.set_clean_up_input_dirs
    )

  def _on_progress_log(self, text):
    self._is_initial_progress = False
    self.progressMessage.append(text)

  def _on_clear_progress_log(self):
    self._is_initial_progress = False
    self.progressMessage.clear()

  def _on_status_text(self, text):
    self._status_is_default = False
    self.statusLabel.setText(text)
    if self.maskEditorWindow is not None and self.decensor.isRunning():
      self.maskEditorWindow.set_status_text(text)

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

  def _refresh_lama_availability(self, is_mosaic=None):
    if is_mosaic is None:
      is_mosaic = self.mosaicButton.isChecked()
    available = lama_weights_available()
    self.lamaModelButton.setEnabled(available and not is_mosaic)
    self.lamaModelButton.setToolTip(lama_radio_tooltip(available))
    if (is_mosaic or not available) and self.lamaModelButton.isChecked():
      self.pepsiModelButton.setChecked(True)
    return available

  def _refresh_anima_availability(self, is_mosaic=None):
    if is_mosaic is None:
      is_mosaic = self.mosaicButton.isChecked()
    available, reason = anima_availability()
    self.animaModelButton.setEnabled(available and not is_mosaic)
    self.animaModelButton.setToolTip(anima_radio_tooltip(available, reason))
    if (is_mosaic or not available) and self.animaModelButton.isChecked():
      self.pepsiModelButton.setChecked(True)
    return available

  def _refresh_model_availability(self, is_mosaic=None):
    lama_ok = self._refresh_lama_availability(is_mosaic=is_mosaic)
    anima_ok = self._refresh_anima_availability(is_mosaic=is_mosaic)
    self._update_model_specific_controls()
    return lama_ok, anima_ok

  def _update_model_specific_controls(self):
    is_pepsi = self.pepsiModelButton.isChecked()
    is_anima = self.animaModelButton.isChecked() and self.animaModelButton.isEnabled()
    self.ttaCheckbox.setEnabled(is_pepsi)
    self.animaSettingsPanel.setVisible(is_anima)
    self.animaSettingsPanel.setEnabled(is_anima)

  def _on_model_selection_changed(self, checked):
    if not checked:
      return
    self._update_model_specific_controls()
    self._emit_settings_to_editor()

  def updateModelAvailability(self, is_mosaic):
    self._refresh_model_availability(is_mosaic=is_mosaic)
    self._emit_settings_to_editor()

  def get_decensor_settings(self):
    is_mosaic = self.mosaicButton.isChecked()
    if (
      not is_mosaic
      and self.animaModelButton.isEnabled()
      and self.animaModelButton.isChecked()
    ):
      model_name = "anima"
    elif (
      not is_mosaic
      and self.lamaModelButton.isEnabled()
      and self.lamaModelButton.isChecked()
    ):
      model_name = "lama"
    else:
      model_name = "pepsi"
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
      "anima_settings": self.animaSettingsPanel.get_settings(),
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
      anima_data = settings.get("anima_settings")

      self.mosaicButton.setChecked(is_mosaic)
      self.barButton.setChecked(not is_mosaic)
      lama_ok, anima_ok = self._refresh_model_availability(is_mosaic=is_mosaic)
      if (is_mosaic or not lama_ok) and model_name == "lama":
        model_name = "pepsi"
      if (is_mosaic or not anima_ok) and model_name == "anima":
        model_name = "pepsi"
      self.pepsiModelButton.setChecked(model_name == "pepsi")
      self.lamaModelButton.setChecked(model_name == "lama")
      self.animaModelButton.setChecked(model_name == "anima")
      self.ttaCheckbox.setChecked(use_tta)
      if anima_data is not None:
        self.animaSettingsPanel.set_settings(anima_data)
      self._update_model_specific_controls()
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
    self.decensor.anima_settings = AnimaSettings.from_dict(
      settings.get("anima_settings")
    )

  def openMaskEditor(self):
    if self.maskEditorWindow is None:
      mask_rgb = tuple(round(v * 255) for v in self.decensor.mask_color)
      self.maskEditorWindow = MaskEditorWindow(
        self.get_decensor_settings(), mask_color=mask_rgb
      )
      self.maskEditorWindow.settings_changed.connect(self.apply_editor_settings)
      self.maskEditorWindow.decensor_requested.connect(self.start_editor_decensor)
      self.maskEditorWindow.cancel_requested.connect(self.cancel_processing)
      if self.decensor.isRunning():
        self.maskEditorWindow.set_busy(
          True,
          self._busy_reason or t("main.status_processing"),
          can_cancel=self._can_cancel,
          reason_key=self._busy_reason_key or "main.status_processing",
        )
    else:
      self.maskEditorWindow.apply_synced_settings(
        self.get_decensor_settings(), auto_switch_tool=False
      )
      self.maskEditorWindow.retranslate_ui()
    self.maskEditorWindow.show()
    self.maskEditorWindow.raise_()
    self.maskEditorWindow.activateWindow()

  def start_editor_decensor(self, ori_img, colored_img, output_stem):
    if self.decensor.isRunning():
      return
    self._busy_reason_key = "main.status_decensoring"
    self._busy_reason = t(self._busy_reason_key)
    self._busy_source = "editor"
    self._can_cancel = True
    self._cancelling = False
    self._is_initial_progress = False
    self.decensorButton.setEnabled(False)
    self._refresh_action_button_texts()
    self.cancelButton.setEnabled(True)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(
        True,
        self._busy_reason,
        can_cancel=True,
        reason_key=self._busy_reason_key,
      )

    self.progressMessage.clear()
    self.progressCursor.insertText(t("main.log_editor_started"))
    self._apply_current_settings_to_decensor()
    self.decensor.reset_cancel()
    self.decensor.set_single_image_job(ori_img, colored_img, output_stem)
    self.decensor.start()

  def _on_single_image_finished(self, results, saved_paths):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.on_decensor_succeeded(results, saved_paths)

  def _on_single_image_failed(self, error_message):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.on_decensor_failed(error_message)

  def _on_single_image_cancelled(self):
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.on_decensor_cancelled()

  def decensorClicked(self):
    if self.decensor.isRunning():
      return
    self._busy_reason_key = "main.status_decensoring"
    self._busy_reason = t(self._busy_reason_key)
    self._busy_source = "batch"
    self._can_cancel = True
    self._cancelling = False
    self._is_initial_progress = False
    self.decensorButton.setEnabled(False)
    self._refresh_action_button_texts()
    self.cancelButton.setEnabled(True)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.set_busy(
        True,
        self._busy_reason,
        can_cancel=True,
        reason_key=self._busy_reason_key,
      )
    self.progressMessage.clear()
    self.progressCursor.insertText(t("main.log_started"))

    self._apply_current_settings_to_decensor()
    self.decensor.reset_cancel()

    self.signals.update_clean_up_input_dirs_flag.emit(
      self.cleanInputDirectoryCheckbox.isChecked()
    )

    self.decensor.start()

  # #centers the main window
  def center(self):
    qr = self.frameGeometry()
    cp = self.screen().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def closeEvent(self, event):
    i18n.remove_listener(self._on_language_changed)
    if self.maskEditorWindow is not None:
      self.maskEditorWindow.close()
    if self.decensor.isRunning():
      self.decensor.request_cancel()
    self.decensor.wait()
    super().closeEvent(event)


def _extract_cli_lang(argv):
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--lang", default=None)
  known, _ = parser.parse_known_args(argv)
  return known.lang


if __name__ == "__main__":
  app = QApplication(sys.argv)
  i18n.init_language(cli_lang=_extract_cli_lang(sys.argv[1:]))
  ex = MainWindow()
  sys.exit(app.exec())
