"""Shared Qt widget for Anima auxiliary prompt and collapsible advanced settings."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
  QCheckBox,
  QComboBox,
  QDoubleSpinBox,
  QGridLayout,
  QGroupBox,
  QLabel,
  QLineEdit,
  QSpinBox,
  QToolButton,
  QVBoxLayout,
  QWidget,
)

from model import (
  ANIMA_DEFAULT_CACHE_PROMPT,
  ANIMA_DEFAULT_CFG,
  ANIMA_DEFAULT_CFG_SKIP_TAIL,
  ANIMA_DEFAULT_LLLITE_STRENGTH,
  ANIMA_DEFAULT_MAX_SIZE,
  ANIMA_DEFAULT_NEGATIVE_PROMPT,
  ANIMA_DEFAULT_SEED,
  ANIMA_DEFAULT_STEPS,
  ANIMA_MAX_CFG_SKIP_TAIL,
  AnimaSettings,
  anima_default_fp8_storage,
  list_anima_files,
)

MAX_SIZE_OPTIONS = ("512", "768", "1024", "1280", "1536")


class AnimaSettingsPanel(QGroupBox):
  """Anima prompt inputs + collapsible advanced settings panel."""

  settings_changed = Signal()

  def __init__(self, *, compact: bool = False, parent=None):
    super().__init__("Anima Settings", parent)
    self._compact = compact
    self._syncing = False
    self._build_ui()
    self.refresh_file_lists()

  def _build_ui(self) -> None:
    root = QVBoxLayout(self)
    root.setSpacing(6)
    if self._compact:
      root.setContentsMargins(6, 8, 6, 6)

    prompt_grid = QGridLayout()
    prompt_grid.setSpacing(4)

    self.positivePromptEdit = QLineEdit()
    self.positivePromptEdit.setPlaceholderText(
      "Optional hints (e.g. body part, @artist)"
    )
    self.positivePromptEdit.setToolTip(
      "Optional auxiliary positive prompt. Leave blank to preserve the original art style."
    )
    self.positivePromptEdit.textChanged.connect(self._emit_changed)

    self.negativePromptEdit = QLineEdit(ANIMA_DEFAULT_NEGATIVE_PROMPT)
    self.negativePromptEdit.setPlaceholderText("Optional negative prompt")
    self.negativePromptEdit.setToolTip(
      "Editable negative prompt. Clear to run without negative terms."
    )
    self.negativePromptEdit.textChanged.connect(self._emit_changed)

    if self._compact:
      prompt_grid.addWidget(QLabel("Prompt:"), 0, 0)
      prompt_grid.addWidget(self.positivePromptEdit, 1, 0)
      prompt_grid.addWidget(QLabel("Negative:"), 2, 0)
      prompt_grid.addWidget(self.negativePromptEdit, 3, 0)
    else:
      prompt_grid.addWidget(QLabel("Prompt (optional):"), 0, 0)
      prompt_grid.addWidget(self.positivePromptEdit, 0, 1)
      prompt_grid.addWidget(QLabel("Negative prompt:"), 1, 0)
      prompt_grid.addWidget(self.negativePromptEdit, 1, 1)

    root.addLayout(prompt_grid)

    # Collapsible advanced section
    self.advancedToggle = QToolButton()
    self.advancedToggle.setText("Advanced Settings")
    self.advancedToggle.setCheckable(True)
    self.advancedToggle.setChecked(False)
    self.advancedToggle.setArrowType(Qt.ArrowType.RightArrow)
    self.advancedToggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    self.advancedToggle.setAutoRaise(True)
    self.advancedToggle.toggled.connect(self._on_advanced_toggled)
    root.addWidget(self.advancedToggle)

    self.advancedContainer = QWidget()
    self.advancedContainer.setVisible(False)
    adv_grid = QGridLayout(self.advancedContainer)
    adv_grid.setContentsMargins(0, 2, 0, 0)
    adv_grid.setSpacing(4)

    self.stepsSpin = QSpinBox()
    self.stepsSpin.setRange(1, 100)
    self.stepsSpin.setValue(ANIMA_DEFAULT_STEPS)
    self.stepsSpin.valueChanged.connect(self._emit_changed)

    self.cfgSpin = QDoubleSpinBox()
    self.cfgSpin.setRange(1.0, 20.0)
    self.cfgSpin.setSingleStep(0.5)
    self.cfgSpin.setDecimals(1)
    self.cfgSpin.setValue(ANIMA_DEFAULT_CFG)
    self.cfgSpin.valueChanged.connect(self._emit_changed)

    self.strengthSpin = QDoubleSpinBox()
    self.strengthSpin.setRange(0.0, 2.0)
    self.strengthSpin.setSingleStep(0.05)
    self.strengthSpin.setDecimals(2)
    self.strengthSpin.setValue(ANIMA_DEFAULT_LLLITE_STRENGTH)
    self.strengthSpin.valueChanged.connect(self._emit_changed)

    self.seedSpin = QSpinBox()
    self.seedSpin.setRange(-1, 2147483647)
    self.seedSpin.setSpecialValueText("Random (-1)")
    self.seedSpin.setValue(ANIMA_DEFAULT_SEED)
    self.seedSpin.valueChanged.connect(self._emit_changed)

    self.maxSizeCombo = QComboBox()
    self.maxSizeCombo.addItems(MAX_SIZE_OPTIONS)
    self.maxSizeCombo.setCurrentText(str(ANIMA_DEFAULT_MAX_SIZE))
    self.maxSizeCombo.currentIndexChanged.connect(self._emit_changed)

    self.cfgSkipTailSpin = QSpinBox()
    self.cfgSkipTailSpin.setRange(0, round(ANIMA_MAX_CFG_SKIP_TAIL * 100))
    self.cfgSkipTailSpin.setSingleStep(1)
    self.cfgSkipTailSpin.setSuffix("%")
    self.cfgSkipTailSpin.setValue(round(ANIMA_DEFAULT_CFG_SKIP_TAIL * 100))
    self.cfgSkipTailSpin.setToolTip(
      "Run the final N% of denoising steps with CFG=1.0 to speed up inference "
      "(0% disables; e.g. 20% skips CFG on the last 20% of steps)."
    )
    self.cfgSkipTailSpin.valueChanged.connect(self._emit_changed)

    self.fp8Checkbox = QCheckBox("FP8 storage (faster on 8GB VRAM)")
    self.fp8Checkbox.setChecked(anima_default_fp8_storage())
    self.fp8Checkbox.setToolTip(
      "Store DiT linear weights in float8_e4m3fn and compute in BF16/FP16 to cut "
      "VRAM in half and avoid block offloading on 8GB GPUs (auto-enabled on <=10.5 GiB GPUs)."
    )
    self.fp8Checkbox.toggled.connect(self._emit_changed)

    self.cachePromptCheckbox = QCheckBox("Cache prompt embeddings")
    self.cachePromptCheckbox.setChecked(ANIMA_DEFAULT_CACHE_PROMPT)
    self.cachePromptCheckbox.setToolTip(
      "Cache encoded prompt embeddings in CPU RAM to skip Qwen3 re-encoding across "
      "variations, batch images, or retries with the same prompt."
    )
    self.cachePromptCheckbox.toggled.connect(self._emit_changed)

    self.diffusionCombo = QComboBox()
    self.diffusionCombo.setToolTip(
      "Select diffusion checkpoint from models/anima/diffusion/."
    )
    self.diffusionCombo.currentIndexChanged.connect(self._emit_changed)

    self.controlnetCombo = QComboBox()
    self.controlnetCombo.setToolTip(
      "Select 4ch Inpaint ControlNet-LLLite from models/anima/controlnet/."
    )
    self.controlnetCombo.currentIndexChanged.connect(self._emit_changed)

    if self._compact:
      rows = [
        ("Steps:", self.stepsSpin),
        ("CFG:", self.cfgSpin),
        ("Tail CFG=1:", self.cfgSkipTailSpin),
        ("LLLite:", self.strengthSpin),
        ("Seed:", self.seedSpin),
        ("Max Size:", self.maxSizeCombo),
        ("Diffusion:", self.diffusionCombo),
        ("ControlNet:", self.controlnetCombo),
      ]
      for r, (label, widget) in enumerate(rows):
        adv_grid.addWidget(QLabel(label), r, 0)
        adv_grid.addWidget(widget, r, 1)
      adv_grid.addWidget(self.fp8Checkbox, len(rows), 0, 1, 2)
      adv_grid.addWidget(self.cachePromptCheckbox, len(rows) + 1, 0, 1, 2)
    else:
      adv_grid.addWidget(QLabel("Steps:"), 0, 0)
      adv_grid.addWidget(self.stepsSpin, 0, 1)
      adv_grid.addWidget(QLabel("CFG:"), 0, 2)
      adv_grid.addWidget(self.cfgSpin, 0, 3)
      adv_grid.addWidget(QLabel("LLLite Strength:"), 0, 4)
      adv_grid.addWidget(self.strengthSpin, 0, 5)

      adv_grid.addWidget(QLabel("Seed:"), 1, 0)
      adv_grid.addWidget(self.seedSpin, 1, 1)
      adv_grid.addWidget(QLabel("Max Size:"), 1, 2)
      adv_grid.addWidget(self.maxSizeCombo, 1, 3)
      adv_grid.addWidget(QLabel("Tail CFG=1:"), 1, 4)
      adv_grid.addWidget(self.cfgSkipTailSpin, 1, 5)

      adv_grid.addWidget(self.fp8Checkbox, 2, 0, 1, 3)
      adv_grid.addWidget(self.cachePromptCheckbox, 2, 3, 1, 3)

      adv_grid.addWidget(QLabel("Diffusion:"), 3, 0)
      adv_grid.addWidget(self.diffusionCombo, 3, 1, 1, 2)
      adv_grid.addWidget(QLabel("ControlNet:"), 3, 3)
      adv_grid.addWidget(self.controlnetCombo, 3, 4, 1, 2)

    root.addWidget(self.advancedContainer)

  def _on_advanced_toggled(self, expanded: bool) -> None:
    self.advancedToggle.setArrowType(
      Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
    )
    self.advancedContainer.setVisible(expanded)

  def _emit_changed(self, *_args) -> None:
    if not self._syncing:
      self.settings_changed.emit()

  def refresh_file_lists(self) -> None:
    self._syncing = True
    try:
      self._populate_combo(self.diffusionCombo, list_anima_files("diffusion"))
      self._populate_combo(self.controlnetCombo, list_anima_files("controlnet"))
    finally:
      self._syncing = False

  @staticmethod
  def _populate_combo(combo: QComboBox, files: list) -> None:
    prev = combo.currentData() or ""
    combo.clear()
    if not files:
      combo.addItem("(none found)", "")
      combo.setEnabled(False)
      return
    for path in files:
      combo.addItem(path.name, path.name)
    combo.setEnabled(len(files) > 1)
    if prev:
      idx = combo.findData(prev)
      if idx >= 0:
        combo.setCurrentIndex(idx)

  def get_settings(self) -> dict:
    diff_file = self.diffusionCombo.currentData() or ""
    ctrl_file = self.controlnetCombo.currentData() or ""
    return AnimaSettings(
      positive_prompt=self.positivePromptEdit.text(),
      negative_prompt=self.negativePromptEdit.text(),
      steps=self.stepsSpin.value(),
      cfg_scale=self.cfgSpin.value(),
      lllite_strength=self.strengthSpin.value(),
      seed=self.seedSpin.value(),
      max_size=int(self.maxSizeCombo.currentText()),
      use_fp8_storage=self.fp8Checkbox.isChecked(),
      cache_prompt_embeds=self.cachePromptCheckbox.isChecked(),
      cfg_skip_tail=self.cfgSkipTailSpin.value() / 100.0,
      diffusion_file=diff_file,
      controlnet_file=ctrl_file,
    ).to_dict()

  def set_settings(self, data: dict | None) -> None:
    settings = AnimaSettings.from_dict(data)
    self._syncing = True
    try:
      if self.positivePromptEdit.text() != settings.positive_prompt:
        self.positivePromptEdit.setText(settings.positive_prompt)
      if self.negativePromptEdit.text() != settings.negative_prompt:
        self.negativePromptEdit.setText(settings.negative_prompt)
      self.stepsSpin.setValue(settings.steps)
      self.cfgSpin.setValue(settings.cfg_scale)
      self.strengthSpin.setValue(settings.lllite_strength)
      self.seedSpin.setValue(settings.seed)
      self.fp8Checkbox.setChecked(bool(settings.use_fp8_storage))
      self.cachePromptCheckbox.setChecked(bool(settings.cache_prompt_embeds))
      self.cfgSkipTailSpin.setValue(round(settings.cfg_skip_tail * 100))
      idx = self.maxSizeCombo.findText(str(settings.max_size))
      if idx >= 0:
        self.maxSizeCombo.setCurrentIndex(idx)
      if settings.diffusion_file:
        d_idx = self.diffusionCombo.findData(settings.diffusion_file)
        if d_idx >= 0:
          self.diffusionCombo.setCurrentIndex(d_idx)
      if settings.controlnet_file:
        c_idx = self.controlnetCombo.findData(settings.controlnet_file)
        if c_idx >= 0:
          self.controlnetCombo.setCurrentIndex(c_idx)
    finally:
      self._syncing = False
