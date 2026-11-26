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
from signals import Signals

# from decensor import Decensor

# from progressWindow import ProgressWindow


class MainWindow(QWidget):
  def __init__(self):
    super().__init__()
    self.signals = Signals()
    self.initUI()
    self.decensor = Decensor(self)
    self.decensor.signals = self.signals
    self.setSignals()
    self.decensor.finished.connect(self.processing_finished)
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

    varLayout = QVBoxLayout()
    varLayout.addWidget(var1Button)
    varLayout.addWidget(var2Button)
    varLayout.addWidget(var3Button)
    # varLayout.addStretch(1)
    self.variationsGroupBox.setLayout(varLayout)

    # Decensor button
    self.decensorButton = QPushButton("Decensor Your Images")
    self.decensorButton.clicked.connect(self.decensorClicked)
    self.decensorButton.setSizePolicy(
      QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
    )

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
    grid_layout.addWidget(self.decensorButton, 3, 0, 1, 3)
    grid_layout.addWidget(self.progressMessage, 4, 0, 1, 3)
    grid_layout.addWidget(self.statusBar, 5, 0, 1, 3)

    # window size settings
    self.resize(900, 600)
    self.center()
    self.setWindowTitle("DeepCreamPy v3.1.0-Unofficial")
    self.show()

  def load_model(self):
    self.decensorButton.setEnabled(False)
    self.decensorButton.setText("Loading Machine Learning Model (Please Wait...)")
    self.decensor.load_only = True
    self.decensor.start()

  def processing_finished(self):
    self.decensorButton.setText("Decensor Your Images")
    self.decensorButton.setEnabled(True)

  def setSignals(self):
    self.signals.update_statusLabel_Text.connect(self.statusLabel.setText)
    self.signals.update_ProgressBar_SET_VALUE.connect(self.progressBar.setValue)
    self.signals.update_ProgressBar_MAX_VALUE.connect(self.progressBar.setMaximum)
    self.signals.update_ProgressBar_MIN_VALUE.connect(self.progressBar.setMinimum)
    # self.signals.insertText_progressCursor.connect(self.progressCursor.insertText)
    self.signals.insertText_progressCursor.connect(self.progressMessage.append)
    self.signals.clear_progressMessage.connect(self.progressMessage.clear)
    self.signals.appendText_progressMessage.connect(self.progressMessage.append)
    self.signals.update_clean_up_input_dirs_flag.connect(
      self.decensor.set_clean_up_input_dirs
    )

  def updateModelAvailability(self, is_mosaic):
    self.lamaModelButton.setEnabled(not is_mosaic)
    if is_mosaic and self.lamaModelButton.isChecked():
      self.pepsiModelButton.setChecked(True)

  def decensorClicked(self):
    self.decensorButton.setEnabled(False)
    self.progressMessage.clear()
    self.progressCursor.insertText("Decensoring has begun!\n")

    self.decensor.is_mosaic = self.mosaicButton.isChecked()
    self.decensor.model_name = (
      "lama"
      if self.lamaModelButton.isChecked() and not self.decensor.is_mosaic
      else "pepsi"
    )
    self.decensor.use_tta = self.ttaCheckbox.isChecked()

    # variations count
    variationsElements = self.variationsGroupBox.children()
    variationsButtons = [
      elem for elem in variationsElements if isinstance(elem, QRadioButton)
    ]
    for vb in variationsButtons:
      if vb.isChecked():
        variations = int(vb.text())
    self.decensor.variations = variations

    self.signals.update_clean_up_input_dirs_flag.emit(
      self.cleanInputDirectoryCheckbox.isChecked()
    )

    self.decensorButton.setEnabled(False)
    self.decensor.start()
    # decensor.decensor_all_images_in_folder()

  # #centers the main window
  def center(self):
    qr = self.frameGeometry()
    cp = self.screen().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def closeEvent(self, event):
    self.decensor.wait()
    super().closeEvent(event)


if __name__ == "__main__":
  app = QApplication(sys.argv)
  ex = MainWindow()
  sys.exit(app.exec())
