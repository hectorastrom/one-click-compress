import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import styles


class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        font = QFont(styles.font_family())
        font.setPointSize(styles.FONT_SIZE_INFO)

        self._model_label = QLabel("Drop in a model and dataset to see compression estimate")
        self._model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._model_label.setFont(font)
        self._model_label.setStyleSheet(f"color: {styles.TEXT_SECONDARY};")

        self._estimate_label = QLabel("")
        self._estimate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._estimate_label.setFont(font)
        self._estimate_label.setStyleSheet(f"color: {styles.TEXT_DISABLED};")

        layout.addWidget(self._model_label)
        layout.addWidget(self._estimate_label)

    def update_info(self, model_path: str):
        try:
            size_bytes = os.path.getsize(model_path)
            size_mb = size_bytes / (1024 * 1024)
            filename = os.path.basename(model_path)
            self._model_label.setText(f"Model: {filename} ({size_mb:.1f} MB)")
            self._model_label.setStyleSheet(f"color: {styles.TEXT_PRIMARY};")

            est_mb = size_mb / 4.0
            self._estimate_label.setText(f"Estimated: ~{est_mb:.1f} MB (4x reduction)")
            self._estimate_label.setStyleSheet(f"color: {styles.ACCENT_SUCCESS};")
        except OSError:
            self._model_label.setText(f"Model: {os.path.basename(model_path)}")
            self._model_label.setStyleSheet(f"color: {styles.TEXT_PRIMARY};")
            self._estimate_label.setText("")

    def clear_info(self):
        self._model_label.setText("Drop a model to see compression estimate")
        self._model_label.setStyleSheet(f"color: {styles.TEXT_SECONDARY};")
        self._estimate_label.setText("")
