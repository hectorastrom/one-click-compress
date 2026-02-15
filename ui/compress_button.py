from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from . import styles


class CompressButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("COMPRESS", parent)
        self.setFixedHeight(50)
        self.setMinimumWidth(220)
        self.set_ready(False)

    def set_ready(self, ready: bool):
        self._ready = ready
        self.setEnabled(ready)
        self.setStyleSheet(styles.button_stylesheet(ready))
        if ready:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
