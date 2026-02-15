from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from . import styles


class CompressionCard(QWidget):
    clicked = pyqtSignal()

    def __init__(
        self,
        title: str,
        description: str,
        enabled: bool = True,
        selected: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._description = description
        self._enabled = enabled
        self._selected = selected
        self._hovered = False

        self.setFixedSize(200, 100)
        if enabled:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if self._enabled:
            self.clicked.emit()

    def enterEvent(self, event):
        if self._enabled:
            self._hovered = True
            self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(1, 1, w - 2, h - 2)

        # Background
        bg = styles.BG_CARD_HOVER if self._hovered else styles.BG_CARD
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(bg))
        p.drawRoundedRect(rect, 14, 14)

        # Border
        if self._selected and self._enabled:
            p.setPen(QPen(QColor(styles.ACCENT_PRIMARY), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, 14, 14)
        elif not self._enabled:
            p.setPen(QPen(QColor(styles.TEXT_DISABLED), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect, 14, 14)

        # Accent dot for selected card
        if self._selected and self._enabled:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(styles.ACCENT_PRIMARY))
            p.drawEllipse(QRectF(w / 2 - 3, 16, 6, 6))

        # Title
        text_color = styles.TEXT_PRIMARY if self._enabled else styles.TEXT_DISABLED
        title_font = QFont(styles.font_family())
        title_font.setPointSize(styles.FONT_SIZE_CARD_TITLE)
        title_font.setWeight(QFont.Weight.Bold)
        p.setFont(title_font)
        p.setPen(QColor(text_color))
        title_y = 28 if (self._selected and self._enabled) else 24
        p.drawText(QRectF(0, title_y, w, 22), Qt.AlignmentFlag.AlignCenter, self._title)

        # Description
        desc_font = QFont(styles.font_family())
        desc_font.setPointSize(styles.FONT_SIZE_CARD_DESC)
        p.setFont(desc_font)
        desc_color = styles.TEXT_SECONDARY if self._enabled else styles.TEXT_DISABLED
        p.setPen(QColor(desc_color))
        p.drawText(QRectF(12, title_y + 24, w - 24, 18), Qt.AlignmentFlag.AlignCenter, self._description)

        # "Coming Soon" badge for disabled cards
        if not self._enabled:
            badge_font = QFont(styles.font_family())
            badge_font.setPointSize(8)
            badge_font.setWeight(QFont.Weight.DemiBold)
            badge_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
            p.setFont(badge_font)

            badge_w, badge_h = 80, 18
            badge_x = (w - badge_w) / 2
            badge_y = h - badge_h - 10

            p.setPen(Qt.PenStyle.NoPen)
            badge_bg = QColor(styles.TEXT_DISABLED)
            badge_bg.setAlphaF(0.4)
            p.setBrush(badge_bg)
            p.drawRoundedRect(QRectF(badge_x, badge_y, badge_w, badge_h), 9, 9)

            p.setPen(QColor(styles.TEXT_SECONDARY))
            p.drawText(
                QRectF(badge_x, badge_y, badge_w, badge_h),
                Qt.AlignmentFlag.AlignCenter,
                "COMING SOON",
            )

        p.end()
