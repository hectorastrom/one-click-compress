from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QRectF,
    QSequentialAnimationGroup,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from . import styles
from .resources import chomnom_closed, chomnom_idle, chomnom_open, chomnom_scaled

# Chomnom image size within the widget
_IMG_SIZE = 140


class BlobState(Enum):
    RESTING = auto()
    HOVER = auto()
    ACCEPTED = auto()


_STATE_PIXMAP = {
    BlobState.RESTING: chomnom_idle,
    BlobState.HOVER: chomnom_open,
    BlobState.ACCEPTED: chomnom_closed,
}


class BlobDropWidget(QWidget):
    file_accepted = pyqtSignal(str)
    file_cleared = pyqtSignal()

    def __init__(
        self,
        label: str,
        file_filter: str,
        file_description: str,
        parent=None,
    ):
        super().__init__(parent)
        self._label = label
        self._file_filter = file_filter
        self._file_description = file_description
        self._state = BlobState.RESTING
        self._accepted_path: str | None = None

        # Cached scaled pixmaps
        self._pixmap = chomnom_scaled(chomnom_idle(), _IMG_SIZE)

        # Animated property backing fields
        self._body_scale = 1.0
        self._glow_opacity = 0.0
        self._bounce_offset = 0.0

        self.setAcceptDrops(True)
        self.setFixedSize(200, 200)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._setup_idle_animation()

    # ── pyqtProperty declarations ──────────────────────────────

    def _get_body_scale(self) -> float:
        return self._body_scale

    def _set_body_scale(self, v: float):
        self._body_scale = v
        self.update()

    body_scale = pyqtProperty(float, _get_body_scale, _set_body_scale)

    def _get_glow_opacity(self) -> float:
        return self._glow_opacity

    def _set_glow_opacity(self, v: float):
        self._glow_opacity = v
        self.update()

    glow_opacity = pyqtProperty(float, _get_glow_opacity, _set_glow_opacity)

    def _get_bounce_offset(self) -> float:
        return self._bounce_offset

    def _set_bounce_offset(self, v: float):
        self._bounce_offset = v
        self.update()

    bounce_offset = pyqtProperty(float, _get_bounce_offset, _set_bounce_offset)

    # ── Idle breathing animation ───────────────────────────────

    def _setup_idle_animation(self):
        self._idle_group = QSequentialAnimationGroup(self)
        self._idle_group.setLoopCount(-1)

        up = QPropertyAnimation(self, b"bounce_offset", self)
        up.setDuration(1000)
        up.setStartValue(0.0)
        up.setEndValue(-3.0)
        up.setEasingCurve(QEasingCurve.Type.InOutSine)

        down = QPropertyAnimation(self, b"bounce_offset", self)
        down.setDuration(1000)
        down.setStartValue(-3.0)
        down.setEndValue(0.0)
        down.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._idle_group.addAnimation(up)
        self._idle_group.addAnimation(down)
        self._idle_group.start()

    # ── State transitions ──────────────────────────────────────

    def _animate_to_state(self, state: BlobState):
        self._state = state
        self._pixmap = chomnom_scaled(_STATE_PIXMAP[state](), _IMG_SIZE)

        targets = {
            BlobState.RESTING: {"body_scale": 1.0, "glow_opacity": 0.0},
            BlobState.HOVER: {"body_scale": 1.1, "glow_opacity": 0.7},
            BlobState.ACCEPTED: {"body_scale": 1.0, "glow_opacity": 0.25},
        }

        vals = targets[state]
        group = QParallelAnimationGroup(self)
        for prop_name, target_val in vals.items():
            anim = QPropertyAnimation(self, prop_name.encode(), self)
            anim.setDuration(250)
            anim.setEndValue(target_val)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            group.addAnimation(anim)
        group.start()

    # ── Drag and drop ──────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.endswith(self._file_filter):
                    event.acceptProposedAction()
                    self._animate_to_state(BlobState.HOVER)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        prev = BlobState.ACCEPTED if self._accepted_path else BlobState.RESTING
        self._animate_to_state(prev)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith(self._file_filter):
                self._accepted_path = path
                self._animate_to_state(BlobState.ACCEPTED)
                self.file_accepted.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self._accepted_path:
            self._accepted_path = None
            self._animate_to_state(BlobState.RESTING)
            self.file_cleared.emit()

    # ── Painting ───────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        img_area_h = self.height() - 50  # reserve space for labels

        # Center point for the image area
        cx = w / 2
        cy = img_area_h / 2 + self._bounce_offset

        p.save()
        p.translate(cx, cy)
        p.scale(self._body_scale, self._body_scale)

        # Glow ring behind the image
        if self._glow_opacity > 0.01:
            glow_color = QColor(styles.ACCENT_GLOW)
            glow_color.setAlphaF(min(self._glow_opacity * 0.5, 1.0))
            glow_pen = QPen(glow_color, 6)
            p.setPen(glow_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = _IMG_SIZE / 2 + 4
            p.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        # Draw the chomnom image centered
        pm = self._pixmap
        p.drawPixmap(
            int(-pm.width() / 2),
            int(-pm.height() / 2),
            pm,
        )

        p.restore()

        # Label text below image
        font = QFont(styles.font_family())
        font.setPointSize(styles.FONT_SIZE_BLOB_LABEL)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)

        label_y = img_area_h + 4
        label_rect = QRectF(0, label_y, w, 18)

        if self._accepted_path:
            p.setPen(QColor(styles.TEXT_PRIMARY))
            filename = Path(self._accepted_path).name
            p.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, filename)
        else:
            p.setPen(QColor(styles.TEXT_SECONDARY))
            p.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._label)

        # File type hint
        if not self._accepted_path:
            hint_font = QFont(styles.font_family())
            hint_font.setPointSize(10)
            p.setFont(hint_font)
            p.setPen(QColor(styles.TEXT_DISABLED))
            hint_rect = QRectF(0, label_y + 17, w, 16)
            p.drawText(hint_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._file_description)

        p.end()
