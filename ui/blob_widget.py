from enum import Enum, auto
from pathlib import Path

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QRectF,
    QSequentialAnimationGroup,
    QTimer,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget

from . import styles
from .resources import (
    chomnom_chew_1,
    chomnom_chew_2,
    chomnom_closed,
    chomnom_idle,
    chomnom_open,
    chomnom_scaled,
)

_IMG_SIZE = 120

# Chewing animation frame sequence: closed -> chew_1 -> chew_2 -> chew_1 -> (repeat)
_CHEW_SEQUENCE_FUNCS = [chomnom_closed, chomnom_chew_1, chomnom_chew_2, chomnom_chew_1]


class BlobState(Enum):
    RESTING = auto()
    HOVER = auto()
    REJECTED = auto()
    ACCEPTED = auto()
    CHEWING = auto()


_STATE_PIXMAP = {
    BlobState.RESTING: chomnom_idle,
    BlobState.HOVER: chomnom_open,
    BlobState.REJECTED: chomnom_idle,
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
        self._is_hovered = False

        self._pixmap = chomnom_scaled(chomnom_idle(), _IMG_SIZE)

        # Animated property backing fields
        self._body_scale = 1.0
        self._bounce_offset = 0.0

        # Chewing animation state
        self._chew_timer: QTimer | None = None
        self._chew_frame_index = 0

        self.setAcceptDrops(True)
        self.setFixedSize(200, 210)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._setup_idle_animation()

    # ── pyqtProperty declarations ──────────────────────────────

    def _get_body_scale(self) -> float:
        return self._body_scale

    def _set_body_scale(self, v: float):
        self._body_scale = v
        self.update()

    body_scale = pyqtProperty(float, _get_body_scale, _set_body_scale)

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
        if state == BlobState.CHEWING:
            return  # chewing is handled by start_chewing()
        self._state = state
        self._pixmap = chomnom_scaled(_STATE_PIXMAP[state](), _IMG_SIZE)

        targets = {
            BlobState.RESTING: {"body_scale": 1.0},
            BlobState.HOVER: {"body_scale": 1.08},
            BlobState.REJECTED: {"body_scale": 0.95},
            BlobState.ACCEPTED: {"body_scale": 1.0},
        }

        vals = targets[state]
        group = QParallelAnimationGroup(self)
        for prop_name, target_val in vals.items():
            anim = QPropertyAnimation(self, prop_name.encode(), self)
            anim.setDuration(200)
            anim.setEndValue(target_val)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            group.addAnimation(anim)
        group.start()

    # ── Chewing animation ──────────────────────────────────────

    def start_chewing(self):
        """Start the chewing loop animation (closed -> chew_1 -> chew_2 -> chew_1 -> ...)"""
        self._state = BlobState.CHEWING
        self._chew_frame_index = 0
        self._pixmap = chomnom_scaled(_CHEW_SEQUENCE_FUNCS[0](), _IMG_SIZE)
        self.setAcceptDrops(False)

        self._chew_timer = QTimer(self)
        self._chew_timer.setInterval(250)
        self._chew_timer.timeout.connect(self._advance_chew_frame)
        self._chew_timer.start()
        self.update()

    def _advance_chew_frame(self):
        self._chew_frame_index = (self._chew_frame_index + 1) % len(_CHEW_SEQUENCE_FUNCS)
        self._pixmap = chomnom_scaled(_CHEW_SEQUENCE_FUNCS[self._chew_frame_index](), _IMG_SIZE)
        self.update()

    def stop_chewing(self):
        """Stop chewing and return to ACCEPTED state."""
        if self._chew_timer is not None:
            self._chew_timer.stop()
            self._chew_timer = None
        self.setAcceptDrops(True)
        self._animate_to_state(BlobState.ACCEPTED)

    # ── Drag and drop ──────────────────────────────────────────

    def _matches_filter(self, path: str) -> bool:
        return path.endswith(self._file_filter)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            for url in event.mimeData().urls():
                if self._matches_filter(url.toLocalFile()):
                    self._animate_to_state(BlobState.HOVER)
                    return
            self._animate_to_state(BlobState.REJECTED)
            return
        event.ignore()

    def dragLeaveEvent(self, event):
        prev = BlobState.ACCEPTED if self._accepted_path else BlobState.RESTING
        self._animate_to_state(prev)

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if self._matches_filter(path):
                self._accepted_path = path
                self._animate_to_state(BlobState.ACCEPTED)
                self.file_accepted.emit(path)
                event.acceptProposedAction()
                return
        prev = BlobState.ACCEPTED if self._accepted_path else BlobState.RESTING
        self._animate_to_state(prev)
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self._accepted_path and self._state != BlobState.CHEWING:
            self._clear_selected_file()

    def enterEvent(self, event):
        self._is_hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._is_hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._accepted_path
            and self._state != BlobState.CHEWING
        ):
            self._clear_selected_file()
            event.accept()
            return
        super().mousePressEvent(event)

    def _clear_selected_file(self):
        self._accepted_path = None
        self._animate_to_state(BlobState.RESTING)
        self.file_cleared.emit()

    # ── Painting ───────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w = self.width()
        label_space = 44
        zone_h = self.height() - label_space
        zone_rect = QRectF(4, 4, w - 8, zone_h - 8)

        # ── Drop zone background with dashed/solid border ──────
        p.setBrush(QColor(styles.BG_CARD))

        if self._state == BlobState.CHEWING:
            pen = QPen(QColor(styles.ACCENT_PRIMARY), 2)
            pen.setStyle(Qt.PenStyle.SolidLine)
        elif self._state == BlobState.ACCEPTED:
            pen = QPen(QColor(styles.ACCENT_SUCCESS), 2)
            pen.setStyle(Qt.PenStyle.SolidLine)
        elif self._state == BlobState.HOVER:
            pen = QPen(QColor(styles.ACCENT_PRIMARY), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])
        elif self._state == BlobState.REJECTED:
            pen = QPen(QColor("#ef4444"), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])
        else:
            pen = QPen(QColor(styles.TEXT_DISABLED), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])

        p.setPen(pen)
        p.drawRoundedRect(zone_rect, 14, 14)

        # Hover affordance for clearing an attached file
        if self._accepted_path and self._is_hovered and self._state != BlobState.CHEWING:
            close_r = 11
            close_cx = zone_rect.right() - 14
            close_cy = zone_rect.top() + 14
            close_rect = QRectF(close_cx - close_r, close_cy - close_r, close_r * 2, close_r * 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 130))
            p.drawEllipse(close_rect)

            x_pen = QPen(QColor(styles.TEXT_PRIMARY), 1.8)
            p.setPen(x_pen)
            inset = 5
            p.drawLine(
                int(close_rect.left() + inset),
                int(close_rect.top() + inset),
                int(close_rect.right() - inset),
                int(close_rect.bottom() - inset),
            )
            p.drawLine(
                int(close_rect.right() - inset),
                int(close_rect.top() + inset),
                int(close_rect.left() + inset),
                int(close_rect.bottom() - inset),
            )

        # ── Chomnom image ──────────────────────────────────────
        cx = w / 2
        cy = zone_h / 2 + self._bounce_offset

        p.save()
        p.translate(cx, cy)
        p.scale(self._body_scale, self._body_scale)

        pm = self._pixmap
        p.drawPixmap(int(-pm.width() / 2), int(-pm.height() / 2), pm)

        # Gray overlay for rejected state
        if self._state == BlobState.REJECTED:
            overlay = QColor(0, 0, 0, 120)
            p.setBrush(overlay)
            p.setPen(Qt.PenStyle.NoPen)
            r = pm.width() / 2
            p.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        p.restore()

        # ── Labels below the drop zone ─────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        font = QFont(styles.font_family())
        font.setPointSize(styles.FONT_SIZE_BLOB_LABEL)
        font.setWeight(QFont.Weight.DemiBold)
        p.setFont(font)

        label_y = zone_h + 2
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
