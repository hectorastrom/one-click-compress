from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

_STATES_DIR = Path(__file__).parent / "chomnom_states"

_cache: dict[str, QPixmap] = {}


def _load(name: str) -> QPixmap:
    if name not in _cache:
        path = _STATES_DIR / name
        pm = QPixmap(str(path))
        if pm.isNull():
            raise FileNotFoundError(f"Missing asset: {path}")
        _cache[name] = pm
    return _cache[name]


def chomnom_idle() -> QPixmap:
    return _load("chomnom_idle.png")


def chomnom_open() -> QPixmap:
    return _load("chomnom_open.png")


def chomnom_closed() -> QPixmap:
    return _load("chomnom_closed.png")


def chomnom_chew_1() -> QPixmap:
    return _load("chomnom_chew_1.png")


def chomnom_chew_2() -> QPixmap:
    return _load("chomnom_chew_2.png")


def chomnom_logo() -> QPixmap:
    return _load("chomnom_logo.png")


def chomnom_scaled(pixmap: QPixmap, size: int) -> QPixmap:
    """Return a smoothly scaled copy of a chomnom pixmap."""
    return pixmap.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def logo_scaled(width: int) -> QPixmap:
    """Return the logo scaled to a given width, keeping aspect ratio."""
    pm = chomnom_logo()
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
