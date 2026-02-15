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

        self._verdict_label = QLabel("")
        self._verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._verdict_label.setFont(font)
        self._verdict_label.setStyleSheet(f"color: {styles.TEXT_DISABLED};")

        layout.addWidget(self._model_label)
        layout.addWidget(self._estimate_label)
        layout.addWidget(self._verdict_label)

    def update_info(self, model_path: str):
        self._verdict_label.setText("")
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

    def show_results(self, results: dict):
        # Size info — compute from output file paths
        delta_pct = results.get("size_delta_pct", 0)
        pte_path = results.get("pte", "")
        int8_path = results.get("int8_pt2", "")

        # Compute actual file sizes from paths
        pte_mb = _file_size_mb(pte_path)
        int8_mb = _file_size_mb(int8_path)

        size_parts = []
        if int8_mb > 0:
            size_parts.append(f"PT2: {int8_mb:.1f} MB")
        if pte_mb > 0:
            size_parts.append(f"PTE: {pte_mb:.1f} MB")
        if delta_pct != 0:
            size_parts.append(f"{delta_pct:+.1f}%")

        self._model_label.setText("  |  ".join(size_parts) if size_parts else "Compression complete")
        self._model_label.setStyleSheet(f"color: {styles.TEXT_PRIMARY};")

        # Metrics line
        cos_mean = results.get("cos_mean", 0)
        latency_ratio = results.get("latency_ratio", 0)
        parts = [f"Cosine: {cos_mean:.4f}"]
        if latency_ratio > 0:
            parts.append(f"Latency: {latency_ratio:.2f}x")
        if results.get("argmax_supported"):
            agreement = results.get("argmax_agreement", 0)
            parts.append(f"Agreement: {agreement:.1%}")

        self._estimate_label.setText("  |  ".join(parts))
        self._estimate_label.setStyleSheet(f"color: {styles.ACCENT_SUCCESS};")

        # Verdict
        verdict_text = ""
        if cos_mean > 0.999:
            verdict_text = "Excellent — outputs near-identical to FP32"
        elif cos_mean > 0.99:
            verdict_text = "Good — minor drift, unlikely to affect accuracy"
        elif cos_mean > 0.95:
            verdict_text = "Acceptable — some drift, validate on your task"
        else:
            verdict_text = "Degraded — significant loss, try more calibration data"

        if cos_mean > 0.95:
            self._verdict_label.setStyleSheet(f"color: {styles.ACCENT_SUCCESS};")
        else:
            self._verdict_label.setStyleSheet("color: #ef4444;")
        self._verdict_label.setText(verdict_text)

    def show_error(self, message: str):
        self._model_label.setText(f"Error: {message}")
        self._model_label.setStyleSheet("color: #ef4444;")
        self._estimate_label.setText("")
        self._verdict_label.setText("")

    def show_progress(self, message: str):
        self._estimate_label.setText(message)
        self._estimate_label.setStyleSheet(f"color: {styles.ACCENT_GLOW};")
        self._verdict_label.setText("")

    def clear_info(self):
        self._model_label.setText("Drop in a model and dataset to see compression estimate")
        self._model_label.setStyleSheet(f"color: {styles.TEXT_SECONDARY};")
        self._estimate_label.setText("")
        self._verdict_label.setText("")


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0
