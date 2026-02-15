from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from . import styles
from .blob_widget import BlobDropWidget
from .compress_button import CompressButton
from .compression_card import CompressionCard
from .info_panel import InfoPanel
from .resources import chomnom_idle, chomnom_scaled


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chomnom Model Compressor")
        self.setMinimumSize(780, 660)
        self.resize(800, 680)
        self.setStyleSheet(styles.main_window_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(40, 28, 40, 24)
        root.setSpacing(20)

        # ── Header ─────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(14)

        icon_label = QLabel()
        icon_pm = chomnom_scaled(chomnom_idle(), 44)
        icon_label.setPixmap(icon_pm)
        icon_label.setFixedSize(44, 44)
        header.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        title = QLabel("One-Click-Compress")
        title_font = QFont(styles.font_family())
        title_font.setPointSize(styles.FONT_SIZE_TITLE)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {styles.TEXT_PRIMARY};")
        title_col.addWidget(title)

        tagline = QLabel("Shrink your model to the edge!")
        tag_font = QFont(styles.font_family())
        tag_font.setPointSize(styles.FONT_SIZE_TAGLINE)
        tagline.setFont(tag_font)
        tagline.setStyleSheet(f"color: {styles.TEXT_SECONDARY};")
        title_col.addWidget(tagline)

        header.addLayout(title_col)
        header.addStretch()
        root.addLayout(header)

        # ── Separator line ─────────────────────────────────────
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {styles.BG_CARD};")
        root.addWidget(sep)

        # ── Compression technique cards ────────────────────────
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(14)
        cards_layout.addStretch()

        self._quant_card = CompressionCard(
            title="Quantization",
            description="INT8 weight compression",
            enabled=True,
            selected=True,
        )
        cards_layout.addWidget(self._quant_card)

        self._prune_card = CompressionCard(
            title="Pruning",
            description="Remove redundant weights",
            enabled=False,
        )
        cards_layout.addWidget(self._prune_card)

        self._distill_card = CompressionCard(
            title="Distillation",
            description="Knowledge transfer",
            enabled=False,
        )
        cards_layout.addWidget(self._distill_card)

        cards_layout.addStretch()
        root.addLayout(cards_layout)

        # ── Blob drop zones ────────────────────────────────────
        blobs_layout = QHBoxLayout()
        blobs_layout.setSpacing(32)
        blobs_layout.addStretch()

        self._dataset_blob = BlobDropWidget(
            label="Dataset",
            file_filter=".py",
            file_description=".py file",
        )
        blobs_layout.addWidget(self._dataset_blob)

        self._model_blob = BlobDropWidget(
            label="Model Weights",
            file_filter=".pt2",
            file_description=".pt2 file",
        )
        blobs_layout.addWidget(self._model_blob)

        blobs_layout.addStretch()
        root.addLayout(blobs_layout)

        # ── Compress button ────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._compress_btn = CompressButton()
        btn_layout.addWidget(self._compress_btn)
        btn_layout.addStretch()
        root.addLayout(btn_layout)

        # ── Info panel ─────────────────────────────────────────
        self._info_panel = InfoPanel()
        root.addWidget(self._info_panel)

        root.addStretch()

        # ── Signal wiring ──────────────────────────────────────
        self._dataset_blob.file_accepted.connect(self._on_file_changed)
        self._dataset_blob.file_cleared.connect(self._on_file_changed)
        self._model_blob.file_accepted.connect(self._on_model_accepted)
        self._model_blob.file_cleared.connect(self._on_model_cleared)
        self._compress_btn.clicked.connect(self._on_compress)

    def _check_readiness(self):
        has_dataset = self._dataset_blob._accepted_path is not None
        has_model = self._model_blob._accepted_path is not None
        self._compress_btn.set_ready(has_dataset and has_model)

    def _on_file_changed(self, *_args):
        self._check_readiness()

    def _on_model_accepted(self, path: str):
        self._info_panel.update_info(path)
        self._check_readiness()

    def _on_model_cleared(self):
        self._info_panel.clear_info()
        self._check_readiness()

    def _on_compress(self):
        dataset = self._dataset_blob._accepted_path
        model = self._model_blob._accepted_path
        print(f"[COMPRESS] Dataset: {dataset}")
        print(f"[COMPRESS] Model:   {model}")
        print("[COMPRESS] Running INT8 quantization... (stub)")
