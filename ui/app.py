from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
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
from .info_panel import InfoPanel
from .resources import logo_scaled


# ── Compression worker (runs in background thread) ────────────


class _CompressionWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_path: str, dataset_path: str):
        super().__init__()
        self._model_path = model_path
        self._dataset_path = dataset_path

    def run(self):
        try:
            self.progress.emit("Loading model and dataset...")
            from compression.compress import compress_and_evaluate

            output_dir = str(Path(self._model_path).parent / "compressed")
            self.progress.emit("Running INT8 quantization...")

            results = compress_and_evaluate(
                model_path=self._model_path,
                dataset_path=self._dataset_path,
                output_dir=output_dir,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main window ───────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chomnom")
        self.setFixedSize(600, 560)
        self.setStyleSheet(styles.main_window_stylesheet())

        self._worker_thread: QThread | None = None

        central = QWidget()
        central.setObjectName("central_surface")
        central.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        central.setStyleSheet(f"QWidget#central_surface {{ background-color: {styles.BG_DEEP}; }}")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(40, 24, 40, 12)
        root.setSpacing(16)

        # ── Header — centered logo ─────────────────────────────
        logo_label = QLabel()
        logo_label.setPixmap(logo_scaled(340))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(logo_label)

        # ── Blob drop zones ────────────────────────────────────
        blobs_layout = QHBoxLayout()
        blobs_layout.setSpacing(32)
        blobs_layout.addStretch()

        self._dataset_blob = BlobDropWidget(
            label="Dataset",
            file_filter=".pt",
            file_description=".pt file",
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

        # ── Signal wiring ──────────────────────────────────────
        self._dataset_blob.file_accepted.connect(self._on_file_changed)
        self._dataset_blob.file_cleared.connect(self._on_file_changed)
        self._model_blob.file_accepted.connect(self._on_model_accepted)
        self._model_blob.file_cleared.connect(self._on_model_cleared)
        self._compress_btn.clicked.connect(self._on_compress)

    # ── Readiness check ────────────────────────────────────────

    def _check_readiness(self):
        has_dataset = self._dataset_blob._accepted_path is not None
        has_model = self._model_blob._accepted_path is not None
        is_running = self._worker_thread is not None and self._worker_thread.isRunning()
        self._compress_btn.set_ready(has_dataset and has_model and not is_running)

    def _on_file_changed(self, *_args):
        self._check_readiness()

    def _on_model_accepted(self, path: str):
        self._info_panel.update_info(path)
        self._check_readiness()

    def _on_model_cleared(self):
        self._info_panel.clear_info()
        self._check_readiness()

    # ── Compression ────────────────────────────────────────────

    def _on_compress(self):
        dataset = self._dataset_blob._accepted_path
        model = self._model_blob._accepted_path
        if not dataset or not model:
            return

        self._compress_btn.set_ready(False)
        self._compress_btn.setText("COMPRESSING...")
        self._info_panel.show_progress("Starting compression pipeline...")

        # Start chewing animation on both blobs
        self._dataset_blob.start_chewing()
        self._model_blob.start_chewing()

        # Launch worker thread
        self._worker_thread = QThread()
        self._worker = _CompressionWorker(model, dataset)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_compress_progress)
        self._worker.finished.connect(self._on_compress_finished)
        self._worker.error.connect(self._on_compress_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)

        self._worker_thread.start()

    def _on_compress_progress(self, message: str):
        self._info_panel.show_progress(message)

    def _on_compress_finished(self, results: dict):
        self._dataset_blob.stop_chewing()
        self._model_blob.stop_chewing()
        self._compress_btn.setText("COMPRESS")
        self._check_readiness()
        self._info_panel.show_results(results)

    def _on_compress_error(self, message: str):
        self._dataset_blob.stop_chewing()
        self._model_blob.stop_chewing()
        self._compress_btn.setText("COMPRESS")
        self._check_readiness()
        self._info_panel.show_error(message)
