# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : live_yolo.py

"""
Live webcam YOLO inference with bounding-box overlay and FPS counter.

Loads any exported YOLOv8 model (.pt2 or .pte), grabs frames from
the default webcam, runs inference as fast as possible, and displays
annotated results in an OpenCV window.

Usage:
    python -m eval.live_yolo weights/yolov8s.pt2
    python -m eval.live_yolo weights/yolov8s_int8.pt2
    python -m eval.live_yolo weights/yolov8s_int8_xnnpack.pte
    python -m eval.live_yolo weights/yolov8s.pt2 --conf 0.5 --iou 0.5
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops

# Register quantized ops so INT8 .pt2 files can be loaded
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    prepare_pt2e,
    convert_pt2e,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_SIZE = 640

# COCO 80-class names (index-aligned with YOLOv8 output columns 4..83)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Distinct colors for the first 20 classes, then cycle
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (182, 247, 67), (0, 152, 51), (131, 120, 255), (255, 128, 0),
    (255, 32, 32), (180, 0, 255), (255, 0, 200), (192, 192, 192),
    (128, 128, 0), (0, 128, 128), (64, 0, 128), (128, 64, 0),
]


# ---------------------------------------------------------------------------
# Model loaders (reused from inference_harness)
# ---------------------------------------------------------------------------

def load_pt2_model(model_path: str):
    """Load a torch.export .pt2 program (FP32 or INT8 q/dq)."""
    ep = torch.export.load(model_path)
    model = ep.module()

    @torch.no_grad()
    def predict(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    return predict


def load_pte_model(model_path: str):
    """Load an ExecuTorch .pte program via the ExecuTorch runtime."""
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")

    def predict(x: torch.Tensor) -> torch.Tensor:
        inputs = [x.contiguous()]
        outputs = method.execute(inputs)
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return out

    return predict


def load_model(model_path: str):
    """Auto-detect format and return a predict_fn(tensor) -> tensor."""
    suffix = Path(model_path).suffix
    if suffix == ".pt2":
        return load_pt2_model(model_path)
    elif suffix == ".pte":
        return load_pte_model(model_path)
    else:
        raise ValueError(
            f"Unsupported model format '{suffix}'. Expected .pt2 or .pte"
        )


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV frame to a (1, 3, 640, 640) float tensor in [0, 1]."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    tensor = torch.from_numpy(resized).float().div(255.0)  # HWC [0,1]
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)           # (1, 3, H, W)
    return tensor


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_detections(
    raw_output: torch.Tensor,
    frame_h: int,
    frame_w: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict]:
    """Decode raw YOLOv8 output into a list of detection dicts.

    Args:
        raw_output: Model output tensor of shape (1, 84, 8400).
        frame_h: Original frame height (for rescaling boxes).
        frame_w: Original frame width (for rescaling boxes).
        conf_threshold: Minimum class confidence to keep.
        iou_threshold: IoU threshold for NMS.

    Returns:
        List of dicts with keys: box (x1,y1,x2,y2), score, class_id, label.
    """
    # raw_output shape: (1, 84, 8400) -> squeeze batch -> (84, 8400)
    pred = raw_output.squeeze(0)  # (84, 8400)
    pred = pred.T                 # (8400, 84)

    # Split into boxes and class scores
    boxes_cxcywh = pred[:, :4]    # (8400, 4) -- cx, cy, w, h in 640-space
    class_scores = pred[:, 4:]    # (8400, 80)

    # Best class per anchor
    max_scores, class_ids = class_scores.max(dim=1)  # (8400,)

    # Confidence filter
    mask = max_scores > conf_threshold
    if mask.sum() == 0:
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # Convert cx,cy,w,h -> x1,y1,x2,y2
    cx, cy, w, h = boxes_cxcywh.unbind(dim=1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)

    # NMS
    keep = torchvision.ops.nms(boxes_xyxy, max_scores, iou_threshold)
    boxes_xyxy = boxes_xyxy[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]

    # Scale boxes from 640x640 back to original frame dimensions
    scale_x = frame_w / MODEL_INPUT_SIZE
    scale_y = frame_h / MODEL_INPUT_SIZE
    boxes_xyxy[:, 0] *= scale_x
    boxes_xyxy[:, 2] *= scale_x
    boxes_xyxy[:, 1] *= scale_y
    boxes_xyxy[:, 3] *= scale_y

    # Build result list
    detections = []
    for i in range(boxes_xyxy.shape[0]):
        cid = int(class_ids[i])
        detections.append({
            "box": boxes_xyxy[i].tolist(),
            "score": float(max_scores[i]),
            "class_id": cid,
            "label": COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid),
        })

    return detections


# ---------------------------------------------------------------------------
# Output diagnostics
# ---------------------------------------------------------------------------

def analyze_output(raw_output: torch.Tensor) -> dict:
    """Compute quick diagnostics for YOLO raw output tensors."""
    y = raw_output.detach()
    if isinstance(y, torch.Tensor):
        y = y.float().cpu()
    else:
        raise TypeError("Expected torch.Tensor output")

    info = {
        "shape": tuple(y.shape),
        "dtype": str(raw_output.dtype),
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "class_nonzero_frac": 0.0,
        "class_abs_max": 0.0,
        "class_q50": 0.0,
        "class_q90": 0.0,
        "class_q99": 0.0,
    }

    # Expected YOLO layout in this project: (1, 84, 8400)
    if y.ndim == 3 and y.shape[1] >= 84:
        p = y.squeeze(0).T
        class_scores = p[:, 4:]
        max_scores, _ = class_scores.max(dim=1)
        info["class_nonzero_frac"] = float((class_scores != 0).float().mean())
        info["class_abs_max"] = float(class_scores.abs().max())
        info["class_q50"] = float(torch.quantile(max_scores, 0.50))
        info["class_q90"] = float(torch.quantile(max_scores, 0.90))
        info["class_q99"] = float(torch.quantile(max_scores, 0.99))

    return info


def print_output_diagnostics(info: dict, prefix: str = "output") -> None:
    print(
        f"  {prefix}: shape={info['shape']} dtype={info['dtype']} "
        f"min={info['min']:.4g} max={info['max']:.4g} mean={info['mean']:.4g}"
    )
    print(
        f"  {prefix}: class_nonzero_frac={info['class_nonzero_frac']:.6f} "
        f"class_abs_max={info['class_abs_max']:.4g} "
        f"q50={info['class_q50']:.4g} q90={info['class_q90']:.4g} "
        f"q99={info['class_q99']:.4g}"
    )


def output_is_degenerate(info: dict) -> bool:
    """Heuristic for broken detector heads (e.g., class channels all zero)."""
    return (
        info["class_nonzero_frac"] < 1e-6
        or info["class_abs_max"] < 1e-8
        or info["class_q99"] < 1e-5
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores onto the frame."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        score = det["score"]
        label = det["label"]
        cid = det["class_id"]
        color = _PALETTE[cid % len(_PALETTE)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1,
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame, text, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw the FPS counter in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


def draw_warning(frame: np.ndarray, message: str) -> np.ndarray:
    """Draw a high-visibility warning banner."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 180), cv2.FILLED)
    cv2.putText(
        frame,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# Camera selection and main loop
# ---------------------------------------------------------------------------

def _backend_name(backend: int) -> str:
    if backend == cv2.CAP_AVFOUNDATION:
        return "AVFoundation"
    return "Default"


def _frame_has_signal(frame: np.ndarray) -> bool:
    """Return True when a frame is not near-uniform black."""
    if frame is None:
        return False
    # Mean checks darkness, std catches frozen/uniform frames.
    return float(frame.mean()) > 2.0 and float(frame.std()) > 2.0


def open_macbook_camera(preferred_camera: int | None) -> tuple[cv2.VideoCapture, int, int, np.ndarray]:
    """Open a working camera stream on macOS by probing indices/backends.

    Returns:
        (capture, camera_index, backend_id, first_good_frame)
    """
    backend_candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    index_candidates = [0, 1, 2, 3]
    if preferred_camera is not None and preferred_camera not in index_candidates:
        index_candidates = [preferred_camera] + index_candidates
    elif preferred_camera is not None:
        index_candidates = [preferred_camera] + [
            i for i in index_candidates if i != preferred_camera
        ]

    for backend in backend_candidates:
        for camera_index in index_candidates:
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Request a reasonable mode. Not all drivers honor these.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

            first_frame = None
            for _ in range(100):
                ret, frame = cap.read()
                if ret and frame is not None:
                    first_frame = frame
                    if _frame_has_signal(frame):
                        print(
                            f"  Camera ready on index {camera_index} "
                            f"({_backend_name(backend)})."
                        )
                        return cap, camera_index, backend, frame
                time.sleep(0.03)

            # Keep searching if this stream is open but still black/unusable.
            cap.release()

    raise RuntimeError(
        "Could not open a usable webcam feed.\n"
        "  - Make sure no other app is using the camera.\n"
        "  - In macOS System Settings > Privacy & Security > Camera,\n"
        "    allow Camera access for your terminal/IDE.\n"
        "  - If using Continuity Camera, set iPhone as camera explicitly."
    )


def run_live(
    model_path: str,
    camera_id: int | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    warmup: int = 5,
    debug_output: bool = False,
) -> None:
    """Open webcam, run YOLO inference, and display annotated frames.

    Args:
        model_path: Path to a .pt2 or .pte YOLO model.
        camera_id: OpenCV camera device index.
        conf_threshold: Minimum detection confidence.
        iou_threshold: NMS IoU threshold.
        warmup: Number of warmup inference passes (not displayed).
    """
    print(f"Loading model: {model_path}")
    predict_fn = load_model(model_path)

    print("Opening webcam...")
    try:
        cap, chosen_camera_id, chosen_backend, frame = open_macbook_camera(camera_id)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    frame_h, frame_w = frame.shape[:2]
    print(f"Using camera index: {chosen_camera_id}")
    print(f"Using backend: {_backend_name(chosen_backend)}")
    print(f"Webcam resolution: {frame_w}x{frame_h}")

    # Warmup passes
    print(f"Warming up ({warmup} frames)...")
    dummy = preprocess_frame(frame)
    warmup_info = None
    for _ in range(warmup):
        warmup_raw = predict_fn(dummy)
        if warmup_info is None:
            warmup_info = analyze_output(warmup_raw)
    print("Warmup complete. Starting live inference (press 'q' to quit).\n")
    if warmup_info is not None:
        print_output_diagnostics(warmup_info, prefix="warmup output")
        if output_is_degenerate(warmup_info):
            print(
                "  WARNING: output appears degenerate (class channels near-zero).\n"
                "  This usually means the .pte conversion damaged detection logits,\n"
                "  not a webcam or drawing bug.",
                file=sys.stderr,
            )

    # FPS tracking with a rolling window
    fps_window: deque[float] = deque(maxlen=30)
    frame_count = 0
    degenerate_warned = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Briefly retry if stream glitches.
                time.sleep(0.02)
                continue

            t0 = time.perf_counter()

            # Preprocess
            tensor = preprocess_frame(frame)

            # Inference
            raw_output = predict_fn(tensor)
            output_info = analyze_output(raw_output)
            is_degenerate = output_is_degenerate(output_info)

            # Postprocess
            detections = postprocess_detections(
                raw_output, frame_h, frame_w,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

            elapsed = time.perf_counter() - t0
            fps_window.append(elapsed)
            fps = len(fps_window) / sum(fps_window) if fps_window else 0.0

            # Draw results
            frame = draw_detections(frame, detections)
            frame = draw_fps(frame, fps)
            if is_degenerate:
                frame = draw_warning(frame, "WARNING: model class logits are near zero")

            cv2.imshow("YOLOv8 Live", frame)

            # Print periodic stats to terminal
            frame_count += 1
            if frame_count % 30 == 0:
                n_det = len(detections)
                brightness = float(frame.mean())
                print(
                    f"  frame {frame_count:>5d} | "
                    f"{elapsed * 1000:.1f} ms | "
                    f"{fps:.1f} FPS | "
                    f"{n_det} detections | "
                    f"brightness {brightness:.1f}"
                )
                if debug_output:
                    print_output_diagnostics(output_info, prefix="live output")
                if is_degenerate and not degenerate_warned:
                    print(
                        "  WARNING: degenerate output detected at runtime. "
                        "Class channels are near-zero.",
                        file=sys.stderr,
                    )
                    degenerate_warned = True

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live webcam YOLO inference with bounding-box overlay",
    )
    parser.add_argument(
        "model",
        help="Path to a YOLO model (.pt2 or .pte)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Preferred camera device index (default: auto-probe)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup inference passes (default: 5)",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Print raw output diagnostics every 30 frames",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    run_live(
        model_path=args.model,
        camera_id=args.camera,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        warmup=args.warmup,
        debug_output=args.debug_output,
    )


if __name__ == "__main__":
    main()
