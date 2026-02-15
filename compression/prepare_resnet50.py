# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : prepare_resnet50.py

"""
End-to-end: download ResNet-50 (ImageNet-1K pretrained), build calibration
data from ImageNette (a 10-class subset of ImageNet), export to .pt2,
and run the INT8 quantization pipeline.

ImageNette is a freely available ~350 MB subset of ImageNet published by
fast.ai containing 10 easily-classifiable classes (~9 500 training images).

Usage:
    python -m compression.prepare_resnet50                   # full pipeline
    python -m compression.prepare_resnet50 --step download   # download only
    python -m compression.prepare_resnet50 --step export     # export only
    python -m compression.prepare_resnet50 --step dataset    # dataset only
    python -m compression.prepare_resnet50 --step quantize   # quantize only
"""

import argparse
import tarfile
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from compression.quantize import universal_compress
from compression.utils import save_dataset, save_torch_export

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path("weights")
DATA_DIR = Path("data")
IMAGENETTE_DIR = DATA_DIR / "imagenette2-320"
IMAGENETTE_TRAIN = IMAGENETTE_DIR / "train"
RESNET_PT2 = WEIGHTS_DIR / "resnet50.pt2"
RESNET_INT8_PT2 = WEIGHTS_DIR / "resnet50_int8.pt2"
CALIBRATION_PT = DATA_DIR / "imagenette_calibration.pt"
IMAGENETTE_URL = (
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
)
INPUT_SIZE = (1, 3, 224, 224)  # B, C, H, W

# ~500 images * 3 * 224 * 224 * 4 bytes = ~290 MB as float32 tensors
NUM_CALIBRATION_IMAGES = 500

# ---------------------------------------------------------------------------
# ImageNette WordNet-ID -> ImageNet-1K class index mapping
# ---------------------------------------------------------------------------
IMAGENETTE_WNID_TO_CLASS: dict[str, int] = {
    "n01440764": 0,     # tench
    "n02102040": 217,   # English springer
    "n02979186": 482,   # cassette player
    "n03000684": 491,   # chain saw
    "n03028079": 497,   # church
    "n03394916": 566,   # French horn
    "n03417042": 569,   # garbage truck
    "n03425413": 571,   # gas pump
    "n03445777": 574,   # golf ball
    "n03888257": 701,   # parachute
}


# ---------------------------------------------------------------------------
# Step 1 -- Acquire assets
# ---------------------------------------------------------------------------

def download_imagenette() -> Path:
    """Download and extract ImageNette-320 (~350 MB, 10 ImageNet classes)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGENETTE_TRAIN.exists() and any(IMAGENETTE_TRAIN.iterdir()):
        print(f"[skip] ImageNette already present at {IMAGENETTE_DIR}")
        return IMAGENETTE_TRAIN

    tgz_path = DATA_DIR / "imagenette2-320.tgz"
    if not tgz_path.exists():
        print(f"Downloading ImageNette from {IMAGENETTE_URL} ...")
        torch.hub.download_url_to_file(IMAGENETTE_URL, str(tgz_path))

    print("Extracting ImageNette...")
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(DATA_DIR, filter="data")
    tgz_path.unlink()  # clean up after extraction

    print(f"ImageNette ready at {IMAGENETTE_DIR}")
    return IMAGENETTE_TRAIN


# ---------------------------------------------------------------------------
# Step 2 -- Calibration dataset
# ---------------------------------------------------------------------------

class ImageNetteDataset(Dataset):
    """ImageNette calibration dataset returning (x, y) tuples.

    x is a float tensor of shape (3, 224, 224) normalized with ImageNet
    channel mean/std.  y is the corresponding ImageNet-1K class index.

    Args:
        image_dir: Root train/ directory with per-class subdirectories.
        num_images: Maximum number of images to include.
    """

    def __init__(
        self,
        image_dir: Path,
        num_images: int = NUM_CALIBRATION_IMAGES,
    ):
        self.samples: list[tuple[Path, int]] = []
        for wnid_dir in sorted(image_dir.iterdir()):
            if not wnid_dir.is_dir():
                continue
            label = IMAGENETTE_WNID_TO_CLASS.get(wnid_dir.name, -1)
            for img_path in sorted(wnid_dir.glob("*.JPEG")):
                self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No .JPEG images found in {image_dir}. "
                "Run the download step first."
            )

        self.samples = self.samples[:num_images]

        # Standard ImageNet preprocessing for ResNet-50
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),  # HWC uint8 -> CHW float [0, 1]
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        print(f"ImageNetteDataset: {len(self)} images from {image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)  # (3, 224, 224)
        return x, label


# ---------------------------------------------------------------------------
# Step 3 -- Export to .pt2
# ---------------------------------------------------------------------------

def export_resnet50_pt2() -> Path:
    """Load ResNet-50 (ImageNet-1K V1 weights) and export to .pt2."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if RESNET_PT2.exists():
        print(f"[skip] {RESNET_PT2} already exists")
        return RESNET_PT2

    print("Loading ResNet-50 with ImageNet-1K pretrained weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.float()

    example_input = torch.randn(*INPUT_SIZE)
    print("Exporting with torch.export ...")
    save_torch_export(model, example_input, str(RESNET_PT2))
    return RESNET_PT2


# ---------------------------------------------------------------------------
# Step 4 -- Quantize
# ---------------------------------------------------------------------------

def quantize_resnet50(calibration_dataset: Dataset) -> Path:
    """Run the universal_compress pipeline on the exported .pt2."""
    print(f"\n{'='*50}")
    print("Running INT8 quantization pipeline")
    print(f"{'='*50}\n")
    output = universal_compress(
        model_path=str(RESNET_PT2),
        calibration_dataset=calibration_dataset,
        output_path=str(RESNET_INT8_PT2),
    )
    return Path(output)


# ---------------------------------------------------------------------------
# Size comparison helper
# ---------------------------------------------------------------------------

def compare_sizes(original: Path, quantized: Path):
    """Print a before/after size comparison."""
    orig_mb = original.stat().st_size / (1024 * 1024)
    quant_mb = quantized.stat().st_size / (1024 * 1024)
    reduction = (1 - quant_mb / orig_mb) * 100
    print(f"\n{'='*50}")
    print(f"Original  (.pt2): {orig_mb:.2f} MB")
    print(f"Quantized (.pt2): {quant_mb:.2f} MB")
    print(f"File size delta:  {reduction:+.1f}%")
    print()
    print(
        "NOTE: The .pt2 stores a q/dq graph (INT8 weights + scales\n"
        "+ dequantize ops), so file size may not shrink yet.\n"
        "True INT8 size reduction happens when lowered to an\n"
        "edge runtime (e.g. ExecuTorch + XNNPACK on Raspberry Pi)."
    )
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-50 preparation and quantization"
    )
    parser.add_argument(
        "--step",
        choices=["download", "export", "dataset", "quantize", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("download", "all"):
        download_imagenette()

    if args.step in ("export", "all"):
        export_resnet50_pt2()

    # Build and (optionally) persist the calibration dataset
    cal_dataset = None
    if args.step in ("dataset", "quantize", "all"):
        cal_dataset = ImageNetteDataset(IMAGENETTE_TRAIN)
        save_dataset(cal_dataset, str(CALIBRATION_PT))

    if args.step in ("quantize", "all"):
        result = quantize_resnet50(cal_dataset)
        compare_sizes(RESNET_PT2, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
