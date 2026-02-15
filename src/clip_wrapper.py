"""
CLIP model wrapper for zero-shot image classification.

Wraps HuggingFace CLIP into a standard nn.Module that:
- Takes image tensors as input
- Returns classification logits over a fixed set of class names
- Supports quantization of the vision encoder

The text encoder is pre-computed and frozen (not quantized).
Only the vision tower is quantized, which is where the bulk of
parameters and compute lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default class names for zero-shot classification
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Prompt templates that improve CLIP zero-shot accuracy
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of a {} in the wild.",
]


class CLIPClassifier(nn.Module):
    """
    CLIP-based zero-shot classifier.

    Wraps the CLIP vision encoder so it looks like a standard classifier:
        logits = model(images)  # shape: (batch_size, num_classes)

    The text embeddings for each class are pre-computed once and frozen.
    Only the vision encoder parameters are exposed for quantization.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        class_names: list[str] | None = None,
        prompt_templates: list[str] | None = None,
        device: str = "cpu",
    ):
        super().__init__()

        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.class_names = class_names or CIFAR10_CLASSES
        self.prompt_templates = prompt_templates or PROMPT_TEMPLATES

        logger.info(f"Loading CLIP model: {model_name}")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Pre-compute text embeddings for all classes (frozen, not quantized)
        self._precompute_text_embeddings(device)

        # Expose vision encoder parameters for quantization
        self.vision_model = self.clip_model.vision_model
        self.visual_projection = self.clip_model.visual_projection

    @torch.no_grad()
    def _precompute_text_embeddings(self, device: str = "cpu") -> None:
        """Pre-compute and cache text embeddings for all classes."""
        self.clip_model.eval()

        all_texts = []
        for class_name in self.class_names:
            for template in self.prompt_templates:
                all_texts.append(template.format(class_name))

        # Tokenize
        text_inputs = self.processor.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # Get text features
        text_outputs = self.clip_model.text_model(**text_inputs)
        text_embeds = self.clip_model.text_projection(
            text_outputs.pooler_output
        )

        # Average over templates for each class
        n_templates = len(self.prompt_templates)
        n_classes = len(self.class_names)
        text_embeds = text_embeds.reshape(n_classes, n_templates, -1).mean(dim=1)

        # Normalize
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Register as buffer (saved with state_dict but not a parameter)
        self.register_buffer("text_embeddings", text_embeds.cpu())
        logger.info(
            f"Pre-computed text embeddings: {text_embeds.shape} "
            f"for {n_classes} classes"
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image tensor -> classification logits.

        Args:
            pixel_values: Image tensor of shape (batch_size, 3, H, W).
                          Already preprocessed (normalized, resized).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        # Get vision features
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = self.visual_projection(vision_outputs.pooler_output)

        # Normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        # Move text embeddings to same device/dtype
        text_embeds = self.text_embeddings.to(
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )

        # Cosine similarity -> logits
        # CLIP uses a learned temperature (logit_scale)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = image_embeds @ text_embeds.t() * logit_scale

        return logits

    def get_vision_model(self) -> nn.Module:
        """Return the vision encoder for scanning/quantization."""
        return self.vision_model


def load_clip_model(
    model_name: str = "openai/clip-vit-large-patch14",
    class_names: list[str] | None = None,
) -> tuple[CLIPClassifier, int, Any]:
    """
    Load a CLIP model wrapped as a classifier.

    Returns:
        (model, input_size, processor) tuple.
    """
    from transformers import CLIPProcessor

    model = CLIPClassifier(model_name=model_name, class_names=class_names)
    processor = CLIPProcessor.from_pretrained(model_name)

    # CLIP models expect 224x224 input
    input_size = processor.image_processor.size.get("shortest_edge", 224)

    model.eval()
    return model, input_size, processor


def get_clip_transform(input_size: int = 224):
    """
    Get the image transform for CLIP models.

    Uses CLIP's expected normalization (different from ImageNet default).
    """
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
