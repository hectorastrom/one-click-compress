# @Time    : 2026-02-15 01:32
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : distill.py

"""
Test script for GPT-automated self distillation.

GPT accepts a `.py` file and produces a replica intelligency compressed across
some dimension (traditionally hidden_dim or number of layers, but this depends
on architecture). 

This new architecture is then opened to training, using the previous logit
outputs of the larger "teacher" model as the reference for how it should behave.
This is a more information-dense objective than CrossEntropyLoss, and leads to
faster and more faithful convergence to good model performance. This is a
standard method used in large language model distillation.

The final, distilled model is saved as a `pt2` file, which can be further
compressed by 4x using post-training quantization with calibration. 
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Model output has no tensor: {type(output).__name__}")


def _load_module_from_path(py_path: str) -> types.ModuleType:
    module_path = Path(py_path).resolve()
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_model_ctor(module: types.ModuleType, factory_fn: str) -> callable:
    if hasattr(module, factory_fn):
        fn = getattr(module, factory_fn)
        if callable(fn):
            return fn
    for _, value in vars(module).items():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise ValueError(
        f"No model constructor found in module. Add `{factory_fn}()` or a zero-arg nn.Module class."
    )


def _build_model(module: types.ModuleType, factory_fn: str) -> nn.Module:
    ctor = _find_model_ctor(module, factory_fn)
    model = ctor()
    if not isinstance(model, nn.Module):
        raise TypeError("Model constructor did not return nn.Module")
    return model


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(k, str) for k in checkpoint.keys()):
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if tensor_values and len(tensor_values) == len(checkpoint):
                return checkpoint
    raise ValueError("Unsupported checkpoint format. Expected state_dict or {'state_dict': ...}.")


def _load_dataset(dataset_path: str) -> TensorDataset:
    data = torch.load(dataset_path, weights_only=True)
    if not isinstance(data, dict) or "x" not in data or "y" not in data:
        raise ValueError("Dataset .pt must contain a dict with 'x' and 'y' tensors.")
    x = data["x"]
    y = data["y"]
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise ValueError("Dataset values for 'x' and 'y' must both be tensors.")
    return TensorDataset(x, y)


def _labels_to_class_ids(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 1:
        return y.long()
    if y.ndim == 2 and y.shape[1] == 1:
        return y.squeeze(1).long()
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(dim=1).long()
    raise ValueError(f"Unsupported label shape: {tuple(y.shape)}")


def _call_openai_for_student_code(
    original_code: str,
    model_name: str,
    api_key: str,
    architecture_path: str,
) -> str:
    system_prompt = (
        "You are a senior PyTorch model compression engineer. "
        "Return only valid Python code and no markdown."
    )
    user_prompt = f"""
Given the model architecture source below, rewrite it into a student model suitable for
distillation with roughly half the parameters.

Rules:
1) Make the smallest, simplest structural edits possible (prefer reducing hidden dimensions,
   channels, width multipliers, or number of repeated blocks/layers).
2) Preserve API compatibility and usage style:
   - Keep public class names and function names if possible.
   - Keep input and output tensor shapes and logits dimension behavior unchanged.
   - Keep imports needed for execution.
3) Avoid complex redesign. Keep code readable and minimal.
4) The output must be a complete Python file.
5) Return only Python code, no explanation.

Source file path: {architecture_path}

--- BEGIN SOURCE ---
{original_code}
--- END SOURCE ---
"""

    payload = {
        "model": model_name,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc.reason}") from exc

    response_json = json.loads(raw)
    output_text = response_json.get("output_text", "").strip()
    if not output_text:
        raise RuntimeError("OpenAI response did not include output_text.")
    # If the model still wraps in fences, strip them.
    fence_match = re.match(r"^```(?:python)?\s*(.*?)\s*```$", output_text, flags=re.DOTALL)
    if fence_match:
        output_text = fence_match.group(1).strip()
    return output_text


def _distill_epoch(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    alpha: float,
) -> float:
    teacher.eval()
    student.train()
    running_loss = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device).float()
        y = _labels_to_class_ids(y).to(device)

        with torch.no_grad():
            teacher_logits = _first_tensor(teacher(x))
        student_logits = _first_tensor(student(x))

        ce = F.cross_entropy(student_logits, y)
        kd = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        loss = alpha * ce + (1.0 - alpha) * (temperature ** 2) * kd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * x.size(0)
        count += int(x.size(0))

    if count == 0:
        return 0.0
    return running_loss / count


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device).float()
        y = _labels_to_class_ids(y).to(device)
        logits = _first_tensor(model(x))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    if total == 0:
        return 0.0
    return correct / total


def run_on_policy_distillation(
    teacher_weights_path: str,
    architecture_path: str,
    dataset_path: str,
    output_dir: str,
    factory_fn: str = "create_model",
    openai_model: str = "gpt-4.1-mini",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 2.0,
    eval_holdout: int = 64,
    seed: int = 7,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")

    original_code = Path(architecture_path).read_text(encoding="utf-8")
    student_code = _call_openai_for_student_code(
        original_code=original_code,
        model_name=openai_model,
        api_key=api_key,
        architecture_path=architecture_path,
    )

    student_arch_path = out_dir / f"{Path(architecture_path).stem}_student.py"
    student_arch_path.write_text(student_code, encoding="utf-8")
    print(f"Saved student architecture -> {student_arch_path}")

    teacher_module = _load_module_from_path(architecture_path)
    student_module = _load_module_from_path(str(student_arch_path))
    teacher = _build_model(teacher_module, factory_fn=factory_fn)
    student = _build_model(student_module, factory_fn=factory_fn)

    checkpoint = torch.load(teacher_weights_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Teacher missing keys: {len(missing)}")
    if unexpected:
        print(f"Teacher unexpected keys: {len(unexpected)}")

    dataset = _load_dataset(dataset_path)
    n_total = len(dataset)
    if n_total < eval_holdout + 1:
        raise ValueError(
            f"Dataset too small ({n_total}). Needs at least {eval_holdout + 1} samples."
        )
    eval_holdout = min(eval_holdout, n_total // 2)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    eval_indices = indices[:eval_holdout]
    train_indices = indices[eval_holdout:]

    train_set = Subset(dataset, train_indices)
    eval_set = Subset(dataset, eval_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device).eval()
    student.to(device).train()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    print(f"Training student on {len(train_set)} samples; eval holdout {len(eval_set)} samples")
    for epoch in range(1, epochs + 1):
        loss = _distill_epoch(
            teacher=teacher,
            student=student,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            alpha=alpha,
        )
        acc = _accuracy(student, eval_loader, device)
        print(f"Epoch {epoch:02d}/{epochs}  loss={loss:.6f}  eval_acc={acc:.4f}")

    final_acc = _accuracy(student, eval_loader, device)
    print(f"Final withheld accuracy ({len(eval_set)} samples): {final_acc:.4f}")

    student = student.to("cpu").eval()
    sample_x, _sample_y = dataset[eval_indices[0]]
    if sample_x.ndim == 3:
        sample_x = sample_x.unsqueeze(0)
    sample_x = sample_x.float()
    exported = torch.export.export(student, (sample_x,), strict=False)

    distilled_pt2_path = out_dir / f"{Path(teacher_weights_path).stem}_distilled.pt2"
    torch.export.save(exported, str(distilled_pt2_path))
    print(f"Saved distilled .pt2 -> {distilled_pt2_path}")

    return {
        "student_architecture": str(student_arch_path),
        "distilled_pt2": str(distilled_pt2_path),
        "withheld_accuracy": final_acc,
        "withheld_samples": len(eval_set),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "On-policy distillation with architecture shrinking via OpenAI. "
            "Inputs: teacher .pt weights, architecture .py, and saved dataset .pt"
        )
    )
    parser.add_argument("weights", help="Path to teacher checkpoint .pt")
    parser.add_argument("architecture", help="Path to architecture .py file")
    parser.add_argument("dataset", help="Path to saved dataset .pt with {'x','y'}")
    parser.add_argument("--output-dir", default="weights", help="Output directory")
    parser.add_argument(
        "--factory-fn",
        default="create_model",
        help="Factory function in architecture module (default: create_model)",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model for architecture rewriting",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Distillation epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Cross entropy weight in blended distillation loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature",
    )
    parser.add_argument(
        "--eval-holdout",
        type=int,
        default=64,
        help="Number of withheld samples used for final accuracy",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    try:
        run_on_policy_distillation(
            teacher_weights_path=args.weights,
            architecture_path=args.architecture,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            factory_fn=args.factory_fn,
            openai_model=args.openai_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            temperature=args.temperature,
            eval_holdout=args.eval_holdout,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
