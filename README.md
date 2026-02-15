# One-Click-Compress

Shrink the size of any PyTorch model in a single click!

## Project Goal
this project aims to build a one-click universal model compression tool designed to shrink pytorch models for real-time edge deployment on hardware like the Raspberry Pi 4. while the pi 4 can hold 100-500mb models in its 2-8gb RAM, we target a post-quantization footprint of 50mb or less to guarantee low-latency inference.

we are prioritizing a pipeline that balances aggressive size reduction with performance stability, starting with int8 quantization as our primary lever.

---

**the four levels of compression**
1. quantization (int8): our starting point. we utilize quantization-aware training (qat) on a provided dataset to secure massive size reductions while preserving accuracy by simulating quantization errors during fine-tuning.

2. structural pruning: unlike unstructured methods, this physically removes network blocks (entire filters or channels). this is the only pruning strategy that genuinely reduces RAM usage and compute operations on ARM architectures.

3. low-rank factorization: this stage uses the best rank-r approximation in the frobenius norm to decompose large weight matrices into smaller, more efficient products.

4. logit distillation: as a final safety measure, we use kl divergence to align
   the compressed student model's logits with the original teacher model,
   recovering accuracy lost during the previous three stages.

---

**technical constraints & hardware notes**
- target hardware: Raspberry Pi 4 / ARM Cortex-A72.
- inference footprint: < 50mb for real-time performance.

---

Built by:
- Christina Lee
- Danny Lin
- Albert Astrom
- Hector Astrom

*for treehacks 2026*
