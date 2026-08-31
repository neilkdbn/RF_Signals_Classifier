# RF Signals Classifier — Current Project State
> **Branch:** `feature/2d-models` | **Last commit:** `feat: add latency/parameter benchmarking and ONNX export for Grad-CAM`

---

## ✅ Chunks 1–5: Completion Status

| Chunk | Title | Status |
|-------|-------|--------|
| 1 | DevOps Scaffolding & 2D Data Ingestion | ✅ Complete |
| 2 | ResNet-18 Baseline (2D) | ✅ Complete |
| 3 | STFT-RADN Architecture | ✅ Complete |
| 4 | Training Loop & Evaluation | ✅ Complete |
| 5 | Benchmarking & ONNX Bridge | ✅ Complete |

---

## 📁 Files Created

| File | Chunk | Purpose |
|------|-------|---------|
| `stft_converter.py` | 1 | Kaiser-windowed STFT converter: raw I/Q → 2D spectrogram (64×5) |
| `test_2d_shapes.py` | 1 | Synthetic unit tests validating tensor shapes in CI (no dataset required) |
| `models_2d.py` | 2 & 3 | Both model architectures: `ResNet18_2D` and `STFTRADN` |
| `train_2d.py` | 4 | Full training & validation loop with checkpointing |
| `.github/workflows/2d_model_ci.yml` | 4 | GitHub Actions CI — runs shape tests on every push/PR |
| `benchmark_2d.py` | 5 | Latency (ms/sample), parameter count, and ONNX export |

---

## 🔍 Architecture Verification

### `AdaptiveAvgPool2d` — confirmed in both models

**`ResNet18_2D`** (`models_2d.py`, line 92):
```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
```
Placed after `layer4`, collapses `[B, 512, H, W]` → `[B, 512, 1, 1]` regardless of spatial size.

**`STFTRADN`** (`models_2d.py`, line 366):
```python
nn.AdaptiveAvgPool2d((1, 1)),  # inside self.backbone
```
Placed at the end of the classification backbone, collapses `[B, 256, H, W]` → `[B, 256, 1, 1]`.

> [!NOTE]
> Using `AdaptiveAvgPool2d` instead of a fixed-size `AvgPool2d` means both models gracefully handle variable input spatial dimensions — important for the strided conv chain on small 64×5 spectrograms.

---

## ⚙️ Optimizer Configuration

**`AdamW`** is used in `train_2d.py` (lines 18, 240):

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=args.lr,          # default: 1e-3  (--lr CLI flag)
    weight_decay=args.wd # default: 1e-4  (--wd CLI flag)
)
```

Paired with a `ReduceLROnPlateau` scheduler that backs off the LR when validation loss plateaus. AdamW is preferred over vanilla Adam here because its decoupled weight decay improves generalization — particularly important for small-dataset RF classification tasks.

---

## 📊 Benchmark Results (CPU, batch=1)

| Model | Trainable Parameters | Inference Latency |
|-------|---------------------|-------------------|
| `ResNet18_2D` | 11,173,323 | ~18.7 ms / sample |
| `STFTRADN` | 852,593 | ~10.6 ms / sample |

> [!TIP]
> STFT-RADN is **13× smaller** and **~1.8× faster** than ResNet-18 on CPU while using richer feature extraction (Dense + CBAM attention). This makes it the preferred model for deployment.

---

## 📦 ONNX Export

The trained STFT-RADN weights can be exported via:
```bash
python benchmark_2d.py
```
Output: `stft_radn_model.onnx` (dynamic batch axis) — ready for **MATLAB Grad-CAM validation** and ONNX Runtime inference.

> [!IMPORTANT]
> `stft_radn_model.onnx` and `stft_radn_model.onnx.data` are listed in `.gitignore` and are **not committed** to the repository. Re-generate them locally by running `benchmark_2d.py` after training.

---

## 🚀 End-to-End Test Commands

```bash
# 1. Run CI shape validation (no dataset needed)
python test_2d_shapes.py

# 2. Smoke test both model architectures
python models_2d.py

# 3. Train on real dataset (requires stft_dataset.npz)
python train_2d.py --model stftradn --epochs 50

# 4. Benchmark latency, param count, and export ONNX
python benchmark_2d.py
```
