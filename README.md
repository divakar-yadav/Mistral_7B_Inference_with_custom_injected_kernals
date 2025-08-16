
---

# Mistral-7B Custom CUDA Attention

End-to-end example project that loads **Mistral-7B** with Hugging Face Transformers, **replaces only the attention score op** `Q·Kᵀ/√d` with your **custom CUDA kernel**, and (optionally) swaps in a custom **A·V** kernel.

The patch is **surgical**: all masking (causal/sliding/padding), RoPE, caching, and projections remain as in HF, except where you opt-in to extra kernels.

---

## Contents

```
.
├── ext/
│   ├── qk_attn.cpp                # C++ binding for QK kernel (if using separate build)
│   ├── qk_attn.cu                 # (Your) Q·Kᵀ CUDA kernel
│   ├── attn_extra.cpp             # C++ bindings for optional extra kernels
│   ├── av_attn.cu                 # Optional A·V kernel (batched matmul)
│   ├── rmsnorm.cu                 # Optional RMSNorm kernel (fused normalize+scale)
│   └── rope_apply.cu              # Optional RoPE apply kernel
├── mistral_patch.py               # Monkey-patches MistralAttention.forward (QK swap, etc.)
├── run_infer.py                   # Simple generation demo
├── setup.py                       # Builds optional extra kernels (attn_extra_ext)
└── requirements.txt               # torch, transformers, accelerate, sentencepiece, (optional) ninja
```

> You already have **`qk_attn_ext`** built/available (from earlier steps). `setup.py` builds **`attn_extra_ext`** which contains A·V / RMSNorm / RoPE (optional).

---

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
# Or minimal:
# pip install torch transformers accelerate sentencepiece ninja
```

Make sure `nvcc` is in your PATH and matches your GPU. Export your compute capability if you want a smaller binary:

```bash
# H100:
export TORCH_CUDA_ARCH_LIST="9.0"
# A100:
# export TORCH_CUDA_ARCH_LIST="8.0"
# L4 / Ada / 4090:
# export TORCH_CUDA_ARCH_LIST="8.9"
```

> Warning messages about CUDA **minor** version mismatch (e.g., 12.0 vs 12.8 used to build PyTorch) are usually fine.

### 2) Build optional extra kernels

This builds **A·V**, **RMSNorm**, **RoPE-apply** into `attn_extra_ext`:

```bash
# Clean old builds if needed
rm -rf build *.so **/*.o **/*.obj .ninja_log .ninja_deps

python setup.py build_ext --inplace
```

You should now see `attn_extra_ext*.so` in the project root.

> Your `qk_attn_ext` is already available (built earlier). If not, build it similarly or inline-JIT it.

### 3) Run a quick generation

```bash
python run_infer.py
```

You’ll see:

* model/config summary
* “Swapped attention in XX layers.”
* Generated text and timing

---

## What’s Being Patched

We monkey-patch **`MistralAttention.forward`** to call:

* `qk_attn_ext.qk_forward(Q_bhd, K_bhd, 1/sqrt(D))` for **attention logits** (instead of `torch.matmul`).
* (Optional) `attn_extra_ext.av_forward(A, V)` for **A·V** aggregation.

Everything else remains HF: projections, RoPE, cache update, masking, output projection.

The patch lives in **`mistral_patch.py`** and is applied by **`swap_in_patched_attention(model)`** after **`force_eager_attention(model)`** (disables FlashAttention path so your override runs).

---

## Environment Toggles

Set these as needed when running `run_infer.py`:

| Env Var          | Default                     | Meaning                                                               |
| ---------------- | --------------------------- | --------------------------------------------------------------------- |
| `QK_USE_TORCH`   | `0`                         | Use PyTorch matmul for Q·Kᵀ (bypass custom kernel).                   |
| `QK_DEBUG`       | `0`                         | Print one-time max/mean abs diff vs PyTorch Q·Kᵀ for sanity.          |
| `ATTN_DEBUG`     | `0`                         | Print attention mask shape once.                                      |
| `USE_AV_KERNEL`  | `1`                         | If `attn_extra_ext` is present, use custom A·V kernel; else fallback. |
| `MODEL_ID`       | `mistralai/Mistral-7B-v0.1` | HF repo id.                                                           |
| `DTYPE`          | `float16`                   | One of `float16`, `bfloat16`, `float32`.                              |
| `MAX_NEW_TOKENS` | `128`                       | Generation length.                                                    |
| `PROMPT`         | *(string)*                  | Override prompt text.                                                 |

Examples:

```bash
QK_USE_TORCH=1 python run_infer.py
QK_DEBUG=1 ATTN_DEBUG=1 python run_infer.py
MODEL_ID=mistralai/Mistral-7B-v0.1 DTYPE=bfloat16 python run_infer.py
```

---

## Verifying Correctness

1. **QK numeric check**

```bash
QK_DEBUG=1 python run_infer.py
# Expect near-zero diffs (max/mean ~ 1e-6 to 1e-4 depending on dtype/tile).
```

2. **Bypass to isolate issues**

```bash
QK_USE_TORCH=1 python run_infer.py
# Confirms problems are not in masking/RoPE/cache integrations.
```

3. **Compare outputs** with and without `USE_AV_KERNEL=1`. They should be numerically close.

---

## Troubleshooting

### Build errors

* **half arithmetic in RMSNorm**
  Use `_hadd()`/`__hadd()`/`__hfma()` or upcast to float. (The provided `rmsnorm.cu` implementation already upcasts to FP32 inside the kernel to avoid ambiguous `__half + __half`.)

* **Multiple rules generate …/qk\_attn.o**
  You had both inline JIT and `setup.py` trying to compile a file with the same object name. Clean and rebuild:

  ```bash
  rm -rf build *.so **/*.o **/*.obj .ninja_log .ninja_deps
  python setup.py build_ext --inplace
  ```

* **Missing extension**
  `ModuleNotFoundError: qk_attn_ext` → ensure the `.so` is in the project root (or PYTHONPATH updated), and you built it.

### Runtime errors

* **Device mismatch**
  Ensure all tensors (Q/K/V, masks, cos/sin) are on the same `cuda:X`. The code moves inputs to `model.device` and crafts masks/cos/sin on that device.

* **Wrong return arity**
  HF attention typically returns `(attn_out, None)` or `(attn_out, attn_weights)` if `output_attentions=True`. The patch follows that. If your local HF changes this, align returns accordingly.

---

## How to Add More Custom Kernels

The easiest next wins (already scaffolded):

1. **A·V (attn × values):** see `ext/av_attn.cu`, called via `attn_extra_ext.av_forward`.
2. **RMSNorm:** `ext/rmsnorm.cu` – fused normalize + scale (gamma).
3. **RoPE apply:** `ext/rope_apply.cu` – apply cos/sin rotation in CUDA.

Wire them in Python where they live (attention path for A·V, residual blocks for RMSNorm, RoPE point in attention). The current patch already provides a toggle for **A·V**. You can add RMSNorm and RoPE kernels analogously once you’re confident.

---

## Performance Tips

* Tune `TILE_M/N/K` in your kernels for your GPU (occupancy, shared mem).
* Use **FP16 inputs with FP32 accumulators** (already done).
* Consider fusing **mask + softmax** or leveraging **FlashAttention**-style tricks once functionality is rock-solid.

---

## Known Good Config (example)

* GPU: H100 (SM90)
* PyTorch: 2.4+
* CUDA: 12.2–12.8 (minor mismatch OK)
* `TORCH_CUDA_ARCH_LIST=9.0`
* Transformers: 4.39+ (Mistral codepath w/ sliding window)
* Model: `mistralai/Mistral-7B-v0.1`
* Dtype: `float16` (bf16 also works if the model was trained/exported that way)

---

## License

This repository stitches together your custom CUDA code with Transformers. Respect the licenses of upstream components (Hugging Face Transformers, Mistral weights). Your CUDA kernels are your own—attach a LICENSE file if you plan to share.

---

## One-liners

* Build extras:

  ```bash
  export TORCH_CUDA_ARCH_LIST=9.0
  python setup.py build_ext --inplace
  ```
* Sanity run:

  ```bash
  QK_DEBUG=1 python run_infer.py
  ```
* Force torch QK for A/B:

  ```bash
  QK_USE_TORCH=1 python run_infer.py
  ```
* Use custom A·V (default if built):

  ```bash
  USE_AV_KERNEL=1 python run_infer.py
  ```

That’s it—happy kernel hacking!

---

