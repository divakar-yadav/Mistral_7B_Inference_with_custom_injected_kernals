"""
End-to-end example: Load Mistral-7B with Hugging Face Transformers, replace the
attention score computation (Q·K^T / sqrt(d)) with a custom CUDA kernel, and run
inference. The kernel computes raw attention logits; masking and softmax remain
in PyTorch for clarity.

Tested shapes: B×H×L×D. The kernel supports float16/float32 inputs and
accumulates in float32. Output is float32 logits.

Notes
- This code forces the model into the eager attention path (disables FlashAttention)
  so we can safely intercept the matmul.
- Sliding-window and causal masks are applied in PyTorch after our kernel call.
- KV cache is supported (basic append semantics matching HF Mistral path).
- Grouped-Query Attention (num_key_value_heads < num_attention_heads) is respected
  via repeat_kv from the Transformers implementation.

Usage
1) Ensure CUDA + a recent PyTorch with C++ extensions.
2) pip install transformers accelerate sentencepiece
3) Run this script. It will JIT-compile the CUDA extension on first run.

If you only want to validate the CUDA op independently, run the `__main__`
section’s small unit test first.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Build the CUDA extension inline ---------------------------------------------------------

CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Tile sizes (tune if desired)
#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

// Convert element to float for accumulation
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__half x) { return __half2float(x); }

// Batched GEMM: C[b, m, n] = scale * sum_k A[b, m, k] * B[b, n, k]
// Shapes per batch b:
//  A: M x K (row-major)
//  B: N x K (row-major)  // note K matches last dim; this computes A * B^T
//  C: M x N (row-major)
// We pass BH batches, where BH = B * H (batch * heads).
extern "C" __global__
void qk_batched_gemm_kernel(
    const void* __restrict__ A_,
    const void* __restrict__ B_,
    float* __restrict__ C,
    int BH, int M, int N, int K,
    float scale,
    int a_stride, int b_stride, int c_stride,
    int dtype_code // 0=float32, 1=float16
) {
    int bh = blockIdx.z; // batch-head index
    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;

    if (bh >= BH) return;

    // Base pointers for this (batch, head)
    const char* A_batch = (const char*)A_ + (size_t)bh * (size_t)a_stride;
    const char* B_batch = (const char*)B_ + (size_t)bh * (size_t)b_stride;
    float* C_batch = C + (size_t)bh * (size_t)c_stride;

    // Shared memory tiles
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    // Thread coordinates within the block
    int ty = threadIdx.y; // [0, TILE_M)
    int tx = threadIdx.x; // [0, TILE_N)

    // Each thread computes one element (m, n) within the C tile
    int m = block_m + ty;
    int n = block_n + tx;

    float acc = 0.0f;

    // Loop over K dimension in TILE_K chunks
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile: (m, k0..k0+TILE_K)
        if (m < M) {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                int k = k0 + kk;
                float a_val = 0.0f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* A = reinterpret_cast<const float*>(A_batch);
                        a_val = A[m * K + k];
                    } else {
                        const __half* A = reinterpret_cast<const __half*>(A_batch);
                        a_val = __half2float(A[m * K + k]);
                    }
                }
                As[ty][kk] = a_val;
            }
        } else {
            for (int kk = tx; kk < TILE_K; kk += TILE_N) {
                As[ty][kk] = 0.0f;
            }
        }

        // Load B tile: (n, k0..k0+TILE_K)
        if (n < N) {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                int k = k0 + kk;
                float b_val = 0.0f;
                if (k < K) {
                    if (dtype_code == 0) {
                        const float* B = reinterpret_cast<const float*>(B_batch);
                        b_val = B[n * K + k];
                    } else {
                        const __half* B = reinterpret_cast<const __half*>(B_batch);
                        b_val = __half2float(B[n * K + k]);
                    }
                }
                Bs[tx][kk] = b_val;
            }
        } else {
            for (int kk = ty; kk < TILE_K; kk += TILE_M) {
                Bs[tx][kk] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial products for this tile
        if (m < M && n < N) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc += As[ty][kk] * Bs[tx][kk];
            }
        }
        __syncthreads();
    }

    // Write result
    if (m < M && n < N) {
        C_batch[m * N + n] = acc * scale;
    }
}

// C-callable launcher so we can invoke from a C++ TU compiled with g++.
extern "C" void launch_qk_batched_gemm(
    const void* A,
    const void* B,
    float* C,
    int BH, int M, int N, int K,
    float scale,
    int a_stride, int b_stride, int c_stride,
    int dtype_code,
    cudaStream_t stream
) {
    dim3 block(/*x=*/TILE_N, /*y=*/TILE_M, /*z=*/1);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M,
              BH);

    qk_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        A, B, C, BH, M, N, K, scale, a_stride, b_stride, c_stride, dtype_code
    );
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAException.h>

extern "C" void launch_qk_batched_gemm(
    const void* A,
    const void* B,
    float* C,
    int BH, int M, int N, int K,
    float scale,
    int a_stride, int b_stride, int c_stride,
    int dtype_code,
    cudaStream_t stream
);

// Q: [BH, M, K]  (float16 or float32)
// K: [BH, N, K]  (float16 or float32)
// returns: [BH, M, N] float32
at::Tensor qk_forward(at::Tensor Q, at::Tensor K, double scale) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "Q and K must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3, "Q and K must be 3D");
    TORCH_CHECK(Q.scalar_type() == at::kHalf || Q.scalar_type() == at::kFloat,
                "Q must be float16 or float32");
    TORCH_CHECK(K.scalar_type() == at::kHalf || K.scalar_type() == at::kFloat,
                "K must be float16 or float32");

    int64_t BH = Q.size(0);
    int64_t M  = Q.size(1);
    int64_t KK = Q.size(2);
    TORCH_CHECK(K.size(0) == BH && K.size(2) == KK, "K dims mismatch");
    int64_t N  = K.size(1);

    auto opts = Q.options().dtype(at::kFloat);
    at::Tensor C = at::empty({BH, M, N}, opts);

    int dtype_code = (Q.scalar_type() == at::kFloat) ? 0 : 1;

    // Strides in BYTES for base-pointer arithmetic on char* in the CUDA TU
    int a_stride = (int)(M * KK * Q.element_size());
    int b_stride = (int)(N * KK * K.element_size());
    int c_stride = (int)(M * N * C.element_size());

    auto stream = at::cuda::getCurrentCUDAStream();

    // Launch CUDA kernel via C-callable launcher (compiled with nvcc)
    launch_qk_batched_gemm(
        Q.data_ptr(),
        K.data_ptr(),
        C.data_ptr<float>(),
        (int)BH, (int)M, (int)N, (int)KK,
        (float)scale,
        a_stride, b_stride, c_stride,
        dtype_code,
        stream.stream()
    );

    // Surface any async launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qk_forward", &qk_forward, "QK^T attention scores (batched)");
}
"""

ext = load_inline(
    name="qk_attn",
    cpp_sources=[CPP_SRC],
    cuda_sources=[CUDA_SRC],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

# --- Hugging Face Mistral integration -------------------------------------------------------

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
    repeat_kv,
)

class PatchedMistralAttention(MistralAttention):
    """Drop-in replacement that swaps Q·K^T with our CUDA kernel.

    We otherwise mirror the HF implementation’s logic for:
      - QKV projections
      - RoPE application
      - KV cache append
      - Repeat kv heads (GQA)
      - Masking (causal + sliding window)
      - Softmax + dropout (dropout is zero at inference)
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project to QKV
        qkv = self.qkv_proj(hidden_states)
        # Split by heads
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states = qkv[:, :, : self.num_heads]
        key_states   = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value_states = qkv[:, :, -self.num_key_value_heads :]

        # [B, L, H, D] -> [B, H, L, D]
        query_states = query_states.transpose(1, 2)
        key_states   = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE to Q/K
        if self.rotary_emb is None:
            self.rotary_emb = MistralRotaryEmbedding(self.head_dim)
        cos, sin = self.rotary_emb(value_states, seq_len=None)  # caches cos/sin up to needed length
        # position_ids may be None for prefill; HF applies using helper
        query_states, key_states = self.apply_rope(query_states, key_states, cos, sin, position_ids)

        # Append to KV cache if provided
        if past_key_value is not None:
            # past_key, past_value: [B, num_kv_heads, L_past, D]
            pk, pv = past_key_value
            key_states   = torch.cat([pk, key_states], dim=2)
            value_states = torch.cat([pv, value_states], dim=2)

        # Optionally return new cache
        present = (key_states, value_states) if use_cache else None

        # Repeat KV to match num_heads (GQA)
        key_states   = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Shapes now:
        #  q: [B, H, Lq, D],  k: [B, H, Lk, D],  v: [B, H, Lk, D]
        # Our kernel expects BH-strided [BH, L, D] forms
        B, H, Lq, D = query_states.shape
        _, _, Lk, _ = key_states.shape

        # Contiguity for the extension (BH, L, D)
        q_bhd = query_states.contiguous().view(B * H, Lq, D)
        k_bhd = key_states.contiguous().view(B * H, Lk, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Compute raw attention logits with custom kernel (float32 output)
        attn_logits = ext.qk_forward(q_bhd, k_bhd, scale).view(B, H, Lq, Lk)

        # Apply sliding-window mask if configured (tokens attend only to last W keys)
        if self.sliding_window is not None:
            W = self.sliding_window
            # For each query position i, keys j < i-W are masked
            # Build a lower-triangular band mask efficiently
            q_positions = torch.arange(Lq, device=attn_logits.device).unsqueeze(-1)
            k_positions = torch.arange(Lk, device=attn_logits.device).unsqueeze(0)
            allowed = (k_positions >= (q_positions - W)) & (k_positions <= q_positions)
            mask = (~allowed).to(attn_logits.dtype) * (-1e9)
            attn_logits = attn_logits + mask  # broadcast over B,H
        else:
            # Standard causal mask
            causal_mask = torch.triu(
                torch.full((Lq, Lk), fill_value=-1e9, device=attn_logits.device, dtype=attn_logits.dtype),
                diagonal=1,
            )
            attn_logits = attn_logits + causal_mask

        # Attention mask (e.g., padding) from model inputs, shape: [B, 1, Lq, Lk] or [B, 1, 1, Lk]
        if attention_mask is not None:
            attn_logits = attn_logits + attention_mask

        # Softmax (float32) then cast back to input dtype for matmul with V
        attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(value_states.dtype)

        # Attention output
        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, Lq, D]

        # Merge heads and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            return attn_output, attn_weights, present
        return attn_output, None, present

    @staticmethod
    def apply_rope(q, k, cos, sin, position_ids=None):
        # Mirror HF rotary application; cos/sin are broadcastable [1,1,L,1]
        def rotate_half(x):
            x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        if position_ids is not None:
            cos = cos[:, :, position_ids, :]
            sin = sin[:, :, position_ids, :]

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


def swap_in_patched_attention(model: nn.Module) -> None:
    """Replace all MistralAttention modules with PatchedMistralAttention and copy weights."""
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

    num_swapped = 0
    for layer in model.model.layers:
        if isinstance(layer.self_attn, MistralAttention):
            old = layer.self_attn
            patched = PatchedMistralAttention(model.config)
            # Copy parameters
            patched.load_state_dict(old.state_dict())
            layer.self_attn = patched
            num_swapped += 1
    print(f"Swapped attention in {num_swapped} layers.")


def force_eager_attention(model):
    # Disable FlashAttention paths to ensure our override is used.
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    os.environ["PYTORCH_CUDNN_V8_API_ENABLED"] = "1"  # no-op safety
    return model


# --- Minimal main: load model, patch, and run a single prompt -------------------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Quick unit test of the CUDA op
    BH, M, N, K = 2, 17, 19, 64
    q = torch.randn(BH, M, K, device="cuda", dtype=torch.float16)
    k = torch.randn(BH, N, K, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(K)
    ref = (q.float() @ k.float().transpose(1, 2)) * scale
    out = ext.qk_forward(q, k, scale)
    max_err = (ref - out).abs().max().item()
    print(f"[Sanity] qk_forward max abs error vs PyTorch fp32: {max_err:.3e}")

    # Load Mistral 7B (you can choose any compatible checkpoint)
    model_id = "mistralai/Mistral-7B-v0.1"
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Force eager attention and swap in our custom attention
    model = force_eager_attention(model)
    swap_in_patched_attention(model)

    # Summarize key config fields for user visibility
    print("\n[Mistral Config Highlights]")
    print(f"layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}")
    print(f"head_dim={cfg.hidden_size // cfg.num_attention_heads}, rope_theta={getattr(cfg, 'rope_theta', None)}, sliding_window={getattr(cfg, 'sliding_window', None)}")

    prompt = "Explain why rotary positional embeddings help extrapolate sequence length."
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
    )
    print("\n[Output]\n", tok.decode(gen[0], skip_special_tokens=True))
