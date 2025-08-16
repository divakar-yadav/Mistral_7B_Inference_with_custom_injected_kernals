# mistral_patch.py
# Patch MistralAttention to replace only the Q·K^T/√d op with your CUDA kernel.
# Optional: replace A·V matmul via attn_extra_ext (USE_AV_KERNEL=1).
#
# Env toggles:
#   QK_USE_TORCH=1    -> bypass custom QK (use torch matmul)
#   QK_DEBUG=1        -> one-time numeric diff vs torch for QK
#   ATTN_DEBUG=1      -> print mask shape once
#   USE_AV_KERNEL=1   -> use attn_extra_ext.av_forward for A·V
#
# Works with HF Mistral 4.34-dev style rotary + cache APIs.

import math, os, types
import torch
import torch.nn.functional as F
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRotaryEmbedding,
    repeat_kv,
)

# Prefer HF’s rotary helper (avoids version-specific gotchas)
try:
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb as hf_apply_rope
except Exception:
    hf_apply_rope = None

# Required: your QK kernel extension built from setup.py
import qk_attn_ext

# Optional: extra kernels (AV/RMSNorm/RoPE) if you built attn_extra_ext
try:
    import attn_extra_ext
except Exception:  # ok if not present
    attn_extra_ext = None

_QK_USE_TORCH = os.environ.get("QK_USE_TORCH", "0") == "1"
_QK_DEBUG     = os.environ.get("QK_DEBUG", "0") == "1"
_ATTN_DEBUG   = os.environ.get("ATTN_DEBUG", "0") == "1"
_USE_AV       = os.environ.get("USE_AV_KERNEL", "1") == "1" and attn_extra_ext is not None

_printed_qk_debug_once = False
_printed_mask_once = False


def _resolve_dims(attn: MistralAttention, hidden_states: torch.Tensor):
    hidden_size = getattr(attn, "hidden_size", None) or getattr(attn, "embed_dim", None) or hidden_states.size(-1)
    num_heads = (
        getattr(attn, "num_heads", None)
        or getattr(attn, "num_attention_heads", None)
        or getattr(attn, "num_q_heads", None)
    )
    num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(attn, "num_kv_heads", None)
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and num_heads is not None:
        head_dim = hidden_size // int(num_heads)

    # Fallbacks using proj sizes
    if (num_heads is None or head_dim is None) and hasattr(attn, "q_proj"):
        q_out = attn.q_proj.out_features
        if num_heads is None and head_dim is not None:
            num_heads = q_out // head_dim
        if head_dim is None and num_heads is not None:
            head_dim = q_out // num_heads

    if num_kv_heads is None and num_heads is not None:
        if hasattr(attn, "k_proj"):
            k_out = attn.k_proj.out_features
            num_kv_heads = max(1, k_out // head_dim) if head_dim else max(1, num_heads // 4)
        else:
            num_kv_heads = max(1, num_heads // 4)

    if not all([hidden_size, num_heads, num_kv_heads, head_dim]):
        raise RuntimeError(
            f"Could not resolve dims: hidden_size={hidden_size}, "
            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
    return int(hidden_size), int(num_heads), int(num_kv_heads), int(head_dim)


def _append_tuple_cache(pk, pv, k_u, v_u):
    if pk is None or pv is None:
        return k_u, v_u
    # pk,pv: [B, KVH, L_past, D]; k_u,v_u: [B, KVH, L_cur, D]
    return torch.cat([pk, k_u], dim=2), torch.cat([pv, v_u], dim=2)


def _maybe_update_cache(past_key_value, k_u, v_u, layer_idx):
    """
    Handle both cache APIs:
      - Cache object with .update(...) → returns updated (k_all, v_all) OR updates in-place
      - Tuple (k,v) → append
      - None → just current k_u, v_u
    Return unrepeated [B, KVH, Lk_total, D]
    """
    if past_key_value is None:
        return k_u, v_u

    # New HF Cache API (object with .update)
    if hasattr(past_key_value, "update"):
        try:
            out = past_key_value.update(k_u, v_u, layer_idx)
        except TypeError:
            out = past_key_value.update(k_u, v_u)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]

        # Some caches update internally; try to fetch
        for attr in ("key_cache", "value_cache", "keys", "values"):
            if hasattr(past_key_value, attr):
                obj = getattr(past_key_value, attr)
                for ka, va in (("keys", "values"), ("key_cache", "value_cache")):
                    if hasattr(obj, ka) and hasattr(obj, va):
                        kc = getattr(obj, ka); vc = getattr(obj, va)
                        try:
                            return kc[layer_idx], vc[layer_idx]
                        except Exception:
                            pass
        # Fallback: return current K,V
        return k_u, v_u

    # Old tuple API
    if isinstance(past_key_value, (list, tuple)):
        if len(past_key_value) == 2:
            pk, pv = past_key_value
            return _append_tuple_cache(pk, pv, k_u, v_u)
        return k_u, v_u

    # Unknown type → no-op
    return k_u, v_u


def _patched_forward(
    self: MistralAttention,
    hidden_states: torch.Tensor,
    attention_mask=None,          # additive mask from HF (already causal/padding/sliding)
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,      # cache handled by API above
    cache_position=None,
    **kwargs,
):
    global _printed_qk_debug_once, _printed_mask_once

    bsz, q_len, _ = hidden_states.size()
    hidden_size, num_heads, num_kv_heads, head_dim = _resolve_dims(self, hidden_states)

    # --- QKV projections (fused or split) ---
    if hasattr(self, "qkv_proj"):  # fused (Mistral)
        qkv = self.qkv_proj(hidden_states)  # [B,L,(H+2*KVH)*D]
        qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
        q = qkv[:, :, : num_heads]
        k = qkv[:, :, num_heads : num_heads + num_kv_heads]
        v = qkv[:, :, -num_kv_heads :]
        query_states   = q.transpose(1, 2)       # [B,H,L,D]
        key_states_u   = k.transpose(1, 2)       # [B,KVH,L,D] (unrepeated)
        value_states_u = v.transpose(1, 2)       # [B,KVH,L,D] (unrepeated)
    else:  # split
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        query_states   = q.view(bsz, q_len, num_heads,     head_dim).transpose(1, 2)
        key_states_u   = k.view(bsz, q_len, num_kv_heads,  head_dim).transpose(1, 2)
        value_states_u = v.view(bsz, q_len, num_kv_heads,  head_dim).transpose(1, 2)

    # --- RoPE ---
    if getattr(self, "rotary_emb", None) is None:
        # For 4.34-dev, constructor takes config (not head_dim)
        if not hasattr(self, "config"):
            raise RuntimeError("rotary_emb missing and no self.config to construct it.")
        self.rotary_emb = MistralRotaryEmbedding(self.config)

    # Positions for current tokens
    if position_ids is None:
        rope_pos = cache_position if cache_position is not None else torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
    else:
        rope_pos = position_ids

    # Ask HF rotary to give cos/sin covering needed length (and cache internally)
    cos, sin = self.rotary_emb(value_states_u, rope_pos)
    if hf_apply_rope is not None:
        query_states, key_states_u = hf_apply_rope(query_states, key_states_u, cos, sin, rope_pos)
    else:
        # Fallback rotate-half
        def rotate_half(t):
            t1, t2 = t[..., : t.size(-1)//2], t[..., t.size(-1)//2 :]
            return torch.cat((-t2, t1), dim=-1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states_u = (key_states_u * cos) + (rotate_half(key_states_u) * sin)

    # --- KV cache (unrepeated) ---
    key_states_u, value_states_u = _maybe_update_cache(past_key_value, key_states_u, value_states_u, getattr(self, "layer_idx", 0))

    # --- Repeat KV to match H (GQA) ---
    num_key_value_groups = getattr(self, "num_key_value_groups", None) or max(1, num_heads // num_kv_heads)
    key_states   = repeat_kv(key_states_u,   num_key_value_groups)   # [B,H,Lk,D]
    value_states = repeat_kv(value_states_u, num_key_value_groups)   # [B,H,Lk,D]

    # --- QK^T / sqrt(d) (custom or torch) ---
    B, H, Lq, D = query_states.shape
    Lk = key_states.size(2)
    q_bhd = query_states.contiguous().view(B * H, Lq, D)
    k_bhd = key_states.contiguous().view(B * H, Lk, D)

    scale = 1.0 / math.sqrt(D)
    if _QK_USE_TORCH:
        attn_logits = torch.matmul(q_bhd.float(), k_bhd.float().transpose(1, 2)) * scale
    else:
        attn_logits = qk_attn_ext.qk_forward(q_bhd, k_bhd, scale)
    attn_logits = attn_logits.view(B, H, Lq, Lk)

    # QK numeric diff (once)
    if _QK_DEBUG and not _printed_qk_debug_once:
        with torch.no_grad():
            ref = torch.matmul(q_bhd.float(), k_bhd.float().transpose(1, 2)) * scale
            diff = (ref - attn_logits.view_as(ref)).abs()
            print(f"[QK_DEBUG] max_abs_diff={diff.max().item():.3e}, mean_abs_diff={diff.mean().item():.3e}")
        _printed_qk_debug_once = True

    # --- Additive mask (already causal/padding/sliding-window from HF) ---
    if attention_mask is not None:
        if _ATTN_DEBUG and not _printed_mask_once:
            try:
                print(f"[MASK] kind=full_additive, mask_shape={tuple(attention_mask.shape)}")
            except Exception:
                pass
            _printed_mask_once = True
        attn_logits = attn_logits + attention_mask.to(dtype=attn_logits.dtype, device=attn_logits.device)

    # --- Softmax + A·V ---
    attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(value_states.dtype)

    if _USE_AV:
        # attn_weights: [B,H,Lq,Lk], value_states: [B,H,Lk,D] -> [B,H,Lq,D]
        aw = attn_weights.contiguous().view(B * H, Lq, Lk)
        vv = value_states.contiguous().view(B * H, Lk, D)
        attn_output = attn_extra_ext.av_forward(aw, vv).view(B, H, Lq, D)
    else:
        attn_output = torch.matmul(attn_weights, value_states)  # [B,H,Lq,D]

    # --- Merge heads + out proj ---
    hs = getattr(self, "hidden_size", None) or getattr(self, "embed_dim", None) or self.o_proj.in_features
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, hs)
    attn_output = self.o_proj(attn_output)

    # HF Mistral expects TWO returns from attention.forward
    if output_attentions:
        return attn_output, attn_weights
    return attn_output, None


def swap_in_patched_attention(model: torch.nn.Module) -> int:
    """Replace forward of all MistralAttention modules in-place. Returns count."""
    num = 0
    for idx, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, MistralAttention):
            if not hasattr(layer.self_attn, "layer_idx"):
                setattr(layer.self_attn, "layer_idx", idx)  # for cache APIs that need it
            layer.self_attn.forward = types.MethodType(_patched_forward, layer.self_attn)
            num += 1
    print(f"Swapped attention in {num} layers.")
    return num


def force_eager_attention(model):
    """Disable FlashAttention path so our override is used."""
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model
