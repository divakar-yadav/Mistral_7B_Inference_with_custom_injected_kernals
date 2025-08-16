# run_infer.py
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from mistral_patch import force_eager_attention, swap_in_patched_attention

def get_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"

def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Env knobs ----
    model_id = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-v0.1")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))
    # Force dtype (auto usually fine; you can override with "float16", "bfloat16", "float32")
    DTYPE = os.environ.get("DTYPE", "").lower()
    if DTYPE == "float16":
        torch_dtype = torch.float16
    elif DTYPE == "bfloat16":
        torch_dtype = torch.bfloat16
    elif DTYPE == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16  # good default for Mistral 7B

    prompt = os.environ.get(
        "PROMPT",
        "Explain what is self attention."
    )

    # ---- Load ----
    print(f"Loading: {model_id}  dtype={torch_dtype}  ...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # ---- Patch attention ----
    model = force_eager_attention(model)
    swapped = swap_in_patched_attention(model)

    # ---- Show config key bits ----
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"Swapped attention in {swapped} layers.")
    print(f"Config: layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, "
          f"heads={cfg.num_attention_heads}, kv_heads={cfg.num_key_value_heads}, "
          f"head_dim={head_dim}, sliding_window={getattr(cfg, 'sliding_window', None)}")

    # ---- Prepare inputs ----
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ---- Generate ----
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    start = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    # ---- Decode & print ----
    text = tok.decode(out[0], skip_special_tokens=True)
    print("\n=== GENERATED TEXT ===")
    print(text)
    print("\n=== DEBUG ===")
    input_len = int(inputs["input_ids"].shape[-1])
    output_len = int(out.shape[-1] - input_len)
    device = next(model.parameters()).device
    print(f"input_len={input_len}, output_len={output_len}")
    print(f"time={elapsed:.3f}s, dtype={torch_dtype}, device={device}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print a more helpful trace if something goes sideways
        import traceback, sys
        print("ERROR during inference:", e, file=sys.stderr)
        traceback.print_exc()
        raise
