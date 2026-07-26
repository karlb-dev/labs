# special-lab-1 shared utilities. Repo philosophy applies: raw HF transformers,
# explicit hooks, self-checks that abort loudly rather than degrade silently.
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", "/content/drive/MyDrive/hf_cache/hub")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

LAB_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(
    os.environ.get(
        "SL1_RUN_DIR",
        "/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726",
    )
)
# v2 delta run (2026-07-26): v1 artifacts (lens, layer_state, metrics) are
# read from RUN_DIR; all v2 outputs go to RUN_DIR_V2. v1 scripts untouched.
RUN_DIR_V2 = Path(
    os.environ.get(
        "SL1_RUN_DIR_V2",
        "/content/drive/MyDrive/interpret/special-lab-1/2026-07-26_v2",
    )
)
LOCAL_WORK = Path("/content/sl1_work")  # fast local scratch for checkpoints

MODELS = {
    "main": "allenai/Olmo-3-32B-Think",       # 64 layers, d_model 5120
    "smoke": "allenai/Olmo-3-7B-Instruct",    # matches VM1's smoke lens
    "think7b": "allenai/Olmo-3-7B-Think",     # cheap Phase-4 prototyping
}

# Fit geometry for the 32B (see PLAN.md): early/late controls + dense middle.
SOURCE_LAYERS_32B = [4, 8, 12, 16] + list(range(20, 45, 2)) + [48, 52, 56, 60]
SEED = 0


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def die(msg: str) -> None:
    print(f"SELF-CHECK FAILED: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def sync_code() -> None:
    subprocess.run(["bash", str(LAB_DIR / "sync_code.sh")], check=False)


def gpu_mem_gb() -> tuple[float, float]:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9, total / 1e9


def load_model(key_or_id: str, *, dtype=torch.bfloat16, device: str = "cuda:0"):
    """Load an HF causal LM fully onto one device and wrap it for jlens.

    Returns (lens_model, hf_model, tokenizer). Single-GPU placement on
    purpose: jlens compile and hook assumptions prefer no device_map.
    """
    import transformers
    import jlens

    model_id = MODELS.get(key_or_id, key_or_id)
    t0 = time.time()
    log(f"loading {model_id} ({dtype}) ...")
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    hf = hf.to(device)
    hf.eval()
    used, total = gpu_mem_gb()
    log(f"loaded in {time.time()-t0:.0f}s; VRAM {used:.1f}/{total:.1f} GB")
    lm = jlens.from_hf(hf, tok)
    return lm, hf, tok


def chat_prompt(tok, user: str, *, system: str | None = None,
                assistant_prefill: str = "", add_generation_prompt: bool = True) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
        return tok.apply_chat_template(messages, tokenize=False,
                                       continue_final_message=True)
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def answer_rank(logits: torch.Tensor, token_id: int) -> int:
    """1-indexed rank of token_id in a [vocab] logits vector."""
    return int((logits > logits[token_id]).sum().item()) + 1


def first_token_id(tok, text: str) -> int:
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        die(f"no tokens for {text!r}")
    return ids[0]


def variant_first_ids(tok, word: str) -> list[int]:
    """First-token ids over case/space variants of `word`.

    Multi-token answers (' lira' -> ' l'+'ira') make single-first-token
    ranks brittle; the paper's evals use min-rank over a synonym set — this
    is the analogous variant set. Prefers variants whose first token spans
    >= 3 chars; falls back to all first tokens if none do.
    """
    w = word.strip()
    variants = {f" {w}", w, f" {w.capitalize()}", f" {w.lower()}",
                f" {w.upper()}"}
    ids, good = [], []
    for v in variants:
        t = tok(v, add_special_tokens=False).input_ids
        if not t:
            continue
        ids.append(t[0])
        if len(tok.decode([t[0]]).strip()) >= min(3, len(w)):
            good.append(t[0])
    out = sorted(set(good or ids))
    if not out:
        die(f"no tokens for {word!r}")
    return out


def ensure_dirs() -> None:
    for sub in ("config/prompts", "lens", "metrics", "figures", "logs",
                "report", "code"):
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
