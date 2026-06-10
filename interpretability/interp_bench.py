#!/usr/bin/env python3
"""Teaching lab bench for mechanistic interpretability experiments.

This file is the executable "microscope" for the interpretability course. The
individual files under ``labs/`` own the experiments: logit lens trajectories,
direct logit attribution, attention routing, probing, patching, and so on.
This file owns the surrounding instrument:

* parse a CLI into a run configuration with hardware-tier defaults;
* create a run directory and tee all console output into it;
* record diagnostics (packages, git state, GPU report, environment) so a run
  can be debugged after the Colab VM is gone;
* load a Hugging Face causal LM and *resolve its anatomy* (where the decoder
  blocks, final norm, and unembedding actually live) into a written report;
* capture the residual stream with explicit, verifiable semantics;
* apply the logit lens with the final-norm handling spelled out;
* verify itself: hook captures are cross-checked against
  ``output_hidden_states``, and the lens at the final depth is cross-checked
  against the model's real output logits, every run;
* dump model state in *human-readable* form: token tables, per-layer stats,
  decoded top-k readouts, and a markdown "state card" per example. Raw
  tensors are only written when ``--save-tensors`` is passed, and always with
  a manifest that says what every tensor is.

Why the bench is deliberately verbose
=====================================

Interpretability claims live or die on instrumentation details: which stream
was captured (pre-norm or post-norm?), which position, which dtype, whether
the readout matches what the model actually computed. Those details are
spelled out in code and comments here instead of being hidden in a library,
because the course's first rule is that you should never trust a plot whose
provenance you cannot explain.

Quick start
===========

Run from this directory (or pass absolute paths). CPU smoke test with gpt2 --
this must work on a laptop with no GPU:

    python interp_bench.py --lab lab1 --tier a

Full Lab 1 on a Colab A100/H100 (bf16, ~24 prompts, a few minutes):

    python interp_bench.py --lab lab1 --tier b

Choose your own model or prompt set:

    python interp_bench.py --lab lab1 \
      --model google/gemma-3-1b-pt \
      --prompt-set full \
      --topk 10

Outputs
=======

Every run creates a directory under ``interpretability/runs`` unless
``--run-dir`` or ``--run-root`` says otherwise. The main artifacts are:

* ``logs/console.log``: everything printed during the run;
* ``run_config.json``: the parsed CLI;
* ``run_metadata.json``: packages, git, device, environment;
* ``diagnostics/``: model anatomy report, hook parity check, lens self-check,
  tokenization report, GPU memory snapshots;
* ``state/<example_id>/``: per-example human-readable model state dumps;
* ``results.csv``, ``metrics.json``, ``tables/``: lab-specific measurements;
* ``plots/*.png`` unless ``--no-plots``;
* ``run_summary.md``: the lab's answers to the standard seven questions;
* ``ledger_suggestions.md``: drafted claims for the student's claim ledger;
* ``artifact_index.json``: a map of every artifact with a one-line purpose.

The claim ledger
================

``claim_ledger.md`` at the course root is the student's running dossier of
claims about one model. Labs draft claims with measured numbers into
``ledger_suggestions.md``; nothing is appended to the real ledger unless
``--append-ledger`` is passed, because writing the claim *is* the coursework.

Residual stream semantics (read this once, carefully)
=====================================================

For a decoder with L blocks, Hugging Face ``output_hidden_states=True``
returns L+1 tensors, but their indexing is subtle and this harness re-maps it:

* ``hidden_states[k]`` for k in 0..L-1 is the stream *entering* block k,
  i.e. the residual stream after k blocks (k=0 is the embedding output).
* ``hidden_states[L]`` is the stream after ALL L blocks *with the final
  norm already applied*. The raw (pre-norm) output of the last block is not
  in the tuple at all.

The bench therefore captures the final norm's *input* with a forward
pre-hook and assembles ``streams[k]`` for k in 0..L as the **pre-norm
residual stream after k blocks**. The logit lens at depth k is then::

    lens(k) = lm_head( final_norm( streams[k] ) )

and lens(L) must reproduce the model's actual output logits -- which the
bench asserts on every run (``diagnostics/logit_lens_self_check.json``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Course map
# ---------------------------------------------------------------------------
#
# The CLI selects a lab by name; each lab is one module under labs/ that
# exposes run(ctx, bundle). Keeping the registry at the top of the file makes
# the course map visible before any implementation details appear.

COURSE_ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = COURSE_ROOT / "runs"
LEDGER_PATH = COURSE_ROOT / "claim_ledger.md"

LAB_PROFILES: dict[str, dict[str, str]] = {
    "lab1": {
        "module": "labs.lab01_residual_logit_lens",
        "run_name": "lab01_residual_logit_lens",
        "description": "Residual stream and logit lens: how a prediction emerges over depth.",
    },
    "lab2": {
        "module": "labs.lab02_direct_logit_attribution",
        "run_name": "lab02_direct_logit_attribution",
        "description": "Direct logit attribution: which components push toward or away from an answer.",
    },
}

# Hardware tiers. Tier A must run on a laptop CPU so every lab is debuggable
# without a GPU; tier B is the primary target (one Colab A100/H100 or any
# 24GB+ card); tier C is a comfortable 40-80GB card for full-precision runs.
TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "a": {"model": "gpt2", "dtype": "float32", "max_examples": 4},
    "b": {"model": "allenai/Olmo-3-1025-7B", "dtype": "bfloat16", "max_examples": 0},
    "c": {"model": "allenai/Olmo-3-1025-7B", "dtype": "float32", "max_examples": 0},
}

# Where decoder blocks and the final norm live for the model families the
# course uses. Resolution is by attribute path probing, and the result is
# written to diagnostics/model_anatomy.{json,md} so nothing is implicit. If a
# new architecture fails to resolve, add its paths here -- one place, on
# purpose.
BLOCKS_PATH_CANDIDATES = (
    "model.layers",            # Llama / Olmo / Gemma / Qwen / Mistral style
    "transformer.h",           # GPT-2 style
    "gpt_neox.layers",         # GPT-NeoX / Pythia style
    "model.decoder.layers",    # OPT style
    "transformer.blocks",      # MPT style
)
FINAL_NORM_PATH_CANDIDATES = (
    "model.norm",                    # Llama / Olmo / Gemma / Qwen / Mistral
    "transformer.ln_f",              # GPT-2
    "gpt_neox.final_layer_norm",     # GPT-NeoX / Pythia
    "model.decoder.final_layer_norm",  # OPT
    "transformer.norm_f",            # MPT
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def now_slug() -> str:
    """Timestamp fragment used in auto-generated run names."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_tag(text: Any) -> str:
    """Return a filesystem-friendly tag for run and artifact names."""
    tag = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(text)).strip("_")
    return tag[:180] or "untitled"


def json_default(obj: Any) -> Any:
    """JSON fallback for dataclasses, paths, tensors, and odd objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        with contextlib.suppress(Exception):
            return obj.tolist()
    return repr(obj)


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write a deterministic, human-readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_text(path: pathlib.Path, text: str) -> None:
    """Write a UTF-8 text artifact, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows with the union of keys in first-seen order.

    Different examples produce different optional fields (ambiguous prompts
    have no target columns). Building the header from all rows keeps exports
    complete without requiring every row to know every possible column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def sha256_file(path: pathlib.Path, *, max_bytes: int | None = None) -> str | None:
    """Return a SHA256 digest for a file, or None if the file is unavailable.

    The digest in artifact_index.json is a cheap reproducibility anchor: when a
    student zips a run directory or copies it out of Colab, the artifact map can
    still tell whether the important CSV or plot changed. Large optional tensor
    blobs can be skipped by passing max_bytes.
    """
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def visible_token(text: str) -> str:
    """Render a token string so whitespace is visible in tables and cards.

    Leading spaces are the most important single character in tokenizer
    behavior ("Paris" and " Paris" are different tokens), so they must never
    be invisible in an artifact a human is supposed to read.
    """
    return (
        text.replace(" ", "␣")  # open-box symbol for space
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def get_by_path(obj: Any, dotted: str) -> Any:
    """Resolve an attribute path like ``model.layers`` on a module tree."""
    return functools.reduce(getattr, dotted.split("."), obj)


# ---------------------------------------------------------------------------
# Console tee
# ---------------------------------------------------------------------------


class ConsoleTee:
    """Mirror Python-level stdout/stderr into ``logs/console.log``.

    This is a deliberate simplification relative to file-descriptor-level
    capture: PyTorch and Transformers report through Python streams and the
    ``warnings``/``logging`` modules, which this catches. Native CUDA-level
    prints (rare in these labs) still reach the terminal but not the log.
    """

    class _Tee:
        def __init__(self, stream: Any, log_file: Any) -> None:
            self._stream = stream
            self._log = log_file

        def write(self, data: str) -> int:
            self._stream.write(data)
            self._log.write(data)
            return len(data)

        def flush(self) -> None:
            self._stream.flush()
            self._log.flush()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._stream, name)

    def __init__(self, log_dir: pathlib.Path) -> None:
        self.log_dir = log_dir
        self._file: Any = None
        self._saved: tuple[Any, Any] | None = None

    def __enter__(self) -> "ConsoleTee":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.log_dir / "console.log").open("a", encoding="utf-8")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = self._Tee(self._saved[0], self._file)
        sys.stderr = self._Tee(self._saved[1], self._file)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
        if self._file is not None:
            self._file.close()


# ---------------------------------------------------------------------------
# Diagnostics: environment, packages, git, GPU
# ---------------------------------------------------------------------------


def env_subset() -> dict[str, str]:
    """Capture environment variables that commonly affect model runs."""
    prefixes = (
        "CUDA_",
        "HF_",
        "HUGGINGFACE_",
        "TRANSFORMERS_",
        "TOKENIZERS_",
        "TORCH_",
        "PYTORCH_",
        "NVIDIA_",
        "MPL",
        "BNB_",
    )
    secret_markers = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")
    captured: dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        if not k.startswith(prefixes):
            continue
        # Run metadata gets zipped and shared; never persist credential values.
        captured[k] = "<redacted>" if any(m in k.upper() for m in secret_markers) else v
    return captured


def package_version(name: str) -> str | None:
    """Return an installed package version, or None when unavailable."""
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def git_state(cwd: pathlib.Path) -> dict[str, Any]:
    """Record repository identity without making git a hard dependency."""

    def run(cmd: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), check=False, capture_output=True, text=True
            )
        except Exception:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    status = run(["git", "status", "--porcelain"])
    return {
        "sha": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def gpu_report(torch: Any) -> dict[str, Any]:
    """Serialize visible accelerator state for the metadata artifact."""
    report: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "devices": [],
    }
    with contextlib.suppress(Exception):
        report["cudnn_version"] = torch.backends.cudnn.version()
    if report["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    return report


def gpu_memory_snapshot(torch: Any, label: str) -> dict[str, Any]:
    """Best-effort snapshot of allocator and device memory state."""
    snap: dict[str, Any] = {"label": label, "time_unix": time.time()}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        snap.update(
            {
                "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
                "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "device_free_gib": round(free_b / 1024**3, 3),
                "device_total_gib": round(total_b / 1024**3, 3),
            }
        )
    return snap


def collect_run_metadata(torch: Any | None = None) -> dict[str, Any]:
    """Collect host, package, git, and environment diagnostics."""
    meta: dict[str, Any] = {
        "time_unix": time.time(),
        "time_iso": datetime.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "env": env_subset(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "tokenizers",
                "accelerate",
                "safetensors",
                "bitsandbytes",
                "numpy",
                "matplotlib",
            )
        },
        "git": git_state(COURSE_ROOT),
    }
    if torch is not None:
        meta["gpu"] = gpu_report(torch)
    return meta


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunContext:
    """Mutable state shared by the bench and the lab module.

    ``artifacts`` accumulates an index entry for every file worth opening, so
    ``artifact_index.json`` can serve as the map when the run directory gets
    crowded.
    """

    run_dir: pathlib.Path
    args: argparse.Namespace
    started_unix: float = dataclasses.field(default_factory=time.time)
    artifacts: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def path(self, *parts: str) -> pathlib.Path:
        """Resolve a run-relative path, creating parent directories."""
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def register_artifact(self, path: pathlib.Path, kind: str, description: str) -> None:
        """Register a generated file in the run's artifact index.

        The index includes size and a digest for ordinary text/plot artifacts.
        Raw tensor blobs are intentionally not hashed by default because they can
        be large; their own manifest explains their contents.
        """
        rel = str(path.relative_to(self.run_dir)) if path.is_relative_to(self.run_dir) else str(path)
        entry: dict[str, Any] = {"path": rel, "kind": kind, "description": description}
        with contextlib.suppress(OSError):
            entry["size_bytes"] = path.stat().st_size
        digest = sha256_file(path, max_bytes=64 * 1024 * 1024)
        if digest:
            entry["sha256"] = digest
        self.artifacts.append(entry)


# ---------------------------------------------------------------------------
# Device, dtype, seeding
# ---------------------------------------------------------------------------


def resolve_device(torch: Any, requested: str) -> str:
    """Map ``--device auto`` to the best available backend."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    """Map a dtype name to a torch dtype, with safe per-device fallbacks.

    bf16 on a GPU without bf16 support silently degrades accuracy through
    emulation, and MPS bf16 support is patchy -- so non-CUDA devices fall back
    to float32 unless the user forces otherwise.
    """
    names = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if requested != "auto":
        dtype = names[requested]
        if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
            print("[bench] WARNING: bf16 requested but not supported; using float16")
            return torch.float16
        return dtype
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def set_determinism(torch: Any, seed: int) -> None:
    """Seed all RNGs. Lab forward passes are deterministic anyway, but later
    labs sample and probe-train, so the habit starts in the bench."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(seed)


def first_module_device(module: Any) -> Any | None:
    """Best-effort device for a module's parameters or buffers."""
    with contextlib.suppress(Exception):
        for param in module.parameters(recurse=True):
            if str(param.device) != "meta":
                return param.device
    with contextlib.suppress(Exception):
        for buffer in module.buffers(recurse=True):
            if str(buffer.device) != "meta":
                return buffer.device
    return None


def infer_input_device(model: Any, fallback: str) -> Any:
    """Find the device where input_ids should be placed.

    For ordinary single-device models this is just the requested device. For
    quantized or device_map="auto" models, feeding inputs to the input embedding
    device is more reliable than assuming every module lives on cuda:0.
    """
    with contextlib.suppress(Exception):
        emb = model.get_input_embeddings()
        dev = first_module_device(emb)
        if dev is not None:
            return dev
    with contextlib.suppress(Exception):
        dev = first_module_device(model)
        if dev is not None:
            return dev
    return fallback


def device_map_summary(model: Any) -> dict[str, Any] | None:
    """Serialize a Hugging Face device map when one exists."""
    mapping = getattr(model, "hf_device_map", None)
    if mapping is None:
        return None
    return {str(k): str(v) for k, v in mapping.items()}


def tensor_cpu_float(tensor: Any) -> Any:
    """Detach a tensor and store it as CPU float32 for diagnostics/metrics."""
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model loading and anatomy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ModelAnatomy:
    """Where the interpretable pieces of this model live, as strings.

    This is the JSON-serializable record; live module references are kept on
    ``ModelBundle``. The point of writing this down is that *no lab should
    ever guess* which module is the residual-stream block list or the final
    norm -- the bench resolves it once and shows its work.
    """

    model_id: str
    revision: str | None
    architecture: str
    blocks_path: str
    n_layers: int
    final_norm_path: str
    final_norm_class: str
    lm_head_class: str
    d_model: int
    vocab_size: int
    tied_embeddings: bool
    param_count: int
    logit_softcap: float | None
    notes: tuple[str, ...] = ()


@dataclasses.dataclass
class ModelBundle:
    """The loaded model plus resolved live references used by every lab."""

    model: Any
    tokenizer: Any
    anatomy: ModelAnatomy
    blocks: Any          # nn.ModuleList of decoder blocks
    final_norm: Any      # the norm applied after the last block
    lm_head: Any         # the unembedding (output projection to vocab)
    device: str          # requested or resolved primary device label
    input_device: Any    # actual device for input_ids, robust to device_map="auto"
    lens_device: Any     # actual device for final_norm/lens matmuls
    torch_dtype: Any     # compute dtype used for lens matmuls
    model_device_map: dict[str, Any] | None = None


def resolve_anatomy(model: Any, model_id: str, revision: str | None) -> tuple[ModelAnatomy, Any, Any, Any]:
    """Probe the module tree for blocks, final norm, and unembedding.

    Returns the serializable anatomy plus the live (blocks, final_norm,
    lm_head) references. Raises with a actionable message when the model is
    not a plain decoder-only LM the candidates cover.
    """
    notes: list[str] = []

    blocks = None
    blocks_path = ""
    for candidate in BLOCKS_PATH_CANDIDATES:
        with contextlib.suppress(AttributeError):
            blocks = get_by_path(model, candidate)
            blocks_path = candidate
            break
    if blocks is None:
        raise RuntimeError(
            f"Could not find decoder blocks on {model_id!r}. Tried: "
            f"{BLOCKS_PATH_CANDIDATES}. If this is a new architecture, add its "
            "block path to BLOCKS_PATH_CANDIDATES in interp_bench.py; if it is "
            "a multimodal or encoder-decoder model, it is out of scope for "
            "this course's labs."
        )

    final_norm = None
    final_norm_path = ""
    for candidate in FINAL_NORM_PATH_CANDIDATES:
        with contextlib.suppress(AttributeError):
            final_norm = get_by_path(model, candidate)
            final_norm_path = candidate
            break
    if final_norm is None:
        raise RuntimeError(
            f"Could not find the final norm on {model_id!r}. Tried: "
            f"{FINAL_NORM_PATH_CANDIDATES}. Add the path to "
            "FINAL_NORM_PATH_CANDIDATES in interp_bench.py."
        )

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError(f"{model_id!r} has no output embeddings / lm_head.")

    config = model.config
    d_model = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    vocab_size = int(lm_head.weight.shape[0])

    tied = bool(getattr(config, "tie_word_embeddings", False))
    if tied:
        notes.append(
            "Input and output embeddings are tied: the unembedding is the "
            "transpose of the token embedding."
        )

    # Gemma-2-style models squash final logits with cap*tanh(logits/cap).
    # The lens must reproduce this or the depth-L self-check would fail for a
    # boring reason.
    softcap = getattr(config, "final_logit_softcapping", None)
    softcap = float(softcap) if softcap else None
    if softcap:
        notes.append(f"Model applies final logit softcapping (cap={softcap}).")

    anatomy = ModelAnatomy(
        model_id=model_id,
        revision=revision,
        architecture=type(model).__name__,
        blocks_path=blocks_path,
        n_layers=len(blocks),
        final_norm_path=final_norm_path,
        final_norm_class=type(final_norm).__name__,
        lm_head_class=type(lm_head).__name__,
        d_model=d_model,
        vocab_size=vocab_size,
        tied_embeddings=tied,
        param_count=sum(p.numel() for p in model.parameters()),
        logit_softcap=softcap,
        notes=tuple(notes),
    )
    return anatomy, blocks, final_norm, lm_head


def write_tokenizer_report(ctx: RunContext, bundle: ModelBundle) -> None:
    """Write tokenizer facts that commonly explain surprising lab results."""
    tok = bundle.tokenizer
    payload = {
        "tokenizer_class": type(tok).__name__,
        "vocab_size": getattr(tok, "vocab_size", None),
        "model_max_length": getattr(tok, "model_max_length", None),
        "bos_token": getattr(tok, "bos_token", None),
        "bos_token_id": getattr(tok, "bos_token_id", None),
        "eos_token": getattr(tok, "eos_token", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
        "pad_token": getattr(tok, "pad_token", None),
        "pad_token_id": getattr(tok, "pad_token_id", None),
        "padding_side": getattr(tok, "padding_side", None),
        "truncation_side": getattr(tok, "truncation_side", None),
        "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "chat_template_used_by_lab1": False,
        "note": (
            "Lab 1 uses raw base-model completions. A chat template, if present "
            "on the tokenizer, is deliberately not applied here."
        ),
    }
    path = ctx.path("diagnostics", "tokenizer_info.json")
    write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Tokenizer special tokens and template status.")


def write_anatomy_report(ctx: RunContext, bundle: ModelBundle) -> None:
    """Write the anatomy as JSON (machines) and Markdown (humans)."""
    a = bundle.anatomy
    write_json(ctx.path("diagnostics", "model_anatomy.json"), a)
    lines = [
        "# Model anatomy",
        "",
        "What the bench resolved before any experiment ran. Every lab's hook",
        "and readout uses these paths; if they look wrong, stop here.",
        "",
        f"| field | value |",
        f"|---|---|",
        f"| model | `{a.model_id}` (revision: {a.revision or 'default'}) |",
        f"| architecture | `{a.architecture}` |",
        f"| parameters | {a.param_count:,} |",
        f"| decoder blocks | `{a.blocks_path}` x {a.n_layers} |",
        f"| final norm | `{a.final_norm_path}` ({a.final_norm_class}) |",
        f"| unembedding | {a.lm_head_class}, vocab {a.vocab_size:,} |",
        f"| d_model | {a.d_model} |",
        f"| tied embeddings | {a.tied_embeddings} |",
        f"| logit softcap | {a.logit_softcap or 'none'} |",
        "",
        "Depth convention used everywhere in this course: `streams[k]` is the",
        "**pre-norm residual stream after k blocks**; k=0 is the embedding",
        f"output and k={a.n_layers} is the input to the final norm.",
        "",
    ]
    if a.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in a.notes)
        lines.append("")
    write_text(ctx.path("diagnostics", "model_anatomy.md"), "\n".join(lines))
    ctx.register_artifact(
        ctx.run_dir / "diagnostics" / "model_anatomy.md",
        "diagnostic",
        "Resolved module paths for blocks, final norm, unembedding.",
    )


def load_model_and_tokenizer(ctx: RunContext) -> ModelBundle:
    """Load tokenizer + causal LM per the run config, then resolve anatomy."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = ctx.args
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype, device)
    print(f"[bench] loading {args.model!r} (device={device}, dtype={dtype})")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    load_kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if args.quantization in ("8bit", "4bit"):
        # Quantization is an opt-in convenience for small GPUs. It changes
        # numerics, so the lens self-check tolerances are looser there and
        # run_summary.md records the setting.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("--quantization requires the bitsandbytes package") from exc
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quantization == "8bit",
            load_in_4bit=args.quantization == "4bit",
            bnb_4bit_compute_dtype=dtype,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except TypeError as exc:
        # Older transformers spell the dtype argument torch_dtype. Re-raise
        # unrelated TypeErrors so unsupported kwargs do not get silently masked.
        if "dtype" not in load_kwargs:
            raise
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    if args.quantization == "none":
        model = model.to(device)
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[bench] model loaded in {load_s:.1f}s")

    anatomy, blocks, final_norm, lm_head = resolve_anatomy(model, args.model, args.model_revision)
    input_device = infer_input_device(model, device)
    lens_device = first_module_device(final_norm) or input_device
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        anatomy=anatomy,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
        device=device,
        input_device=input_device,
        lens_device=lens_device,
        torch_dtype=dtype,
        model_device_map=device_map_summary(model),
    )
    write_anatomy_report(ctx, bundle)
    write_tokenizer_report(ctx, bundle)
    if bundle.model_device_map is not None:
        path = ctx.path("diagnostics", "model_device_map.json")
        write_json(path, bundle.model_device_map)
        ctx.register_artifact(path, "diagnostic", "Hugging Face device map for quantized/offloaded runs.")
    mem_path = ctx.path("diagnostics", "gpu_memory_after_load.json")
    write_json(mem_path, gpu_memory_snapshot(torch, "after_load"))
    ctx.register_artifact(mem_path, "diagnostic", "GPU memory snapshot after model load.")
    return bundle


# ---------------------------------------------------------------------------
# Residual stream capture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ForwardCapture:
    """One prompt's worth of model state, with explicit semantics.

    ``streams`` has shape ``[n_layers + 1, seq_len, d_model]`` in float32:
    the **pre-norm** residual stream after k blocks, for k = 0..n_layers
    (see the module docstring for how this is assembled from
    ``output_hidden_states`` plus a final-norm pre-hook).

    ``final_logits_last`` is the model's *actual* output logits at the final
    position, float32 -- the ground truth the logit lens is checked against.
    """

    prompt: str
    input_ids: list[int]
    tokens_raw: list[str]    # tokenizer-internal pieces, e.g. 'ĠParis'
    tokens_text: list[str]   # decoded text per token, e.g. ' Paris'
    streams: Any             # torch.Tensor [L+1, seq, d_model] float32
    final_logits_last: Any   # torch.Tensor [vocab] float32


def run_with_residual_cache(bundle: ModelBundle, prompt: str) -> ForwardCapture:
    """Run one prompt and capture the full pre-norm residual stream.

    Prompts run one at a time, unbatched. With <100 short prompts per lab the
    cost is irrelevant, and skipping batching removes the entire class of
    padding/attention-mask bugs from the course's foundation.
    """
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    # The final block's pre-norm output is not in hidden_states (see module
    # docstring), so capture the final norm's input as it flows past.
    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = tensor_cpu_float(hook_args[0])

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.no_grad():
            out = bundle.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        handle.remove()

    if "final_prenorm" not in captured:
        raise RuntimeError(
            "The final-norm pre-hook never fired. The resolved final_norm "
            "module is probably wrong for this architecture -- check "
            "diagnostics/model_anatomy.md."
        )

    hs = out.hidden_states  # tuple of L+1 tensors, indexing per module docstring
    n_layers = bundle.anatomy.n_layers
    if len(hs) != n_layers + 1:
        raise RuntimeError(
            f"Expected {n_layers + 1} hidden states, got {len(hs)}. The "
            "residual indexing assumptions do not hold for this architecture."
        )

    # streams[k] = pre-norm residual after k blocks:
    #   k in 0..L-1  -> hidden_states[k]   (input to block k)
    #   k = L        -> the final norm's captured input (output of block L-1)
    streams = torch.stack(
        [tensor_cpu_float(h[0]) for h in hs[:-1]] + [captured["final_prenorm"][0]]
    )

    ids = input_ids[0].detach().cpu().tolist()
    return ForwardCapture(
        prompt=prompt,
        input_ids=ids,
        tokens_raw=tokenizer.convert_ids_to_tokens(ids),
        tokens_text=[tokenizer.decode([i]) for i in ids],
        streams=streams,
        final_logits_last=tensor_cpu_float(out.logits[0, -1]),
    )


def run_hook_parity_check(ctx: RunContext, bundle: ModelBundle, prompt: str) -> dict[str, Any]:
    """Cross-check per-block forward hooks against the assembled stream cache.

    A mismatch means the harness is not measuring the object it says it is
    measuring. By default, this aborts the run. ``--allow-hook-mismatch`` turns
    the abort into a diagnostic warning for architecture bring-up work.
    """
    import torch

    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = tensor_cpu_float(out)

        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = run_with_residual_cache(bundle, prompt)
    finally:
        for handle in handles:
            handle.remove()

    n_layers = bundle.anatomy.n_layers
    by_layer_rows: list[dict[str, Any]] = []
    max_diff = 0.0
    max_mean_diff = 0.0
    compared = 0
    missing_layers: list[int] = []
    for k in range(n_layers):
        if k not in block_outputs:
            missing_layers.append(k)
            continue
        # Block k's output is the stream after k+1 blocks == streams[k+1].
        hook_out = block_outputs[k][0]
        expected = capture.streams[k + 1]
        abs_diff = (hook_out - expected).abs()
        layer_max = float(abs_diff.max())
        layer_mean = float(abs_diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean_diff = max(max_mean_diff, layer_mean)
        compared += 1
        by_layer_rows.append(
            {
                "layer": k,
                "max_abs_diff": layer_max,
                "mean_abs_diff": layer_mean,
                "hook_l2": float(hook_out.norm()),
                "expected_l2": float(expected.norm()),
                "shape": "x".join(str(x) for x in hook_out.shape),
                "ok_at_tolerance": layer_max <= ctx.args.hook_tolerance,
            }
        )

    by_layer_path = ctx.path("diagnostics", "hook_parity_by_layer.csv")
    write_csv(by_layer_path, by_layer_rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Layer-level hook versus hidden-state parity check.")

    ok = (not missing_layers) and compared == n_layers and max_diff <= ctx.args.hook_tolerance
    result = {
        "prompt": prompt,
        "blocks_compared": compared,
        "n_layers": n_layers,
        "missing_layers": missing_layers,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean_diff,
        "tolerance": ctx.args.hook_tolerance,
        "ok": bool(ok),
        "allow_hook_mismatch": bool(ctx.args.allow_hook_mismatch),
        "explanation": (
            "Forward hooks on each decoder block were compared against the "
            "streams assembled from output_hidden_states plus the final-norm "
            "pre-hook. A mismatch means the residual stream capture semantics "
            "are not verified for this architecture or library version."
        ),
    }
    path = ctx.path("diagnostics", "hook_parity.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Summary of hook captures versus assembled residual streams.")
    status = "OK" if result["ok"] else "MISMATCH"
    print(f"[bench] hook parity check: {status} (max |diff| = {max_diff:g}, compared {compared}/{n_layers})")
    if not result["ok"] and not ctx.args.allow_hook_mismatch:
        raise RuntimeError(
            "Hook parity check failed. The harness did not verify its residual "
            "capture semantics. See diagnostics/hook_parity*."
        )
    return result

# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------


def logit_lens_all_depths(bundle: ModelBundle, streams_at_position: Any) -> Any:
    """Apply final_norm + unembedding to residual streams at one position.

    The cached streams live on CPU float32 so diagnostics do not pin GPU memory.
    This function moves only the depth x d_model slice needed for the lens to
    the final-norm device, runs the model's own readout modules, then returns
    CPU float32 logits. At depth L this should reproduce the model's real logits.
    """
    import torch

    with torch.no_grad():
        norm_device = first_module_device(bundle.final_norm) or bundle.lens_device
        head_device = first_module_device(bundle.lm_head) or norm_device
        x = streams_at_position.to(device=norm_device, dtype=bundle.torch_dtype)
        normed = bundle.final_norm(x)
        if str(head_device) != str(norm_device):
            normed = normed.to(head_device)
        logits = bundle.lm_head(normed).float()
        if bundle.anatomy.logit_softcap:
            cap = bundle.anatomy.logit_softcap
            logits = cap * torch.tanh(logits / cap)
    return logits.detach().to(device="cpu", dtype=torch.float32)

def run_lens_self_check(ctx: RunContext, bundle: ModelBundle, capture: ForwardCapture) -> dict[str, Any]:
    """Verify lens(depth=L) reproduces the model's actual final logits.

    Top-1 agreement is required. The numeric logit difference is reported, not
    used as a hard failure by default, because recomputing the final projection
    outside the model may avoid fused kernels or quantized execution paths.
    """
    import torch

    lens_logits = logit_lens_all_depths(bundle, capture.streams[:, -1, :])
    lens_final = lens_logits[-1]
    real_final = capture.final_logits_last

    lens_top = torch.topk(lens_final, k=min(5, lens_final.numel()))
    real_top = torch.topk(real_final, k=min(5, real_final.numel()))
    lens_top1 = int(lens_top.indices[0])
    real_top1 = int(real_top.indices[0])
    max_diff = float((lens_final - real_final).abs().max())
    mean_diff = float((lens_final - real_final).abs().mean())
    rel = max_diff / max(1e-9, float(real_final.abs().max()))
    top5_overlap = len(set(int(x) for x in lens_top.indices) & set(int(x) for x in real_top.indices))

    top1_matches = lens_top1 == real_top1
    # A bf16 recomputation can legitimately flip top-1 on a near-tie prompt:
    # different matmul shapes reduce in different orders. Accept a mismatch
    # only when the model's own logit gap between the two candidates is within
    # the observed numeric noise floor and both candidates sit in each other's
    # top-5. Anything larger means the capture or norm path is wrong.
    near_tie_ok = False
    real_gap = None
    if not top1_matches:
        real_gap = float(real_final[real_top1] - real_final[lens_top1])
        near_tie_ok = real_gap <= max_diff * 1.5 and top5_overlap >= 4

    result = {
        "prompt": capture.prompt,
        "top1_matches": top1_matches,
        "near_tie_accepted": near_tie_ok,
        "real_logit_gap_top1_vs_lens_top1": real_gap,
        "lens_top1": bundle.tokenizer.decode([lens_top1]),
        "model_top1": bundle.tokenizer.decode([real_top1]),
        "lens_top5": [bundle.tokenizer.decode([int(i)]) for i in lens_top.indices],
        "model_top5": [bundle.tokenizer.decode([int(i)]) for i in real_top.indices],
        "top5_overlap": top5_overlap,
        "max_abs_logit_diff": max_diff,
        "mean_abs_logit_diff": mean_diff,
        "max_rel_logit_diff": rel,
        "quantization": ctx.args.quantization,
        "ok": top1_matches or near_tie_ok,
        "explanation": (
            "lens(L) = lm_head(final_norm(stream after all blocks)) recomputed "
            "outside the model must reproduce the model's own top prediction. "
            "If top-1 disagrees, the capture or norm/logit post-processing is "
            "wrong and no mid-depth lens readout in this run can be trusted."
        ),
    }
    path = ctx.path("diagnostics", "logit_lens_self_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that the lens at final depth reproduces the model's logits.")
    status = "OK" if top1_matches else ("OK (near-tie within numeric noise)" if near_tie_ok else "FAILED")
    print(
        f"[bench] lens self-check: {status} "
        f"(top1 lens={result['lens_top1']!r} model={result['model_top1']!r}, "
        f"max |dlogit| = {max_diff:.4f}, top5 overlap={top5_overlap}/5)"
    )
    if not result["ok"]:
        raise RuntimeError(
            "Logit lens self-check failed: the lens at final depth does not "
            "reproduce the model's own prediction. See diagnostics/logit_lens_self_check.json."
        )
    return result

# ---------------------------------------------------------------------------
# Lens trajectory: the per-depth measurement pack used by Lab 1 (and reused
# by later labs as the "running prediction" baseline view).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LensTrajectory:
    """Per-depth readout metrics at the final position, as plain lists."""

    n_depths: int
    top1_ids: list[int]
    top1_texts: list[str]
    top1_probs: list[float]
    top2_ids: list[int]
    top2_texts: list[str]
    top2_probs: list[float]
    top1_margin: list[float]
    entropy_bits: list[float]
    kl_to_final_bits: list[float]
    cosine_to_final: list[float]
    cosine_to_prev: list[float | None]
    resid_l2: list[float]
    stream_delta_l2: list[float]
    p_target: list[float] | None
    p_distractor: list[float] | None
    logit_target: list[float] | None
    logit_distractor: list[float] | None
    target_rank: list[int] | None
    distractor_rank: list[int] | None
    topk_rows: list[dict[str, Any]]  # depth, rank, token_id, token, prob, logit


def compute_lens_trajectory(
    bundle: ModelBundle,
    capture: ForwardCapture,
    *,
    target_id: int | None = None,
    distractor_id: int | None = None,
    topk: int = 5,
) -> LensTrajectory:
    """Compute the standard per-depth readout at the final token position.

    Besides the obvious p(target) and top-k outputs, this records metrics that
    help students debug interpretations: rank can improve before probability,
    entropy can drop before correctness, and KL-to-final can converge while the
    top-1 token still flips.
    """
    import torch

    streams_last = capture.streams[:, -1, :]  # [L+1, d_model] float32 on CPU
    logits = logit_lens_all_depths(bundle, streams_last)  # [L+1, vocab] CPU float32
    probs = torch.softmax(logits, dim=-1)
    vocab = logits.shape[-1]
    k = max(1, min(topk, vocab))

    # Entropy in bits: 0 = certain, log2(vocab) = uniform. Bits are intuitive:
    # 2.3 bits means roughly five equally plausible options.
    entropy = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=-1)

    final_probs = probs[-1].clamp_min(1e-12)
    kl_to_final = (final_probs.unsqueeze(0) * (torch.log2(final_probs.unsqueeze(0)) - torch.log2(probs.clamp_min(1e-12)))).sum(dim=-1)

    final_stream = streams_last[-1]
    cosine = torch.nn.functional.cosine_similarity(
        streams_last, final_stream.unsqueeze(0), dim=-1
    )
    resid_l2 = streams_last.norm(dim=-1)
    deltas = torch.zeros_like(resid_l2)
    if streams_last.shape[0] > 1:
        deltas[1:] = (streams_last[1:] - streams_last[:-1]).norm(dim=-1)
    cosine_prev: list[float | None] = [None]
    if streams_last.shape[0] > 1:
        prev_cos = torch.nn.functional.cosine_similarity(streams_last[1:], streams_last[:-1], dim=-1)
        cosine_prev.extend(float(v) for v in prev_cos)

    top = torch.topk(probs, k=k, dim=-1)
    top2 = torch.topk(probs, k=min(2, vocab), dim=-1)
    if top2.indices.shape[-1] == 1:
        top2_ids = top2.indices[:, 0]
        top2_probs = torch.zeros_like(top2.values[:, 0])
        margin = top2.values[:, 0]
    else:
        top2_ids = top2.indices[:, 1]
        top2_probs = top2.values[:, 1]
        margin = top2.values[:, 0] - top2.values[:, 1]

    topk_rows: list[dict[str, Any]] = []
    for depth in range(logits.shape[0]):
        for rank in range(k):
            token_id = int(top.indices[depth, rank])
            topk_rows.append(
                {
                    "depth": depth,
                    "rank": rank + 1,
                    "token_id": token_id,
                    "token": bundle.tokenizer.decode([token_id]),
                    "prob": round(float(top.values[depth, rank]), 6),
                    "logit": round(float(logits[depth, token_id]), 4),
                }
            )

    top1_ids = [int(i) for i in top.indices[:, 0]]

    def ranks_for(token_id: int | None) -> list[int] | None:
        if token_id is None:
            return None
        target_logits = logits[:, token_id].unsqueeze(-1)
        return [int(v) + 1 for v in (logits > target_logits).sum(dim=-1)]

    traj = LensTrajectory(
        n_depths=int(logits.shape[0]),
        top1_ids=top1_ids,
        top1_texts=[bundle.tokenizer.decode([i]) for i in top1_ids],
        top1_probs=[float(v) for v in top.values[:, 0]],
        top2_ids=[int(i) for i in top2_ids],
        top2_texts=[bundle.tokenizer.decode([int(i)]) for i in top2_ids],
        top2_probs=[float(v) for v in top2_probs],
        top1_margin=[float(v) for v in margin],
        entropy_bits=[float(v) for v in entropy],
        kl_to_final_bits=[float(v) for v in kl_to_final],
        cosine_to_final=[float(v) for v in cosine],
        cosine_to_prev=cosine_prev,
        resid_l2=[float(v) for v in resid_l2],
        stream_delta_l2=[float(v) for v in deltas],
        p_target=None,
        p_distractor=None,
        logit_target=None,
        logit_distractor=None,
        target_rank=ranks_for(target_id),
        distractor_rank=ranks_for(distractor_id),
        topk_rows=topk_rows,
    )
    if target_id is not None:
        traj.p_target = [float(v) for v in probs[:, target_id]]
        traj.logit_target = [float(v) for v in logits[:, target_id]]
    if distractor_id is not None:
        traj.p_distractor = [float(v) for v in probs[:, distractor_id]]
        traj.logit_distractor = [float(v) for v in logits[:, distractor_id]]
    return traj

# ---------------------------------------------------------------------------
# Component capture: per-block attention and MLP contributions (Lab 2+)
# ---------------------------------------------------------------------------
#
# Direct logit attribution needs the exact tensor each sub-block ADDS to the
# residual stream. Where that tensor lives differs by architecture:
#
#   GPT-2 (pre-norm):   x + attn(ln_1(x)); then + mlp(ln_2(.))
#                       -> the attn/mlp module outputs ARE the contributions.
#   Olmo-2/3 (post-norm): x + post_attention_layernorm(attn(x));
#                         then + post_feedforward_layernorm(mlp(.))
#                       -> the *norm* outputs are the contributions; the raw
#                          attn/mlp outputs never touch the stream.
#
# Name heuristics are a trap here: Llama-style models also have a module
# called `post_attention_layernorm`, but there it is a PRE-norm for the MLP.
# So the bench does not guess. It runs one probe forward, captures *both*
# candidate sources per block, and keeps whichever pair actually reconstructs
# each block's residual delta (streams[k+1] - streams[k]). The decision and
# its reconstruction error are written to diagnostics/component_anatomy.json.
# Same contract as the other self-checks: if nothing reconstructs, abort.

ATTN_MODULE_CANDIDATES = ("self_attn", "attn")
MLP_MODULE_CANDIDATES = ("mlp",)
POST_ATTN_NORM_CANDIDATES = ("post_attention_layernorm",)
POST_MLP_NORM_CANDIDATES = ("post_feedforward_layernorm",)


@dataclasses.dataclass
class ComponentAnatomy:
    """Verified per-block contribution hook points for this model."""

    attn_module_path: str        # e.g. 'self_attn' (relative to each block)
    mlp_module_path: str
    attn_source: str             # 'module' or 'post_norm'
    mlp_source: str
    attn_hook_path: str          # the path actually hooked for contributions
    mlp_hook_path: str
    max_block_recon_rel_err: float
    probe_prompt: str


@dataclasses.dataclass
class ComponentCapture:
    """One prompt's residual streams plus per-block contribution vectors.

    ``attn_contrib`` / ``mlp_contrib`` have shape ``[n_layers, d_model]``,
    float32 on CPU: the tensor block k added to the residual stream at the
    FINAL position. Together with ``capture.streams[0]`` (the embedding
    stream) they decompose the final pre-norm stream exactly:

        streams[L][-1] == streams[0][-1] + sum_k attn_contrib[k] + mlp_contrib[k]

    (up to the accumulation rounding of the model's compute dtype; the
    decomposition check below measures and enforces this).
    """

    capture: ForwardCapture
    attn_contrib: Any   # torch.Tensor [L, d_model] float32 cpu
    mlp_contrib: Any    # torch.Tensor [L, d_model] float32 cpu


def _first_module_path(block: Any, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if getattr(block, name, None) is not None:
            return name
    return None


def _contrib_hook(store: dict, key: tuple) -> Any:
    """Forward hook capturing a module's output at the final position."""

    def hook(module: Any, hook_args: tuple, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        store[key] = tensor_cpu_float(out[0, -1])

    return hook


def resolve_component_anatomy(
    ctx: RunContext, bundle: ModelBundle, probe_prompt: str, *, rel_tolerance: float = 0.02
) -> ComponentAnatomy:
    """Probe and VERIFY where each block's attn/mlp contributions live.

    Captures both the raw submodule outputs and (when present) the post-norm
    outputs in a single forward, then selects the (attn, mlp) source pair
    whose sum reconstructs every block's residual delta at the final
    position. ``rel_tolerance`` is relative to the delta's norm and must
    absorb only the model dtype's residual-add rounding (bf16: ~1%).
    """
    import torch

    block0 = bundle.blocks[0]
    attn_path = _first_module_path(block0, ATTN_MODULE_CANDIDATES)
    mlp_path = _first_module_path(block0, MLP_MODULE_CANDIDATES)
    if attn_path is None or mlp_path is None:
        raise RuntimeError(
            f"Could not find attention ({ATTN_MODULE_CANDIDATES}) or MLP "
            f"({MLP_MODULE_CANDIDATES}) submodules on the decoder block. Add "
            "this architecture's paths to interp_bench.py."
        )
    post_attn_path = _first_module_path(block0, POST_ATTN_NORM_CANDIDATES)
    post_mlp_path = _first_module_path(block0, POST_MLP_NORM_CANDIDATES)

    store: dict[tuple, Any] = {}
    handles = []
    for i, block in enumerate(bundle.blocks):
        handles.append(getattr(block, attn_path).register_forward_hook(_contrib_hook(store, ("attn", "module", i))))
        handles.append(getattr(block, mlp_path).register_forward_hook(_contrib_hook(store, ("mlp", "module", i))))
        if post_attn_path:
            handles.append(
                getattr(block, post_attn_path).register_forward_hook(_contrib_hook(store, ("attn", "post_norm", i)))
            )
        if post_mlp_path:
            handles.append(
                getattr(block, post_mlp_path).register_forward_hook(_contrib_hook(store, ("mlp", "post_norm", i)))
            )
    try:
        capture = run_with_residual_cache(bundle, probe_prompt)
    finally:
        for h in handles:
            h.remove()

    n_layers = bundle.anatomy.n_layers
    deltas = [capture.streams[k + 1, -1] - capture.streams[k, -1] for k in range(n_layers)]

    attn_sources = ["module"] + (["post_norm"] if post_attn_path else [])
    mlp_sources = ["module"] + (["post_norm"] if post_mlp_path else [])
    results: dict[tuple[str, str], float] = {}
    for a_src in attn_sources:
        for m_src in mlp_sources:
            worst = 0.0
            for k in range(n_layers):
                recon = store[("attn", a_src, k)] + store[("mlp", m_src, k)]
                denom = max(float(deltas[k].norm()), 1e-9)
                worst = max(worst, float((recon - deltas[k]).norm()) / denom)
            results[(a_src, m_src)] = worst

    (best_a, best_m), best_err = min(results.items(), key=lambda kv: kv[1])
    diag = {
        "probe_prompt": probe_prompt,
        "candidates_tried": {f"attn={a},mlp={m}": err for (a, m), err in results.items()},
        "selected": {"attn_source": best_a, "mlp_source": best_m},
        "max_block_recon_rel_err": best_err,
        "rel_tolerance": rel_tolerance,
        "explanation": (
            "For each candidate hook-point pair, every block's captured "
            "attn+mlp contribution must reconstruct that block's residual "
            "delta streams[k+1]-streams[k] at the final position. The "
            "selected pair is the verified place this model adds its "
            "components to the residual stream."
        ),
    }
    path = ctx.path("diagnostics", "component_anatomy.json")
    write_json(path, diag)
    ctx.register_artifact(path, "diagnostic", "Verified hook points for per-block attn/MLP contributions.")

    if best_err > rel_tolerance:
        raise RuntimeError(
            f"No contribution hook-point pair reconstructs the per-block residual "
            f"deltas (best: attn={best_a}, mlp={best_m}, max rel err {best_err:.4f} > "
            f"tolerance {rel_tolerance}). This architecture adds components to the "
            "residual stream somewhere the candidates do not cover; see "
            "diagnostics/component_anatomy.json."
        )

    print(
        f"[bench] component anatomy: attn={best_a}, mlp={best_m} "
        f"(max block reconstruction rel err = {best_err:.2e})"
    )
    return ComponentAnatomy(
        attn_module_path=attn_path,
        mlp_module_path=mlp_path,
        attn_source=best_a,
        mlp_source=best_m,
        attn_hook_path=attn_path if best_a == "module" else post_attn_path,
        mlp_hook_path=mlp_path if best_m == "module" else post_mlp_path,
        max_block_recon_rel_err=best_err,
        probe_prompt=probe_prompt,
    )


def run_with_component_cache(
    bundle: ModelBundle, prompt: str, comp_anatomy: ComponentAnatomy
) -> ComponentCapture:
    """Run one prompt capturing residual streams AND per-block contributions."""
    import torch

    store: dict[tuple, Any] = {}
    handles = []
    for i, block in enumerate(bundle.blocks):
        handles.append(
            getattr(block, comp_anatomy.attn_hook_path).register_forward_hook(_contrib_hook(store, ("attn", i)))
        )
        handles.append(
            getattr(block, comp_anatomy.mlp_hook_path).register_forward_hook(_contrib_hook(store, ("mlp", i)))
        )
    try:
        capture = run_with_residual_cache(bundle, prompt)
    finally:
        for h in handles:
            h.remove()

    n_layers = bundle.anatomy.n_layers
    return ComponentCapture(
        capture=capture,
        attn_contrib=torch.stack([store[("attn", i)] for i in range(n_layers)]),
        mlp_contrib=torch.stack([store[("mlp", i)] for i in range(n_layers)]),
    )


def run_decomposition_check(
    ctx: RunContext, bundle: ModelBundle, comp: ComponentCapture, *, rel_tolerance: float = 0.02
) -> dict[str, Any]:
    """Self-check 3: components must sum to the final pre-norm stream.

    embeddings + sum(attn contributions) + sum(mlp contributions) must equal
    streams[L] at the final position, up to the model compute dtype's
    residual-accumulation rounding. If this fails, every attribution number
    downstream is bookkeeping fiction; abort.
    """
    final_stream = comp.capture.streams[-1, -1]
    recon = comp.capture.streams[0, -1] + comp.attn_contrib.sum(dim=0) + comp.mlp_contrib.sum(dim=0)
    abs_err = float((recon - final_stream).norm())
    rel_err = abs_err / max(float(final_stream.norm()), 1e-9)
    result = {
        "prompt": comp.capture.prompt,
        "rel_err": rel_err,
        "abs_err": abs_err,
        "final_stream_norm": float(final_stream.norm()),
        "rel_tolerance": rel_tolerance,
        "ok": rel_err <= rel_tolerance,
        "explanation": (
            "The DLA ledger is only meaningful if embeddings + all captured "
            "attn/MLP contributions reconstruct the final pre-norm residual "
            "stream. rel_err measures the unexplained remainder; the "
            "tolerance absorbs the model dtype's residual-add rounding only."
        ),
    }
    path = ctx.path("diagnostics", "dla_decomposition_check.json")
    write_json(path, result)
    ctx.register_artifact(path, "diagnostic", "Proof that captured components sum to the final residual stream.")
    status = "OK" if result["ok"] else "FAILED"
    print(f"[bench] decomposition check: {status} (rel err = {rel_err:.2e})")
    if not result["ok"]:
        raise RuntimeError(
            "Component decomposition check failed: captured contributions do "
            "not sum to the final residual stream. See "
            "diagnostics/dla_decomposition_check.json."
        )
    return result


def run_with_component_ablation(
    bundle: ModelBundle,
    prompt: str,
    comp_anatomy: ComponentAnatomy,
    component_type: str,
    layer: int,
) -> Any:
    """Forward pass with one component's FINAL-POSITION output zeroed.

    This is *direct-path* ablation: it removes exactly the contribution DLA
    scored (the write to the final position's residual stream) and nothing
    else. Contributions at earlier positions, which can reach the output
    indirectly through later attention, are deliberately left intact so the
    causal effect is commensurable with the attribution score.
    Returns the model's final-position logits, float32 on CPU.
    """
    import torch

    hook_path = comp_anatomy.attn_hook_path if component_type == "attn" else comp_anatomy.mlp_hook_path
    module = getattr(bundle.blocks[layer], hook_path)

    def ablate_hook(mod: Any, hook_args: tuple, output: Any) -> Any:
        if isinstance(output, tuple):
            out = output[0].clone()
            out[0, -1] = 0
            return (out,) + tuple(output[1:])
        out = output.clone()
        out[0, -1] = 0
        return out

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)

    handle = module.register_forward_hook(ablate_hook)
    try:
        with torch.no_grad():
            out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return tensor_cpu_float(out.logits[0, -1])


# ---------------------------------------------------------------------------
# Human-readable state dumps
# ---------------------------------------------------------------------------
#
# The design rule: every dumped fact about model state must be inspectable
# with a text editor. Token tables are CSV with the whitespace made visible;
# activations are summarized as per-layer statistics and *decoded* through the
# lens rather than printed as raw arrays. Raw tensors are opt-in and always
# ship with a manifest explaining each entry.


def dump_example_state(
    ctx: RunContext,
    bundle: ModelBundle,
    example_id: str,
    capture: ForwardCapture,
    traj: LensTrajectory,
    *,
    target: str | None = None,
    distractor: str | None = None,
) -> pathlib.Path:
    """Write the per-example state directory. Returns its path."""
    state_dir = ctx.run_dir / "state" / sanitize_tag(example_id)

    # --- tokens.csv: exactly what the model saw, position by position.
    token_rows = []
    cumulative = ""
    for i, (tid, raw, text) in enumerate(zip(capture.input_ids, capture.tokens_raw, capture.tokens_text)):
        cumulative += text
        token_rows.append(
            {
                "position": i,
                "token_id": tid,
                "token_raw": raw,
                "token_text": text,
                "token_visible": visible_token(text),
                "cumulative_text_visible": visible_token(cumulative),
                "is_final": i == len(capture.input_ids) - 1,
            }
        )
    write_csv(state_dir / "tokens.csv", token_rows)

    # --- residual_norms_by_position.csv: depth x position L2-norm grid.
    # This is the "shape" of the forward pass: where the stream grows, which
    # positions carry unusually large state, whether the BOS column is an
    # outlier (it usually is -- worth seeing once).
    norms = capture.streams.norm(dim=-1)  # [L+1, seq]
    norm_rows = []
    for depth in range(norms.shape[0]):
        row: dict[str, Any] = {"depth": depth}
        for pos in range(norms.shape[1]):
            row[f"pos_{pos}"] = round(float(norms[depth, pos]), 3)
        norm_rows.append(row)
    write_csv(state_dir / "residual_norms_by_position.csv", norm_rows)

    # --- residual_stats_final_pos.csv: per-depth summary stats at the
    # position being read out. Numbers a human can sanity-check, not arrays.
    final_streams = capture.streams[:, -1, :]
    stat_rows = []
    for depth in range(final_streams.shape[0]):
        v = final_streams[depth]
        stat_rows.append(
            {
                "depth": depth,
                "l2_norm": round(float(v.norm()), 4),
                "rms": round(float(v.pow(2).mean().sqrt()), 5),
                "mean": round(float(v.mean()), 6),
                "std": round(float(v.std()), 5),
                "abs_max": round(float(v.abs().max()), 4),
            }
        )
    write_csv(state_dir / "residual_stats_final_pos.csv", stat_rows)

    # --- logit_lens_topk.csv: the decoded view of "what the stream says".
    write_csv(state_dir / "logit_lens_topk.csv", traj.topk_rows)

    # --- lens_trajectory.csv: one row per depth, the lab's working table.
    traj_rows = []
    for depth in range(traj.n_depths):
        row = {
            "depth": depth,
            "top1_token_id": traj.top1_ids[depth],
            "top1_token": traj.top1_texts[depth],
            "top1_prob": round(traj.top1_probs[depth], 6),
            "top2_token_id": traj.top2_ids[depth],
            "top2_token": traj.top2_texts[depth],
            "top2_prob": round(traj.top2_probs[depth], 6),
            "top1_margin": round(traj.top1_margin[depth], 6),
            "entropy_bits": round(traj.entropy_bits[depth], 4),
            "kl_to_final_bits": round(traj.kl_to_final_bits[depth], 4),
            "cosine_to_final": round(traj.cosine_to_final[depth], 5),
            "cosine_to_prev": "" if traj.cosine_to_prev[depth] is None else round(traj.cosine_to_prev[depth], 5),
            "resid_l2": round(traj.resid_l2[depth], 3),
            "stream_delta_l2": round(traj.stream_delta_l2[depth], 3),
        }
        if traj.p_target is not None:
            row["p_target"] = round(traj.p_target[depth], 6)
            row["logit_target"] = round(traj.logit_target[depth], 4)
            row["target_rank"] = traj.target_rank[depth] if traj.target_rank is not None else ""
        if traj.p_distractor is not None:
            row["p_distractor"] = round(traj.p_distractor[depth], 6)
            row["logit_distractor"] = round(traj.logit_distractor[depth], 4)
            row["distractor_rank"] = traj.distractor_rank[depth] if traj.distractor_rank is not None else ""
        if traj.p_target is not None and traj.p_distractor is not None:
            row["logit_diff_target_minus_distractor"] = round(
                traj.logit_target[depth] - traj.logit_distractor[depth], 4
            )
        traj_rows.append(row)
    write_csv(state_dir / "lens_trajectory.csv", traj_rows)

    # --- state_card.md: the narrative view, designed to be read top to
    # bottom by a human deciding whether the run makes sense.
    state_card_path = state_dir / "state_card.md"
    write_text(state_card_path, render_state_card(bundle, example_id, capture, traj, target, distractor))
    ctx.register_artifact(state_card_path, "state", f"Per-example state card for {example_id}.")

    # --- optional raw tensors, with a manifest so nothing is a mystery blob.
    if ctx.args.save_tensors:
        import torch

        tensor_path = state_dir / "residual_streams.pt"
        torch.save({"streams": capture.streams.cpu()}, tensor_path)
        write_json(
            state_dir / "tensor_manifest.json",
            {
                "residual_streams.pt": {
                    "key": "streams",
                    "shape": list(capture.streams.shape),
                    "dtype": "float32",
                    "semantics": (
                        "Pre-norm residual stream; index [k, p, :] is the "
                        "stream after k blocks at token position p. k=0 is "
                        "the embedding output."
                    ),
                }
            },
        )
    return state_dir


def render_state_card(
    bundle: ModelBundle,
    example_id: str,
    capture: ForwardCapture,
    traj: LensTrajectory,
    target: str | None,
    distractor: str | None,
) -> str:
    """Render the per-example markdown state card."""
    a = bundle.anatomy
    L = a.n_layers
    lines = [
        f"# State card: {example_id}",
        "",
        f"- model: `{a.model_id}` ({L} blocks, d_model {a.d_model})",
        f"- prompt: `{capture.prompt}`",
        f"- target: `{visible_token(target) if target else '(none)'}`"
        + (f" | distractor: `{visible_token(distractor)}`" if distractor else ""),
        f"- depth convention: 0 = embeddings, k = after k blocks, {L} = pre-final-norm",
        "- all readouts are at the final token position",
        "",
        "## Tokens as the model saw them",
        "",
        "| pos | id | token |",
        "|---:|---:|---|",
    ]
    for i, (tid, text) in enumerate(zip(capture.input_ids, capture.tokens_text)):
        lines.append(f"| {i} | {tid} | `{visible_token(text)}` |")

    header = "| depth | top-1 | p(top1) | margin | entropy | KL->final | cos->final | delta L2 |"
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|"
    if traj.p_target is not None:
        header += " p(target) |"
        sep += "---:|"
    if traj.p_target is not None and traj.p_distractor is not None:
        header += " logit diff (t-d) |"
        sep += "---:|"
    lines += ["", "## Layer-by-layer readout", "", header, sep]
    for depth in range(traj.n_depths):
        row = (
            f"| {depth} | `{visible_token(traj.top1_texts[depth])}` "
            f"| {traj.top1_probs[depth]:.3f} | {traj.top1_margin[depth]:.3f} "
            f"| {traj.entropy_bits[depth]:.2f} | {traj.kl_to_final_bits[depth]:.2f} "
            f"| {traj.cosine_to_final[depth]:.3f} | {traj.stream_delta_l2[depth]:.2f} |"
        )
        if traj.p_target is not None:
            row += f" {traj.p_target[depth]:.4f} |"
        if traj.p_target is not None and traj.p_distractor is not None:
            diff = traj.logit_target[depth] - traj.logit_distractor[depth]
            row += f" {diff:+.2f} |"
        lines.append(row)

    # Top-5 snapshots at a handful of depths: enough to see the prediction
    # form without printing every depth's full distribution.
    snapshot_depths = sorted({0, L // 4, L // 2, (3 * L) // 4, L - 1, L})
    lines += ["", "## Top-5 readout at selected depths", ""]
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for row in traj.topk_rows:
        by_depth.setdefault(row["depth"], []).append(row)
    for depth in snapshot_depths:
        entries = by_depth.get(depth, [])
        rendered = " | ".join(
            f"`{visible_token(e['token'])}` {e['prob']:.3f}" for e in entries
        )
        lines.append(f"- depth {depth}: {rendered}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "fact": "#1f77b4",
    "ambiguous": "#7f7f7f",
    "counterfactual": "#d62728",
    "control": "#9467bd",
}


def new_figure(figsize: tuple[float, float] = (8.0, 5.0)) -> tuple[Any, Any]:
    """Create a matplotlib figure with the bench's house style."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3)
    return fig, ax


def save_figure(ctx: RunContext, fig: Any, name: str, description: str) -> None:
    """Save a plot under plots/ and register it in the artifact index."""
    import matplotlib.pyplot as plt

    path = ctx.path("plots", name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ctx.register_artifact(path, "plot", description)
    print(f"[bench] wrote plots/{name}")


def close_figure(fig: Any) -> None:
    """Release a figure that is being abandoned instead of saved."""
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# Claim ledger
# ---------------------------------------------------------------------------

LEDGER_HEADER = """# Claim ledger

A running dossier of claims about one model. Every entry carries an evidence
tag (OBS | ATTR | DECODE | CAUSAL | SELF-REPORT), the artifact that backs it,
and the observation that would kill it. Labs draft suggested claims into
their run directory; what lands here is the student's own judgment.

Format:

    [L01-C1] OBS | <claim text with numbers and scope>
    Artifact: runs/<run>/<file> | Falsifier: <what would kill this claim>

---
"""


def ensure_ledger() -> None:
    """Create the course-root claim ledger with its header if missing."""
    if not LEDGER_PATH.exists():
        write_text(LEDGER_PATH, LEDGER_HEADER)
        print(f"[bench] initialized claim ledger at {LEDGER_PATH}")


def format_claim(claim: Mapping[str, str]) -> str:
    """Render one claim in the ledger's two-line format."""
    return (
        f"[{claim['id']}] {claim['tag']} | {claim['text']}\n"
        f"Artifact: {claim['artifact']} | Falsifier: {claim['falsifier']}\n"
    )


def write_ledger_suggestions(ctx: RunContext, lab_id: str, claims: Sequence[Mapping[str, str]]) -> None:
    """Write drafted claims to the run dir; append to the real ledger only on
    explicit request, because writing the claim is the student's job."""
    body = [
        f"# Suggested ledger claims from {lab_id} ({ctx.run_dir.name})",
        "",
        "These are drafts with measured numbers filled in. Edit them until you",
        "would defend them, then move them into claim_ledger.md (or re-run with",
        "--append-ledger to copy them verbatim, and edit there).",
        "",
        "```text",
    ]
    body += [format_claim(c) for c in claims]
    body += ["```", ""]
    path = ctx.path("ledger_suggestions.md")
    write_text(path, "\n".join(body))
    ctx.register_artifact(path, "summary", "Drafted claim-ledger entries with measured numbers.")

    if ctx.args.append_ledger:
        ensure_ledger()
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(f"\n<!-- appended by {ctx.run_dir.name} -->\n")
            for claim in claims:
                f.write(format_claim(claim) + "\n")
        print(f"[bench] appended {len(claims)} claims to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab bench for the mechanistic interpretability course.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lab", choices=sorted(LAB_PROFILES), default="lab1", help="Which lab to run.")
    parser.add_argument("--model", default=None, help="HF model id; defaults to the tier's model.")
    parser.add_argument("--model-revision", default=None, help="Pinned HF revision (commit/tag).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download models/tokenizers; use local cache only.")
    parser.add_argument("--attn-implementation", default="auto", help="Optional HF attention implementation, e.g. eager, sdpa, flash_attention_2.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Pass low_cpu_mem_usage=True during model loading.")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--quantization", default="none", choices=("none", "8bit", "4bit"),
                        help="Optional bitsandbytes quantization for small GPUs.")
    parser.add_argument("--tier", default="auto", choices=("auto", "a", "b", "c"),
                        help="Hardware tier: a = CPU smoke (gpt2), b = 24GB+/Colab GPU, c = 40-80GB.")
    parser.add_argument("--prompt-set", default="small",
                        help="small | medium | full | path to a custom prompts .json file.")
    parser.add_argument("--max-examples", type=int, default=-1,
                        help="Cap examples; -1 = tier default, 0 = no cap.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k tokens recorded per depth.")
    parser.add_argument("--include-controls", action="store_true", help="Lab 1: include optional weak/scrambled control prompts.")
    parser.add_argument("--ablate-top", type=int, default=3,
                        help="Lab 2: per example, ablate this many top-|attribution| components "
                             "plus matched controls to compare attribution vs causal effect (0 = skip).")
    parser.add_argument("--dla-tolerance", type=float, default=0.02,
                        help="Relative tolerance for the component decomposition self-check "
                             "(absorbs the compute dtype's residual-add rounding; bf16 needs ~0.02).")
    parser.add_argument("--hook-tolerance", type=float, default=0.0, help="Allowed max absolute diff in hook parity diagnostics.")
    parser.add_argument("--allow-hook-mismatch", action="store_true", help="Warn instead of aborting on hook parity mismatch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--showcase", default=None,
                        help="Example id to feature in the biography plot (default: first counterfactual example).")
    parser.add_argument("--save-tensors", action="store_true",
                        help="Also save raw residual tensors (with a manifest) per example.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    parser.add_argument("--append-ledger", action="store_true",
                        help="Append suggested claims to claim_ledger.md (default: drafts only).")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Parent directory for run dirs.")
    parser.add_argument("--run-name", default=None, help="Run directory name (auto-suffixed if taken).")
    parser.add_argument("--run-dir", default=None, help="Exact run directory (overrides root/name).")
    return parser.parse_args(argv)


def apply_tier_defaults(args: argparse.Namespace) -> None:
    """Resolve tier 'auto' and fill model/dtype/max-examples defaults.

    The tier shortcuts are convenience presets, not hidden modes: after this
    function runs the rest of the bench sees ordinary --model/--dtype values,
    and every resolved value lands in run_config.json.
    """
    if args.tier == "auto":
        # Tier choice needs to know whether a GPU exists, but torch is not
        # imported yet (env config must happen first), so probe lazily later?
        # No: tier only affects defaults, and the only signal needed is CUDA
        # availability. Import of torch here is acceptable because tier=auto
        # implies no env-sensitive flags were requested.
        try:
            import torch

            args.tier = "b" if torch.cuda.is_available() else "a"
        except ImportError:
            args.tier = "a"
        print(f"[bench] tier auto-resolved to '{args.tier}'")
    spec = TIER_DEFAULTS[args.tier]
    if args.model is None:
        args.model = spec["model"]
    if args.dtype == "auto":
        args.dtype = spec["dtype"]
    if args.max_examples < 0:
        args.max_examples = spec["max_examples"]


def make_run_dir(args: argparse.Namespace) -> pathlib.Path:
    """Resolve the run directory, avoiding accidental overwrite by default."""
    if args.run_dir:
        return pathlib.Path(args.run_dir).expanduser().resolve()
    run_root = pathlib.Path(args.run_root).expanduser().resolve()
    profile = LAB_PROFILES[args.lab]
    run_name = args.run_name or f"{profile['run_name']}-{now_slug()}-{uuid.uuid4().hex[:6]}"
    base = run_root / sanitize_tag(run_name)
    if not base.exists():
        return base
    for i in range(2, 10_000):
        candidate = base.with_name(f"{base.name}_{i:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find an unused run directory for {base}")


def configure_env(run_dir: pathlib.Path) -> None:
    """Set environment that should exist before torch/transformers import."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")  # plots are files, not windows
    mpl_config = run_dir / "matplotlib_config"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_tier_defaults(args)
    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_env(run_dir)

    with ConsoleTee(run_dir / "logs"):
        print(f"[bench] run directory: {run_dir}")
        ctx = RunContext(run_dir=run_dir, args=args)
        run_config_path = ctx.path("run_config.json")
        write_json(run_config_path, vars(args))
        ctx.register_artifact(run_config_path, "config", "Resolved CLI arguments after tier defaults.")

        # Heavy imports happen after env config so MPLBACKEND etc. apply.
        import torch

        set_determinism(torch, args.seed)
        metadata_path = ctx.path("run_metadata.json")
        write_json(metadata_path, collect_run_metadata(torch))
        ctx.register_artifact(metadata_path, "diagnostic", "Host, package, git, GPU, and environment metadata.")

        bundle = load_model_and_tokenizer(ctx)
        ensure_ledger()

        # Labs import this module by name. When this file runs as a script
        # the module is '__main__', so alias it to keep one shared instance.
        sys.modules.setdefault("interp_bench", sys.modules[__name__])
        if str(COURSE_ROOT) not in sys.path:
            sys.path.insert(0, str(COURSE_ROOT))
        lab_module = importlib.import_module(LAB_PROFILES[args.lab]["module"])

        print(f"[bench] running {args.lab}: {LAB_PROFILES[args.lab]['description']}")
        t0 = time.perf_counter()
        try:
            lab_module.run(ctx, bundle)
        finally:
            # A failed run still leaves partial artifacts; index whatever
            # was registered so the run directory stays auditable.
            write_json(ctx.path("artifact_index.json"), {"artifacts": ctx.artifacts})
        elapsed = time.perf_counter() - t0

        end_mem_path = ctx.path("diagnostics", "gpu_memory_at_end.json")
        write_json(end_mem_path, gpu_memory_snapshot(torch, "end"))
        ctx.register_artifact(end_mem_path, "diagnostic", "GPU memory snapshot at the end of the run.")
        write_json(ctx.path("artifact_index.json"), {"artifacts": ctx.artifacts})
        print(f"[bench] lab finished in {elapsed:.1f}s")
        print(f"[bench] start with: {run_dir / 'run_summary.md'}")
        print(f"[bench] artifact map: {run_dir / 'artifact_index.json'}")


if __name__ == "__main__":
    main()
