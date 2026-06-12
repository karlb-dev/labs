#!/usr/bin/env python3
"""Micro-benchmark for the bench's generation engines.

Measures the continuous-batching engine against lockstep ``model.generate``
on a synthetic, *reproducible* workload shaped like Lab 10's: many greedy
jobs with heavy-tailed output lengths. Per-job length caps are taken from a
fixed schedule and EOS is suppressed, so every engine generates EXACTLY the
same number of tokens — wall-clock and memory become comparable, and the
generated token ids themselves must MATCH between engines (greedy + same
prompts = same tokens), which the harness asserts on a sample.

Usage:
  python bench_inference.py --model allenai/Olmo-3-7B-Think --jobs 48 \
      --max-concurrent 16 --engines continuous,lockstep
  python bench_inference.py --model allenai/Olmo-3-32B-Think --jobs 24 \
      --max-concurrent 8 --scale 0.5 --engines continuous

Reports per engine: wall s, generated tokens, tok/s, peak CUDA GiB, and
(for continuous) the engine's own telemetry. Writes a JSON row per engine to
--out for tracking optimization progress across commits.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

# Heavy-tailed length schedule (fractions of --max-new), shaped like measured
# Olmo-Think CoT lengths: most jobs mid-length, a fat tail of cap-hitters.
LENGTH_FRACS = [0.15, 0.3, 0.3, 0.4, 0.45, 0.5, 0.5, 0.55, 0.6, 0.6,
                0.7, 0.75, 0.8, 0.9, 1.0, 1.0]

PROMPT_POOL = [
    "Explain why the sky is blue, step by step.",
    "What is 17 * 23? Think it through.",
    "Name three rivers in Europe and describe each briefly.",
    "Which is heavier, a kilogram of feathers or of iron? Reason carefully.",
    "Summarize the plot of a famous tragedy in your own words.",
    "How does a refrigerator keep food cold? Walk through the cycle.",
    "Compare two sorting algorithms and their tradeoffs.",
    "Why do seasons exist? Explain with the Earth's tilt.",
]


def build_jobs(bundle: Any, n_jobs: int, max_new: int, scale: float) -> tuple[list[str], list[int]]:
    import interp_bench as bench

    rendered, caps = [], []
    for i in range(n_jobs):
        user = PROMPT_POOL[i % len(PROMPT_POOL)] + f" (variant {i})"
        if bench.supports_chat_template(bundle):
            text = bundle.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        else:
            text = user
        rendered.append(text)
        frac = LENGTH_FRACS[i % len(LENGTH_FRACS)]
        caps.append(max(8, int(max_new * frac * scale)))
    return rendered, caps


class GPUSampler:
    """Background NVML/CUDA sampler (lifted from olmo_lora_bench's
    CUDAMetricSampler, slimmed): mean/max GPU utilization and reserved memory
    while an engine runs — the 'is the GPU actually fed' number."""

    def __init__(self, interval_sec: float = 0.5):
        import threading

        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._interval = interval_sec
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._handle = None

    def _loop(self):
        import torch

        while not self._stop.wait(self._interval):
            rec: dict[str, Any] = {}
            if torch.cuda.is_available():
                rec["alloc_gib"] = torch.cuda.memory_allocated() / 1024**3
                rec["reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
            if self._nvml is not None:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    rec["gpu_util_pct"] = int(util.gpu)
                except Exception:
                    pass
            self.samples.append(rec)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        utils = [s["gpu_util_pct"] for s in self.samples if "gpu_util_pct" in s]
        res = [s["reserved_gib"] for s in self.samples if "reserved_gib" in s]
        out: dict[str, Any] = {"n_samples": len(self.samples)}
        if utils:
            out["gpu_util_mean_pct"] = round(sum(utils) / len(utils), 1)
            out["gpu_util_max_pct"] = max(utils)
        if res:
            out["reserved_max_gib"] = round(max(res), 2)
        return out


def run_continuous(bundle, rendered, caps, max_concurrent) -> tuple[list[str], dict[str, Any]]:
    import interp_bench as bench

    outs = bench.generate_continuous(
        bundle, rendered, caps, max_concurrent=max_concurrent,
        eos_token_id=-1,  # suppress EOS: exact-length jobs (engine treats <0 as never)
        progress_label="bench")
    return outs, dict(bench.LAST_GENERATION_STATS)


def run_lockstep(bundle, rendered, caps, batch_size) -> list[str]:
    """Lockstep reference: chunks of batch_size, each chunk runs to its own
    max cap (the per-batch straggler cost this engine exists to remove)."""
    import torch

    tok = bundle.tokenizer
    outs = []
    pad_id = tok.pad_token_id or tok.eos_token_id
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start:start + batch_size]
        chunk_caps = caps[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False)
        ids = enc["input_ids"].to(bundle.input_device)
        mask = enc["attention_mask"].to(bundle.input_device)
        with torch.no_grad():
            # min_new_tokens masks EOS so every row runs to the chunk cap —
            # the straggler cost lockstep actually pays on heavy-tailed jobs.
            out = bundle.model.generate(input_ids=ids, attention_mask=mask,
                                        max_new_tokens=max(chunk_caps),
                                        min_new_tokens=max(chunk_caps), do_sample=False,
                                        num_beams=1, pad_token_id=pad_id)
        for r, cap in enumerate(chunk_caps):
            new_ids = out[r, ids.shape[1]:ids.shape[1] + cap]
            outs.append(tok.decode(new_ids, skip_special_tokens=False))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Think")
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink all length caps")
    ap.add_argument("--max-concurrent", type=int, default=16)
    ap.add_argument("--engines", default="continuous,lockstep")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--verify-sample", type=int, default=4,
                    help="jobs to cross-check for identical tokens across engines")
    ap.add_argument("--out", default="runs/bench_inference_log.jsonl")
    args = ap.parse_args()

    import torch

    import interp_bench as bench

    # minimal bundle via the bench loader (reuses anatomy/device plumbing)
    class _Args:
        model = args.model
        model_revision = None
        trust_remote_code = False
        local_files_only = False
        attn_implementation = "auto"
        low_cpu_mem_usage = True
        quantization = "none"
        device = "auto"
        dtype = args.dtype
        lab = "lab10"
        tier = "b"

    run_dir = pathlib.Path("runs") / "bench_inference_scratch"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = bench.RunContext(run_dir=run_dir, args=_Args())
    bundle = bench.load_model_and_tokenizer(ctx)
    rendered, caps = build_jobs(bundle, args.jobs, args.max_new, args.scale)
    total_planned = sum(caps)
    print(f"[bench-inf] {args.jobs} jobs, planned tokens {total_planned}, "
          f"caps {min(caps)}..{max(caps)}, max_concurrent {args.max_concurrent}")

    results: dict[str, dict[str, Any]] = {}
    sample_texts: dict[str, list[str]] = {}
    for engine in args.engines.split(","):
        engine = engine.strip()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with GPUSampler() as sampler:
            if engine == "continuous":
                outs, stats = run_continuous(bundle, rendered, caps, args.max_concurrent)
            elif engine == "lockstep":
                outs = run_lockstep(bundle, rendered, caps, args.max_concurrent)
                stats = {}
            else:
                raise SystemExit(f"unknown engine {engine!r}")
        wall = time.perf_counter() - t0
        stats = {**stats, **sampler.summary()}
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        n_tokens = sum(len(bundle.tokenizer(o, add_special_tokens=False)["input_ids"]) for o in outs)
        row = {"engine": engine, "model": args.model, "jobs": args.jobs,
               "max_concurrent": args.max_concurrent, "planned_tokens": total_planned,
               "decoded_tokens": n_tokens, "wall_s": round(wall, 1),
               "tok_per_s": round(n_tokens / wall, 1), "peak_cuda_gib": round(peak, 2),
               "engine_stats": stats}
        results[engine] = row
        sample_texts[engine] = outs[: args.verify_sample]
        print(f"[bench-inf] {engine:11s} wall {wall:7.1f}s  {row['tok_per_s']:7.1f} tok/s  "
              f"peak {peak:5.1f} GiB")
        with pathlib.Path(args.out).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    if len(sample_texts) == 2:
        a, b = sample_texts.values()
        same = sum(x == y for x, y in zip(a, b))
        print(f"[bench-inf] determinism cross-check: {same}/{len(a)} sample jobs identical "
              f"across engines{' — MISMATCH' if same < len(a) else ''}")


if __name__ == "__main__":
    main()
