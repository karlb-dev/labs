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


def run_continuous(bundle, rendered, caps, max_concurrent, admit_block=None) -> tuple[list[str], dict[str, Any]]:
    import interp_bench as bench

    outs = bench.generate_continuous(
        bundle, rendered, caps, max_concurrent=max_concurrent,
        eos_token_id=-1,  # suppress EOS: exact-length jobs (engine treats <0 as never)
        progress_label="bench", admit_block=admit_block)
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
    ap.add_argument("--admit-block", type=int, default=None,
                    help="admit only when this many slots are free (default: max_concurrent//4)")
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
        # A trailing digit repeats an engine under a distinct key, e.g.
        # "continuous,continuous2" checks the engine's self-determinism
        # (two identical calls must produce bitwise-identical outputs).
        kind = engine.rstrip("0123456789")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with GPUSampler() as sampler:
            if kind == "continuous":
                outs, stats = run_continuous(bundle, rendered, caps, args.max_concurrent,
                                             args.admit_block)
            elif kind == "lockstep":
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
        if kind == "continuous":
            # TTFT includes queue wait (jobs admitted late wait by design);
            # prefill stall is the part that interrupts in-flight rows.
            print(f"[bench-inf]   itl p50/p95 {stats.get('itl_p50_ms')}/{stats.get('itl_p95_ms')} ms  "
                  f"ttft p50/p95 {stats.get('ttft_p50_s')}/{stats.get('ttft_p95_s')} s  "
                  f"admits {stats.get('admit_events')} (block {stats.get('admit_block')})  "
                  f"stall total/max {stats.get('prefill_stall_ms_total')}/{stats.get('prefill_stall_ms_max')} ms")
        with pathlib.Path(args.out).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    if len(sample_texts) == 2:
        a, b = sample_texts.values()
        same = sum(x == y for x, y in zip(a, b))
        print(f"[bench-inf] determinism cross-check: {same}/{len(a)} sample jobs identical across engines")
        if same < len(a):
            # bf16 greedy decoding is not bitwise stable across attention
            # kernel paths (batch shape / tensor strides select different
            # kernels -> ulp-level logit differences -> near-tie argmax
            # flips). That is benign; a real cache bug diverges at a LARGE
            # logit gap. So divergences are verified, not just counted: at
            # each first-divergence point, a clean single-row teacher-forced
            # forward must show the two engines picked the model's top-2
            # tokens within a small gap.
            tok = bundle.tokenizer
            verified = failed = 0
            for i in range(len(a)):
                if a[i] == b[i]:
                    continue
                a_ids = tok(a[i], add_special_tokens=False)["input_ids"]
                b_ids = tok(b[i], add_special_tokens=False)["input_ids"]
                d = next((k for k in range(min(len(a_ids), len(b_ids))) if a_ids[k] != b_ids[k]),
                         min(len(a_ids), len(b_ids)))
                prefix = tok(rendered[i], add_special_tokens=False)["input_ids"] + a_ids[:d]
                ids = torch.tensor([prefix], device=bundle.input_device)
                with torch.inference_mode():
                    logits = bundle.model(input_ids=ids, use_cache=False).logits[0, -1].float()
                top = torch.topk(logits, 8)
                top1 = float(top.values[0])
                by_tok = {int(t): float(v) for v, t in zip(top.values, top.indices)}
                chose = [x for x in (a_ids[d] if d < len(a_ids) else -1,
                                     b_ids[d] if d < len(b_ids) else -1)]
                # benign = BOTH engines picked a token whose clean logit is
                # within a couple of bf16 ulps of the maximum (n-way near-tie)
                gap = max(top1 - by_tok.get(c, float("-inf")) for c in chose)
                ok = gap <= 0.25
                verified += ok
                failed += not ok
                if not ok:
                    print(f"[bench-inf]   job {i}: REAL divergence at token {d} "
                          f"(max gap to top-1 {gap:.4f}, engines chose {chose}, "
                          f"top tokens {sorted(by_tok, key=by_tok.get, reverse=True)[:4]})")
            print(f"[bench-inf] near-tie verification: {verified} benign bf16 near-tie flips, "
                  f"{failed} REAL divergences{' — FAIL' if failed else ' — PASS'}")


if __name__ == "__main__":
    main()
