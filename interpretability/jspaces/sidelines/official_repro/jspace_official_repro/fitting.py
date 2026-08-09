"""Hardened OLMo fit producer (OLMO_FIT_CONTRACT).

Wraps upstream ``jlens.fitting.jacobian_for_prompt`` without touching the
estimator: identical accumulation order (running fp32 sum, mean at the
end), identical skip semantics (``next_idx`` vs ``n_done``). Adds atomic
local checkpoints every 3 accepted prompts, Drive recovery copies every
15 with two-deep rotation, milestone saves, per-prompt diagnostics, and a
runtime-sentinel resume gate. Checkpoints keep the upstream key layout so
``jlens.fit`` could resume from them unchanged.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path

import torch

from jlens.fitting import jacobian_for_prompt
from jlens.lens import JacobianLens

from .manifests import file_sha256, json_sha256

LOCAL_CKPT_EVERY = 3
DRIVE_CKPT_EVERY = 15
DRIVE_KEEP = 2
SENTINEL_MAX_REL_DIFF = 0.5  # campaign's prospective runtime-identity contract


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def runtime_sentinel(
    model, prompt: str, source_layers: list[int], *, target_layer: int,
    dim_batch: int, max_seq_len: int = 128, skip_first: int = 16,
) -> dict:
    """Per-layer Jacobian Frobenius norms on one excluded prompt."""
    start = time.perf_counter()
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, target_layer=target_layer,
        dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
    )
    norms = {str(layer): float(J.norm()) for layer, J in jacobians.items()}
    return {
        "norms": norms,
        "seq_len": seq_len,
        "n_valid": n_valid,
        "wall_seconds": time.perf_counter() - start,
        "norms_sha256": json_sha256(norms),
    }


def sentinel_rel_diff(a: dict, b: dict) -> float:
    return max(
        abs(a["norms"][k] - b["norms"][k]) / max(abs(a["norms"][k]), 1e-30)
        for k in a["norms"]
    )


class HalfFit:
    """One half-fit (A or B) with checkpointing and diagnostics."""

    def __init__(
        self,
        *,
        half: str,
        prompts: list[dict],  # rows from the fit manifest (text + fit_index)
        source_layers: list[int],
        target_layer: int,
        dim_batch: int,
        local_dir: Path,
        drive_dir: Path | None,
        sentinel: dict,
        max_seq_len: int = 128,
        skip_first: int = 16,
        milestones: tuple[int, ...] = (125, 250, 500),
    ) -> None:
        self.half = half
        self.prompts = prompts
        self.source_layers = sorted(source_layers)
        self.target_layer = target_layer
        self.dim_batch = dim_batch
        self.max_seq_len = max_seq_len
        self.skip_first = skip_first
        self.milestones = milestones
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.drive_dir = Path(drive_dir) if drive_dir else None
        if self.drive_dir:
            self.drive_dir.mkdir(parents=True, exist_ok=True)
        self.sentinel = sentinel
        self.ckpt_path = self.local_dir / f"fit_half_{half}.ckpt"
        self.header_path = self.local_dir / f"fit_half_{half}.header.json"
        self.diag_path = self.local_dir / f"fit_half_{half}.diagnostics.jsonl"

    # ------------------------------------------------------------ state

    def _header(self, n_done: int, next_idx: int, ckpt_sha: str) -> dict:
        return {
            "half": self.half,
            "n_done": n_done,
            "next_idx": next_idx,
            "source_layers": self.source_layers,
            "target_layer": self.target_layer,
            "dim_batch": self.dim_batch,
            "max_seq_len": self.max_seq_len,
            "skip_first": self.skip_first,
            "sentinel_norms_sha256": self.sentinel["norms_sha256"],
            "ckpt_sha256": ckpt_sha,
            "n_prompts_total": len(self.prompts),
        }

    def _write_checkpoint(self, jacobian_sum, n_done, next_idx) -> None:
        _atomic_save(
            {
                "jacobian_sum": jacobian_sum,
                "n_done": n_done,
                "next_idx": next_idx,
                "source_layers": self.source_layers,
                "target_layer": self.target_layer,
                "skip_first": self.skip_first,
            },
            self.ckpt_path,
        )
        header = self._header(n_done, next_idx, file_sha256(self.ckpt_path))
        tmp = self.header_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(header, indent=2))
        os.replace(tmp, self.header_path)

    def _publish_drive(self, n_done: int) -> None:
        if not self.drive_dir:
            return
        stamp = f"{self.half}_n{n_done:04d}"
        ckpt_copy = self.drive_dir / f"recovery_{stamp}.ckpt"
        shutil.copy2(self.ckpt_path, ckpt_copy)
        if file_sha256(ckpt_copy) != file_sha256(self.ckpt_path):
            raise RuntimeError("Drive recovery copy hash mismatch")
        shutil.copy2(self.header_path, self.drive_dir / f"recovery_{stamp}.header.json")
        shutil.copy2(self.diag_path, self.drive_dir / f"diagnostics_{self.half}.jsonl")
        # Rotation: keep newest DRIVE_KEEP recovery ckpts for this half.
        recoveries = sorted(self.drive_dir.glob(f"recovery_{self.half}_n*.ckpt"))
        for old in recoveries[:-DRIVE_KEEP]:
            old.unlink()
            header = old.with_suffix("").with_suffix(".header.json")
            if header.exists():
                header.unlink()

    def _resume_state(self):
        if not self.ckpt_path.exists():
            return None
        header = json.loads(self.header_path.read_text())
        for key in ("source_layers", "target_layer", "dim_batch",
                    "max_seq_len", "skip_first"):
            if header[key] != getattr(self, key):
                raise RuntimeError(f"resume forbidden: header {key} differs")
        if header["sentinel_norms_sha256"] != self.sentinel["norms_sha256"]:
            raise RuntimeError(
                "resume forbidden: runtime sentinel differs from fit header"
            )
        if file_sha256(self.ckpt_path) != header["ckpt_sha256"]:
            raise RuntimeError("resume forbidden: checkpoint hash != header")
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        return state

    # -------------------------------------------------------------- run

    def run(self, model, *, stop_after: int | None = None) -> dict:
        """Fit until the prompt list (or ``stop_after`` accepted prompts) is
        exhausted. Returns summary; lens saved via :meth:`save_lens`."""
        sqrt_d = math.sqrt(model.d_model)
        state = self._resume_state()
        if state is not None:
            jacobian_sum = state["jacobian_sum"]
            n_done, next_idx = state["n_done"], state["next_idx"]
        else:
            jacobian_sum = {
                layer: torch.zeros(model.d_model, model.d_model,
                                   dtype=torch.float32)
                for layer in self.source_layers
            }
            n_done, next_idx = 0, 0
        diag = self.diag_path.open("a")
        target = stop_after if stop_after is not None else len(self.prompts)
        skipped = []
        while next_idx < len(self.prompts) and n_done < target:
            row = self.prompts[next_idx]
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            try:
                per_prompt, seq_len, n_valid = jacobian_for_prompt(
                    model, row["text"], self.source_layers,
                    target_layer=self.target_layer, dim_batch=self.dim_batch,
                    max_seq_len=self.max_seq_len, skip_first=self.skip_first,
                )
            except ValueError as exc:
                skipped.append({"fit_index": row["fit_index"], "why": str(exc)})
                diag.write(json.dumps({"fit_index": row["fit_index"],
                                       "skipped": str(exc)}) + "\n")
                next_idx += 1
                continue
            prompt_norm = max(
                per_prompt[l].norm().item() for l in self.source_layers
            ) / sqrt_d
            if n_done > 0:
                rel_move = max(
                    ((per_prompt[l] - jacobian_sum[l] / n_done).norm()
                     / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())).item()
                    for l in self.source_layers
                )
            else:
                rel_move = float("nan")
            finite = all(torch.isfinite(per_prompt[l]).all()
                         for l in self.source_layers)
            if not finite:
                raise RuntimeError(
                    f"non-finite Jacobian at fit_index {row['fit_index']}"
                )
            for layer in self.source_layers:
                jacobian_sum[layer] += per_prompt[layer]
            n_done += 1
            next_idx += 1
            diag.write(json.dumps({
                "fit_index": row["fit_index"], "half": self.half,
                "seq_len": seq_len, "n_valid": n_valid,
                "prompt_norm_over_sqrt_d": prompt_norm,
                "running_mean_rel_move": rel_move,
                "wall_seconds": time.perf_counter() - start,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "n_done": n_done,
            }) + "\n")
            diag.flush()
            if n_done % LOCAL_CKPT_EVERY == 0:
                self._write_checkpoint(jacobian_sum, n_done, next_idx)
            if n_done % DRIVE_CKPT_EVERY == 0:
                self._publish_drive(n_done)
            if n_done in self.milestones:
                self.save_lens(jacobian_sum, n_done,
                               self.local_dir / f"milestone_{self.half}_n{n_done}.pt")
        self._write_checkpoint(jacobian_sum, n_done, next_idx)
        self._publish_drive(n_done)
        diag.close()
        return {"half": self.half, "n_done": n_done, "next_idx": next_idx,
                "skipped": skipped}

    def save_lens(self, jacobian_sum, n_done: int, path: Path) -> JacobianLens:
        mean = {l: jacobian_sum[l] / n_done for l in self.source_layers}
        lens = JacobianLens(jacobians=mean, n_prompts=n_done,
                            d_model=next(iter(mean.values())).shape[0])
        lens.save(str(path))
        return lens

    def final_lens(self) -> JacobianLens:
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        mean = {
            l: state["jacobian_sum"][l] / state["n_done"]
            for l in self.source_layers
        }
        return JacobianLens(jacobians=mean, n_prompts=state["n_done"],
                            d_model=next(iter(mean.values())).shape[0])
