# One typed ScoringSpec + tokenizer wrapper for EVERYTHING that scores
# (nextsteps §2.7): capability scoring, intervention scoring, generation
# grading, audits. Freezes the Amendment-1 conventions:
#
#   * BOS: assay-wide BOS units. jlens.from_hf(force_bos=True) mutates
#     the shared tokenizer, so every scorer must produce ids in the SAME
#     units whether or not jlens has been constructed yet.
#   * Piecewise concatenation: UN-rstripped prompt ids + alias ids
#     tokenized separately with add_special_tokens=False, concatenated.
#   * Whitespace: prompts must end at the exact scoring boundary; a
#     trailing-whitespace prompt is a BANK DEFECT in Phase 3 (rejected at
#     authoring, not silently patched at scoring).
#   * Alias sets are frozen, and prefix-overlapping alias pairs (one
#     token sequence a prefix of the other) are rejected or classified
#     separately: under logsumexp aggregation they double-count the
#     shared prefix event.
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ScoringSpec:
    force_bos: bool = True
    max_prompt_tokens: int = 512
    max_answer_tokens: int = 24
    reject_trailing_whitespace: bool = True
    normalization: str = "lower_alnum_space"     # generation grading

    def normalize(self, s: str) -> str:
        if self.normalization != "lower_alnum_space":
            raise ValueError(f"unknown normalization {self.normalization!r}")
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


DEFAULT_SPEC = ScoringSpec()

# For re-measuring FROZEN Phase 2 items only. Amendment 1 scored the
# un-rstripped prompt, and 5/325 bank items carry a trailing-space
# artifact; an audit of those items must reproduce their tokenization
# rather than reject it. New Phase 3 banks use DEFAULT_SPEC, which
# refuses the artifact at authoring time. Any producer using this spec
# must record how many items relied on it.
LEGACY_PHASE2_SPEC = ScoringSpec(reject_trailing_whitespace=False)


class ScoringSession:
    """Wraps ONE tokenizer with ONE spec; every id this session produces
    is in the same units. Constructing it asserts the BOS convention
    instead of trusting whoever touched the tokenizer last."""

    def __init__(self, tok, spec: ScoringSpec = DEFAULT_SPEC,
                 device: str | torch.device = "cpu"):
        self.tok = tok
        self.spec = spec
        self.device = device
        if spec.force_bos:
            if tok.bos_token_id is None:
                raise ValueError("spec demands BOS units but the tokenizer "
                                 "has no bos_token_id")
            probe = tok("probe", return_tensors="pt").input_ids[0]
            if int(probe[0]) != int(tok.bos_token_id):
                # match the jlens.from_hf(force_bos=True) mutation
                tok.add_bos_token = True
                probe = tok("probe", return_tensors="pt").input_ids[0]
                if int(probe[0]) != int(tok.bos_token_id):
                    raise ValueError("could not establish BOS-prefixed "
                                     "tokenization on this tokenizer")

    # ------------------------------------------------------------ encode
    def prompt_ids(self, prompt: str) -> torch.Tensor:
        if self.spec.reject_trailing_whitespace and prompt != prompt.rstrip():
            raise ValueError("prompt has trailing whitespace — Phase 3 "
                             "banks must end at the scoring boundary")
        ids = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.spec.max_prompt_tokens).input_ids
        return ids.to(self.device)

    def answer_ids(self, alias: str) -> torch.Tensor:
        ids = self.tok(alias, add_special_tokens=False, return_tensors="pt",
                       truncation=True,
                       max_length=self.spec.max_answer_tokens).input_ids
        if ids.shape[1] == 0:
            raise ValueError(f"alias {alias!r} tokenizes to nothing")
        return ids.to(self.device)

    def full_ids(self, prompt: str, alias: str) -> tuple[torch.Tensor, int]:
        p = self.prompt_ids(prompt)
        a = self.answer_ids(alias)
        return torch.cat([p, a], dim=1), p.shape[1]

    # ------------------------------------------------------------- score
    @staticmethod
    def answer_seq_lp(full_ids: torch.Tensor, logits: torch.Tensor,
                      n_prompt: int) -> float:
        """Sum logprob of the answer tokens (teacher-forced, full
        sequence) — the frozen primary endpoint's per-alias input."""
        lps = torch.log_softmax(logits[:-1].float(), dim=-1)
        tgt = full_ids[0, 1:].cpu()
        rows = torch.arange(n_prompt - 1, full_ids.shape[1] - 1)
        return float(lps[rows, tgt[rows]].sum())

    # -------------------------------------------------------------- audit
    def alias_audit(self, aliases: list[str]) -> dict:
        """Tokenize every alias; flag prefix-overlapping pairs (§2.7)."""
        seqs = {a: self.answer_ids(a)[0].tolist() for a in aliases}
        overlaps = []
        items = list(seqs.items())
        for i, (a1, s1) in enumerate(items):
            for a2, s2 in items[i + 1:]:
                shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
                if longer[:len(shorter)] == shorter:
                    overlaps.append([a1, a2])
        return {"token_ids": seqs, "prefix_overlaps": overlaps,
                "ok": not overlaps}

    # --------------------------------------------------------- generation
    def grade_generation(self, generated: str, accepted: list[str]) -> dict:
        """Deterministic, LLM-free grading (primary endpoints)."""
        gnorm = self.spec.normalize(generated)
        for a in accepted:
            anorm = self.spec.normalize(a)
            if anorm and (gnorm == anorm or gnorm.startswith(anorm + " ")
                          or gnorm.startswith(anorm)):
                return {"correct": True, "matched_alias": a}
        return {"correct": False, "matched_alias": None}


@dataclass
class ScoredArm:
    """Per (item, alias, condition) result row fields shared by every
    Phase 3 grid (the §6.9 schema's scoring core)."""
    lp_by_alias: dict = field(default_factory=dict)

    def aggregates(self) -> dict:
        import numpy as np
        lps = np.array(list(self.lp_by_alias.values()))
        return {"lp_logsumexp": float(np.logaddexp.reduce(lps)),
                "lp_max": float(lps.max())}
