"""Single typed scoring and generation-grading contract for Phase 4."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ScoringSpec:
    version: str = "p4-scoring-v1"
    force_bos: bool = True
    concatenation: str = "piecewise-unrstripped"
    alias_aggregation: str = "prefix-disjoint-logsumexp"
    whitespace_policy: str = "reject-trailing"
    answer_boundary: str = "normalized-token-prefix"
    generation_normalization: str = "nfkd-lower-alnum-space"
    max_prompt_tokens: int = 512
    max_answer_tokens: int = 24
    max_generation_tokens: int = 256
    reasoning_parser_version: str = "p4-phase-parser-v1"

    def as_dict(self) -> dict:
        return asdict(self)

    def normalize_generation(self, text: str) -> str:
        if self.generation_normalization != "nfkd-lower-alnum-space":
            raise ValueError(
                f"unknown normalization {self.generation_normalization!r}")
        folded = unicodedata.normalize("NFKD", text).encode(
            "ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


DEFAULT_SPEC = ScoringSpec()


def canonical_alias_for(session: "ScoringSession", aliases: Sequence[str],
                        canonical_answer: str) -> str:
    """Resolve the bank-declared canonical spelling before score selection.

    Exact spelling is checked before the grading normalization because the
    latter deliberately folds accents (for example, Río and Rio).  A
    normalized fallback is allowed only when it is unique.
    """
    exact_matches = [
        alias for alias in aliases
        if alias.strip() == canonical_answer.strip()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"canonical answer {canonical_answer!r} has "
            f"{len(exact_matches)} exact aliases")
    target = session.spec.normalize_generation(canonical_answer)
    matches = [
        alias for alias in aliases
        if session.spec.normalize_generation(alias) == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"canonical answer {canonical_answer!r} has {len(matches)} "
            "exact normalized aliases")
    return matches[0]


def token_manifest_sha256(alias_token_ids: Mapping[str, Sequence[int]]) -> str:
    rows = [
        {"alias": alias, "token_ids": [int(value) for value in ids]}
        for alias, ids in sorted(alias_token_ids.items())
    ]
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_prefix(left: Sequence[int], right: Sequence[int]) -> bool:
    return len(left) <= len(right) and list(right[:len(left)]) == list(left)


def prefix_disjoint_aliases(
        aliases: Sequence[str],
        token_ids: Mapping[str, Sequence[int]],
        canonical_alias: str,
) -> list[str]:
    """Choose the largest safe alias set before outcomes are scored.

    Ties prefer inclusion of the canonical alias, then the historical first
    alias, then the lexicographically smallest ordinal tuple.
    """
    aliases = list(aliases)
    if not aliases or canonical_alias not in aliases:
        raise ValueError("aliases must be nonempty and contain canonical")
    if set(aliases) != set(token_ids):
        raise ValueError("token IDs must be supplied for every alias")
    safe = []
    for size in range(1, len(aliases) + 1):
        for indices in itertools.combinations(range(len(aliases)), size):
            candidate = [aliases[index] for index in indices]
            if any(
                    _is_prefix(token_ids[a], token_ids[b])
                    or _is_prefix(token_ids[b], token_ids[a])
                    for a, b in itertools.combinations(candidate, 2)):
                continue
            safe.append(indices)
    if not safe:
        raise ValueError("no prefix-disjoint alias subset exists")
    first = aliases[0]
    best = min(
        safe,
        key=lambda indices: (
            -len(indices),
            -(canonical_alias in [aliases[index] for index in indices]),
            -(first in [aliases[index] for index in indices]),
            indices,
        ),
    )
    return [aliases[index] for index in best]


def aggregate_alias_lps(
        lp_by_alias: Mapping[str, float],
        selected_aliases: Sequence[str],
        *,
        method: str = DEFAULT_SPEC.alias_aggregation,
) -> float:
    if method != "prefix-disjoint-logsumexp":
        raise ValueError(f"Phase 4 primary aggregation is frozen: {method!r}")
    aliases = list(selected_aliases)
    if not aliases:
        raise ValueError("selected alias set is empty")
    missing = [alias for alias in aliases if alias not in lp_by_alias]
    if missing:
        raise ValueError(f"missing alias scores: {missing}")
    values = torch.tensor(
        [float(lp_by_alias[alias]) for alias in aliases],
        dtype=torch.float64,
    )
    return float(torch.logsumexp(values, dim=0).item())


class ScoringSession:
    """One tokenizer, one immutable scoring specification."""

    def __init__(self, tokenizer, spec: ScoringSpec = DEFAULT_SPEC,
                 device: str | torch.device = "cpu"):
        self.tokenizer = tokenizer
        self.spec = spec
        self.device = torch.device(device)
        self.bos_prefixed = False
        if spec.force_bos and tokenizer.bos_token_id is not None:
            probe = tokenizer("probe", return_tensors="pt").input_ids[0]
            if not len(probe) or int(probe[0]) != int(tokenizer.bos_token_id):
                tokenizer.add_bos_token = True
                probe = tokenizer("probe", return_tensors="pt").input_ids[0]
            if not len(probe) or int(probe[0]) != int(tokenizer.bos_token_id):
                raise ValueError("could not establish BOS token units")
            self.bos_prefixed = True

    def prompt_ids(self, prompt: str) -> torch.Tensor:
        if (self.spec.whitespace_policy == "reject-trailing"
                and prompt != prompt.rstrip()):
            raise ValueError("prompt has trailing whitespace")
        result = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.spec.max_prompt_tokens,
        ).input_ids
        return result.to(self.device)

    def answer_ids(self, alias: str) -> torch.Tensor:
        result = self.tokenizer(
            alias,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.spec.max_answer_tokens,
        ).input_ids
        if result.shape[1] == 0:
            raise ValueError(f"alias {alias!r} tokenizes to nothing")
        return result.to(self.device)

    def full_ids(self, prompt: str, alias: str) -> tuple[torch.Tensor, int]:
        prompt_ids = self.prompt_ids(prompt)
        answer_ids = self.answer_ids(alias)
        return torch.cat([prompt_ids, answer_ids], dim=1), prompt_ids.shape[1]

    @staticmethod
    def answer_sequence_lp(full_ids: torch.Tensor, logits: torch.Tensor,
                           prompt_length: int) -> float:
        answer_logits = logits[
            prompt_length - 1:full_ids.shape[1] - 1].float()
        targets = full_ids[0, prompt_length:].to(answer_logits.device)
        if answer_logits.shape[0] != targets.shape[0]:
            raise ValueError("answer logits/targets length mismatch")
        token_lps = torch.log_softmax(
            answer_logits, dim=-1).gather(
                1, targets.unsqueeze(1)).squeeze(1)
        return float(token_lps.sum().item())

    def freeze_alias_manifest(self, aliases: Sequence[str],
                              canonical_alias: str) -> dict:
        aliases = list(aliases)
        token_ids = {
            alias: [int(value) for value in self.answer_ids(alias)[0].tolist()]
            for alias in aliases
        }
        selected = prefix_disjoint_aliases(
            aliases, token_ids, canonical_alias)
        return {
            "aliases": aliases,
            "canonical_alias": canonical_alias,
            "token_ids": token_ids,
            "prefix_disjoint_aliases": selected,
            "token_manifest_sha256": token_manifest_sha256(token_ids),
            "aggregation": self.spec.alias_aggregation,
        }

    @staticmethod
    def clean_first_token_ranks(
            logits_at_boundary: torch.Tensor,
            alias_first_token_ids: Mapping[str, int]) -> dict:
        if not alias_first_token_ids:
            raise ValueError("protected-answer rank requires alias metadata")
        row = logits_at_boundary.float()
        if row.ndim != 1:
            raise ValueError("boundary logits must be one-dimensional")
        ranks = {
            alias: int((row > row[int(token_id)]).sum().item()) + 1
            for alias, token_id in alias_first_token_ids.items()
        }
        return {
            "rank_metadata_present": True,
            "rank_by_alias": ranks,
            "min_rank": min(ranks.values()),
        }

    def grade_alias(self, generated: str,
                    aliases: Sequence[str]) -> str | None:
        normalized = self.spec.normalize_generation(generated)
        matches = []
        for alias in aliases:
            candidate = self.spec.normalize_generation(alias)
            if candidate and (
                    normalized == candidate
                    or normalized.startswith(candidate + " ")):
                matches.append(alias)
        if not matches:
            return None
        return max(matches, key=lambda value: len(
            self.spec.normalize_generation(value)))

    def grade_counterfactual_generation(
            self, generated: str, *,
            original_aliases: Sequence[str],
            counterfactual_aliases: Sequence[str]) -> dict:
        original = self.grade_alias(generated, original_aliases)
        counterfactual = self.grade_alias(generated, counterfactual_aliases)
        if original and counterfactual:
            outcome = "ambiguous"
        elif original:
            outcome = "original"
        elif counterfactual:
            outcome = "counterfactual"
        else:
            outcome = "other_invalid"
        return {
            "outcome": outcome,
            "matched_original_alias": original,
            "matched_counterfactual_alias": counterfactual,
        }


def logsumexp_reference(values: Sequence[float]) -> float:
    """Small pure-Python reference used by tokenizer golden tests."""
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum)
                                  for value in values))
