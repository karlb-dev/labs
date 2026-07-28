# N1.4 — corrected protected dynamic ablation (nextsteps_2_2 §2.3).
#
# THE DEFECTS THIS REPAIRS (all confirmed in protected_dynamic.py):
#
#  1. GENERATION PREFILL BROADCAST. `protected_generate` took ONE logit
#     vector (the final prompt position) and broadcast that single
#     protection set across every prompt position during the ablated
#     prefill. The teacher-forced path builds a per-position set correctly;
#     the generation path did not. Any chat-mode or generation-based
#     confirmatory cell would therefore have run a different protocol than
#     the one described.
#
#  2. GLOBAL DOSE COLLAPSE. `take = min(k, avail.min())` — ONE position
#     with few positive-scoring rows shrank the dose k for the ENTIRE
#     sequence. Selection is now row-wise: each position takes its own
#     min(k, its own availability), and a starved position cannot touch
#     its neighbours.
#
#  3. LOGGING SEMANTICS.
#       n_steps            counted hook applications, not decode tokens
#       protected_hits_blocked  counted protected rows with finite positive
#                          scores, not protected rows that would actually
#                          have ENTERED the top-k
#       last_selected      kept only the first flattened position
#       removed_energy     pooled layers, positions and forwards into one
#                          ambiguous number
#     v2 logs per (layer, phase): decode-token count separate from hook
#     fires, would-have-entered blocking, per-position selected counts,
#     per-position removed energy, requested vs available vs achieved k,
#     and the singular spectrum.
#
#  4. PHASE CONTROL. Prefill-only / decode-only / both are now explicit
#     and provable from fire counts, so "generation-only intervention" is
#     a checked property rather than prose.
#
# The v1 module is preserved unchanged for exact historical reproduction
# of pilot evidence; nothing new should import it.
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PositionRecord:
    """One (layer, phase, forward, position) intervention."""
    layer: int
    phase: str
    forward_index: int
    position: int
    requested_k: int
    available_positive: int
    selected_k: int
    effective_rank: int
    removed_energy_frac: float
    protected_blocked: int          # protected rows that WOULD have entered
    selected_ids: list | None = None
    selected_scores: list | None = None
    singular_values: list | None = None


@dataclass
class V2Log:
    decode_tokens: int = 0                  # actual generated tokens
    hook_fires: dict = field(default_factory=lambda: {"prefill": 0, "decode": 0})
    positions: list = field(default_factory=list)   # list[PositionRecord]

    def summary(self) -> dict:
        if not self.positions:
            return {"decode_tokens": self.decode_tokens,
                    "hook_fires": dict(self.hook_fires), "n_positions": 0}
        import statistics as st
        rk = [p.effective_rank for p in self.positions]
        en = [p.removed_energy_frac for p in self.positions]
        bl = [p.protected_blocked for p in self.positions]
        starved = sum(1 for p in self.positions if p.selected_k < p.requested_k)
        return {
            "decode_tokens": self.decode_tokens,
            "hook_fires": dict(self.hook_fires),
            "n_positions": len(self.positions),
            "effective_rank_mean": round(st.mean(rk), 4),
            "effective_rank_min": min(rk), "effective_rank_max": max(rk),
            "removed_energy_frac_mean": round(st.mean(en), 6),
            "removed_energy_frac_max": round(max(en), 6),
            "protected_blocked_total": int(sum(bl)),
            "positions_below_requested_k": starved,
        }


class ProtectedDynamicAblatorV2:
    """Per-position protected dynamic deflation with row-wise dose.

    mode dict:
      dicts        {layer: [V, d] unit-row dictionary}
      k            requested dose per position
      protect_sets None | LongTensor [T, pk] aligned to sequence positions
                   | LongTensor [pk] (single-position decode step)
      nonneg       mask non-positive scores (default True)
      active_phases  {"prefill", "decode"} subset; hooks are inert elsewhere
      record_ids   store selected ids/scores per position (audit; heavier)
    """

    def __init__(self, layers, band):
        self._layers = layers
        self.band = band
        self._handles = []
        self.mode = None
        self.phase = "prefill"
        self.forward_index = 0
        self.log = V2Log()

    # -------------------------------------------------------------- core
    def _apply(self, h, layer_idx):
        m = self.mode
        D = m["dicts"][layer_idx]
        B, T, d = h.shape
        if B != 1:
            raise NotImplementedError("v2 assay is per-item (batch size 1)")
        flat = h.reshape(-1, d).float()                       # [T, d]
        scores = (flat.to(D.dtype) @ D.T).float()             # [T, V]
        if m.get("nonneg", True):
            scores = torch.where(scores > 0, scores,
                                 torch.full_like(scores, float("-inf")))

        k = int(m["k"])
        # ---- protection, per position, with WOULD-HAVE-ENTERED accounting
        blocked_per_pos = torch.zeros(T, dtype=torch.long)
        ps = m.get("protect_sets")
        if ps is not None:
            if ps.dim() == 1:
                idx = ps.to(scores.device).unsqueeze(0).expand(T, -1)
            else:
                if ps.shape[0] != T:
                    raise ValueError(
                        f"protect_sets has {ps.shape[0]} rows for {T} positions "
                        f"— v2 refuses to broadcast or pad (the v1 defect)")
                idx = ps.to(scores.device)
            # protection ids are TOKEN ids, so the dictionary must be
            # vocab-indexed. A mis-sized dictionary would either crash here
            # or (worse) protect the wrong rows silently.
            if int(idx.max()) >= D.shape[0]:
                raise ValueError(
                    f"protect id {int(idx.max())} exceeds dictionary rows "
                    f"{D.shape[0]}: the dictionary must be indexed by token id")
            # a protected row is BLOCKED only if it would have made the
            # top-k among the still-valid rows (v1 counted any finite score)
            pre = scores.clone()
            kth = _kth_valid_value(pre, k)                     # [T]
            prot_scores = pre.gather(1, idx)
            would_enter = torch.isfinite(prot_scores) & \
                (prot_scores >= kth.unsqueeze(1))
            blocked_per_pos = would_enter.sum(dim=1).cpu()
            scores.scatter_(1, idx, float("-inf"))

        # ---- ROW-WISE dose: each position takes its own availability
        finite = torch.isfinite(scores)
        avail = finite.sum(dim=1)                              # [T]
        take = int(min(k, int(avail.max().item()))) if avail.numel() else 0
        if take <= 0:
            self.log.hook_fires[self.phase] += 1
            return h
        top_v, top_i = scores.topk(take, dim=1)                # [T, take]
        valid = torch.isfinite(top_v)                          # per-row mask

        dirs = D[top_i].float()                                # [T, take, d]
        dirs = dirs * valid.unsqueeze(-1)                      # zero invalid rows
        h2_before = (flat * flat).sum(dim=1)                   # [T]
        U, S, _ = torch.linalg.svd(dirs.transpose(1, 2), full_matrices=False)
        thr = (S[:, :1] * 1e-4).clamp_min(1e-7)
        rank_mask = S > thr
        Ur = U * rank_mask.unsqueeze(1)
        coef = torch.einsum("tdk,td->tk", Ur, flat)
        flat = flat - torch.einsum("tdk,tk->td", Ur, coef)
        h2_after = (flat * flat).sum(dim=1)

        self.log.hook_fires[self.phase] += 1
        eff = rank_mask.sum(dim=1).cpu()
        sel_k = valid.sum(dim=1).cpu()
        frac = (1.0 - h2_after / h2_before.clamp_min(1e-30)).cpu()
        rec_ids = m.get("record_ids", False)
        for t in range(T):
            self.log.positions.append(PositionRecord(
                layer=layer_idx, phase=self.phase,
                forward_index=self.forward_index, position=t,
                requested_k=k, available_positive=int(avail[t]),
                selected_k=int(sel_k[t]), effective_rank=int(eff[t]),
                removed_energy_frac=float(frac[t]),
                protected_blocked=int(blocked_per_pos[t]),
                selected_ids=(top_i[t][valid[t]].tolist() if rec_ids else None),
                selected_scores=([round(float(x), 5)
                                  for x in top_v[t][valid[t]]] if rec_ids else None),
                singular_values=([round(float(x), 5) for x in S[t][rank_mask[t]]]
                                 if rec_ids else None)))
        return flat.reshape(B, T, d).to(h.dtype)

    def _hook(self, layer_idx):
        def fn(mod, inp, out):
            if self.mode is None:
                return out
            if self.phase not in self.mode.get("active_phases",
                                               {"prefill", "decode"}):
                return out
            h = out[0] if not torch.is_tensor(out) else out
            h_new = self._apply(h, layer_idx)
            return h_new if torch.is_tensor(out) else (h_new, *out[1:])
        return fn

    def __enter__(self):
        for l in self.band:
            self._handles.append(
                self._layers[l].register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _kth_valid_value(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Per row, the k-th largest finite score (-inf if fewer than k exist).
    Defines 'would have entered the top-k' for protection accounting."""
    T, V = scores.shape
    kk = min(k, V)
    vals = scores.topk(kk, dim=1).values
    n_finite = torch.isfinite(vals).sum(dim=1)
    out = torch.full((T,), float("-inf"), device=scores.device)
    has = n_finite >= kk
    if has.any():
        out[has] = vals[has, kk - 1]
    return out


# --------------------------------------------------------------- drivers
@torch.no_grad()
def protected_teacher_forced_v2(hf, model_encode, ab: ProtectedDynamicAblatorV2,
                                dicts, text, *, k=10, protect=10,
                                protected=True, max_length=512,
                                record_ids=False):
    """Two-pass teacher-forced scoring; protection sets are per position."""
    ids = model_encode(text, max_length=max_length)
    ab.mode = None
    clean = hf(input_ids=ids, use_cache=False).logits[0]
    psets = clean.topk(protect, dim=-1).indices if protected else None
    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": dicts, "k": k, "nonneg": True, "protect_sets": psets,
               "active_phases": {"prefill"}, "record_ids": record_ids}
    ablated = hf(input_ids=ids, use_cache=False).logits[0]
    ab.mode = None
    return ids, ablated.float().cpu(), clean.float().cpu()


@torch.no_grad()
def protected_generate_v2(hf, tok, ab: ProtectedDynamicAblatorV2, dicts, prompt,
                          *, k=10, protect=10, max_new=48, protected=True,
                          phases=("prefill", "decode"), record_ids=False,
                          eos_id=None):
    """Greedy generation with a parallel CLEAN kv stream supplying the
    protection sets.

    phases:
      ("prefill","decode")  paper-all-positions: the ablated prefill uses
                            the FULL [T, pk] per-position protection matrix
                            (v1 broadcast one row here — the defect)
      ("decode",)           decode-only: hooks provably inert during
                            prefill, so the prompt KV cache is untouched
    """
    active = set(phases)
    ids = tok(prompt, return_tensors="pt").input_ids.to(
        next(hf.parameters()).device)
    ab.log = V2Log()

    ab.mode = None
    clean = hf(input_ids=ids, use_cache=True)
    clean_past = clean.past_key_values
    prefill_protect = (clean.logits[0].topk(protect, dim=-1).indices
                       if protected else None)          # [T, pk] per position

    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
               "protect_sets": prefill_protect, "active_phases": active,
               "record_ids": record_ids}
    out = hf(input_ids=ids, use_cache=True)
    abl_past = out.past_key_values

    eos = tok.eos_token_id if eos_id is None else eos_id
    toks = []
    nxt = int(out.logits[0, -1].argmax())
    clean_step_logits = clean.logits[0, -1]
    for step in range(max_new):
        toks.append(nxt)
        ab.log.decode_tokens += 1
        if nxt == eos:
            break
        tok_in = torch.tensor([[nxt]], device=ids.device)
        ab.mode = None
        c = hf(input_ids=tok_in, past_key_values=clean_past, use_cache=True)
        clean_past = c.past_key_values
        clean_step_logits = c.logits[0, -1]
        ab.phase, ab.forward_index = "decode", step + 1
        ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
                   "protect_sets": (clean_step_logits.topk(protect).indices
                                    if protected else None),
                   "active_phases": active, "record_ids": record_ids}
        a = hf(input_ids=tok_in, past_key_values=abl_past, use_cache=True)
        abl_past = a.past_key_values
        nxt = int(a.logits[0, -1].argmax())
    ab.mode = None
    return tok.decode(toks, skip_special_tokens=True), toks
