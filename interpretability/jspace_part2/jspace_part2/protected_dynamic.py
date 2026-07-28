# R1 — the paper's output-protected dynamic top-k J ablation.
#
# Protocol (addendum §0.1/§1.2): at each token position, remove the k most
# active J-directions from the residual stream at every band layer, but
# PROTECT any token direction that is in the clean pass's top-`protect`
# output tokens at that position — never delete the token the model is
# presently trying to emit.
#
# Design decisions (recorded for the crosswalk):
# - Selection score = h · D_row (nonnegative selection: rows with score<=0
#   masked), matching the paper's non-negative activation convention.
# - LIVE application = masked selection + EXACT batched rank-safe
#   projection per position (per-position SVD of the selected rows,
#   relative singular-value threshold, orthonormal column projection).
#   Part-1's 2-pass deflation under-removes in high-coherence regimes
#   (verified in tests); both dyn arms use the exact projector equally, so
#   the protected-vs-unprotected contrast remains single-variable. This is
#   a deliberate, recorded departure from part-1 dyn mechanics (R3
#   rank-safety applied to the live instrument).
# - Generation keeps TWO KV streams over the same emitted tokens: a CLEAN
#   stream (hooks off) whose logits define the per-step protection set,
#   and an ABLATED stream (hooks on). Cost: 2 forwards/token.
# - Teacher-forced scoring: pass 1 clean forward over prompt+answer gives
#   per-position protection sets; pass 2 hooked forward applies
#   per-position protected deflation; answer scored with the full-sequence
#   conditional logprob at the ablated logits.
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StepLog:
    n_steps: int = 0
    protected_hits_blocked: int = 0     # selections prevented by protection
    selected_counts: list = field(default_factory=list)   # eff. rank per step
    removed_energy: list = field(default_factory=list)    # per step (frac)
    last_selected: list = field(default_factory=list)     # ids, first position


class ProtectedDynamicAblator:
    """Forward hooks applying per-position protected dynamic deflation.

    mode: None | dict with
      dicts: {layer: [V, d] fp16 unit rows}
      k: int
      protect_sets: None (no protection) OR
                    per-position LongTensor [T, protect_k] (teacher-forced) OR
                    1-D LongTensor [protect_k] (single decode step)
      nonneg: bool (mask non-positive scores)
    """

    def __init__(self, layers, band):
        self._layers = layers
        self.band = band
        self._handles = []
        self.mode = None
        self.log = StepLog()

    def _apply(self, h, layer_idx):
        m = self.mode
        if "inject" in m:
            # swap/injection mode (G4 positive control): optionally project
            # out a per-layer basis (the bridge direction), then ADD
            # alpha_rel * ||h(pos)|| * unit-direction at every position.
            B, T, d = h.shape
            flat = h.reshape(-1, d).float()
            Q = m.get("remove", {}).get(layer_idx)
            if Q is not None:
                Qf = Q.to(flat.device, torch.float32)
                flat = flat - (flat @ Qf) @ Qf.T
            v = m["inject"][layer_idx].to(flat.device, torch.float32)
            v = v / v.norm()
            scale = m.get("alpha_rel", 0.1) * flat.norm(dim=1, keepdim=True)
            flat = flat + scale * v[None, :]
            self.log.n_steps += 1
            return flat.reshape(B, T, d).to(h.dtype)
        D = m["dicts"][layer_idx]
        B, T, d = h.shape
        flat = h.reshape(-1, d).float()
        scores = (flat.to(D.dtype) @ D.T).float()            # [BT, V]
        if m.get("nonneg", True):
            scores = torch.where(scores > 0, scores,
                                 torch.full_like(scores, float("-inf")))
        ps = m.get("protect_sets")
        cap = m.get("capture")          # optional list: per-(layer,pos) audit
        if ps is not None:
            if ps.dim() == 1:
                idx = ps.to(scores.device).unsqueeze(0).expand(scores.shape[0], -1)
            else:  # [T, pk] aligned with sequence positions
                if ps.shape[0] < T:  # guard: pad by repeating last row
                    pad = ps[-1:].expand(T - ps.shape[0], -1)
                    ps = torch.cat([ps, pad], 0)
                idx = ps[:T].to(scores.device).repeat(B, 1)
            fin = torch.isfinite(scores.gather(1, idx))
            scores.scatter_(1, idx, float("-inf"))
            self.log.protected_hits_blocked += int(fin.sum())
            if cap is not None:
                cap.append({"kind": "protect", "layer": layer_idx,
                            "ids": idx.reshape(B, T, -1)[0]
                                .to("cpu", torch.int32).numpy(),
                            "blocked": fin.reshape(B, T, -1)[0].cpu().numpy()})
        k = m["k"]
        finite = torch.isfinite(scores)
        avail = finite.sum(dim=1)
        take = int(min(k, int(avail.min().item()))) if avail.numel() else 0
        if take <= 0:
            return h
        top = scores.topk(take, dim=1).indices                # [BT, take]
        if cap is not None:
            cap.append({"kind": "selected", "layer": layer_idx,
                        "ids": top.reshape(B, T, -1)[0]
                            .to("cpu", torch.int32).numpy(),
                        "scores": scores.gather(1, top).reshape(B, T, -1)[0]
                            .to("cpu", torch.float32).numpy()})
        dirs = D[top].float()                                 # [BT, take, d]
        h2_before = (flat * flat).sum(dim=1)
        # exact rank-safe projection: batched SVD of selected rows
        U, S, _ = torch.linalg.svd(dirs.transpose(1, 2), full_matrices=False)
        thr = (S[:, :1] * 1e-4).clamp_min(1e-7)
        rank_mask = (S > thr)                                 # [BT, take]
        Ur = U * rank_mask.unsqueeze(1)                       # drop null cols
        coef = torch.einsum("bdk,bd->bk", Ur, flat)
        flat = flat - torch.einsum("bdk,bk->bd", Ur, coef)
        h2_after = (flat * flat).sum(dim=1)
        self.log.n_steps += 1
        self.log.selected_counts.append(
            float(rank_mask.float().sum(dim=1).mean()))       # eff. rank
        self.log.last_selected = top[0].tolist()              # audit hook
        denom = float(h2_before.sum().item()) or 1.0
        self.log.removed_energy.append(
            1.0 - float(h2_after.sum().item()) / denom)
        return flat.reshape(B, T, d).to(h.dtype)

    def _hook(self, layer_idx):
        def fn(mod, inp, out):
            if self.mode is None:
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


@torch.no_grad()
def protected_generate(hf, tok, ab: ProtectedDynamicAblator, dicts, prompt,
                       *, k=10, protect=10, max_new=48, protected=True):
    """Greedy generation with per-step protection from a parallel clean
    KV stream. protected=False runs identical mechanics without the mask
    (the part-1-style unprotected arm)."""
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    ab.mode = None
    clean_out = hf(input_ids=ids, use_cache=True)
    clean_past = clean_out.past_key_values
    clean_logits = clean_out.logits[0, -1]

    abl_past = None
    ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
               "protect_sets": (clean_logits.topk(protect).indices
                                if protected else None)}
    out = hf(input_ids=ids, use_cache=True)
    abl_past = out.past_key_values
    toks = []
    nxt = int(out.logits[0, -1].argmax())
    for _ in range(max_new):
        toks.append(nxt)
        if nxt == tok.eos_token_id:
            break
        step = torch.tensor([[nxt]], device="cuda")
        ab.mode = None
        c = hf(input_ids=step, past_key_values=clean_past, use_cache=True)
        clean_past = c.past_key_values
        ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
                   "protect_sets": (c.logits[0, -1].topk(protect).indices
                                    if protected else None)}
        a = hf(input_ids=step, past_key_values=abl_past, use_cache=True)
        abl_past = a.past_key_values
        nxt = int(a.logits[0, -1].argmax())
    ab.mode = None
    return tok.decode(toks, skip_special_tokens=True), toks


@torch.no_grad()
def protected_teacher_forced(hf, model_encode, ab: ProtectedDynamicAblator,
                             dicts, text, *, k=10, protect=10,
                             protected=True, max_length=512, capture=None):
    """Two-pass teacher-forced scoring: clean pass defines per-position
    protection; hooked pass returns ablated logits [T, V] (fp32 cpu).
    capture: optional list receiving per-(layer,pos) selected/blocked-id
    audit records from the hooked pass (see ProtectedDynamicAblator)."""
    ids = model_encode(text, max_length=max_length)
    ab.mode = None
    clean = hf(input_ids=ids, use_cache=False).logits[0]      # [T, V]
    psets = clean.topk(protect, dim=-1).indices if protected else None
    ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
               "protect_sets": psets, "capture": capture}
    ablated = hf(input_ids=ids, use_cache=False).logits[0]
    ab.mode = None
    return ids, ablated.float().cpu()
