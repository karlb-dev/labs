# Phase 3 intervention arms (nextsteps §6.1–§6.4), built on the frozen
# v2 ablator semantics: per-position prefill protection, row-wise dose,
# phase control, batch-size-1 assay, would-have-entered accounting.
#
# Arms this module provides (naming per §15.2):
#   meanJ_label_protected   exact Phase 2 / paper arm (v2 behaviour,
#                           plus optional per-position overlap logging)
#   meanJ_span_safe         label selection, then residualize selected
#                           rows against the protected-row span before
#                           building the rank-safe basis; lost rank is an
#                           observable, never refilled (§6.1)
#   instant_rank_energy_matched / overlap_matched / persistent_matched
#                           controls consuming the J arm's logged profile
#   logit_* variants        same machinery on the logit dictionary
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from jspace_part2.protected_dynamic_v2 import (PositionRecord,
                                               ProtectedDynamicAblatorV2,
                                               V2Log)
from .controls import (PersistentFrame, _seed_for,
                       build_instant_matched_subspace,
                       build_overlap_matched_subspace,
                       build_prot_energy_matched_subspace)
from jspace_part2.lib import orthonormal_basis_from_rows


@dataclass
class OverlapPositionRecord:
    layer: int
    phase: str
    forward_index: int
    position: int
    rank_selected: int
    rank_protected: int
    projector_overlap: float
    overlap_normalized: float
    answer_dir_survival: float | None
    removed_energy_in_prot_frac: float | None
    lost_rank: int                     # span-safe only; 0 for label arm
    null_row_frac: float               # span-safe only


@dataclass
class P3Log(V2Log):
    overlap: list = field(default_factory=list)  # OverlapPositionRecord

    def overlap_summary(self) -> dict:
        if not self.overlap:
            return {"n_positions": 0}
        import statistics as st
        ov = [r.projector_overlap for r in self.overlap]
        ovn = [r.overlap_normalized for r in self.overlap]
        lost = [r.lost_rank for r in self.overlap]
        surv = [r.answer_dir_survival for r in self.overlap
                if r.answer_dir_survival is not None]
        rem = [r.removed_energy_in_prot_frac for r in self.overlap
               if r.removed_energy_in_prot_frac is not None]
        return {
            "n_positions": len(self.overlap),
            "projector_overlap_mean": round(st.mean(ov), 6),
            "projector_overlap_max": round(max(ov), 6),
            "overlap_normalized_mean": round(st.mean(ovn), 6),
            "lost_rank_mean": round(st.mean(lost), 4),
            "lost_rank_max": max(lost),
            "answer_dir_survival_min": (round(min(surv), 6) if surv else None),
            "answer_dir_survival_mean": (round(st.mean(surv), 6)
                                         if surv else None),
            "removed_energy_in_prot_frac_mean": (round(st.mean(rem), 6)
                                                 if rem else None),
        }


class Phase3JAblator(ProtectedDynamicAblatorV2):
    """meanJ/logit arms with label or span-safe protection.

    Extra mode keys over v2:
      span_safe       bool — residualize selected rows against the
                      protected-row span before the basis (default False:
                      exact v2 behaviour)
      record_overlap  bool — per-position OverlapPositionRecord
      answer_id       int | None — token id for answer-direction survival
    """

    def __init__(self, layers, band):
        super().__init__(layers, band)
        self.log = P3Log()

    def _apply(self, h, layer_idx):  # noqa: C901 (mirrors v2 structure)
        m = self.mode
        D = m["dicts"][layer_idx]
        B, T, d = h.shape
        if B != 1:
            raise NotImplementedError("assay is per-item (batch size 1)")
        flat = h.reshape(-1, d).float()
        scores = (flat.to(D.dtype) @ D.T).float()
        if m.get("nonneg", True):
            scores = torch.where(scores > 0, scores,
                                 torch.full_like(scores, float("-inf")))

        k = int(m["k"])
        blocked_per_pos = torch.zeros(T, dtype=torch.long)
        ps = m.get("protect_sets")
        idx = None
        if ps is not None:
            if ps.dim() == 1:
                idx = ps.to(scores.device).unsqueeze(0).expand(T, -1)
            else:
                if ps.shape[0] != T:
                    raise ValueError(
                        f"protect_sets has {ps.shape[0]} rows for {T} "
                        f"positions — refusing to broadcast (v1 defect)")
                idx = ps.to(scores.device)
            if int(idx.max()) >= D.shape[0]:
                raise ValueError("protect id exceeds dictionary rows")
            pre = scores.clone()
            kth = _kth_valid_value(pre, k)
            prot_scores = pre.gather(1, idx)
            would_enter = torch.isfinite(prot_scores) & \
                (prot_scores >= kth.unsqueeze(1))
            blocked_per_pos = would_enter.sum(dim=1).cpu()
            scores.scatter_(1, idx, float("-inf"))

        # §6.5 mediation arms: restrict the CANDIDATE set to given token
        # rows (bridge-only / answer-only / unrelated-content lesions).
        # Default-off; the golden-tested label/span-safe paths never
        # reach this branch.
        rs = m.get("restrict_sets")
        if rs is not None:
            ridx = rs.to(scores.device)
            if ridx.dim() == 1:
                ridx = ridx.unsqueeze(0).expand(T, -1)
            allowed = torch.full_like(scores, float("-inf"))
            allowed.scatter_(1, ridx, 0.0)
            scores = scores + allowed

        finite = torch.isfinite(scores)
        avail = finite.sum(dim=1)
        take = int(min(k, int(avail.max().item()))) if avail.numel() else 0
        if take <= 0:
            self.log.hook_fires[self.phase] += 1
            return h
        top_v, top_i = scores.topk(take, dim=1)
        valid = torch.isfinite(top_v)
        dirs = D[top_i].float() * valid.unsqueeze(-1)        # [T, take, d]

        span_safe = bool(m.get("span_safe", False))
        lost_rank_t = torch.zeros(T, dtype=torch.long)
        null_frac_t = torch.zeros(T)
        qp_masked = None
        if idx is not None and (span_safe or m.get("record_overlap", False)):
            prot_dirs = D[idx].float()                       # [T, pk, d]
            up, sp, _ = torch.linalg.svd(prot_dirs.transpose(1, 2),
                                         full_matrices=False)
            thr_p = (sp[:, :1] * 1e-4).clamp_min(1e-7)
            qp_masked = up * (sp > thr_p).unsqueeze(1)       # [T, d, pk]

        if span_safe and qp_masked is not None:
            # label rank BEFORE residualization (the lost-rank observable)
            _, s_lab, _ = torch.linalg.svd(dirs.transpose(1, 2),
                                           full_matrices=False)
            thr_lab = (s_lab[:, :1] * 1e-4).clamp_min(1e-7)
            lab_rank = (s_lab > thr_lab).sum(dim=1).cpu()
            coef_p = torch.einsum("tkd,tdp->tkp", dirs, qp_masked)
            dirs_res = dirs - torch.einsum("tkp,tdp->tkd", coef_p, qp_masked)
            norm_before = dirs.norm(dim=2).clamp_min(1e-30)
            survival = dirs_res.norm(dim=2) / norm_before
            null_frac_t = ((survival < 1e-4) & valid).float().sum(dim=1) / \
                valid.float().sum(dim=1).clamp_min(1)
            dirs = dirs_res

        h2_before = (flat * flat).sum(dim=1)
        U, S, _ = torch.linalg.svd(dirs.transpose(1, 2), full_matrices=False)
        if span_safe and qp_masked is not None:
            # anchor the rank threshold to the PRE-residualization scale so
            # ~1e-7 residual junk cannot fabricate rank (same rule as
            # protected_span.span_safe_j_basis)
            thr = (s_lab[:, :1] * 1e-4).clamp_min(1e-7)
        else:
            thr = (S[:, :1] * 1e-4).clamp_min(1e-7)
        rank_mask = S > thr
        Ur = U * rank_mask.unsqueeze(1)
        coef = torch.einsum("tdk,td->tk", Ur, flat)
        flat_new = flat - torch.einsum("tdk,tk->td", Ur, coef)
        h2_after = (flat_new * flat_new).sum(dim=1)

        # §6.5 counterfactual bridge swap: after removing the selected
        # span, inject a unit direction scaled per position to the
        # removed norm (energy-matched substitution). Default-off.
        inj = m.get("inject_dir")
        if isinstance(inj, dict):
            inj = inj.get(layer_idx)
        if inj is not None:
            u = torch.nn.functional.normalize(
                inj.to(flat_new.device).float(), dim=0)
            scale = (h2_before - h2_after).clamp_min(0).sqrt()
            flat_new = flat_new + scale.unsqueeze(1) * u.unsqueeze(0)

        eff = rank_mask.sum(dim=1).cpu()
        if span_safe and qp_masked is not None:
            lost_rank_t = (lab_rank - eff).clamp_min(0)

        if m.get("record_overlap", False) and qp_masked is not None:
            mm = torch.einsum("tdk,tdp->tkp", Ur, qp_masked)
            sv = torch.linalg.svdvals(mm)                    # [T, min(k,pk)]
            ov = (sv ** 2).sum(dim=1).cpu()
            rp = (sp > thr_p).sum(dim=1).cpu()
            aid = m.get("answer_id")
            if aid is not None:
                arow = torch.nn.functional.normalize(
                    D[int(aid)].float(), dim=0)
                acoef = torch.einsum("tdk,d->tk", Ur, arow)
                asurv = (arow.unsqueeze(0)
                         - torch.einsum("tdk,tk->td", Ur, acoef)).norm(dim=1)
            removed = flat - flat_new                        # [T, d]
            rem_in_prot = torch.einsum("td,tdp->tp", removed, qp_masked)
            rn2 = (removed * removed).sum(dim=1)
            remfrac = ((rem_in_prot ** 2).sum(dim=1)
                       / rn2.clamp_min(1e-30)).cpu()
            for t in range(T):
                self.log.overlap.append(OverlapPositionRecord(
                    layer=layer_idx, phase=self.phase,
                    forward_index=self.forward_index, position=t,
                    rank_selected=int(eff[t]), rank_protected=int(rp[t]),
                    projector_overlap=round(float(ov[t]), 6),
                    overlap_normalized=round(
                        float(ov[t]) / max(min(int(eff[t]), int(rp[t])), 1), 6),
                    answer_dir_survival=(round(float(asurv[t]), 6)
                                         if aid is not None else None),
                    removed_energy_in_prot_frac=(
                        round(float(remfrac[t]), 6) if rn2[t] > 0 else None),
                    lost_rank=int(lost_rank_t[t]),
                    null_row_frac=round(float(null_frac_t[t]), 6)))

        self.log.hook_fires[self.phase] += 1
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
                                  for x in top_v[t][valid[t]]]
                                 if rec_ids else None)))
        return flat_new.reshape(B, T, d).to(h.dtype)


def _kth_valid_value(scores: torch.Tensor, k: int) -> torch.Tensor:
    T, V = scores.shape
    kk = min(k, V)
    vals = scores.topk(kk, dim=1).values
    n_finite = torch.isfinite(vals).sum(dim=1)
    out = torch.full((T,), float("-inf"), device=scores.device)
    has = n_finite >= kk
    if has.any():
        out[has] = vals[has, kk - 1]
    return out


@dataclass
class MatchedRecord3:
    layer: int
    phase: str
    forward_index: int
    position: int
    target_rank: int
    achieved_rank: int
    target_energy_frac: float
    achieved_energy_frac: float
    clamped: bool
    max_protected_cos: float
    protected_effective_rank: int
    overlap_target: float | None = None
    overlap_achieved: float | None = None


@dataclass
class Matched3Log(V2Log):
    matched: list = field(default_factory=list)
    ENERGY_REL_FLOOR = 1e-3

    def matched_summary(self) -> dict:
        if not self.matched:
            return {"n_positions": 0}
        import statistics as st
        fl = self.ENERGY_REL_FLOOR
        meas = [m for m in self.matched
                if not m.clamped and m.target_energy_frac >= fl]
        below = [m for m in self.matched
                 if not m.clamped and m.target_energy_frac < fl]
        rel = [abs(m.achieved_energy_frac - m.target_energy_frac)
               / m.target_energy_frac for m in meas]
        ab = [abs(m.achieved_energy_frac - m.target_energy_frac)
              for m in below]
        ovt = [m.overlap_target for m in self.matched
               if m.overlap_target is not None]
        ova = [m.overlap_achieved for m in self.matched
               if m.overlap_achieved is not None]
        return {
            "n_positions": len(self.matched),
            "rank_match_frac": sum(1 for m in self.matched
                                   if m.achieved_rank == m.target_rank)
            / len(self.matched),
            "n_above_floor": len(meas), "n_below_floor": len(below),
            "energy_rel_err_median": (round(st.median(rel), 6) if rel else None),
            "energy_rel_err_max": (round(max(rel), 6) if rel else None),
            "energy_abs_err_max_below_floor": (round(max(ab), 8)
                                               if ab else None),
            "clamped_frac": sum(1 for m in self.matched if m.clamped)
            / len(self.matched),
            "max_protected_cos": round(max(m.max_protected_cos
                                           for m in self.matched), 8),
            "protected_rank_min": min(m.protected_effective_rank
                                      for m in self.matched),
            "overlap_target_mean": (round(st.mean(ovt), 6) if ovt else None),
            "overlap_achieved_mean": (round(st.mean(ova), 6) if ova else None),
        }


class Phase3MatchedAblator(ProtectedDynamicAblatorV2):
    """Matched controls with pluggable geometry (§15.2 names):
    mode["variant"] ∈ {"instant_rank_energy_matched", "overlap_matched",
    "persistent_matched"}. Consumes the J arm's per-position profile
    {layer: {rank, energy[, overlap]}}; protected bases are RANK-SAFE."""

    def __init__(self, layers, band):
        super().__init__(layers, band)
        self.log = Matched3Log()
        self._frames: dict = {}

    def _apply(self, h, layer_idx):
        m = self.mode
        prof = m["profile"].get(layer_idx)
        if prof is None:
            self.log.hook_fires[self.phase] += 1
            return h
        B, T, d = h.shape
        if B != 1:
            raise NotImplementedError("assay is per-item (batch size 1)")
        ranks, energies = prof["rank"], prof["energy"]
        overlaps = prof.get("overlap")
        if len(ranks) != T:
            raise ValueError(f"profile has {len(ranks)} positions for a "
                             f"{T}-position forward — refusing to broadcast")
        D = m["dicts"][layer_idx]
        ps = m.get("protect_sets")
        if ps is not None and ps.dim() == 2 and ps.shape[0] != T:
            raise ValueError("protect_sets row count mismatch")
        variant = m.get("variant", "instant_rank_energy_matched")

        flat = h.reshape(-1, d).float()
        for t in range(T):
            r = int(ranks[t])
            if r <= 0:
                continue
            e = float(energies[t])
            prot = None
            if ps is not None:
                pidx = ps[t] if ps.dim() == 2 else ps
                prot = D[pidx.to(D.device)].float()
            seed = _seed_for(m["seed_base"], layer_idx, self.forward_index, t)
            ov_t = ov_a = None
            if variant == "instant_rank_energy_matched":
                basis, info = build_instant_matched_subspace(
                    flat[t], r, e, prot, seed)
            elif variant == "overlap_matched":
                ov_t = float(overlaps[t]) if overlaps is not None else 0.0
                basis, info = build_overlap_matched_subspace(
                    flat[t], r, e, ov_t, prot, seed)
                ov_a = info.get("overlap_achieved")
            elif variant == "prot_energy_matched":
                pe = prof.get("prot_energy")
                ov_t = float(pe[t]) if pe is not None else 0.0
                basis, info = build_prot_energy_matched_subspace(
                    flat[t], r, e, ov_t, prot, seed)
                ov_a = info.get("prot_energy_achieved")
            elif variant == "persistent_matched":
                key = layer_idx
                if key not in self._frames:
                    self._frames[key] = PersistentFrame(
                        d, max_rank=max(int(x) for x in ranks) or 1,
                        seed=_seed_for(m["seed_base"], layer_idx, 0, 0))
                basis, info = self._frames[key].subspace_at(
                    flat[t], r, e, prot)
            else:
                raise ValueError(f"unknown matched variant {variant!r}")

            before = float(flat[t] @ flat[t])
            coef = basis.T @ flat[t]
            flat[t] = flat[t] - basis @ coef
            after = float(flat[t] @ flat[t])
            ach = 1.0 - after / max(before, 1e-30)
            if variant in ("overlap_matched", "prot_energy_matched"):
                # overlap with prot span is INTENDED for these arms; the
                # protection-contract cosine is n/a by design
                mpc = 0.0
            else:
                mpc = (float((basis.T @ prot.T).abs().max())
                       if prot is not None and prot.numel() else 0.0)
            self.log.matched.append(MatchedRecord3(
                layer=layer_idx, phase=self.phase,
                forward_index=self.forward_index, position=t,
                target_rank=r, achieved_rank=basis.shape[1],
                target_energy_frac=e, achieved_energy_frac=ach,
                clamped=info["clamped"], max_protected_cos=mpc,
                protected_effective_rank=info.get(
                    "protected_effective_rank", 0),
                overlap_target=ov_t, overlap_achieved=ov_a))
        self.log.hook_fires[self.phase] += 1
        return flat.reshape(B, T, d).to(h.dtype)


def profile_from_p3log(log: V2Log, phase: str = "prefill",
                       forward_index: int = 0,
                       overlap_records: list | None = None) -> dict:
    """(rank, energy[, overlap]) per (layer, position) from a J-arm log."""
    prof: dict = {}
    for p in log.positions:
        if p.phase != phase or p.forward_index != forward_index:
            continue
        prof.setdefault(p.layer, {})[p.position] = \
            [p.effective_rank, p.removed_energy_frac, 0.0, 0.0]
    if overlap_records:
        for r in overlap_records:
            if r.phase == phase and r.forward_index == forward_index \
                    and r.layer in prof and r.position in prof[r.layer]:
                prof[r.layer][r.position][2] = r.projector_overlap
                prof[r.layer][r.position][3] = \
                    r.removed_energy_in_prot_frac or 0.0
    out = {}
    for lay, dd in prof.items():
        T = max(dd) + 1
        rank = torch.zeros(T, dtype=torch.long)
        energy = torch.zeros(T)
        overlap = torch.zeros(T)
        prot_energy = torch.zeros(T)
        for t, (r, e, o, pe) in dd.items():
            rank[t], energy[t], overlap[t], prot_energy[t] = r, e, o, pe
        out[lay] = {"rank": rank, "energy": energy, "overlap": overlap,
                    "prot_energy": prot_energy}
    return out


@torch.no_grad()
def teacher_forced_arm(hf, encode_ids, ablator, dicts, full_ids,
                       *, k=10, protect=10, span_safe=False,
                       record_overlap=False, answer_id=None,
                       protected=True):
    """One teacher-forced pass of a J/logit arm over pre-built ids
    (piecewise Amendment-1 tokenization happens in scoring.py, not here).
    Returns (ablated_logits, clean_logits, log)."""
    ablator.mode = None
    ablator.log = type(ablator.log)()
    clean = hf(input_ids=full_ids, use_cache=False).logits[0]
    psets = clean.topk(protect, dim=-1).indices if protected else None
    ablator.phase, ablator.forward_index = "prefill", 0
    ablator.mode = {"dicts": dicts, "k": k, "nonneg": True,
                    "protect_sets": psets, "active_phases": {"prefill"},
                    "span_safe": span_safe, "record_overlap": record_overlap,
                    "answer_id": answer_id}
    abl = hf(input_ids=full_ids, use_cache=False).logits[0]
    log = ablator.log
    ablator.mode = None
    return abl.float().cpu(), clean.float().cpu(), log


@torch.no_grad()
def teacher_forced_matched_arm(hf, layers, band, dicts, full_ids, profile,
                               *, variant, protect_sets, seed_base):
    """One matched-control pass consuming a J-arm profile."""
    ab = Phase3MatchedAblator(layers, band)
    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": dicts, "profile": profile, "variant": variant,
               "protect_sets": protect_sets, "seed_base": seed_base,
               "active_phases": {"prefill"}}
    with ab:
        logits = hf(input_ids=full_ids, use_cache=False).logits[0]
    ab.mode = None
    return logits.float().cpu(), ab.log
