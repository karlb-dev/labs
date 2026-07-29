# The geometry-matched primary control (freeze-blocking condition 2).
#
# DESIGN DECISION (2026-07-29, VM9): the primary control for HP3 is
# `dyn_energy_rank_matched_random` — per (item, layer, position) a RANDOM
# subspace matched to the J arm's achieved geometry at that exact site:
#
#   * SAME effective rank r as the J arm's rank-safe projector there;
#   * SAME removed-energy fraction e = ||P_S h||^2 / ||h||^2, matched
#     EXACTLY by construction, not by rejection sampling;
#   * orthogonal to the protected dictionary rows at that position, so the
#     control respects the same output-protection contract as the J arm;
#   * otherwise uniformly random, seeded deterministically per
#     (seed_base, layer, forward, position).
#
# WHY THIS OPTION AND NOT THE OTHERS (prereg candidate §5 naming):
#   dynJ_rotated            preserves the dictionary Gram/spectrum but not
#                           the DOSE: rotated rows misalign with h, so
#                           selection scores, selected-k and removed energy
#                           collapse toward the isotropic-random regime —
#                           it re-introduces the very dose confound the
#                           matched control exists to remove.
#   dyn_spectrum_matched_nonJ  matches the dictionary's singular spectrum,
#                           but a span projection is invariant to row
#                           spectrum given the span; dictionary-level
#                           spectrum is not the quantity that drives
#                           intervention severity.
#   dynJ_label_shuffled     identical row ensemble but selection now keys
#                           on wrong labels; per-item dose unmatched and
#                           the protection mask interacts with labels.
# For a span-projection intervention, the subspace's complete geometric
# relation to the current state h IS (rank, removed-energy) — there is a
# single principal angle between a subspace and a vector. Matching both
# exactly therefore equates everything about the dose and leaves only the
# DIRECTION CONTENT different between arms, which is exactly the
# specificity claim HP3 makes. The isotropic random arm remains and is
# named dynR_mechanics_control.
#
# CONSTRUCTION per position t (h = residual at t, prot = D[protect_ids[t]]):
#   u_1..u_r  <- random Gaussian, orthogonalised against span(h, prot),
#               then QR-orthonormalised            (all  _|_ h, _|_ prot)
#   p_hat     <- unit(h - proj_{span(prot)} h)     (reachable h direction)
#   a         <- sqrt(e) * ||h|| / ||h_perp||      (so (h.v1)^2 = e ||h||^2)
#   v_1       <- a * p_hat + sqrt(1-a^2) * u_1
#   S         <- span{v_1, u_2..u_r}   (orthonormal; rank exactly r;
#                removed energy exactly e; entirely _|_ prot rows)
# If e exceeds the reachable maximum ||h_perp||^2/||h||^2 (possible only
# when the protected span holds nearly all of h), e is clamped and the
# position is flagged; the dev-validation gate bounds how often this may
# happen.
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .protected_dynamic_v2 import ProtectedDynamicAblatorV2, V2Log


@dataclass
class MatchedRecord:
    layer: int
    phase: str
    forward_index: int
    position: int
    target_rank: int
    achieved_rank: int
    target_energy_frac: float
    achieved_energy_frac: float
    clamped: bool
    max_protected_cos: float        # max |v_i . prot_row| over basis x rows


@dataclass
class MatchedLog(V2Log):
    matched: list = field(default_factory=list)     # list[MatchedRecord]

    def matched_summary(self) -> dict:
        if not self.matched:
            return {"n_positions": 0}
        import statistics as st
        rel = [abs(m.achieved_energy_frac - m.target_energy_frac) /
               max(m.target_energy_frac, 1e-12)
               for m in self.matched if not m.clamped and m.target_energy_frac > 0]
        return {
            "n_positions": len(self.matched),
            "rank_match_frac": sum(1 for m in self.matched
                                   if m.achieved_rank == m.target_rank)
            / len(self.matched),
            "energy_rel_err_median": (round(st.median(rel), 6) if rel else None),
            "energy_rel_err_max": (round(max(rel), 6) if rel else None),
            "clamped_frac": sum(1 for m in self.matched if m.clamped)
            / len(self.matched),
            "max_protected_cos": round(max(m.max_protected_cos
                                           for m in self.matched), 8),
        }


def _seed_for(seed_base: int, layer: int, forward: int, pos: int) -> int:
    # stable, collision-poor mixing; must not depend on wall clock
    return (seed_base * 1_000_003 + layer * 10_007 + forward * 101 + pos) \
        % (2 ** 63 - 1)


def build_matched_subspace(h: torch.Tensor, rank: int, energy_frac: float,
                           prot_rows: torch.Tensor | None,
                           seed: int) -> tuple[torch.Tensor, dict]:
    """Return an orthonormal [d, rank] basis whose projection removes
    exactly `energy_frac` of ||h||^2, orthogonal to every row of
    `prot_rows`, random otherwise. CPU generator for cross-device
    determinism; all math in float32.
    """
    d = h.shape[0]
    h32 = h.float()
    hn2 = float(h32 @ h32)
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(rank, d, generator=g).to(h.device)

    if prot_rows is not None and prot_rows.numel():
        P = prot_rows.float()
        M = torch.cat([h32.unsqueeze(0), P], dim=0)           # [1+pk, d]
    else:
        M = h32.unsqueeze(0)
    Q, _ = torch.linalg.qr(M.T, mode="reduced")               # [d, 1+pk]
    G = G - (G @ Q) @ Q.T                                     # _|_ h, _|_ prot
    U, _ = torch.linalg.qr(G.T, mode="reduced")               # [d, rank]

    # reachable h direction inside the protected complement
    if prot_rows is not None and prot_rows.numel():
        Qp, _ = torch.linalg.qr(P.T, mode="reduced")
        h_perp = h32 - Qp @ (Qp.T @ h32)
    else:
        h_perp = h32
    hp2 = float(h_perp @ h_perp)
    e_max = hp2 / max(hn2, 1e-30)
    clamped = energy_frac > e_max * 0.999999
    e = min(energy_frac, e_max * 0.999999)
    e = max(e, 0.0)

    if hp2 > 0 and e > 0:
        p_hat = h_perp / hp2 ** 0.5
        a = (e * hn2 / hp2) ** 0.5
        v1 = a * p_hat + (1.0 - a * a) ** 0.5 * U[:, 0]
        basis = torch.cat([v1.unsqueeze(1), U[:, 1:rank]], dim=1)
    else:
        basis = U[:, :rank]

    info = {"clamped": bool(clamped), "e_target": float(energy_frac),
            "e_max": float(e_max)}
    return basis, info


class MatchedControlAblatorV2(ProtectedDynamicAblatorV2):
    """dyn_energy_rank_matched_random: consumes a per-(layer, position)
    (rank, energy) profile recorded from the J arm on the SAME item and
    replaces the J span with a matched random subspace.

    mode dict (differs from the J ablator):
      profile      {layer: {"rank": LongTensor [T], "energy": FloatTensor [T]}}
      dicts        the J dictionaries — used ONLY to materialise protected
                   rows for the orthogonality constraint
      protect_sets None | [T, pk] LongTensor (token ids, per position)
      seed_base    int; per-position seeds derive deterministically
    """

    def __init__(self, layers, band):
        super().__init__(layers, band)
        self.log = MatchedLog()

    def _apply(self, h, layer_idx):
        m = self.mode
        prof = m["profile"].get(layer_idx)
        if prof is None:
            self.log.hook_fires[self.phase] += 1
            return h
        B, T, d = h.shape
        if B != 1:
            raise NotImplementedError("v2 assay is per-item (batch size 1)")
        ranks, energies = prof["rank"], prof["energy"]
        if len(ranks) != T:
            raise ValueError(f"profile has {len(ranks)} positions for a "
                             f"{T}-position forward — refusing to broadcast")
        D = m["dicts"][layer_idx]
        ps = m.get("protect_sets")
        if ps is not None and ps.dim() == 2 and ps.shape[0] != T:
            raise ValueError(f"protect_sets has {ps.shape[0]} rows for {T} "
                             f"positions")

        flat = h.reshape(-1, d).float()
        for t in range(T):
            r = int(ranks[t])
            if r <= 0:
                continue
            e = float(energies[t])
            if ps is None:
                prot = None
            else:
                idx = ps[t] if ps.dim() == 2 else ps
                prot = D[idx.to(D.device)].float()
            seed = _seed_for(m["seed_base"], layer_idx, self.forward_index, t)
            basis, info = build_matched_subspace(flat[t], r, e, prot, seed)
            before = float(flat[t] @ flat[t])
            coef = basis.T @ flat[t]
            flat[t] = flat[t] - basis @ coef
            after = float(flat[t] @ flat[t])
            ach = 1.0 - after / max(before, 1e-30)
            mpc = (float((basis.T @ prot.T).abs().max())
                   if prot is not None and prot.numel() else 0.0)
            self.log.matched.append(MatchedRecord(
                layer=layer_idx, phase=self.phase,
                forward_index=self.forward_index, position=t,
                target_rank=r, achieved_rank=basis.shape[1],
                target_energy_frac=e, achieved_energy_frac=ach,
                clamped=info["clamped"], max_protected_cos=mpc))
        self.log.hook_fires[self.phase] += 1
        return flat.reshape(B, T, d).to(h.dtype)


def profile_from_log(log: V2Log, phase: str = "prefill",
                     forward_index: int = 0) -> dict:
    """Extract the per-(layer, position) (rank, energy) profile the J arm
    achieved, keyed for MatchedControlAblatorV2.mode['profile']."""
    prof: dict = {}
    for p in log.positions:
        if p.phase != phase or p.forward_index != forward_index:
            continue
        lay = prof.setdefault(p.layer, {})
        lay[p.position] = (p.effective_rank, p.removed_energy_frac)
    out = {}
    for lay, d in prof.items():
        T = max(d) + 1
        rank = torch.zeros(T, dtype=torch.long)
        energy = torch.zeros(T)
        for t, (r, e) in d.items():
            rank[t], energy[t] = r, e
        out[lay] = {"rank": rank, "energy": energy}
    return out


@torch.no_grad()
def teacher_forced_matched_pair_v2(hf, model_encode, layers, band, dicts,
                                   text, *, k=10, protect=10, seed_base=0,
                                   max_length=512):
    """Three passes on one item: clean -> J arm (v2, logged) -> matched
    control consuming the J arm's achieved profile. Returns
    (ids, clean, ablated_J, ablated_C, j_log, c_log)."""
    from .protected_dynamic_v2 import protected_teacher_forced_v2

    ab_j = ProtectedDynamicAblatorV2(layers, band)
    with ab_j:
        ids, abl_j, clean = protected_teacher_forced_v2(
            hf, model_encode, ab_j, dicts, text, k=k, protect=protect,
            protected=True, max_length=max_length)
    profile = profile_from_log(ab_j.log)

    clean_dev = clean.to(next(hf.parameters()).device)
    psets = clean_dev.topk(protect, dim=-1).indices
    ab_c = MatchedControlAblatorV2(layers, band)
    ab_c.phase, ab_c.forward_index = "prefill", 0
    ab_c.mode = {"dicts": dicts, "profile": profile, "protect_sets": psets,
                 "seed_base": seed_base, "active_phases": {"prefill"}}
    with ab_c:
        abl_c = hf(input_ids=ids, use_cache=False).logits[0]
    ab_c.mode = None
    return ids, clean, abl_j, abl_c.float().cpu(), ab_j.log, ab_c.log
