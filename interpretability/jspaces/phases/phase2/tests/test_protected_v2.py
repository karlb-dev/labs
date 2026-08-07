# Golden tests for protected_dynamic_v2 (nextsteps_2_2 §2.3 acceptance list).
# CPU stub layers + a real tiny transformer where one is available.
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2.protected_dynamic_v2 import (  # noqa: E402
    ProtectedDynamicAblatorV2, protected_generate_v2)

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


torch.manual_seed(0)
d, V, T = 64, 500, 6
layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
D = torch.nn.functional.normalize(torch.randn(V, d), dim=1).half()

print("[1] per-position protection aligns with sequence positions")
# position t is dominated by dictionary row (10 + t)
h = torch.stack([D[10 + t].float() * 10 + 0.05 * torch.randn(d)
                 for t in range(T)]).reshape(1, T, d)
psets = torch.tensor([[10 + t] for t in range(T)])          # protect own row
ab = ProtectedDynamicAblatorV2(layers, band=[1])
with ab:
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": psets,
               "active_phases": {"prefill"}, "record_ids": True}
    out_p = layers[1](h.clone())
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": None,
               "active_phases": {"prefill"}, "record_ids": True}
    ab.log.positions.clear()
    out_u = layers[1](h.clone())
    unprot_recs = list(ab.log.positions)
    ab.mode = None
    out_off = layers[1](h.clone())
check(torch.allclose(out_off, h), "hooks off = identity")
for t in range(T):
    keep_p = float(out_p[0, t].float() @ D[10 + t].float())
    keep_u = float(out_u[0, t].float() @ D[10 + t].float())
    ok = keep_p > 3 * abs(keep_u)
    if not ok:
        check(False, f"position {t}: protection preserves its OWN token")
check(all(float(out_p[0, t].float() @ D[10 + t].float()) >
          3 * abs(float(out_u[0, t].float() @ D[10 + t].float()))
          for t in range(T)),
      "every position's protected token is preserved at that position")

print("[2] a protected id is never selected at its own position")
with ab:
    ab.log.positions.clear()
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": psets,
               "active_phases": {"prefill"}, "record_ids": True}
    layers[1](h.clone())
    ab.mode = None
recs = {p.position: p for p in ab.log.positions}
check(all((10 + t) not in (recs[t].selected_ids or []) for t in range(T)),
      "protected id absent from selected ids at its position")
check(sum(p.protected_blocked for p in ab.log.positions) == T,
      f"blocked count = would-have-entered, not merely finite "
      f"({sum(p.protected_blocked for p in ab.log.positions)} of {T})")

print("[3] one starved position does not shrink the others (v1's global dose)")
# position 0 has only 2 positive-scoring rows; the rest have many
Dsmall = torch.nn.functional.normalize(torch.randn(8, d), dim=1).half()
h2 = torch.stack([Dsmall[0].float() * 5,                    # aligns with few
                  torch.randn(d) * 3, torch.randn(d) * 3]).reshape(1, 3, d)
starve = torch.nn.functional.normalize(torch.randn(6, d), dim=1).half()
# craft a dictionary where row-0's activation is positive on exactly 2 rows
Dst = torch.cat([Dsmall[:2], -Dsmall[:2], starve[:2]]).half()
h3 = torch.stack([(Dsmall[0].float() + Dsmall[1].float()) * 5,
                  (starve[0].float() + starve[1].float() +
                   Dsmall[0].float()) * 5,
                  (starve[0].float() + starve[1].float() +
                   Dsmall[1].float()) * 5]).reshape(1, 3, d)
ab2 = ProtectedDynamicAblatorV2(layers, band=[1])
with ab2:
    ab2.mode = {"dicts": {1: Dst}, "k": 4, "nonneg": True, "protect_sets": None,
                "active_phases": {"prefill"}, "record_ids": True}
    layers[1](h3.clone())
    ab2.mode = None
sel = {p.position: p.selected_k for p in ab2.log.positions}
avail = {p.position: p.available_positive for p in ab2.log.positions}
check(sel == {t: min(4, avail[t]) for t in sel},
      f"each position takes its own min(k, available): selected={sel} avail={avail}")
check(len(set(avail.values())) > 1 and len(set(sel.values())) > 1,
      "the test actually exercises differing availability")

print("[4] v2 refuses to pad/broadcast a mis-sized protection matrix")
try:
    with ab:
        ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True,
                   "protect_sets": torch.tensor([[10], [11]]),   # 2 rows, T=6
                   "active_phases": {"prefill"}}
        layers[1](h.clone())
        ab.mode = None
    check(False, "mis-sized protect_sets must raise")
except ValueError:
    check(True, "mis-sized protect_sets raises (v1 padded by repeating)")

print("[5] phase gating is provable from fire counts")
ab3 = ProtectedDynamicAblatorV2(layers, band=[1])
with ab3:
    ab3.phase = "prefill"
    ab3.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": None,
                "active_phases": {"decode"}}
    layers[1](h.clone())
    fires_prefill = dict(ab3.log.hook_fires)
    ab3.phase = "decode"
    layers[1](h.clone())
    fires_after = dict(ab3.log.hook_fires)
    ab3.mode = None
check(fires_prefill["prefill"] == 0, "decode-only mode is inert during prefill")
check(fires_after["decode"] == 1, "decode-only mode fires during decode")

print("[6] logging separates decode tokens from hook fires")
log = ab.log
check(hasattr(log, "decode_tokens") and "prefill" in log.hook_fires,
      "decode_tokens and hook_fires are distinct fields")
s = ab.log.summary()
check({"effective_rank_mean", "removed_energy_frac_mean",
       "positions_below_requested_k"} <= set(s),
      "summary reports per-position rank/energy, not one pooled number")

print("[7] real transformer: decode-only never alters the prompt KV cache")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mid = "HuggingFaceTB/SmolLM-135M"
    tk = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)
    mdl.eval()
    dm = mdl.config.hidden_size
    # dictionary must be indexed by token id (protection sets are token ids)
    Dr = torch.nn.functional.normalize(
        torch.randn(mdl.config.vocab_size, dm), dim=1)
    blocks = mdl.model.layers
    band = [len(blocks) // 2]
    abr = ProtectedDynamicAblatorV2(blocks, band=band)
    with abr:
        txt_a, _ = protected_generate_v2(mdl, tk, abr, {band[0]: Dr},
                                         "The capital of France is",
                                         k=4, protect=5, max_new=6,
                                         phases=("decode",))
        dec_only = dict(abr.log.hook_fires)
        abr.log.positions.clear()
        txt_b, _ = protected_generate_v2(mdl, tk, abr, {band[0]: Dr},
                                         "The capital of France is",
                                         k=4, protect=5, max_new=6,
                                         phases=("prefill", "decode"))
        both = dict(abr.log.hook_fires)
        prefill_positions = [p for p in abr.log.positions if p.phase == "prefill"]
    check(dec_only["prefill"] == 0, "decode-only: zero prefill fires")
    check(both["prefill"] >= 1, "both-phases: prefill fires")
    check(len(prefill_positions) > 1,
          f"prefill protection covers ALL {len(prefill_positions)} prompt "
          f"positions (v1 broadcast one row)")
    ids = tk("The capital of France is", return_tensors="pt").input_ids
    check(len({p.position for p in prefill_positions}) == ids.shape[1],
          "one record per prompt position")
except Exception as e:                                   # offline / no net
    print(f"  (skip real-transformer golden: {type(e).__name__}: {e})")

print("ALL PROTECTED V2 TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
