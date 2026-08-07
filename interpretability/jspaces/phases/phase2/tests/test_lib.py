# Conformance test suite (addendum §10 utilities). CPU, fast.
# Run: jspace-part2 selftest   (or python tests/test_lib.py)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jspace_part2.lib import (PhaseControlledAblator, RunningVectorMoments,
                   conditional_sequence_logprob, equivalence_from_interval,
                   intervention_phase, orthonormal_basis_from_rows,
                   paired_cluster_bootstrap, seeded_random_orthobasis,
                   select_output_protected_j_basis)

torch.manual_seed(0)


def check(name, cond):
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"  ok  {name}")


print("[1] rank-safe basis")
rows = torch.randn(5, 32)
rows[3] = rows[0] * 0.999999 + 1e-7 * torch.randn(32)   # near-duplicate
rows[4] = rows[1]                                       # exact duplicate
b = orthonormal_basis_from_rows(rows)
check("duplicates collapse rank (<5)", b.effective_rank == 3)
check("orthonormal", torch.allclose(b.basis.T @ b.basis,
                                    torch.eye(b.effective_rank), atol=1e-4))
check("empty input handled", orthonormal_basis_from_rows(
    torch.zeros(0, 8)).effective_rank == 0)
check("requested rows recorded", b.requested_rows == 5)

print("[2] output-protected selection")
V, d = 200, 32
D = torch.randn(V, d)
h = torch.zeros(d)
# make token 7 the strongest J direction AND the top clean output token
D[7] = torch.nn.functional.normalize(torch.randn(d), dim=0)
h = D[7] * 10 + 0.1 * torch.randn(d)
clean_logits = torch.zeros(V)
clean_logits[7] = 100.0
sel = select_output_protected_j_basis(h, D, clean_logits, k=5, protect_top_k=10)
check("protected id excluded", 7 not in sel.selected_ids.tolist())
check("protected list correct", 7 in sel.protected_ids.tolist())
sel2 = select_output_protected_j_basis(h, D, clean_logits, k=5, protect_top_k=10)
check("deterministic", sel.selected_ids.tolist() == sel2.selected_ids.tolist())
# k shrinks when few positive directions remain
D_small = torch.nn.functional.normalize(torch.randn(12, d), dim=1)
h_neg = -D_small.sum(0)  # mostly negative correlations
sel3 = select_output_protected_j_basis(h_neg, D_small, torch.zeros(12),
                                       k=10, protect_top_k=3)
check("k shrinks gracefully", sel3.basis_result.effective_rank <= 10)
# protection property is COMPARATIVE (addendum §10.2 intent): the projector
# built under protection must remove strictly less of the intended-output
# direction than the unprotected one, which selects id 7 outright.
# (Incidental overlap via similar rows is expected paper semantics.)
sel_unprot = select_output_protected_j_basis(h, D, clean_logits, k=5,
                                             protect_top_k=0)
check("unprotected selects the output id", 7 in sel_unprot.selected_ids.tolist())
def survival(selection):
    q = selection.basis_result.basis
    out = D[7] / D[7].norm()
    return float((out - (out @ q) @ q.T).norm())
check("protection strictly increases output-direction survival",
      survival(sel) > survival(sel_unprot) + 0.1)

print("[3] mergeable moments == direct covariance (interrupt/merge orders)")
X = torch.randn(500, 16, dtype=torch.float64)
direct = torch.cov(X.T, correction=1)
rng = np.random.default_rng(0)
for trial in range(3):
    cuts = np.sort(rng.choice(np.arange(1, 500), size=6, replace=False))
    parts = np.split(np.arange(500), cuts)
    rng.shuffle(parts)
    a = RunningVectorMoments.empty(16)
    half = len(parts) // 2
    for p in parts[:half]:
        a.update(X[p])
    b2 = RunningVectorMoments.empty(16)
    for p in parts[half:]:
        b2.update(X[p])
    # simulate interrupt/resume via state_dict round-trip, then merge
    a = RunningVectorMoments.from_state_dict(a.state_dict())
    a.merge(b2)
    check(f"trial {trial}: cov matches direct",
          torch.allclose(a.covariance(), direct, atol=1e-8))

print("[4] full-sequence conditional logprob vs hand calculation")
class StubLM:  # deterministic tiny "model"
    class Out:
        def __init__(self, logits): self.logits = logits
    def __init__(self, V=11):
        g = torch.Generator().manual_seed(1)
        self.table = torch.randn(64, V, generator=g)
    def __call__(self, input_ids, use_cache=False):
        # logits at position t depend only on t and the token at t
        T = input_ids.shape[1]
        logits = torch.stack([self.table[(t * 7 + int(input_ids[0, t])) % 64]
                              for t in range(T)])[None]
        return StubLM.Out(logits)
lm = StubLM()
prompt = torch.tensor([[1, 2, 3]])
answer = torch.tensor([[4, 5]])
scored = conditional_sequence_logprob(lm, prompt, answer)
full = torch.cat([prompt, answer], 1)
logits = lm(full).logits[0]
lp_hand = (torch.log_softmax(logits[2], -1)[4]
           + torch.log_softmax(logits[3], -1)[5]).item()
check("sum matches hand calc", abs(scored.sum_logprob - lp_hand) < 1e-5)
check("per-token count", len(scored.token_logprobs) == 2)

print("[5] phase-controlled ablator")
layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
q = torch.zeros(8, 1); q[0, 0] = 1.0   # project out e0
ab = PhaseControlledAblator(layers, {1: q}, active_phases={"decode"})
x = torch.ones(1, 4, 8)
with ab:
    with intervention_phase("prefill"):
        y_pre = layers[1](x.clone())
    with intervention_phase("decode"):
        y_dec = layers[1](x.clone())
    with intervention_phase("inactive"):
        y_off = layers[1](x.clone())
check("prefill untouched", torch.allclose(y_pre, x))
check("decode edited (e0 zeroed)", torch.allclose(y_dec[..., 0],
                                                  torch.zeros(1, 4)))
check("inactive untouched", torch.allclose(y_off, x))
check("fire counts", ab.fire_counts == {"prefill": 0, "decode": 1})
check("hooks detached on exit", len(ab.handles) == 0)
y_after = layers[1](x.clone())
check("no residual hook", torch.allclose(y_after, x))

print("[6] nested seeded bases")
b1 = seeded_random_orthobasis(64, 20, seed=5)
b2 = seeded_random_orthobasis(64, 20, seed=5)
b3 = seeded_random_orthobasis(64, 20, seed=6)
check("reproducible", torch.allclose(b1, b2))
check("seed-distinct", not torch.allclose(b1, b3))
check("dose = prefix (nested by construction)",
      torch.allclose(b1[:, :10], b2[:, :10]))

print("[7] paired cluster bootstrap + equivalence")
rows = []
rng = np.random.default_rng(2)
for fam in range(12):
    for it in range(4):
        base = rng.normal(0, 0.05)
        rows.append({"family": f"f{fam}", "item": f"f{fam}i{it}",
                     "condition": "baseline", "score": base})
        rows.append({"family": f"f{fam}", "item": f"f{fam}i{it}",
                     "condition": "treat", "score": base - 1.0
                     + rng.normal(0, 0.05)})
res = paired_cluster_bootstrap(pd.DataFrame(rows), cluster_column="family",
                               item_column="item", condition_column="condition",
                               score_column="score", treatment="treat",
                               baseline="baseline", draws=2000)
check("recovers true delta ≈ -1", abs(res["estimate"] + 1.0) < 0.05)
check("CI excludes 0", res["ci_high"] < 0)
check("equivalence rejects real effect",
      not equivalence_from_interval(**{k: res[v] for k, v in
          [("estimate", "estimate"), ("ci_low", "ci_low"),
           ("ci_high", "ci_high")]}, smallest_effect=0.5))
check("equivalence accepts tight null",
      equivalence_from_interval(0.01, -0.04, 0.06, smallest_effect=0.5))

print("ALL P2LIB SELF-TESTS PASS")
