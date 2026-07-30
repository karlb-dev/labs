# Phase 3 power design (nextsteps §14.8): simulate FAMILY-LEVEL paired
# contrasts — never the defective Phase 2 prefix families — and freeze a
# family floor, not only an item floor.
#
# Calibration comes from development-tier Phase 3 measurements (the
# §4.1b span audits: real `meanJ_span_safe` per-item deltas on all three
# primaries): between-family SD, within-family SD, zero-inflation share,
# tail prevalence, and the cross-model per-item correlation. The
# generative model per cell:
#
#   family effect  ~ (1 - z0)·N(theta, fam_sd²) + z0·0     (zero-inflated)
#   item value     = family effect + N(0, noise_sd²)
#                    + tail·(-tail_size) w.p. p_tail        (heavy tail)
#   model pairing  : per-item shared component with the measured
#                    cross-model correlation (P3-P1 differences)
#
# P3-P1 power: family sign-flip on the model-difference of family means.
# P3-P2 power: within-item J/control label exchange on tail rates.
# Simulation draws are reduced (2k) — power tolerance ±2%; the frozen
# analysis itself uses the full §14.2 draw counts.
#
# Usage: python -m jspace_phase3.experiments.power_sim
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..stats import family_signflip_test, within_item_label_exchange_tail

EVIDENCE_ID = "p3-power-sim-v3"
# v2 was self-refuting: (a) its P3-P1 generator applied the cancellation
# factor only to the FAMILY variance while the §14.1 within-fact chain
# cancels item noise too, so every MDE cell came out null; (b) its
# P3-P2 null was not null (non-tail J and C components had different
# means through sd-1.0 noise, leaking a +0.17 tail-rate difference at
# dtail=0). v3 applies the cancellation factor to both variance levels
# and makes the P3-P2 non-tail components symmetric under the null.
SUPERSEDES = "p3-power-sim-v2"  # v1 applied the RAW span-safe family SD
# to P3-P1's family effects; the within-fact composition difference
# cancels family intercepts, so v2 sweeps a cancellation axis
# fam_sd in {measured, 0.6, 0.4, 0.25} and reports MDEs per scenario
TIER = "phase3-development"
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
N_SIM = 400
ALPHA_HOLM_FIRST = 0.05 / 3          # worst-case Holm slot


def dev_calibration() -> dict:
    per_model = {}
    per_item = {}
    for slug in SLUGS:
        df = pd.read_parquet(metrics_dir(slug) / "span_audit" /
                             f"span_audit_items_{slug}.parquet")
        d = df.lp_meanJ_span_safe - df.lp_baseline
        fam = df.canonical_family
        fam_means = d.groupby(fam).mean()
        within = d - fam.map(fam_means)
        per_model[slug] = {
            "fam_sd": round(float(fam_means.std()), 4),
            "noise_sd": round(float(within.std()), 4),
            "zero_share": round(float((d.abs() < 0.05).mean()), 4),
            "tail_prev": round(float((d < -1.0).mean()), 4)}
        per_item[slug] = d.reset_index(drop=True)
    cors = []
    for a in range(len(SLUGS)):
        for b in range(a + 1, len(SLUGS)):
            cors.append(float(np.corrcoef(per_item[SLUGS[a]],
                                          per_item[SLUGS[b]])[0, 1]))
    return {"per_model": per_model,
            "cross_model_item_corr": round(float(np.mean(cors)), 4)}


def sim_p3p1_once(rng, *, n_fam, facts, theta, fam_sd, noise_sd,
                  z0, rho) -> float:
    """One simulated family-level model-difference vector -> p-value."""
    fam_eff = rng.normal(theta, fam_sd, n_fam)
    fam_eff[rng.random(n_fam) < z0] = 0.0
    shared = rng.normal(0, noise_sd, (n_fam, facts))
    a = shared * np.sqrt(rho) + rng.normal(
        0, noise_sd, (n_fam, facts)) * np.sqrt(1 - rho)
    b = shared * np.sqrt(rho) + rng.normal(
        0, noise_sd, (n_fam, facts)) * np.sqrt(1 - rho)
    diff = (a - b) + fam_eff[:, None]        # per-item model difference
    fam_vals = diff.mean(axis=1)
    return family_signflip_test(fam_vals, draws=2000,
                                seed=int(rng.integers(2**31)))["p"]


def sim_p3p2_once(rng, *, n_fam, facts, dtail, base_tail,
                  noise_sd) -> float:
    rows = []
    for f in range(n_fam):
        for _ in range(facts):
            hit_j = rng.random() < base_tail + dtail
            hit_c = rng.random() < base_tail
            # non-tail components SYMMETRIC so dtail=0 is a true null
            rows.append({"delta_J": -1.5 if hit_j else
                         rng.normal(-0.1, 0.35),
                         "delta_C": -1.5 if hit_c else
                         rng.normal(-0.1, 0.35),
                         "canonical_family": f"f{f}"})
    return within_item_label_exchange_tail(
        pd.DataFrame(rows), draws=2000,
        seed=int(rng.integers(2**31)))["p"]


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    cal = dev_calibration()
    fam_sd = float(np.mean([m["fam_sd"]
                            for m in cal["per_model"].values()]))
    noise_sd = float(np.mean([m["noise_sd"]
                              for m in cal["per_model"].values()]))
    z0 = float(np.mean([m["zero_share"]
                        for m in cal["per_model"].values()]))
    rho = max(cal["cross_model_item_corr"], 0.0)
    rng = np.random.default_rng(4242)

    p1_power = {}
    for fs_name, c in (("raw", 1.0), ("cancel06", 0.6),
                       ("cancel04", 0.4), ("cancel025", 0.25)):
        for n_fam in (18, 24, 30, 36):
            for theta in (0.0, 0.10, 0.15, 0.20, 0.30, 0.50):
                ps = [sim_p3p1_once(rng, n_fam=n_fam, facts=3,
                                    theta=theta, fam_sd=fam_sd * c,
                                    noise_sd=noise_sd * c, z0=z0, rho=rho)
                      for _ in range(N_SIM)]
                p1_power[f"{fs_name}_fam{n_fam}_theta{theta}"] = round(
                    float(np.mean(np.array(ps) < ALPHA_HOLM_FIRST)), 3)

    p2_power = {}
    for n_fam in (18, 24, 30, 36):
        for dtail in (0.0, 0.10, 0.15, 0.20, 0.30):
            ps = [sim_p3p2_once(rng, n_fam=n_fam, facts=3, dtail=dtail,
                                base_tail=0.05, noise_sd=noise_sd)
                  for _ in range(N_SIM // 2)]
            p2_power[f"fam{n_fam}_dtail{dtail}"] = round(
                float(np.mean(np.array(ps) < ALPHA_HOLM_FIRST)), 3)

    def mde(power: dict, prefix, target=0.9):
        out = {}
        for key, val in power.items():
            if not key.startswith(tuple([prefix] if isinstance(prefix, str)
                                        else list(prefix))):
                continue
            base, theta = key.rsplit("_", 1)
            out.setdefault(base, [])
            out[base].append((float(theta.replace("theta", "")
                                    .replace("dtail", "")), val))
        return {b: next((t for t, p in sorted(v) if p >= target), None)
                for b, v in out.items()}

    payload = {
        "calibration": cal,
        "sim_params": {"fam_sd": round(fam_sd, 4),
                       "noise_sd": round(noise_sd, 4),
                       "zero_share": round(z0, 4), "rho": round(rho, 4),
                       "alpha": ALPHA_HOLM_FIRST, "n_sim": N_SIM},
        "p3p1_power": p1_power, "p3p2_power": p2_power,
        "p3p1_mde90": mde(p1_power, ("raw", "cancel")),
        "p3p2_mde90": mde(p2_power, "fam"),
    }
    cmd = "python -m jspace_phase3.experiments.power_sim"
    out = metrics_dir("cross_model") / "power_sim.json"
    write_result3(payload, out, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=4242))
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             supersedes=SUPERSEDES,
             what=(f"§14.8 power simulation from dev calibration "
                   f"(fam_sd {fam_sd:.2f}, noise {noise_sd:.2f}, "
                   f"zero {z0:.2f}, rho {rho:.2f}): family floors vs "
                   f"MDE at 90% power for P3-P1/P3-P2"),
             outputs=[out])
    print(json.dumps({k: payload[k] for k in
                      ("sim_params", "p3p1_mde90", "p3p2_mde90")},
                     indent=1))


if __name__ == "__main__":
    main()
