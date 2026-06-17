import json, os
RUNS=[("A SmolLM2-135M","lab36_mp_smollm"),("B Olmo-3-7B-Instruct","lab36_mp_olmo7b_instruct"),
      ("B Olmo-3-7B-Think","lab36_mp_olmo7b_think"),("Gemma-4-E4B","lab36_mp_gemma4_e4b"),
      ("C Olmo-3.1-32B-Instruct","lab36_mp_olmo31_32b_instruct"),("C Olmo-3.1-32B-Think","lab36_mp_olmo31_32b_think")]
def f(m,k):
    v=m.get(k,"") if m else ""
    if v in ("",None): return "—"
    try: return f"{float(v):.3f}"
    except: return str(v)
print("### Readout probes (n=60 heldout; control should be ~1.0)\n")
print("| Model | proj ctrl | proj sent | proj null | trained ctrl | trained sent | trained null | perdir mean | perdir>null |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for label,run in RUNS:
    p=f"runs/{run}/metrics.json"
    if not os.path.exists(p): print(f"| {label} | (pending) | | | | | | | |"); continue
    m=json.load(open(p))
    print(f"| {label} | {f(m,'b5_readout_reportquery_control_auc')} | {f(m,'b5_readout_sentinel_auc')} | {f(m,'b5_readout_sentinel_null_p95_auc')} | {f(m,'b5_readout_trained_pooled_control_auc')} | {f(m,'b5_readout_trained_pooled_auc')} | {f(m,'b5_readout_trained_pooled_null_p95_auc')} | {f(m,'b5_readout_trained_perdir_auc_mean')} | {m.get('b5_readout_trained_perdir_above_null_count','—')}/{m.get('b5_readout_trained_perdir_n_directions','—')} |")
print("\n### Sentinel verbalized sweeps (d′)\n")
print("| Model | dose2 | dose8 | place early | place mid | place late |")
print("|---|--:|--:|--:|--:|--:|")
for label,run in RUNS:
    p=f"runs/{run}/metrics.json"
    if not os.path.exists(p): print(f"| {label} | (pending) | | | | |"); continue
    m=json.load(open(p))
    # dose2 from summary table
    import csv
    s={r['comparison']:r for r in csv.DictReader(open(f"runs/{run}/tables/injection_detection_summary.csv"))} if os.path.exists(f"runs/{run}/tables/injection_detection_summary.csv") else {}
    d2=s.get('sentinel_dose_sweep::dose_2',{}).get('d_prime','—')
    print(f"| {label} | {d2} | {f(m,'b5_sentinel_sweep_max_dose_d_prime')} | {f(m,'b5_sentinel_placement_early_d_prime')} | {f(m,'b5_sentinel_placement_mid_d_prime')} | {f(m,'b5_sentinel_placement_late_d_prime')} |")
