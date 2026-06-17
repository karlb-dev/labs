import json, os, sys
RUNS=[("A SmolLM2-135M","lab36_ro_smollm"),("B Olmo-3-7B-Instruct","lab36_ro_olmo7b_instruct"),
      ("B Olmo-3-7B-Think","lab36_ro_olmo7b_think"),("Gemma-4-E4B","lab36_ro_gemma4_e4b"),
      ("C Olmo-3.1-32B-Instruct","lab36_ro_olmo31_32b_instruct"),("C Olmo-3.1-32B-Think","lab36_ro_olmo31_32b_think")]
def f(m,k):
    v=m.get(k,"") if m else ""
    if v in ("",None): return "—"
    try: return f"{float(v):.3f}"
    except: return str(v)
print("| Model | readout control AUC | readout sentinel AUC | null p95 | above null? | concept AUC | sweep d′@8× | verbalized sentinel d′ |")
print("|---|--:|--:|--:|:--:|--:|--:|--:|")
for label,run in RUNS:
    p=f"runs/{run}/metrics.json"
    if not os.path.exists(p): 
        print(f"| {label} | (pending) | | | | | | |"); continue
    m=json.load(open(p))
    print(f"| {label} | {f(m,'b5_readout_reportquery_control_auc')} | {f(m,'b5_readout_sentinel_auc')} | {f(m,'b5_readout_sentinel_null_p95_auc')} | {'yes' if str(m.get('b5_readout_sentinel_above_null'))=='1' else 'no'} | {f(m,'b5_readout_concept_auc')} | {f(m,'b5_sentinel_sweep_max_dose_d_prime')} | {f(m,'b5_sentinel_d_prime')} |")
