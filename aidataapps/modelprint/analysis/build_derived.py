#!/usr/bin/env python3
"""Build prompt-aware and likelihood representations from SQL-resident vectors."""
import argparse, hashlib, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from sklearn.linear_model import Ridge

parser = argparse.ArgumentParser()
parser.add_argument("--run")
parser.add_argument("--ridge-alpha", type=float, default=10.0)
args = parser.parse_args()
lab = Path(__file__).resolve().parents[1]
run = Path(args.run or lab / (lab / ".current-run").read_text().strip()).resolve()
freeze = json.loads((run / "manifests/campaign-freeze.json").read_text())
campaign = int(freeze["campaignId"]); run_id = run.name; models = freeze["campaign"]["profiles"]
robustness_path = run / "manifests/robustness-freeze.json"
robustness = int(json.loads(robustness_path.read_text())["campaignId"]) if robustness_path.exists() else None
campaign_ids = [campaign] + ([robustness] if robustness else []); campaign_sql = ",".join(map(str, campaign_ids))
conn = pymssql.connect(server=os.getenv("SQLSERVER_HOST", "127.0.0.1"), port=int(os.getenv("SQLSERVER_PORT", "1433")), user="sa",
                       password=os.environ["MSSQL_SA_PASSWORD"], database=os.getenv("MSSQL_DATABASE", "ModelPrint"),
                       autocommit=False, login_timeout=60, timeout=3600)

sql_text = f"""SELECT g.generation_id,g.campaign_id,g.prompt_variant_id,g.decode_config_id,g.split,v.carrier_id,JSON_VALUE(d.config_json,'$.key') decode_key,t.text_artifact_id,
 CAST(o.embedding AS nvarchar(max)) output_vector,CAST(p.embedding AS nvarchar(max)) prompt_vector
FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
JOIN dbo.semantic_vectors o ON o.text_artifact_id=t.text_artifact_id AND o.embedding_profile_id='qwen3-embedding-0.6b' AND o.representation_id='whole-raw-final-v1'
JOIN dbo.prompt_embeddings p ON p.prompt_variant_id=g.prompt_variant_id AND p.embedding_profile_id='qwen3-embedding-0.6b'
WHERE g.campaign_id IN ({campaign_sql}) AND g.truncated=0 ORDER BY g.generation_id"""
df = pd.read_sql(sql_text, conn)
if df.empty: raise SystemExit("No complete Qwen whole-output and prompt embeddings are available")
parse = lambda value: np.asarray(json.loads(value), dtype=np.float32)
output = np.stack(df.output_vector.map(parse)); prompt = np.stack(df.prompt_vector.map(parse))
normalize = lambda value: value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
train = ((df.campaign_id == campaign) & (df.split == "train") & (df.decode_key != "hv") & (df.carrier_id != "structured-v1")).to_numpy()

representations = {
    "residual-diff-v1": normalize(output - prompt),
}
prompt_unit = normalize(prompt)
representations["residual-reject-v1"] = normalize(output - np.sum(output * prompt_unit, axis=1, keepdims=True) * prompt_unit)

ridge = Ridge(alpha=args.ridge_alpha, fit_intercept=True, solver="cholesky")
ridge.fit(prompt[train], output[train])
ridge_raw = output - ridge.predict(prompt).astype(np.float32)
ridge_mean = ridge_raw[train].mean(axis=0, keepdims=True); ridge_sd = ridge_raw[train].std(axis=0, keepdims=True)
representations["residual-ridge-v1"] = normalize((ridge_raw - ridge_mean) / np.maximum(ridge_sd, 1e-6))

# Diagnostic oracle: subtract the four-model mean for exactly the same prompt and decode cell.
key = df.prompt_variant_id.astype(str) + "\0" + df.decode_config_id.astype(str)
centered = np.full_like(output, np.nan)
complete_center_groups = 0
for _, indices in pd.Series(np.arange(len(df))).groupby(key).groups.items():
    idx = np.asarray(list(indices), dtype=int)
    if len(idx) < len(models): continue
    centered[idx] = output[idx] - output[idx].mean(axis=0, keepdims=True); complete_center_groups += 1
center_mask = np.isfinite(centered).all(axis=1)
centered[center_mask] = normalize(centered[center_mask])

ridge_path = run / "manifests/residual-ridge-v1.npz"
np.savez_compressed(ridge_path, coef=ridge.coef_.astype(np.float32), intercept=ridge.intercept_.astype(np.float32),
                    residual_mean=ridge_mean.astype(np.float32), residual_sd=ridge_sd.astype(np.float32), alpha=np.asarray([args.ridge_alpha]))
ridge_hash = hashlib.sha256(ridge_path.read_bytes()).hexdigest()
manifest_hash = hashlib.sha256((freeze["campaignHash"] + ridge_hash).encode()).hexdigest()

cursor = conn.cursor()
def insert_residual(name, matrix, mask=None):
    selected = np.ones(len(df), dtype=bool) if mask is None else mask
    rows = [(int(gid), name, json.dumps(vector.tolist()), manifest_hash, run_id)
            for gid, vector in zip(df.generation_id[selected], matrix[selected])]
    for start in range(0, len(rows), 500):
        cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.residual_vectors WHERE generation_id=%s AND embedding_profile_id='qwen3-embedding-0.6b' AND representation_id=%s)
          INSERT dbo.residual_vectors(generation_id,embedding_profile_id,representation_id,embedding,training_manifest_hash,created_by_run)
          VALUES(%s,'qwen3-embedding-0.6b',%s,CAST(%s AS vector(1024)),%s,%s)""",
          [(row[0], row[1], *row) for row in rows[start:start+500]])
        conn.commit()
    return len(rows)

counts = {name: insert_residual(name, matrix) for name, matrix in representations.items()}
counts["prompt-centered-v1"] = insert_residual("prompt-centered-v1", centered, center_mask)

cursor.execute(f"""UPDATE l SET bits_per_ref_token=-l.ll_sum_nats/LOG(2.0)/NULLIF(g.reference_token_count,0)
FROM dbo.likelihood_scores l JOIN dbo.generations g ON g.generation_id=l.generation_id
WHERE g.campaign_id IN ({campaign_sql}) AND l.bits_per_ref_token IS NULL"""); conn.commit()
likelihood = pd.read_sql(f"""SELECT g.generation_id,g.campaign_id,g.split,l.scoring_model_profile_id,l.prompted,l.bits_per_char
FROM dbo.generations g JOIN dbo.likelihood_scores l ON l.generation_id=g.generation_id
WHERE g.campaign_id IN ({campaign_sql}) AND JSON_VALUE((SELECT config_json FROM dbo.decode_configs d WHERE d.decode_config_id=g.decode_config_id),'$.key') IN ('det','nat-0','nat-1')""", conn)
likelihood_rows = 0
if not likelihood.empty:
    likelihood["channel"] = likelihood.scoring_model_profile_id + likelihood.prompted.map({1: "-prompted", 0: "-unprompted"})
    expected = [f"{model}-prompted" for model in models] + [f"{model}-unprompted" for model in models]
    wide = likelihood.pivot_table(index=["generation_id", "campaign_id", "split"], columns="channel", values="bits_per_char", aggfunc="first").reset_index()
    if all(column in wide for column in expected):
        wide = wide.dropna(subset=expected); raw = wide[expected].to_numpy(dtype=np.float32); training = ((wide.campaign_id == campaign) & wide.split.eq("train")).to_numpy()
        mean = raw[training].mean(axis=0); sd = raw[training].std(axis=0); standardized = (raw - mean) / np.maximum(sd, 1e-6)
        profile_manifest = hashlib.sha256((freeze["campaignHash"] + json.dumps({"columns": expected, "mean": mean.tolist(), "sd": sd.tolist()}, sort_keys=True)).encode()).hexdigest()
        rows = [(int(gid), "likelihood-profile8-v1", json.dumps(vector.tolist()), profile_manifest)
                for gid, vector in zip(wide.generation_id, standardized)]
        for start in range(0, len(rows), 1000):
            cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.likelihood_profile_vectors WHERE generation_id=%s AND representation_id=%s)
              INSERT dbo.likelihood_profile_vectors(generation_id,representation_id,embedding,training_manifest_hash) VALUES(%s,%s,CAST(%s AS vector(8)),%s)""",
              [(row[0], row[1], *row) for row in rows[start:start+1000]])
            conn.commit()
        likelihood_rows = len(rows)
        (run / "manifests/likelihood-profile8-v1.json").write_text(json.dumps({"schemaVersion":1,"columns":expected,"mean":mean.tolist(),"sd":sd.tolist(),"trainingManifestHash":profile_manifest},indent=2)+"\n")

manifest = {"schemaVersion":1,"campaignIds":campaign_ids,"campaignHash":freeze["campaignHash"],"rows":len(df),"trainRows":int(train.sum()),
            "representations":counts,"completePromptCenteredGroups":complete_center_groups,"ridge":{"alpha":args.ridge_alpha,"artifact":str(ridge_path),"sha256":ridge_hash},
            "likelihoodProfileRows":likelihood_rows}
(run / "manifests/derived-features.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2)); conn.close()
