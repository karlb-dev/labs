#!/usr/bin/env python3
"""Fit frozen ModelPrint baselines and persist row-level predictions.

The script deliberately keeps retrieval and supervised decoding separate. It
never reports these probe predictions as SQL nearest-neighbor results.
"""
import argparse, hashlib, json, math, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support, top_k_accuracy_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260822
RNG = np.random.default_rng(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--run")
parser.add_argument("--tier", choices=["smoke", "dev", "standard", "full"], default="full")
parser.add_argument("--permutations", type=int, default=200)
parser.add_argument("--bootstrap", type=int, default=1000)
args = parser.parse_args()
lab = Path(__file__).resolve().parents[1]
run_dir = Path(args.run or (lab / (lab / ".current-run").read_text().strip())).resolve()
freeze = json.loads((run_dir / "manifests/campaign-freeze.json").read_text())
campaign_id = int(freeze["campaignId"])
run_id = run_dir.name
models = freeze["campaign"]["profiles"]
tier = json.loads((lab / f"data/manifests/tier-{args.tier}.json").read_text())
tier_variants = set(tier["variantIds"])

conn = pymssql.connect(server=os.getenv("SQLSERVER_HOST", "127.0.0.1"), port=int(os.getenv("SQLSERVER_PORT", "1433")), user="sa",
                       password=os.environ["MSSQL_SA_PASSWORD"], database=os.getenv("MSSQL_DATABASE", "ModelPrint"), autocommit=False,
                       login_timeout=60, timeout=3600)

base_sql = """SELECT g.generation_id,g.model_profile_id,g.prompt_variant_id,v.prompt_group_id,v.carrier_id,p.prompt_source_id source_id,p.family,p.domain,p.stratum,p.split,
 JSON_VALUE(d.config_json,'$.key') decode_key,g.reference_token_count,g.length_band,g.truncated,g.self_name_found,g.template_residue_found,g.final_text,
 t.text_artifact_id,CAST(s.embedding AS nvarchar(max)) style_vector
FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
LEFT JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1'
WHERE g.campaign_id=%s"""
df = pd.read_sql(base_sql, conn, params=(campaign_id,))
df = df[df.prompt_variant_id.isin(tier_variants) & df.model_profile_id.isin(models)].copy()
df = df[(df.truncated == 0) & (df.final_text.str.len() > 0)].copy()
if df.model_profile_id.nunique() < 2:
    raise SystemExit("Need at least two completed target profiles before evaluation")
label_to_id = {label: index for index, label in enumerate(models)}
df["label"] = df.model_profile_id.map(label_to_id)

def parse_vector(value):
    if value is None or (isinstance(value, float) and np.isnan(value)): return None
    return np.asarray(json.loads(value), dtype=np.float32)

representations = {}
if df.style_vector.notna().all(): representations["style512-v1"] = np.stack(df.style_vector.map(parse_vector))
for profile, name in [("qwen3-embedding-0.6b", "semantic1024-qwen-v1"), ("bge-large-en-v1.5", "semantic1024-bge-v1")]:
    sem = pd.read_sql("""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
    JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
    JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id=%s AND s.representation_id='whole-raw-final-v1'
    WHERE g.campaign_id=%s""", conn, params=(profile, campaign_id))
    mapping = dict(zip(sem.generation_id, sem.vector))
    if len(mapping) == len(df) and all(value in mapping for value in df.generation_id): representations[name] = np.stack([parse_vector(mapping[value]) for value in df.generation_id])

if not representations:
    raise SystemExit("No complete vector representation is available; run npm run features:build")

train = (df.split == "train").to_numpy(); calibration = (df.split == "calibration").to_numpy()
test_masks = {"test_id": (df.split == "test_id").to_numpy(), "test_source_holdout": (df.split == "test_source_holdout").to_numpy()}
y = df.label.to_numpy(dtype=int)

def softmax(logits, temperature=1.0):
    scaled = logits / max(temperature, 1e-4); scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled); return exp / exp.sum(axis=1, keepdims=True)

def temperature_fit(logits, labels):
    result = minimize_scalar(lambda log_t: log_loss(labels, softmax(logits, math.exp(log_t)), labels=np.arange(len(models))), bounds=(-4, 4), method="bounded")
    return float(math.exp(result.x))

def adaptive_ece(labels, probabilities, bins=10):
    confidence = probabilities.max(axis=1); predicted = probabilities.argmax(axis=1); order = np.argsort(confidence)
    parts = np.array_split(order, bins); return float(sum(len(part) / len(labels) * abs((predicted[part] == labels[part]).mean() - confidence[part].mean()) for part in parts if len(part)))

def bootstrap_interval(frame, predictions, replicates):
    groups = frame.prompt_group_id.unique(); scores = []
    group_indices = {group: np.flatnonzero(frame.prompt_group_id.to_numpy() == group) for group in groups}
    for _ in range(replicates):
        sampled = RNG.choice(groups, len(groups), replace=True); indices = np.concatenate([group_indices[group] for group in sampled])
        scores.append(f1_score(frame.label.to_numpy()[indices], predictions[indices], average="macro", labels=np.arange(len(models)), zero_division=0))
    return [float(np.quantile(scores, .025)), float(np.quantile(scores, .975))]

def permutation_null(frame, predictions, replicates):
    truth = frame.label.to_numpy(); groups = frame.prompt_group_id.to_numpy(); scores=[]
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    for _ in range(replicates):
        permuted = truth.copy()
        for indices in group_indices: permuted[indices] = RNG.permutation(permuted[indices])
        scores.append(f1_score(permuted, predictions, average="macro", labels=np.arange(len(models)), zero_division=0))
    return {"mean": float(np.mean(scores)), "p95": float(np.quantile(scores,.95)), "values": scores}

def suite_metrics(frame, probabilities, permutations=args.permutations):
    truth = frame.label.to_numpy(); prediction = probabilities.argmax(axis=1); confidence = probabilities.max(axis=1)
    bootstrap = bootstrap_interval(frame, prediction, args.bootstrap); null = permutation_null(frame, prediction, permutations)
    order = np.argsort(-confidence); half = order[:max(1, len(order)//2)]
    coverage85 = 0.0
    for fraction in np.linspace(.05, 1, 20):
        selected = order[:max(1, int(len(order)*fraction))]
        if accuracy_score(truth[selected], prediction[selected]) >= .85: coverage85 = float(fraction)
    classwise = precision_recall_fscore_support(truth,prediction,labels=np.arange(len(models)),zero_division=0)
    return {"rows":len(frame),"groups":int(frame.prompt_group_id.nunique()),"accuracy":float(accuracy_score(truth,prediction)),
      "balancedAccuracy":float(balanced_accuracy_score(truth,prediction)),"macroF1":float(f1_score(truth,prediction,average="macro",labels=np.arange(len(models)),zero_division=0)),
      "macroF1Ci95":bootstrap,"top2Accuracy":float(top_k_accuracy_score(truth,probabilities,k=min(2,len(models)),labels=np.arange(len(models)))),
      "nll":float(log_loss(truth,probabilities,labels=np.arange(len(models)))),"brier":float(np.mean(np.sum((probabilities-np.eye(len(models))[truth])**2,axis=1))),
      "eceAdaptive10":adaptive_ece(truth,probabilities),"selectiveAccuracyAt50Coverage":float(accuracy_score(truth[half],prediction[half])),
      "coverageAt85Accuracy":coverage85,"permutationNull":{"mean":null["mean"],"p95":null["p95"]},
      "perClass":{models[i]:{"precision":float(classwise[0][i]),"recall":float(classwise[1][i]),"f1":float(classwise[2][i]),"support":int(classwise[3][i])} for i in range(len(models))},
      "confusion":confusion_matrix(truth,prediction,labels=np.arange(len(models))).tolist(),"predictions":prediction,"confidence":confidence,"nullValues":null["values"]}

metrics={"schemaVersion":1,"runId":run_id,"campaignId":campaign_id,"campaignHash":freeze["campaignHash"],"tier":args.tier,"models":models,
         "rows":len(df),"dataRoles":df.split.value_counts().to_dict(),"representations":{},"seed":SEED,"permutations":args.permutations,"bootstrap":args.bootstrap}
prediction_frames=[]; fitted={}
for name, X in representations.items():
    pipeline = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED,C=1.0))
    pipeline.fit(X[train],y[train]); cal_logits=pipeline.decision_function(X[calibration]); temperature=temperature_fit(cal_logits,y[calibration])
    rep={"temperature":temperature,"suites":{}}
    for suite,mask in test_masks.items():
        if not mask.any(): continue
        probabilities=softmax(pipeline.decision_function(X[mask]),temperature); frame=df.loc[mask].reset_index(drop=True)
        result=suite_metrics(frame,probabilities); predictions=result.pop("predictions"); confidence=result.pop("confidence"); null_values=result.pop("nullValues")
        rep["suites"][suite]=result
        out=frame[["generation_id","model_profile_id","prompt_group_id","source_id","family","carrier_id","decode_key","length_band","split"]].copy()
        out["representation"]=name; out["suite"]=suite; out["predicted_model_profile_id"]=[models[index] for index in predictions]; out["confidence"]=confidence
        out["probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in probabilities]; prediction_frames.append(out)
        pd.DataFrame({"value":null_values}).to_csv(run_dir/f"tables/permutation-{name}-{suite}.csv",index=False)
        pd.DataFrame(result["confusion"],index=models,columns=models).to_csv(run_dir/f"tables/confusion-{name}-{suite}.csv")
    metrics["representations"][name]=rep; fitted[name]=pipeline

# Learned fingerprint floor: PCA-128 then LDA-3, padded to 64 and normalized.
if "semantic1024-qwen-v1" in representations and "style512-v1" in representations:
    combined=np.concatenate([representations["semantic1024-qwen-v1"],representations["style512-v1"]],axis=1)
    scaler=StandardScaler(); train_scaled=scaler.fit_transform(combined[train]); all_scaled=scaler.transform(combined)
    pca=PCA(n_components=min(128,train_scaled.shape[1],train_scaled.shape[0]-1),svd_solver="randomized",random_state=SEED)
    train_pca=pca.fit_transform(train_scaled); all_pca=pca.transform(all_scaled)
    lda=LinearDiscriminantAnalysis(n_components=min(3,len(models)-1)); lda.fit(train_pca,y[train]); projected=lda.transform(all_pca).astype(np.float32)
    fingerprint=np.zeros((len(df),64),dtype=np.float32); fingerprint[:,:projected.shape[1]]=projected
    norms=np.linalg.norm(fingerprint,axis=1,keepdims=True); fingerprint/=np.maximum(norms,1e-12)
    pipeline=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=SEED).fit(fingerprint[train],y[train]); temperature=temperature_fit(pipeline.decision_function(fingerprint[calibration]),y[calibration])
    rep={"temperature":temperature,"projection":"pca128-lda3-padded64","suites":{}}
    for suite,mask in test_masks.items():
        if not mask.any(): continue
        probabilities=softmax(pipeline.decision_function(fingerprint[mask]),temperature); frame=df.loc[mask].reset_index(drop=True); result=suite_metrics(frame,probabilities)
        predictions=result.pop("predictions"); confidence=result.pop("confidence"); null_values=result.pop("nullValues"); rep["suites"][suite]=result
        out=frame[["generation_id","model_profile_id","prompt_group_id","source_id","family","carrier_id","decode_key","length_band","split"]].copy()
        out["representation"]="fingerprint64-v1";out["suite"]=suite;out["predicted_model_profile_id"]=[models[index] for index in predictions];out["confidence"]=confidence
        out["probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in probabilities];prediction_frames.append(out)
        pd.DataFrame({"value":null_values}).to_csv(run_dir/f"tables/permutation-fingerprint64-v1-{suite}.csv",index=False)
    metrics["representations"]["fingerprint64-v1"]=rep; representations["fingerprint64-v1"]=fingerprint; fitted["fingerprint64-v1"]=pipeline
    projection_path=run_dir/"manifests/fingerprint64-projection.pkl";projection_path.write_bytes(pickle.dumps({"scaler":scaler,"pca":pca,"lda":lda}))
    projection_hash=hashlib.sha256(projection_path.read_bytes()).hexdigest()
    cursor=conn.cursor()
    corpus_hash=hashlib.sha256((freeze["campaignHash"]+args.tier+projection_hash).encode()).hexdigest()
    rows_to_insert=[(int(gid),"fingerprint64-v1",projection_hash,json.dumps(vector.tolist()),run_id) for gid,vector in zip(df.generation_id,fingerprint)]
    for chunk_start in range(0,len(rows_to_insert),1000):
        cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.fingerprint_vectors WHERE generation_id=%s AND representation_id=%s)
        INSERT dbo.fingerprint_vectors(generation_id,representation_id,projection_manifest_hash,embedding,created_by_run) VALUES(%s,%s,%s,CAST(%s AS vector(64)),%s)""",
        [(r[0],r[1],*r) for r in rows_to_insert[chunk_start:chunk_start+1000]])
    conn.commit()
    metrics["fingerprintProjection"]={"path":str(projection_path),"sha256":projection_hash,"corpusManifestHash":corpus_hash}

predictions=pd.concat(prediction_frames,ignore_index=True) if prediction_frames else pd.DataFrame()
predictions.to_parquet(run_dir/"tables/predictions.parquet",index=False)
predictions.to_csv(run_dir/"tables/predictions.csv",index=False)

# Persist model metadata and row predictions; artifacts remain rebuildable from parquet.
cursor=conn.cursor()
artifact={"metricsPath":"metrics/attribution.json","tier":args.tier,"representations":list(metrics["representations"]),"models":models}
cursor.execute("INSERT dbo.attribution_models(run_id,model_kind,training_manifest_hash,artifact_json) VALUES(%s,%s,%s,%s); SELECT SCOPE_IDENTITY()",
               (run_id,"multinomial-probes",freeze["campaignHash"],json.dumps(artifact)))
attribution_id=int(cursor.fetchone()[0])
cursor.execute("INSERT dbo.calibration_models(attribution_model_id,method,calibration_manifest_hash,artifact_json) VALUES(%s,%s,%s,%s)",
               (attribution_id,"temperature-scaling",freeze["campaignHash"],json.dumps({k:v["temperature"] for k,v in metrics["representations"].items()})))
for (representation,suite),frame in predictions.groupby(["representation","suite"]):
    cursor.execute("INSERT dbo.prediction_runs(run_id,suite,representation_id,method,config_json) VALUES(%s,%s,%s,%s,%s); SELECT SCOPE_IDENTITY()",
                   (run_id,suite,representation,"linear-probe",json.dumps({"tier":args.tier,"calibrated":True})))
    prediction_run_id=int(cursor.fetchone()[0]); records=[]
    for row in frame.itertuples():
        probabilities=json.loads(row.probabilities); candidate=[key for key,value in probabilities.items() if value>=.1]
        decision="attributed" if row.confidence>=.5 else "ambiguous"
        records.append((prediction_run_id,int(row.generation_id),row.model_profile_id,row.predicted_model_profile_id,decision,float(row.confidence),json.dumps(candidate),1-float(row.confidence)))
    cursor.executemany("INSERT dbo.predictions(prediction_run_id,generation_id,true_model_profile_id,predicted_model_profile_id,decision,calibrated_probability,candidate_set_json,novelty_score) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",records)
conn.commit()

(run_dir/"metrics/attribution.json").write_text(json.dumps(metrics,indent=2,default=lambda value: int(value) if isinstance(value,np.integer) else float(value)))
summary={"runId":run_id,"campaignId":campaign_id,"tier":args.tier,"rows":len(df),"profiles":int(df.model_profile_id.nunique()),"representations":list(metrics["representations"]),"predictionRows":len(predictions)}
print(json.dumps(summary,indent=2))
conn.close()
