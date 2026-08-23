#!/usr/bin/env python3
"""Calibrate same-served-profile verification and known-k batch grouping."""
import argparse, json, math, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from scipy.optimize import brentq
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=20260822
parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--bootstrap",type=int,default=1000);parser.add_argument("--permutations",type=int,default=200);args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
campaign=int(freeze["campaignId"]);run_id=run.name;conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",
 password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=3600)
rows=pd.read_sql("""SELECT g.generation_id,g.model_profile_id,g.prompt_variant_id,v.prompt_group_id,v.carrier_id,p.domain,p.split,JSON_VALUE(d.config_json,'$.key') decode_key,
 g.reference_token_count,g.length_band FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
 JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id=%s AND g.truncated=0 AND g.reference_token_count>=16 AND v.carrier_id<>'structured-v1'
 AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')""",conn,params=(campaign,)).sort_values("generation_id").reset_index(drop=True)
def parse(value):return np.asarray(json.loads(value),dtype=np.float32)
def vectors(query):
 data=pd.read_sql(query,conn);return {int(row.generation_id):parse(row.vector) for row in data.itertuples()}
spaces={
 "semantic":vectors(f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
 JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id
 AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1' WHERE g.campaign_id={campaign}"""),
 "style":vectors(f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
 JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE g.campaign_id={campaign}"""),
}
for name in ["residual-ridge-v1","fingerprint64-v1"]:
 table="residual_vectors" if name.startswith("residual") else "fingerprint_vectors";mapping=vectors(f"SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.{table} WHERE representation_id='{name}'")
 if mapping:spaces[name]=mapping
eligible=set.intersection(*[set(mapping) for mapping in spaces.values()]);rows=rows[rows.generation_id.isin(eligible)].reset_index(drop=True);metadata=rows.set_index("generation_id")
def stable_sample(values,n,seed):
 rng=np.random.default_rng(seed);values=list(values);return [values[index] for index in rng.choice(len(values),min(n,len(values)),replace=False)] if values else []
def construct(frame,target,seed):
 rng=np.random.default_rng(seed);records={key:[] for key in ["positive-easy","positive-hard","negative-hard","negative-easy","negative-cross-domain"]}
 # Same prompt variant, same profile, different natural seed.
 for _,part in frame.groupby(["model_profile_id","prompt_variant_id"]):
  left=part[part.decode_key=="nat-0"];right=part[part.decode_key=="nat-1"]
  if len(left) and len(right):records["positive-easy"].append((int(left.iloc[0].generation_id),int(right.iloc[0].generation_id)))
 # Same rendered prompt/decode, different profiles.
 for _,part in frame.groupby(["prompt_variant_id","decode_key"]):
  ids=part.sort_values("model_profile_id").generation_id.tolist()
  for i in range(len(ids)):
   for j in range(i+1,len(ids)):records["negative-hard"].append((int(ids[i]),int(ids[j])))
 by_model={key:value for key,value in frame.groupby("model_profile_id")};by_carrier_length={key:value for key,value in frame.groupby(["carrier_id","length_band"])}
 attempts=0
 while attempts<target*30 and any(len(records[key])<target for key in ["positive-hard","negative-easy","negative-cross-domain"]):
  attempts+=1;left=frame.iloc[int(rng.integers(len(frame)))]
  same=by_model[left.model_profile_id];candidates=same[(same.prompt_group_id!=left.prompt_group_id)&(same.domain!=left.domain)]
  if len(candidates):right=candidates.iloc[int(rng.integers(len(candidates)))];records["positive-hard"].append((int(left.generation_id),int(right.generation_id)))
  different=frame[(frame.model_profile_id!=left.model_profile_id)&(frame.prompt_group_id!=left.prompt_group_id)]
  if len(different):right=different.iloc[int(rng.integers(len(different)))];records["negative-easy"].append((int(left.generation_id),int(right.generation_id)))
  matched=by_carrier_length.get((left.carrier_id,left.length_band),frame.iloc[0:0]);matched=matched[(matched.model_profile_id!=left.model_profile_id)&(matched.prompt_group_id!=left.prompt_group_id)&(matched.domain!=left.domain)]
  if len(matched):right=matched.iloc[int(rng.integers(len(matched)))];records["negative-cross-domain"].append((int(left.generation_id),int(right.generation_id)))
 output=[]
 for cell,pairs in records.items():
  unique=sorted({tuple(sorted(pair)) for pair in pairs});selected=stable_sample(unique,target,seed+sum(map(ord,cell)))
  output.extend({"left":left,"right":right,"pair_cell":cell,"same_source":int(cell.startswith("positive"))} for left,right in selected)
 return pd.DataFrame(output)
targets={"train":5000,"calibration":1500,"test_id":2000,"test_source_holdout":1500};pairs=[]
for index,(split,target) in enumerate(targets.items()):
 frame=rows[rows.split==split];part=construct(frame,target,SEED+index);part["split"]=split;pairs.append(part)
pairs=pd.concat(pairs,ignore_index=True).sort_values(["split","pair_cell","left","right"]).drop_duplicates(["split","left","right"]).reset_index(drop=True)
def cosine(left,right):
 return float(1-left@right/max(np.linalg.norm(left)*np.linalg.norm(right),1e-12))
feature_names=[f"{name}_distance" for name in spaces]+["length_difference","carrier_match","decode_match"]
def feature(left_id,right_id):
 left=metadata.loc[left_id];right=metadata.loc[right_id];return [cosine(mapping[left_id],mapping[right_id]) for mapping in spaces.values()]+[
  abs(float(left.reference_token_count)-float(right.reference_token_count)),int(left.carrier_id==right.carrier_id),int(left.decode_key==right.decode_key)]
X=np.asarray([feature(row.left,row.right) for row in pairs.itertuples()],dtype=np.float32);y=pairs.same_source.to_numpy();train=pairs.split.eq("train").to_numpy();cal=pairs.split.eq("calibration").to_numpy();test=pairs.split.isin(["test_id","test_source_holdout"]).to_numpy()
model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,class_weight="balanced",random_state=SEED));model.fit(X[train],y[train]);cal_logit=model.decision_function(X[cal])
platt=LogisticRegression(C=1e6).fit(cal_logit.reshape(-1,1),y[cal]);prob=platt.predict_proba(model.decision_function(X).reshape(-1,1))[:,1]
def eer(truth,score):
 fpr,tpr,_=roc_curve(truth,score);fnr=1-tpr;index=np.nanargmin(np.abs(fpr-fnr));return float((fpr[index]+fnr[index])/2)
def score(mask):
 truth=y[mask];value=prob[mask];return {"rows":int(mask.sum()),"auroc":float(roc_auc_score(truth,value)),"auprc":float(average_precision_score(truth,value)),"brier":float(brier_score_loss(truth,value)),"eer":eer(truth,value),"accuracy":float(((value>=.5)==truth).mean())}
hard=test&pairs.pair_cell.isin(["positive-hard","negative-hard"]).to_numpy();metrics={"overall":score(test),"hard":score(hard),"cells":{}}
for cell,indices in pairs[test].groupby("pair_cell").groups.items():
 mask=np.zeros(len(pairs),dtype=bool);mask[np.asarray(list(indices),dtype=int)]=True;metrics["cells"][cell]={"rows":int(mask.sum()),"meanProbability":float(prob[mask].mean()),"accuracy":float(((prob[mask]>=.5)==y[mask]).mean())}
rng=np.random.default_rng(SEED);hard_indices=np.flatnonzero(hard);hard_groups=(pairs.left.astype(str)+":"+pairs.right.astype(str)).to_numpy();group_map={key:np.flatnonzero(hard_groups[hard_indices]==key) for key in np.unique(hard_groups[hard_indices])};group_keys=list(group_map);boot=[]
for _ in range(args.bootstrap):
 local=np.concatenate([group_map[key] for key in rng.choice(group_keys,len(group_keys),replace=True)]);idx=hard_indices[local]
 if len(np.unique(y[idx]))==2:boot.append(roc_auc_score(y[idx],prob[idx]))
metrics["hard"]["aurocCi95"]=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]

# Full label-permutation control: pair labels are re-derived after within-prompt-group profile shuffling, then every fit/calibration stage is repeated.
endpoint_ids=np.unique(np.concatenate([pairs.left,pairs.right]));endpoint_meta=metadata.loc[endpoint_ids][["model_profile_id","prompt_group_id"]];id_position={int(value):index for index,value in enumerate(endpoint_ids)}
group_indices=[np.asarray(values,dtype=int) for values in endpoint_meta.reset_index().groupby("prompt_group_id").indices.values()];base_labels=endpoint_meta.model_profile_id.to_numpy();left_pos=pairs.left.map(id_position).to_numpy();right_pos=pairs.right.map(id_position).to_numpy();null=[]
for repeat in range(args.permutations):
 shuffled=base_labels.copy()
 for idx in group_indices:values=shuffled[idx].copy();rng.shuffle(values);shuffled[idx]=values
 perm=(shuffled[left_pos]==shuffled[right_pos]).astype(int)
 if len(np.unique(perm[train]))<2 or len(np.unique(perm[cal]))<2 or len(np.unique(perm[hard]))<2:continue
 candidate=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=300,class_weight="balanced",random_state=SEED+repeat));candidate.fit(X[train],perm[train]);calibrator=LogisticRegression(C=1e6).fit(candidate.decision_function(X[cal]).reshape(-1,1),perm[cal]);value=calibrator.predict_proba(candidate.decision_function(X[hard]).reshape(-1,1))[:,1];null.append(roc_auc_score(perm[hard],value))
metrics["hard"]["permutationNull"]={"replicates":len(null),"mean":float(np.mean(null)),"p95":float(np.quantile(null,.95))};base_brier=float(y[hard].mean()*(1-y[hard].mean()))
supported=metrics["hard"]["aurocCi95"][0]>.70 and metrics["hard"]["brier"]<base_brier;metrics["taxonomy"]="SAME_SOURCE_SIGNAL" if supported else "NO_SUPPORTED_SIGNAL";metrics["baseRateBrier"]=base_brier
cursor=conn.cursor();cursor.execute("INSERT dbo.pair_runs(run_id,config_json) VALUES(%s,%s); SELECT SCOPE_IDENTITY()",(run_id,json.dumps({"features":feature_names,"calibration":"platt","threshold":.5})));pair_run=int(cursor.fetchone()[0])
records=[]
for index,row in pairs[test].iterrows():records.append((pair_run,int(row.left),int(row.right),row.pair_cell,int(y[index]),float(prob[index]),int(prob[index]>=.5),json.dumps(dict(zip(feature_names,map(float,X[index]))))))
cursor.executemany("INSERT dbo.pair_predictions(pair_run_id,left_generation_id,right_generation_id,pair_cell,same_source_label,probability,predicted_same_source,features_json) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",records)
cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
 VALUES(%s,'CAL','hybrid-pair-features','logistic-platt','hard-pairs','auroc',%s,%s,%s,%s,%s,%s,NULL,%s); SELECT SCOPE_IDENTITY()""",(run_id,metrics["hard"]["auroc"],*metrics["hard"]["aurocCi95"],metrics["hard"]["permutationNull"]["mean"],metrics["hard"]["permutationNull"]["p95"],metrics["hard"]["rows"],json.dumps(metrics)));metric_id=int(cursor.fetchone()[0])
cursor.execute("INSERT dbo.claims(run_id,research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale,metric_result_id) VALUES(%s,'RQ8','CAL','hybrid-pair-features','logistic-platt','hard-pairs',%s,%s,%s,%s)",
 (run_id,metrics["taxonomy"],supported,f"Hard-pair AUROC lower bound {metrics['hard']['aurocCi95'][0]:.4f}; Brier {metrics['hard']['brier']:.4f} versus base {base_brier:.4f}.",metric_id));conn.commit()

# Known-k grouping diagnostic: 20 held-out outputs, five per profile, average linkage on calibrated pair dissimilarity.
test_rows=rows[rows.split=="test_id"];group_scores=[];group_run_records=[]
for batch in range(30):
 selected=pd.concat([part.sample(n=min(5,len(part)),random_state=SEED+batch) for _,part in test_rows.groupby("model_profile_id")],ignore_index=True);n=len(selected);matrix=np.zeros((n,n))
 for left in range(n):
  for right in range(left+1,n):
   value=np.asarray(feature(int(selected.iloc[left].generation_id),int(selected.iloc[right].generation_id)),dtype=np.float32);p=float(platt.predict_proba(model.decision_function(value.reshape(1,-1)).reshape(-1,1))[0,1]);matrix[left,right]=matrix[right,left]=1-p
 labels=AgglomerativeClustering(n_clusters=len(selected.model_profile_id.unique()),metric="precomputed",linkage="average").fit_predict(matrix);truth=pd.factorize(selected.model_profile_id)[0];ari=float(adjusted_rand_score(truth,labels));group_scores.append(ari)
 cursor.execute("INSERT dbo.group_runs(run_id,config_json) VALUES(%s,%s); SELECT SCOPE_IDENTITY()",(run_id,json.dumps({"batch":batch,"method":"average-linkage-known-k","items":n})));group_run=int(cursor.fetchone()[0])
 cursor.executemany("INSERT dbo.group_assignments(group_run_id,generation_id,source_group,ambiguous,diagnostics_json) VALUES(%s,%s,%s,0,%s)",[(group_run,int(gid),int(label),json.dumps({"batchAri":ari})) for gid,label in zip(selected.generation_id,labels)]);conn.commit()
metrics["groupingKnownK"]={"batches":len(group_scores),"meanAri":float(np.mean(group_scores)),"p05Ari":float(np.quantile(group_scores,.05))}
pairs.assign(probability=prob).to_parquet(run/"tables/pairs.parquet",index=False);pd.DataFrame({"value":null}).to_csv(run/"tables/permutation-pairs.csv",index=False)
(run/"metrics/pairs.json").write_text(json.dumps(metrics,indent=2)+"\n");(run/"manifests/pair-model.pkl").write_bytes(pickle.dumps({"model":model,"platt":platt,"features":feature_names,"threshold":.5}))
(run/"reports/SAME_SOURCE_REPORT.md").write_text(f"# Same-source Verification\n\nDisposition: `{metrics['taxonomy']}`. Hard-pair AUROC `{metrics['hard']['auroc']:.4f}` (95% grouped bootstrap `{metrics['hard']['aurocCi95'][0]:.4f}`–`{metrics['hard']['aurocCi95'][1]:.4f}`); Brier `{metrics['hard']['brier']:.4f}` versus base-rate `{base_brier:.4f}`.\n")
print(json.dumps({"runId":run_id,"pairRunId":pair_run,"pairs":len(pairs),"metrics":metrics},indent=2));conn.close()
