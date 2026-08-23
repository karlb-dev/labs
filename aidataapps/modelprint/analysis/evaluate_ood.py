#!/usr/bin/env python3
"""Calibrate OOD abstention and evaluate human and mixed-source controls."""
import argparse, json, math, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from sklearn.metrics import accuracy_score, balanced_accuracy_score

parser=argparse.ArgumentParser();parser.add_argument("--run");args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve()
freeze=json.loads((run/"manifests/campaign-freeze.json").read_text());campaign=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"];run_id=run.name
conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",password=os.environ["MSSQL_SA_PASSWORD"],
 database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=3600)
items=pd.read_sql("SELECT evaluation_item_id,source_id,source_row_id,prompt_group_id,split_role,item_text,metadata_json FROM dbo.evaluation_items WHERE run_id=%s ORDER BY evaluation_item_id",conn,params=(run_id,))
if items.empty:raise SystemExit("No evaluation controls; run npm run controls:build")
items["metadata"]=items.metadata_json.map(json.loads);item_position={int(value):index for index,value in enumerate(items.evaluation_item_id)}
meta=pd.read_sql("SELECT generation_id,model_profile_id,prompt_group_id,split FROM dbo.generations WHERE campaign_id=%s AND truncated=0 AND reference_token_count>=16",conn,params=(campaign,))

def parse(value):return np.asarray(json.loads(value),dtype=np.float32)
def softmax(logits,temp):
 values=logits/max(float(temp),1e-4);values-=values.max(axis=1,keepdims=True);values=np.exp(values);return values/values.sum(axis=1,keepdims=True)
def normalized(values):
 values=np.asarray(values,dtype=np.float32);return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-12)
def percentile(reference,values):
 ordered=np.sort(np.asarray(reference));return np.searchsorted(ordered,np.asarray(values),side="right")/max(len(ordered),1)

definitions={
 "semantic1024-qwen-raw-final-v1":("semantic1024-qwen-v1",f"""SELECT g.generation_id,g.model_profile_id,g.split,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
  JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
  JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1' WHERE g.campaign_id={campaign}"""),
 "semantic1024-bge-raw-final-v1":("semantic1024-bge-v1",f"""SELECT g.generation_id,g.model_profile_id,g.split,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
  JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
  JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='bge-large-en-v1.5' AND s.representation_id='whole-raw-final-v1' WHERE g.campaign_id={campaign}"""),
 "style512-raw-final-v1":("style512-v1",f"""SELECT g.generation_id,g.model_profile_id,g.split,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
  JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
  JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE g.campaign_id={campaign}"""),
 "fingerprint64-v1":(None,f"SELECT g.generation_id,g.model_profile_id,g.split,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.fingerprint_vectors s ON s.generation_id=g.generation_id AND s.representation_id='fingerprint64-v1' WHERE g.campaign_id={campaign}"),
}
eval_vectors=pd.read_sql("SELECT v.evaluation_item_id,v.representation_id,v.vector_json FROM dbo.evaluation_vectors v JOIN dbo.evaluation_items e ON e.evaluation_item_id=v.evaluation_item_id WHERE e.run_id=%s",conn,params=(run_id,))
eval_map={(int(row.evaluation_item_id),row.representation_id):parse(row.vector_json) for row in eval_vectors.itertuples()}
projection_path=run/"manifests/fingerprint64-projection.pkl"
if projection_path.exists():
 projection=pickle.loads(projection_path.read_bytes())
 for row in items.itertuples():
  q=eval_map.get((int(row.evaluation_item_id),"semantic1024-qwen-v1"));s=eval_map.get((int(row.evaluation_item_id),"style512-v1"))
  if q is None or s is None:continue
  values=projection["lda"].transform(projection["pca"].transform(projection["scaler"].transform(np.concatenate([q,s]).reshape(1,-1))))[0]
  vector=np.zeros(64,dtype=np.float32);vector[:len(values)]=values;vector/=max(np.linalg.norm(vector),1e-12);eval_map[(int(row.evaluation_item_id),"fingerprint64-v1")]=vector

channels={};calibration_ids=None
for representation,(evaluation_representation,query) in definitions.items():
 model_path=run/f"manifests/probe-{representation}.pkl"
 if not model_path.exists():continue
 model_info=pickle.loads(model_path.read_bytes());known=pd.read_sql(query,conn).drop_duplicates("generation_id");train_rows=known[known.split=="train"];cal_rows=known[known.split=="calibration"]
 if train_rows.empty or cal_rows.empty:continue
 train_matrix=np.stack(train_rows.vector.map(parse));cal_matrix=np.stack(cal_rows.vector.map(parse));cal_prob=softmax(model_info["pipeline"].decision_function(cal_matrix),model_info["temperature"])
 # Support centroids are always fit from training rows, then applied to calibration/evaluation rows.
 train_unit=normalized(train_matrix);centroids=[]
 for model in models:
  center=train_unit[train_rows.model_profile_id.to_numpy()==model].mean(0);center/=max(np.linalg.norm(center),1e-12);centroids.append(center)
 centroids=np.stack(centroids);cal_support=1-np.max(normalized(cal_matrix)@centroids.T,axis=1)
 ids=[];vectors=[]
 for row in items.itertuples():
  vector=eval_map.get((int(row.evaluation_item_id),evaluation_representation or representation))
  if vector is not None:ids.append(int(row.evaluation_item_id));vectors.append(vector)
 if len(ids)!=len(items):continue
 matrix=np.stack(vectors);prob=softmax(model_info["pipeline"].decision_function(matrix),model_info["temperature"]);support=1-np.max(normalized(matrix)@centroids.T,axis=1)
 channels[representation]={"prob":prob,"support":support,"calProb":cal_prob,"calSupport":cal_support,"calIds":cal_rows.generation_id.to_numpy(),"q":float(model_info["conformalQ"])}

if len(channels)<2:raise SystemExit(f"Need at least two complete OOD channels; found {list(channels)}")
calibration_ids=sorted(set.intersection(*[set(map(int,value["calIds"])) for value in channels.values()]));cal_position={name:{int(value):index for index,value in enumerate(channel["calIds"])} for name,channel in channels.items()}
eval_prob=np.stack([value["prob"] for value in channels.values()]);hybrid=np.exp(np.mean(np.log(np.clip(eval_prob,1e-9,1)),axis=0));hybrid/=hybrid.sum(1,keepdims=True)
cal_prob=np.stack([value["calProb"][[cal_position[name][value] for value in calibration_ids]] for name,value in channels.items()]);cal_hybrid=np.exp(np.mean(np.log(np.clip(cal_prob,1e-9,1)),axis=0));cal_hybrid/=cal_hybrid.sum(1,keepdims=True)
entropy=lambda p:-np.sum(p*np.log(np.clip(p,1e-9,1)),axis=1)/math.log(len(models))
eval_labels=eval_prob.argmax(2);eval_disagreement=1-np.max(np.stack([(eval_labels==model_index).mean(0) for model_index in range(len(models))]),axis=0)
cal_disagreement=1-np.max(np.stack([(cal_prob.argmax(2)==model_index).mean(0) for model_index in range(len(models))]),axis=0)
eval_support=np.mean(np.stack([percentile(value["calSupport"],value["support"]) for value in channels.values()]),axis=0)
cal_support=np.mean(np.stack([percentile(value["calSupport"],value["calSupport"][[cal_position[name][identifier] for identifier in calibration_ids]]) for name,value in channels.items()]),axis=0)
q=max(value["q"] for value in channels.values());eval_set=(hybrid>=(1-q)).sum(1)/len(models);cal_set=(cal_hybrid>=(1-q)).sum(1)/len(models)
eval_score=np.mean(np.stack([percentile(entropy(cal_hybrid),entropy(hybrid)),eval_support,eval_disagreement,eval_set]),axis=0)
cal_score=np.mean(np.stack([percentile(entropy(cal_hybrid),entropy(cal_hybrid)),cal_support,cal_disagreement,cal_set]),axis=0)
dev_mask=items.split_role.eq("ood-development").to_numpy();dev_scores=eval_score[dev_mask]
if not len(dev_scores):raise SystemExit("OOD development controls are empty")
truth=np.concatenate([np.zeros(len(cal_score),dtype=int),np.ones(len(dev_scores),dtype=int)]);scores=np.concatenate([cal_score,dev_scores]);candidates=np.unique(scores);best=None
for threshold in candidates:
 prediction=(scores>threshold).astype(int);candidate=(balanced_accuracy_score(truth,prediction),-float(threshold))
 if best is None or candidate>best[0]:best=(candidate,float(threshold))
threshold=best[1];unknown=eval_score>threshold;prediction=hybrid.argmax(1);confidence=hybrid.max(1)
output=items[["evaluation_item_id","source_id","source_row_id","prompt_group_id","split_role"]].copy();output["novelty_score"]=eval_score;output["unknown"]=unknown
output["predicted_model_profile_id"]=[models[value] for value in prediction];output["confidence"]=confidence;output["probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in hybrid]
output.to_parquet(run/"tables/ood-predictions.parquet",index=False);output.to_csv(run/"tables/ood-predictions.csv",index=False)

source_rows=[]
for (role,source),frame in output.groupby(["split_role","source_id"]):
 source_rows.append({"split_role":role,"source":source,"rows":len(frame),"unknown_rejection_rate":float(frame.unknown.mean()),"unknown_acceptance_rate":float(1-frame.unknown.mean()),
  "mean_novelty":float(frame.novelty_score.mean()),"mean_confidence_if_accepted":float(frame.loc[~frame.unknown,"confidence"].mean()) if (~frame.unknown).any() else None})
source_summary=pd.DataFrame(source_rows);source_summary.to_csv(run/"tables/ood-by-source.csv",index=False)

# Paragraph-level mixed-source localization uses only paragraph items. Each true
# boundary is a source change by construction, so boundary recall measures how
# often consecutive attributed paragraphs also change label.
paragraph=output[output.source_id=="mixed-source-paragraph"].copy();paragraph["parent"]=[value.get("parentId") for value in items.loc[paragraph.index,"metadata"]]
paragraph["ordinal"]=[int(value.get("ordinal",0)) for value in items.loc[paragraph.index,"metadata"]];paragraph["true_model"]=[value.get("sourceModelProfileId") for value in items.loc[paragraph.index,"metadata"]]
accepted=paragraph[~paragraph.unknown];paragraph_accuracy=float((accepted.predicted_model_profile_id==accepted.true_model).mean()) if len(accepted) else 0.0
boundaries=[];multiple=[]
for parent,frame in paragraph.sort_values(["parent","ordinal"]).groupby("parent"):
 values=frame.predicted_model_profile_id.to_numpy();valid=(~frame.unknown).to_numpy();changes=[bool(valid[i] and valid[i+1] and values[i]!=values[i+1]) for i in range(len(values)-1)];boundaries.extend(changes);multiple.append(len(set(values[valid]))>1 if valid.any() else False)
boundary_recall=float(np.mean(boundaries)) if boundaries else 0.0;boundary_f1=2*boundary_recall/(1+boundary_recall) if boundary_recall else 0.0
mixed_metrics={"paragraphRows":len(paragraph),"acceptedParagraphRows":len(accepted),"paragraphAccuracyAccepted":paragraph_accuracy,"boundaryRecall":boundary_recall,
 "boundaryF1":boundary_f1,"multipleSourcesDetectedRate":float(np.mean(multiple)) if multiple else 0.0,"documents":len(multiple)}
(run/"tables/mixed-source-localization.json").write_text(json.dumps(mixed_metrics,indent=2)+"\n")

human=source_summary[source_summary.source=="human-controls"];human_accept=float(human.iloc[0].unknown_acceptance_rate) if len(human) else 1.0
taxonomy="OOD_UNRELIABLE" if human_accept>.20 else "SELECTIVE_ATTRIBUTION"
metrics={"schemaVersion":1,"runId":run_id,"channels":list(channels),"noveltyFeatures":["hybrid_entropy_percentile","training_centroid_support_percentile","channel_disagreement","conformal_set_fraction"],
 "omittedNoveltyFeatures":["chunk_disagreement","phrase_evidence_scarcity"],"threshold":threshold,"developmentBalancedAccuracy":best[0][0],"calibrationFalseUnknownRate":float((cal_score>threshold).mean()),
 "sources":source_rows,"mixedSourceLocalization":mixed_metrics,"taxonomy":taxonomy,"missingFrozenSources":[source for source in ["qwen-smoke","qwen-3.6-27b-pinned"] if source not in set(items.source_id)]}
(run/"metrics/ood.json").write_text(json.dumps(metrics,indent=2)+"\n")
cursor=conn.cursor();cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,rows_count,groups_count,detail_json)
 VALUES(%s,'CAL','hybrid-text','novelty-threshold','human-controls','unknown_acceptance_rate',%s,%s,NULL,%s); SELECT SCOPE_IDENTITY()""",(run_id,human_accept,int(human.iloc[0].rows) if len(human) else 0,json.dumps(metrics)));metric_id=int(cursor.fetchone()[0])
cursor.execute("INSERT dbo.claims(run_id,research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale,metric_result_id) VALUES(%s,'RQ5','CAL','hybrid-text','novelty-threshold','human-controls',%s,%s,%s,%s)",
 (run_id,taxonomy,taxonomy!="OOD_UNRELIABLE",f"Human-control unknown acceptance {human_accept:.4f}; development balanced accuracy {best[0][0]:.4f}.",metric_id));conn.commit()
(run/"reports/OOD_REPORT.md").write_text(f"# Out-of-distribution Abstention\n\nDisposition: `{taxonomy}`. The threshold was frozen on calibration rows plus transformed development controls. Human-control unknown acceptance was `{human_accept:.4f}`.\n\n"
 "Qwen smoke and Qwen 3.6 rows are reported only if their retained evaluation controls are present; missing sources are explicit in `metrics/ood.json`. Human-versus-model comparisons here are descriptive and are not a detector claim.\n")
print(json.dumps({"runId":run_id,"channels":list(channels),"threshold":threshold,"sources":source_rows,"mixed":mixed_metrics,"taxonomy":taxonomy},indent=2));conn.close()
