#!/usr/bin/env python3
"""Evaluate exact SQL segment retrieval and generation-level chunk voting."""
import argparse, hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from sklearn.metrics import accuracy_score, f1_score

SEED=20260822
parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--k",type=int,default=10);parser.add_argument("--candidate-k",type=int,default=100)
parser.add_argument("--workers",type=int,default=24);parser.add_argument("--max-per-suite",type=int,default=500);parser.add_argument("--bootstrap",type=int,default=1000);parser.add_argument("--permutations",type=int,default=200);args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
primary=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"];run_id=run.name;robust_path=run/"manifests/robustness-freeze.json";robust=int(json.loads(robust_path.read_text())["campaignId"]) if robust_path.exists() else None
campaign_ids=[primary]+([robust] if robust else []);campaign_sql=",".join(map(str,campaign_ids));db=dict(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",
 password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),login_timeout=60,timeout=3600);conn=pymssql.connect(**db,autocommit=False)
meta=pd.read_sql(f"""SELECT g.generation_id,g.campaign_id,g.model_profile_id,v.prompt_group_id,v.carrier_id,p.split,JSON_VALUE(d.config_json,'$.key') decode_key,JSON_VALUE(v.metadata_json,'$.suite') variant_suite
FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id IN({campaign_sql}) AND g.truncated=0 AND g.reference_token_count>=16""",conn)
def suite(row):
 if int(row.campaign_id)==primary:
  if row.split=="test_id" and row.decode_key in {"det","nat-0","nat-1"} and row.carrier_id!="structured-v1":return "test_id"
  if row.split=="test_source_holdout" and row.decode_key in {"det","nat-0","nat-1"} and row.carrier_id!="structured-v1":return "test_source_holdout"
  if row.split=="test_id" and row.carrier_id=="structured-v1" and row.decode_key!="hv":return "test_carrier_holdout"
  if row.split=="test_id" and row.decode_key=="hv" and row.carrier_id!="structured-v1":return "test_decode_shift"
 elif row.carrier_id=="persona-v1":return "persona"
 elif row.carrier_id=="rag-grounded-v1":return "rag_grounded"
 return None
meta["suite"]=meta.apply(suite,axis=1);meta=meta[meta.suite.notna()].copy();meta["sample_key"]=meta.apply(lambda row:hashlib.sha256(f"{row.suite}:{row.generation_id}".encode()).hexdigest(),axis=1)
meta=meta.sort_values("sample_key").groupby("suite",group_keys=False).head(args.max_per_suite).drop(columns="sample_key");selected=set(map(int,meta.generation_id))
segments=pd.read_sql(f"""SELECT g.generation_id,o.segment_id,o.segmenter_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
JOIN dbo.output_segments o ON o.text_artifact_id=t.text_artifact_id AND o.is_primary_eligible=1 JOIN dbo.semantic_vectors s ON s.segment_id=o.segment_id
AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id=CONCAT('segment-',o.segmenter_id) WHERE g.campaign_id IN({campaign_sql})""",conn)
segments=segments[segments.generation_id.isin(selected)].merge(meta,on="generation_id",how="inner")
if segments.empty:raise SystemExit("No query segment embeddings; run npm run features:build")
thread_state=threading.local()
def thread_conn():
 if not hasattr(thread_state,"conn"):thread_state.conn=pymssql.connect(**db,autocommit=True)
 return thread_state.conn
def exact_one(row):
 started=time.perf_counter();cursor=thread_conn().cursor(as_dict=True);cursor.execute(f"""WITH candidates AS (SELECT TOP ({args.candidate_k}) c.vector_id,c.model_profile_id,c.prompt_group_id,c.text_artifact_id,
  VECTOR_DISTANCE('cosine',c.embedding,CAST(%s AS vector(1024))) distance FROM dbo.search_segment_train c JOIN dbo.output_segments o ON o.segment_id=c.segment_id
  WHERE o.segmenter_id=%s AND c.prompt_group_id<>%s ORDER BY distance,c.vector_id), ranked AS
  (SELECT *,ROW_NUMBER() OVER(PARTITION BY prompt_group_id ORDER BY distance,vector_id) rn_prompt,ROW_NUMBER() OVER(PARTITION BY text_artifact_id ORDER BY distance,vector_id) rn_text FROM candidates)
     SELECT TOP ({args.k}) vector_id,model_profile_id,prompt_group_id,distance FROM ranked WHERE rn_prompt=1 AND rn_text=1 ORDER BY distance,vector_id OPTION (MAXDOP 1)""",
  (row.vector,row.segmenter_id,row.prompt_group_id));neighbors=list(cursor);return int(row.segment_id),int(row.generation_id),(time.perf_counter()-started)*1000,neighbors
cursor=conn.cursor();all_metrics={"schemaVersion":1,"runId":run_id,"campaignIds":campaign_ids,"k":args.k,"candidateK":args.candidate_k,"maxPerSuite":args.max_per_suite,"segmenters":{}};predictions=[];neighbor_export=[]
for segmenter,query_rows in segments.groupby("segmenter_id"):
 query_rows=query_rows.sort_values(["generation_id","segment_id"]);cursor.execute("""INSERT dbo.search_runs(run_id,representation_id,requested_search_mode,actual_search_mode,index_version,metric,candidate_k,returned_k,started_at)
  VALUES(%s,%s,'exact','exact','none','cosine',%s,%s,SYSUTCDATETIME()); SELECT SCOPE_IDENTITY()""",(run_id,f"chunk:{segmenter}",args.candidate_k,args.k));search_run=int(cursor.fetchone()[0]);conn.commit()
 with ThreadPoolExecutor(max_workers=args.workers) as executor:results=list(executor.map(exact_one,[row for row in query_rows.itertuples()]))
 scores={};latencies=[];neighbor_labels={};neighbor_groups={};neighbor_artifacts={}
 for segment_id,generation_id,latency,neighbors in results:
  latencies.append(latency);neighbor_labels[segment_id]=[]
  for rank,row in enumerate(neighbors):
   weight=float(np.exp(-float(row["distance"])/.10));scores.setdefault(generation_id,np.zeros(len(models)))[models.index(row["model_profile_id"])]+=weight
   neighbor_labels[segment_id].append((int(row["vector_id"]),row["model_profile_id"],weight));neighbor_export.append({"search_run_id":search_run,"segmenter_id":segmenter,"segment_id":segment_id,"generation_id":generation_id,"rank":rank+1,**row})
   cursor.execute("INSERT dbo.neighbor_results(search_run_id,query_id,rank,neighbor_id,distance,neighbor_model_profile_id,neighbor_prompt_group_id) VALUES(%s,%s,%s,%s,%s,%s,%s)",
    (search_run,segment_id,rank+1,row["vector_id"],row["distance"],row["model_profile_id"],row["prompt_group_id"]))
 cursor.execute("UPDATE dbo.search_runs SET finished_at=SYSUTCDATETIME() WHERE search_run_id=%s",(search_run,));conn.commit()
 frame=meta[meta.generation_id.isin(scores)].copy().sort_values("generation_id").reset_index(drop=True);prob=np.stack([scores[int(value)]/max(scores[int(value)].sum(),1e-12) for value in frame.generation_id]);truth=frame.model_profile_id.map({model:index for index,model in enumerate(models)}).to_numpy();pred=prob.argmax(1)
 rep={"searchRunId":search_run,"queryGenerations":len(frame),"querySegments":len(query_rows),"latencyMs":{"p50":float(np.median(latencies)),"p95":float(np.quantile(latencies,.95))},"suites":{}}
 # Corpus labels are permuted at the text-artifact level within prompt group so every segment of an output retains one shuffled label.
 corpus=pd.read_sql("""SELECT c.vector_id,c.text_artifact_id,c.prompt_group_id,c.model_profile_id FROM dbo.search_segment_train c JOIN dbo.output_segments o ON o.segment_id=c.segment_id WHERE o.segmenter_id=%s""",conn,params=(segmenter,))
 artifact=corpus.drop_duplicates(["text_artifact_id"]);artifact_groups=[np.asarray(values,dtype=int) for values in artifact.groupby("prompt_group_id").indices.values()];artifact_labels=artifact.model_profile_id.map({m:i for i,m in enumerate(models)}).to_numpy();vector_artifact=dict(zip(corpus.vector_id,corpus.text_artifact_id));artifact_position={int(value):index for index,value in enumerate(artifact.text_artifact_id)}
 for suite_name,suite_frame in frame.groupby("suite"):
  index=suite_frame.index.to_numpy();local_truth=truth[index];local_pred=pred[index];groups=suite_frame.prompt_group_id.to_numpy();rng=np.random.default_rng(SEED+sum(map(ord,str(segmenter)+suite_name)))
  group_indices={group:np.flatnonzero(groups==group) for group in np.unique(groups)};boot=[]
  for _ in range(args.bootstrap):
   take=np.concatenate([group_indices[group] for group in rng.choice(list(group_indices),len(group_indices),replace=True)]);boot.append(f1_score(local_truth[take],local_pred[take],average="macro",labels=np.arange(len(models)),zero_division=0))
  null=[]
  for _ in range(args.permutations):
   shuffled=artifact_labels.copy()
   for values in artifact_groups:
    part=shuffled[values].copy();rng.shuffle(part);shuffled[values]=part
   shuffled_truth=local_truth.copy()
   for values in group_indices.values():
    part=shuffled_truth[values].copy();rng.shuffle(part);shuffled_truth[values]=part
   null_score=np.zeros((len(suite_frame),len(models)));generation_position={int(value):position for position,value in enumerate(suite_frame.generation_id)}
   for row in query_rows[query_rows.generation_id.isin(suite_frame.generation_id)].itertuples():
    for vector_id,_,weight in neighbor_labels[int(row.segment_id)]:null_score[generation_position[int(row.generation_id)],shuffled[artifact_position[vector_artifact[vector_id]]]]+=weight
   null.append(f1_score(shuffled_truth,null_score.argmax(1),average="macro",labels=np.arange(len(models)),zero_division=0))
  metric={"rows":len(suite_frame),"groups":int(suite_frame.prompt_group_id.nunique()),"accuracy":float(accuracy_score(local_truth,local_pred)),
   "macroF1":float(f1_score(local_truth,local_pred,average="macro",labels=np.arange(len(models)),zero_division=0)),"macroF1Ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
   "permutationNull":{"mean":float(np.mean(null)),"p95":float(np.quantile(null,.95))}};rep["suites"][suite_name]=metric
  pd.DataFrame({"value":null}).to_csv(run/f"tables/permutation-chunk-{segmenter}-{suite_name}.csv",index=False)
  cursor.execute("INSERT dbo.prediction_runs(run_id,suite,representation_id,method,config_json) VALUES(%s,%s,%s,'chunk-vote',%s); SELECT SCOPE_IDENTITY()",(run_id,suite_name,f"chunk:{segmenter}",json.dumps({"searchRunId":search_run,"k":args.k})));prediction_run=int(cursor.fetchone()[0]);rows=[]
  for local,row_index in enumerate(index):
   row=frame.iloc[row_index];rows.append((prediction_run,int(row.generation_id),row.model_profile_id,models[pred[row_index]],"retrieval_only",None,"[]",None));predictions.append({"generation_id":int(row.generation_id),"model_profile_id":row.model_profile_id,"prompt_group_id":row.prompt_group_id,"suite":suite_name,"segmenter_id":segmenter,"predicted_model_profile_id":models[pred[row_index]],"confidence":float(prob[row_index].max())})
  cursor.executemany("INSERT dbo.predictions(prediction_run_id,generation_id,true_model_profile_id,predicted_model_profile_id,decision,calibrated_probability,candidate_set_json,novelty_score) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",rows)
  cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
   VALUES(%s,'RETRIEVE',%s,'chunk-vote',%s,'macro_f1',%s,%s,%s,%s,%s,%s,%s,%s)""",(run_id,f"chunk:{segmenter}",suite_name,metric["macroF1"],*metric["macroF1Ci95"],metric["permutationNull"]["mean"],metric["permutationNull"]["p95"],metric["rows"],metric["groups"],json.dumps(metric)));conn.commit()
 all_metrics["segmenters"][segmenter]=rep
pd.DataFrame(predictions).to_parquet(run/"tables/chunk-predictions.parquet",index=False);pd.DataFrame(neighbor_export).to_parquet(run/"tables/chunk-neighbors.parquet",index=False)
(run/"metrics/chunk-retrieval.json").write_text(json.dumps(all_metrics,indent=2)+"\n");print(json.dumps({"runId":run_id,"segmenters":list(all_metrics["segmenters"]),"predictionRows":len(predictions),"neighborRows":len(neighbor_export)},indent=2));conn.close()
