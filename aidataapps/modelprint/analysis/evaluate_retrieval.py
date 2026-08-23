#!/usr/bin/env python3
"""Run exact SQL retrieval, retain neighbors, and evaluate kNN vote evidence."""
import argparse, hashlib, json, math, os, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--run")
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--candidate-k", type=int, default=200)
parser.add_argument("--tau", type=float, default=0.10)
parser.add_argument("--workers", type=int, default=24)
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--permutations", type=int, default=200)
parser.add_argument("--max-queries-per-suite", type=int)
parser.add_argument("--representations", help="Comma-separated representation IDs; default is every available space")
parser.add_argument("--equivalence", type=int, default=500)
args = parser.parse_args()
if not (1 <= args.k <= args.candidate_k <= 2000): raise SystemExit("Require 1 <= k <= candidate-k <= 2000")
lab = Path(__file__).resolve().parents[1]; run = Path(args.run or lab / (lab / ".current-run").read_text().strip()).resolve()
freeze = json.loads((run / "manifests/campaign-freeze.json").read_text()); primary = int(freeze["campaignId"]); models = freeze["campaign"]["profiles"]
robust_path = run / "manifests/robustness-freeze.json"; robustness = int(json.loads(robust_path.read_text())["campaignId"]) if robust_path.exists() else None
campaign_ids = [primary] + ([robustness] if robustness else []); campaign_sql = ",".join(map(str, campaign_ids)); run_id = run.name
db = dict(server=os.getenv("SQLSERVER_HOST", "127.0.0.1"), port=int(os.getenv("SQLSERVER_PORT", "1433")), user="sa",
          password=os.environ["MSSQL_SA_PASSWORD"], database=os.getenv("MSSQL_DATABASE", "ModelPrint"), login_timeout=60, timeout=3600)
conn = pymssql.connect(**db, autocommit=False)

meta = pd.read_sql(f"""SELECT g.generation_id,g.campaign_id,g.model_profile_id,v.prompt_group_id,v.carrier_id,p.prompt_source_id source_id,p.family,p.stratum,p.split,
 JSON_VALUE(d.config_json,'$.key') decode_key,JSON_VALUE(v.metadata_json,'$.suite') variant_suite,g.length_band,g.truncated,g.reference_token_count
FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id IN ({campaign_sql})""", conn)
def suite_for(row):
    if int(row.campaign_id) == primary:
        shifted = row.decode_key == "hv"; held_carrier = row.carrier_id == "structured-v1"
        if row.split == "test_id" and not shifted and not held_carrier: return "test_id"
        if row.split == "test_source_holdout" and not shifted and not held_carrier: return "test_source_holdout"
        if row.split == "test_id" and not shifted and held_carrier: return "test_carrier_holdout"
        if row.split == "test_id" and shifted and not held_carrier: return "test_decode_shift"
        return None
    if row.carrier_id == "persona-v1": return "persona"
    if row.carrier_id == "rag-grounded-v1": return "rag_grounded"
    if row.variant_suite == "pressure": return "pressure"
    return None
meta["suite"] = meta.apply(suite_for, axis=1)
queries = meta[(meta.suite.notna()) & (meta.truncated == 0) & (meta.reference_token_count.fillna(0) >= 16)].copy()
if args.max_queries_per_suite:
    queries["sample_key"] = queries.apply(lambda row: hashlib.sha256(f"{row.suite}:{row.generation_id}".encode()).hexdigest(), axis=1)
    queries = queries.sort_values("sample_key").groupby("suite", group_keys=False).head(args.max_queries_per_suite).drop(columns="sample_key")

common_semantic = f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id WHERE g.campaign_id IN ({campaign_sql})"""
common_style = f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE g.campaign_id IN ({campaign_sql})"""
representations = {
  "semantic1024-qwen-raw-final-v1": {"table":"search_semantic_train","dim":1024,"query":common_semantic+" AND s.embedding_profile_id='qwen3-embedding-0.6b' AND t.text_view_id='raw-final-v1' AND s.representation_id='whole-raw-final-v1'", "where":"embedding_profile_id='qwen3-embedding-0.6b' AND text_view_id='raw-final-v1'"},
  "semantic1024-qwen-name-masked-v1": {"table":"search_semantic_train","dim":1024,"query":common_semantic+" AND s.embedding_profile_id='qwen3-embedding-0.6b' AND t.text_view_id='name-masked-v1' AND s.representation_id='whole-name-masked-v1'", "where":"embedding_profile_id='qwen3-embedding-0.6b' AND text_view_id='name-masked-v1'"},
  "semantic1024-bge-raw-final-v1": {"table":"search_semantic_train","dim":1024,"query":common_semantic+" AND s.embedding_profile_id='bge-large-en-v1.5' AND t.text_view_id='raw-final-v1' AND s.representation_id='whole-raw-final-v1'", "where":"embedding_profile_id='bge-large-en-v1.5' AND text_view_id='raw-final-v1'"},
  "semantic1024-bge-name-masked-v1": {"table":"search_semantic_train","dim":1024,"query":common_semantic+" AND s.embedding_profile_id='bge-large-en-v1.5' AND t.text_view_id='name-masked-v1' AND s.representation_id='whole-name-masked-v1'", "where":"embedding_profile_id='bge-large-en-v1.5' AND text_view_id='name-masked-v1'"},
  "style512-raw-final-v1": {"table":"search_style_train","dim":512,"query":common_style+" AND t.text_view_id='raw-final-v1'", "where":"text_view_id='raw-final-v1'"},
  "style512-name-masked-v1": {"table":"search_style_train","dim":512,"query":common_style+" AND t.text_view_id='name-masked-v1'", "where":"text_view_id='name-masked-v1'"},
}
for name in ["residual-diff-v1","residual-reject-v1","residual-ridge-v1","prompt-centered-v1"]:
    representations[name] = {"table":"search_residual_train","dim":1024,
      "query":f"SELECT g.generation_id,CAST(r.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.residual_vectors r ON r.generation_id=g.generation_id WHERE g.campaign_id IN ({campaign_sql}) AND r.representation_id='{name}'",
      "where":f"representation_id='{name}'"}
representations["fingerprint64-v1"] = {"table":"search_fingerprint_train","dim":64,
  "query":f"SELECT g.generation_id,CAST(v.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.fingerprint_vectors v ON v.generation_id=g.generation_id WHERE g.campaign_id IN ({campaign_sql}) AND v.representation_id='fingerprint64-v1'", "where":"1=1"}
representations["likelihood-profile8-v1"] = {"table":"search_likelihood_train","dim":8,
  "query":f"SELECT g.generation_id,CAST(v.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.likelihood_profile_vectors v ON v.generation_id=g.generation_id WHERE g.campaign_id IN ({campaign_sql}) AND v.representation_id='likelihood-profile8-v1'", "where":"1=1"}
if args.representations:
    requested = args.representations.split(","); missing = set(requested) - set(representations)
    if missing: raise SystemExit(f"Unknown representations: {sorted(missing)}")
    representations = {key: representations[key] for key in requested}

thread_state = threading.local()
def thread_conn():
    if not hasattr(thread_state, "conn"): thread_state.conn = pymssql.connect(**db, autocommit=True)
    return thread_state.conn

def exact_one(item, definition):
    generation_id, prompt_group_id, vector_json = item; started = time.perf_counter(); cursor = thread_conn().cursor(as_dict=True); candidate_k=args.candidate_k
    while True:
      cursor.execute(f"""WITH candidates AS (SELECT TOP ({candidate_k}) vector_id,model_profile_id,prompt_group_id,text_artifact_id,
        VECTOR_DISTANCE('cosine',embedding,CAST(%s AS vector({definition['dim']}))) distance FROM dbo.{definition['table']}
        WHERE {definition['where']} AND prompt_group_id<>%s ORDER BY distance,vector_id), ranked AS
        (SELECT *,ROW_NUMBER() OVER(PARTITION BY prompt_group_id ORDER BY distance,vector_id) rn_prompt,
         ROW_NUMBER() OVER(PARTITION BY text_artifact_id ORDER BY distance,vector_id) rn_text FROM candidates)
        SELECT TOP ({args.k}) vector_id,model_profile_id,prompt_group_id,distance FROM ranked WHERE rn_prompt=1 AND rn_text=1 ORDER BY distance,vector_id""",
        (vector_json, prompt_group_id)); rows = list(cursor)
      if len(rows)>=args.k:return generation_id,(time.perf_counter()-started)*1000,rows,candidate_k
      if candidate_k>=2000:raise RuntimeError(f"generation {generation_id}: exact search returned {len(rows)}/{args.k} after {candidate_k} candidates")
      candidate_k=min(2000,candidate_k*2)

def interval(frame, prediction, replicates, rng):
    groups = frame.prompt_group_id.unique(); indices = {group:np.flatnonzero(frame.prompt_group_id.to_numpy()==group) for group in groups}; values=[]
    for _ in range(replicates):
        chosen=rng.choice(groups,len(groups),replace=True); take=np.concatenate([indices[group] for group in chosen])
        values.append(f1_score(frame.true_label.to_numpy()[take],prediction[take],average="macro",labels=np.arange(len(models)),zero_division=0))
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

all_metrics={"schemaVersion":1,"runId":run_id,"campaignId":primary,"k":args.k,"candidateK":args.candidate_k,"candidateKPolicy":"double-on-dedup-shortfall-to-2000","tau":args.tau,"representations":{}}
all_neighbors=[]; all_predictions=[]; equivalence_result=None
for rep_index,(name,definition) in enumerate(representations.items()):
    vector_rows=pd.read_sql(definition["query"],conn); vector_map=dict(zip(vector_rows.generation_id,vector_rows.vector))
    selected=queries[queries.generation_id.isin(vector_map)].copy()
    if selected.empty: continue
    selected["vector"] = selected.generation_id.map(vector_map)
    selected_by_id = selected.set_index("generation_id")
    started_at=pd.Timestamp.utcnow(); cursor=conn.cursor(); query_hash=hashlib.sha256((name+definition["table"]+definition["where"]+"adaptive-candidate-to-2000").encode()).hexdigest()
    cursor.execute("""INSERT dbo.search_runs(run_id,representation_id,requested_search_mode,actual_search_mode,index_name,index_version,metric,candidate_k,returned_k,fallback_reason,query_plan_hash,started_at)
      VALUES(%s,%s,'exact','exact',NULL,NULL,'cosine',%s,%s,NULL,%s,%s); SELECT SCOPE_IDENTITY()""",
      (run_id,name,args.candidate_k,args.k,query_hash,started_at.to_pydatetime())); search_run=int(cursor.fetchone()[0]); conn.commit()
    items=[(int(row.generation_id),row.prompt_group_id,row.vector) for row in selected.itertuples()]
    with ThreadPoolExecutor(max_workers=args.workers) as executor: results=list(executor.map(lambda item:exact_one(item,definition),items))
    neighbor_records=[]; latency_records=[]
    maximum_candidate_used=max(row[3] for row in results)
    if maximum_candidate_used>args.candidate_k:
        cursor.execute("UPDATE dbo.search_runs SET candidate_k=%s,fallback_reason=%s WHERE search_run_id=%s",(maximum_candidate_used,"exact candidate expansion after prompt/text dedup shortfall",search_run));conn.commit()
    for generation_id,latency,neighbors,candidate_used in results:
        query_meta=selected_by_id.loc[generation_id]; latency_records.append({"representation":name,"query_id":generation_id,"suite":query_meta.suite,"latency_ms":latency,"candidate_k":candidate_used})
        for rank,row in enumerate(neighbors,1):
            neighbor_records.append({"search_run_id":search_run,"query_id":generation_id,"rank":rank,"neighbor_id":int(row["vector_id"]),"distance":float(row["distance"]),
              "neighbor_model_profile_id":row["model_profile_id"],"neighbor_prompt_group_id":row["prompt_group_id"],"representation":name,"suite":query_meta.suite})
    for start in range(0,len(neighbor_records),1000):
        cursor.executemany("INSERT dbo.neighbor_results(search_run_id,query_id,rank,neighbor_id,distance,neighbor_model_profile_id,neighbor_prompt_group_id) VALUES(%s,%s,%s,%s,%s,%s,%s)",
          [(r["search_run_id"],r["query_id"],r["rank"],r["neighbor_id"],r["distance"],r["neighbor_model_profile_id"],r["neighbor_prompt_group_id"]) for r in neighbor_records[start:start+1000]])
        conn.commit()
    cursor.execute("UPDATE dbo.search_runs SET finished_at=SYSUTCDATETIME() WHERE search_run_id=%s",(search_run,));conn.commit()
    neighbors=pd.DataFrame(neighbor_records); latencies=pd.DataFrame(latency_records); all_neighbors.append(neighbors); latencies.to_csv(run/f"tables/retrieval-latency-{name}.csv",index=False)
    votes=pd.read_sql("""WITH weights AS (SELECT query_id,neighbor_model_profile_id,SUM(EXP(-distance/%s)) weight,COUNT(*) neighbor_count,MIN(distance) nearest_distance
      FROM dbo.neighbor_results WHERE search_run_id=%s GROUP BY query_id,neighbor_model_profile_id), totals AS
      (SELECT *,weight/SUM(weight) OVER(PARTITION BY query_id) vote_share FROM weights)
      SELECT query_id,neighbor_model_profile_id,vote_share,neighbor_count,nearest_distance FROM totals""",conn,params=(args.tau,search_run))
    corpus=pd.read_sql(f"SELECT vector_id,prompt_group_id,model_profile_id FROM dbo.{definition['table']} WHERE {definition['where']}",conn)
    corpus_labels=corpus.model_profile_id.map({model:i for i,model in enumerate(models)}).to_numpy(); corpus_pos={int(value):i for i,value in enumerate(corpus.vector_id)}
    corpus_groups=[np.asarray(indices,dtype=int) for indices in corpus.groupby("prompt_group_id").indices.values()]
    rep_metrics={"searchRunId":search_run,"corpusRows":len(corpus),"queryRows":len(selected),"candidateKMaximumUsed":maximum_candidate_used,"suites":{},"latencyMs":{"median":float(latencies.latency_ms.median()),"p95":float(latencies.latency_ms.quantile(.95))}}
    for suite,frame in selected.groupby("suite"):
        frame=frame.reset_index(drop=True); query_position={int(value):i for i,value in enumerate(frame.generation_id)}; probability=np.zeros((len(frame),len(models)))
        suite_votes=votes[votes.query_id.isin(query_position)]
        for row in suite_votes.itertuples(): probability[query_position[int(row.query_id)],models.index(row.neighbor_model_profile_id)]=float(row.vote_share)
        prediction=probability.argmax(axis=1); frame["true_label"]=frame.model_profile_id.map({m:i for i,m in enumerate(models)})
        rng=np.random.default_rng(20260822+rep_index*101+sum(map(ord,suite))); ci=interval(frame,prediction,args.bootstrap,rng)
        suite_neighbors=neighbors[neighbors.query_id.isin(query_position)].copy(); nq=suite_neighbors.query_id.map(query_position).to_numpy(); npos=suite_neighbors.neighbor_id.map(corpus_pos).to_numpy(); weights=np.exp(-suite_neighbors.distance.to_numpy()/args.tau)
        query_group_indices=[np.asarray(indices,dtype=int) for indices in frame.groupby("prompt_group_id").indices.values()]; null=[]
        for _ in range(args.permutations):
            shuffled=corpus_labels.copy()
            for idx in corpus_groups:
                values=shuffled[idx].copy();rng.shuffle(values);shuffled[idx]=values
            shuffled_truth=frame.true_label.to_numpy().copy()
            for idx in query_group_indices:
                values=shuffled_truth[idx].copy();rng.shuffle(values);shuffled_truth[idx]=values
            score=np.zeros((len(frame),len(models))); np.add.at(score,(nq,shuffled[npos]),weights); null_prediction=score.argmax(axis=1)
            null.append(f1_score(shuffled_truth,null_prediction,average="macro",labels=np.arange(len(models)),zero_division=0))
        metrics={"rows":len(frame),"groups":int(frame.prompt_group_id.nunique()),"accuracy":float(accuracy_score(frame.true_label,prediction)),
          "macroF1":float(f1_score(frame.true_label,prediction,average="macro",labels=np.arange(len(models)),zero_division=0)),"macroF1Ci95":ci,
          "top2Accuracy":float(top_k_accuracy_score(frame.true_label,probability,k=2,labels=np.arange(len(models)))),
          "permutationNull":{"mean":float(np.mean(null)),"p95":float(np.quantile(null,.95))},"searchRunId":search_run}
        rep_metrics["suites"][suite]=metrics; pd.DataFrame({"value":null}).to_csv(run/f"tables/permutation-retrieval-{name}-{suite}.csv",index=False)
        cursor.execute("INSERT dbo.prediction_runs(run_id,suite,representation_id,method,config_json) VALUES(%s,%s,%s,'knn-vote',%s); SELECT SCOPE_IDENTITY()",
          (run_id,suite,name,json.dumps({"searchRunId":search_run,"k":args.k,"tau":args.tau,"calibrated":False}))); prediction_run=int(cursor.fetchone()[0]); conn.commit()
        pred_rows=[]; candidate_rows=[]
        for index,row in frame.iterrows():
            pred_rows.append((prediction_run,int(row.generation_id),row.model_profile_id,models[prediction[index]],"retrieval_only",None,"[]",None))
            for model_index,model in enumerate(models): candidate_rows.append((prediction_run,int(row.generation_id),model,float(probability[index,model_index]),"{}"))
            all_predictions.append({"prediction_run_id":prediction_run,"generation_id":int(row.generation_id),"model_profile_id":row.model_profile_id,
              "prompt_group_id":row.prompt_group_id,"suite":suite,"representation":name,"method":"knn-vote","predicted_model_profile_id":models[prediction[index]],
              "confidence":float(probability[index].max()),"probabilities":json.dumps({models[i]:float(probability[index,i]) for i in range(len(models))})})
        cursor.executemany("INSERT dbo.predictions(prediction_run_id,generation_id,true_model_profile_id,predicted_model_profile_id,decision,calibrated_probability,candidate_set_json,novelty_score) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",pred_rows)
        cursor.executemany("INSERT dbo.prediction_candidates(prediction_run_id,generation_id,model_profile_id,probability,features_json) VALUES(%s,%s,%s,%s,%s)",candidate_rows)
        detail=json.dumps(metrics)
        cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
          VALUES(%s,'RETRIEVE',%s,'knn-vote',%s,'macro_f1',%s,%s,%s,%s,%s,%s,%s,%s)""",
          (run_id,name,suite,metrics["macroF1"],ci[0],ci[1],metrics["permutationNull"]["mean"],metrics["permutationNull"]["p95"],len(frame),metrics["groups"],detail));conn.commit()
    all_metrics["representations"][name]=rep_metrics

    if equivalence_result is None and args.equivalence:
        sample=selected.sort_values("generation_id").head(args.equivalence); corpus_vectors=pd.read_sql(f"SELECT vector_id,prompt_group_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.{definition['table']} WHERE {definition['where']} ORDER BY vector_id",conn)
        matrix=np.stack(corpus_vectors.vector.map(lambda value:np.asarray(json.loads(value),dtype=np.float32))); matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12)
        ids=corpus_vectors.vector_id.to_numpy(); groups=corpus_vectors.prompt_group_id.to_numpy(); sql_map={int(q):part.sort_values('rank') for q,part in neighbors.groupby('query_id')}; matched=0; max_error=0.0; checked=0
        for row in sample.itertuples():
            q=np.asarray(json.loads(row.vector),dtype=np.float32);q/=max(np.linalg.norm(q),1e-12);distance=1-matrix@q; allowed=np.flatnonzero(groups!=row.prompt_group_id)
            order=allowed[np.lexsort((ids[allowed],distance[allowed]))[:maximum_candidate_used]]; seen=set(); chosen=[]
            for pos in order:
                if groups[pos] in seen: continue
                seen.add(groups[pos]);chosen.append(pos)
                if len(chosen)==args.k:break
            sql_rows=sql_map.get(int(row.generation_id));
            if sql_rows is None:continue
            checked+=1; numpy_ids=ids[chosen].tolist();sql_ids=sql_rows.neighbor_id.tolist();matched+=int(numpy_ids==sql_ids)
            if numpy_ids==sql_ids:max_error=max(max_error,float(np.max(np.abs(distance[chosen]-sql_rows.distance.to_numpy()))))
        equivalence_result={"representation":name,"checked":checked,"exactListMatches":matched,"listAgreement":matched/max(checked,1),"maxDistanceErrorOnMatchingLists":max_error,"tolerance":1e-4}
        if checked<min(args.equivalence,len(selected)) or matched!=checked or max_error>1e-4: raise RuntimeError(f"SQL/NumPy exact-neighbor equivalence failed: {equivalence_result}")

neighbors_out=pd.concat(all_neighbors,ignore_index=True) if all_neighbors else pd.DataFrame(); predictions_out=pd.DataFrame(all_predictions)
neighbors_out.to_parquet(run/"tables/exact-neighbors.parquet",index=False); predictions_out.to_parquet(run/"tables/retrieval-predictions.parquet",index=False)
all_metrics["equivalence"]=equivalence_result
(run/"metrics/retrieval.json").write_text(json.dumps(all_metrics,indent=2)+"\n");(run/"reports/SQL_NUMPY_EQUIVALENCE.md").write_text("# SQL/NumPy Exact-Neighbor Equivalence\n\n```json\n"+json.dumps(equivalence_result,indent=2)+"\n```\n")
print(json.dumps({"runId":run_id,"representations":list(all_metrics["representations"]),"neighborRows":len(neighbors_out),"predictionRows":len(predictions_out),"equivalence":equivalence_result},indent=2));conn.close()
