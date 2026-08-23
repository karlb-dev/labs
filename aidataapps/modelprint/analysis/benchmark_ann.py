#!/usr/bin/env python3
"""Benchmark SQL Server exact search against proven DiskANN execution.

The benchmark rebuilds disposable, deterministic corpus prefixes. Scientific
neighbors stay exact; ANN is evaluated only as an engineering optimization.
"""
import argparse, collections, hashlib, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql

SEED = 20260822
parser = argparse.ArgumentParser()
parser.add_argument("--run")
parser.add_argument("--queries", type=int, default=1000)
parser.add_argument("--k", default="1,5,10,20")
parser.add_argument("--prefixes", default="1000,10000,25000,50000")
args = parser.parse_args()
ks = sorted({int(value) for value in args.k.split(",")}); maximum_k = max(ks)
prefix_targets = sorted({int(value) for value in args.prefixes.split(",")})
if not ks or ks[0] < 1 or maximum_k > 100: raise SystemExit("k values must be 1..100")
if args.queries < 20: raise SystemExit("--queries must be at least 20")

lab = Path(__file__).resolve().parents[1]
run = Path(args.run or lab / (lab / ".current-run").read_text().strip()).resolve()
freeze = json.loads((run / "manifests/campaign-freeze.json").read_text())
campaign = int(freeze["campaignId"]); run_id = run.name
db = dict(server=os.getenv("SQLSERVER_HOST", "127.0.0.1"), port=int(os.getenv("SQLSERVER_PORT", "1433")), user="sa",
          password=os.environ["MSSQL_SA_PASSWORD"], database=os.getenv("MSSQL_DATABASE", "ModelPrint"), login_timeout=60, timeout=7200)
conn = pymssql.connect(**db, autocommit=False); cursor = conn.cursor()

def vector_ddl(statement):
  conn.commit();conn.autocommit(True)
  try:cursor.execute(statement)
  finally:conn.autocommit(False)

spaces = {
  "semantic1024-segment-union-v1": {
    "dimension": 1024, "scratch": "__mp_ann_bench_1024", "index": "vix_mp_ann_bench_1024",
    "source": "dbo.search_segment_train",
    "query": f"""SELECT TOP ({args.queries}) g.generation_id,g.model_profile_id,v.prompt_group_id,CAST(s.embedding AS nvarchar(max)) vector
      FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
      JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
      JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1'
      WHERE g.campaign_id={campaign} AND g.split IN('test_id','test_source_holdout') AND g.truncated=0 AND g.reference_token_count>=16
      ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(8),g.generation_id)),g.generation_id"""
  },
  "fingerprint64-v1": {
    "dimension": 64, "scratch": "__mp_ann_bench_64", "index": "vix_mp_ann_bench_64",
    "source": "dbo.search_fingerprint_train",
    "query": f"""SELECT TOP ({args.queries}) g.generation_id,g.model_profile_id,v.prompt_group_id,CAST(s.embedding AS nvarchar(max)) vector
      FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
      JOIN dbo.fingerprint_vectors s ON s.generation_id=g.generation_id AND s.representation_id='fingerprint64-v1'
      WHERE g.campaign_id={campaign} AND g.split IN('test_id','test_source_holdout') AND g.truncated=0 AND g.reference_token_count>=16
      ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(8),g.generation_id)),g.generation_id"""
  },
}

def create_search_run(representation, requested, actual, index, candidate_k, returned_k, plan_hash, fallback=None):
  cursor.execute("""INSERT dbo.search_runs(run_id,representation_id,requested_search_mode,actual_search_mode,index_name,index_version,metric,candidate_k,returned_k,fallback_reason,query_plan_hash,started_at)
    VALUES(%s,%s,%s,%s,%s,'legacy-unversioned','cosine',%s,%s,%s,%s,SYSUTCDATETIME()); SELECT SCOPE_IDENTITY()""",
    (run_id, representation, requested, actual, index, candidate_k, returned_k, fallback, plan_hash))
  value = int(cursor.fetchone()[0]); conn.commit(); return value

def finish_search_run(identifier):
  cursor.execute("UPDATE dbo.search_runs SET finished_at=SYSUTCDATETIME() WHERE search_run_id=%s", (identifier,)); conn.commit()

def dedupe(rows, limit):
  seen_groups=set();seen_artifacts=set();output=[]
  for row in rows:
    vector_id, model, group, artifact, distance = row
    if group in seen_groups or artifact in seen_artifacts: continue
    seen_groups.add(group);seen_artifacts.add(artifact);output.append((int(vector_id),str(model),str(group),float(distance)))
    if len(output) == limit: break
  return output

def exact_one(space, query):
  dimension=space["dimension"];table=space["scratch"];candidate=max(200,maximum_k*10);started=time.perf_counter()
  local=conn.cursor();local.execute(f"""SELECT TOP ({candidate}) vector_id,model_profile_id,prompt_group_id,text_artifact_id,
    VECTOR_DISTANCE('cosine',embedding,CAST(%s AS vector({dimension}))) distance FROM dbo.{table}
    WHERE prompt_group_id<>%s ORDER BY distance,vector_id""",(query.vector,query.prompt_group_id));raw=local.fetchall()
  return (time.perf_counter()-started)*1000,dedupe(raw,maximum_k)

def ann_one(space, query, multiplier):
  dimension=space["dimension"];table=space["scratch"];candidate=maximum_k*multiplier;started=time.perf_counter();local=conn.cursor()
  local.execute(f"""DECLARE @q vector({dimension})=CAST(%s AS vector({dimension}));
    SELECT TOP ({candidate}) p.vector_id,p.model_profile_id,p.prompt_group_id,p.text_artifact_id,s.distance
    FROM VECTOR_SEARCH(TABLE=dbo.{table} AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine',TOP_N={candidate}) s
    WHERE p.prompt_group_id<>%s ORDER BY s.distance,p.vector_id""",(query.vector,query.prompt_group_id));raw=local.fetchall()
  return (time.perf_counter()-started)*1000,dedupe(raw,maximum_k),candidate-len(raw)

def capture_plan(space, query, multiplier):
  dimension=space["dimension"];table=space["scratch"];candidate=maximum_k*multiplier;local=conn.cursor();values=[]
  try:
    local.execute("SET STATISTICS XML ON")
    local.execute(f"""DECLARE @q vector({dimension})=CAST(%s AS vector({dimension}));
      SELECT TOP ({candidate}) p.vector_id,p.model_profile_id,p.prompt_group_id,p.text_artifact_id,s.distance
      FROM VECTOR_SEARCH(TABLE=dbo.{table} AS p,COLUMN=embedding,SIMILAR_TO=@q,METRIC='cosine',TOP_N={candidate}) s
      WHERE p.prompt_group_id<>%s ORDER BY s.distance,p.vector_id""",(query.vector,query.prompt_group_id))
    while True:
      if local.description:
        for row in local.fetchall():
          for value in row:
            if isinstance(value,str) and ("ShowPlanXML" in value or "<ShowPlanXML" in value): values.append(value)
      if not local.nextset(): break
  finally:
    try: local.execute("SET STATISTICS XML OFF")
    except Exception: pass
  xml="\n".join(values);plan_hash=hashlib.sha256(xml.encode()).hexdigest() if xml else None
  present=bool(xml and space["index"] in xml and ("Vector Index Seek" in xml or "VectorSearch" in xml or "VECTOR_SEARCH" in xml))
  return {"sha256":plan_hash,"bytes":len(xml),"indexOperatorPresent":present,"xml":xml}

def vote(rows, k):
  scores=collections.defaultdict(float)
  for _,model,_,distance in rows[:k]:scores[model]+=float(np.exp(-distance/.10))
  return min(scores,key=lambda model:(-scores[model],model)) if scores else None

def persist_neighbors(search_run, query_id, rows):
  cursor.executemany("INSERT dbo.neighbor_results(search_run_id,query_id,rank,neighbor_id,distance,neighbor_model_profile_id,neighbor_prompt_group_id) VALUES(%s,%s,%s,%s,%s,%s,%s)",
    [(search_run,int(query_id),rank+1,row[0],row[3],row[1],row[2]) for rank,row in enumerate(rows)])

summary={"schemaVersion":1,"runId":run_id,"campaignId":campaign,"queryTarget":args.queries,"k":ks,"oversamplingMultipliers":[1,2,5,10],"spaces":{}}
table_rows=[]
try:
  conn.autocommit(True);cursor.execute("ALTER DATABASE SCOPED CONFIGURATION SET PREVIEW_FEATURES=ON");conn.autocommit(False)
  for representation,space in spaces.items():
    source_count=int(pd.read_sql(f"SELECT COUNT(*) count FROM {space['source']}",conn).iloc[0]["count"])
    if source_count<100: summary["spaces"][representation]={"disposition":"NOT_RUN_INSUFFICIENT_CORPUS","achievedMaximum":source_count};continue
    prefixes=sorted(set([value for value in prefix_targets if value<=source_count]+[source_count]));queries=pd.read_sql(space["query"],conn)
    if len(queries)<20: summary["spaces"][representation]={"disposition":"NOT_RUN_INSUFFICIENT_QUERIES","queryRows":len(queries)};continue
    space_rows=[]
    for prefix in prefixes:
      table=space["scratch"];index=space["index"];dimension=space["dimension"]
      vector_ddl(f"IF OBJECT_ID(N'dbo.{table}',N'U') IS NOT NULL DROP TABLE dbo.{table};")
      cursor.execute(f"""CREATE TABLE dbo.{table}(vector_id bigint NOT NULL PRIMARY KEY CLUSTERED,model_profile_id varchar(80) NOT NULL,
        prompt_group_id varchar(120) NOT NULL,text_artifact_id bigint NOT NULL,embedding vector({dimension}) NOT NULL);
        INSERT dbo.{table}(vector_id,model_profile_id,prompt_group_id,text_artifact_id,embedding)
        SELECT TOP ({prefix}) vector_id,model_profile_id,prompt_group_id,text_artifact_id,embedding FROM {space['source']}
        ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(8),vector_id)),vector_id;""");conn.commit()
      build_started=time.perf_counter();vector_ddl(f"CREATE VECTOR INDEX {index} ON dbo.{table}(embedding) WITH(TYPE='DISKANN',METRIC='COSINE');");build_seconds=time.perf_counter()-build_started
      metadata=pd.read_sql(f"""SELECT i.name index_name,v.build_parameters,v.vector_index_type,v.distance_metric,
        (SELECT SUM(reserved_page_count)*8192 FROM sys.dm_db_partition_stats p WHERE p.object_id=v.object_id AND p.index_id=v.index_id) reserved_bytes
        FROM sys.vector_indexes v JOIN sys.indexes i ON i.object_id=v.object_id AND i.index_id=v.index_id WHERE i.name='{index}'""",conn).iloc[0].to_dict()
      exact_run=create_search_run(f"{representation}@{prefix}","exact","exact",None,max(200,maximum_k*10),maximum_k,None)
      exact={};exact_latencies=[]
      for query in queries.itertuples():
        latency,neighbors=exact_one(space,query);exact[int(query.generation_id)]=neighbors;exact_latencies.append(latency);persist_neighbors(exact_run,query.generation_id,neighbors)
      conn.commit();finish_search_run(exact_run)
      prefix_result={"corpusSize":prefix,"indexBuildSeconds":build_seconds,"indexReservedBytes":int(metadata.get("reserved_bytes") or 0),
        "indexMetadata":metadata,"exact":{"searchRunId":exact_run,"latencyP50Ms":float(np.median(exact_latencies)),"latencyP95Ms":float(np.quantile(exact_latencies,.95))},"oversampling":{}}
      for multiplier in [1,2,5,10]:
        plan=capture_plan(space,queries.iloc[0],multiplier);plan_path=run/f"environment/ann-plan-{representation}-{prefix}-x{multiplier}.xml"
        if plan["xml"]:plan_path.write_text(plan.pop("xml"))
        else:plan.pop("xml")
        proven=bool(plan["indexOperatorPresent"]);actual="ann_legacy" if proven else "execution_mode_unknown";fallback=None if proven else "captured plan did not name the vector index operator"
        ann_run=create_search_run(f"{representation}@{prefix}","ann_legacy",actual,index,maximum_k*multiplier,maximum_k,plan["sha256"],fallback)
        benchmark_config={"k":ks,"oversamplingMultiplier":multiplier,"candidateTopN":maximum_k*multiplier,"queryRows":len(queries),"plan":plan,
          "indexBuildSeconds":build_seconds,"indexReservedBytes":prefix_result["indexReservedBytes"],"exactSearchRunId":exact_run,"annSearchRunId":ann_run}
        cursor.execute("INSERT dbo.ann_benchmark_runs(run_id,representation_id,corpus_size,index_name,index_version,plan_evidence,config_json) VALUES(%s,%s,%s,%s,'legacy-unversioned',%s,%s); SELECT SCOPE_IDENTITY()",
          (run_id,representation,prefix,index,int(proven),json.dumps(benchmark_config)));benchmark_run=int(cursor.fetchone()[0]);conn.commit()
        latencies=[];recalls={k:[] for k in ks};agreements={k:[] for k in ks};losses=[];per_class={k:collections.defaultdict(list) for k in ks};records=[]
        for query in queries.itertuples():
          latency,neighbors,loss=ann_one(space,query,multiplier);latencies.append(latency);losses.append(loss);persist_neighbors(ann_run,query.generation_id,neighbors)
          truth=exact[int(query.generation_id)]
          for k in ks:
            recall=len({row[0] for row in truth[:k]} & {row[0] for row in neighbors[:k]})/k
            agreement=int(vote(truth,k)==vote(neighbors,k));recalls[k].append(recall);agreements[k].append(agreement);per_class[k][query.model_profile_id].append(agreement)
            records.append((benchmark_run,int(query.generation_id),k,recall,latency,agreement,agreement,loss,fallback,actual))
        cursor.executemany("INSERT dbo.ann_benchmark_rows(ann_benchmark_run_id,query_id,k,recall,latency_ms,vote_agreement,decision_agreement,candidate_loss,fallback_reason,execution_mode) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",records);conn.commit();finish_search_run(ann_run)
        result={"annBenchmarkRunId":benchmark_run,"annSearchRunId":ann_run,"planEvidence":proven,"plan":plan,
          "latencyP50Ms":float(np.median(latencies)),"latencyP95Ms":float(np.quantile(latencies,.95)),"meanCandidateLoss":float(np.mean(losses)),
          "recall":{str(k):float(np.mean(recalls[k])) for k in ks},"voteAgreement":{str(k):float(np.mean(agreements[k])) for k in ks},
          "minimumClassDecisionAgreement":{str(k):float(min(np.mean(values) for values in per_class[k].values())) for k in ks}}
        prefix_result["oversampling"][str(multiplier)]=result
        for k in ks:table_rows.append({"representation":representation,"dimensions":dimension,"corpus_size":prefix,"queries":len(queries),"k":k,"oversampling":multiplier,
          "exact_p50_ms":prefix_result["exact"]["latencyP50Ms"],"exact_p95_ms":prefix_result["exact"]["latencyP95Ms"],"ann_p50_ms":result["latencyP50Ms"],"ann_p95_ms":result["latencyP95Ms"],
          "recall":result["recall"][str(k)],"vote_agreement":result["voteAgreement"][str(k)],"minimum_class_decision_agreement":result["minimumClassDecisionAgreement"][str(k)],
          "mean_candidate_loss":result["meanCandidateLoss"],"plan_evidence":proven,"index_build_seconds":build_seconds,"index_reserved_bytes":prefix_result["indexReservedBytes"]})
      space_rows.append(prefix_result)
      vector_ddl(f"DROP INDEX {index} ON dbo.{table};");cursor.execute(f"DROP TABLE dbo.{table};");conn.commit()
    maximum=space_rows[-1];best=max(maximum["oversampling"].values(),key=lambda value:value["recall"].get("10",0))
    if maximum["exact"]["latencyP95Ms"]<250:disposition="ANN_UNNEEDED_AT_SCALE"
    elif best["planEvidence"] and best["recall"].get("10",0)>=.95 and best["voteAgreement"].get("10",0)>=.98 and best["minimumClassDecisionAgreement"].get("10",0)>=.95 and best["latencyP95Ms"]<maximum["exact"]["latencyP95Ms"]:disposition="ANN_PRESERVES"
    else:disposition="ANN_DISTORTS"
    summary["spaces"][representation]={"achievedMaximum":source_count,"queryRows":len(queries),"prefixes":space_rows,"disposition":disposition}
    cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,rows_count,groups_count,detail_json)
      VALUES(%s,'ENG',%s,'diskann-prefix-sweep','achieved-maximum','recall_at_10',%s,%s,%s,%s)""",
      (run_id,representation,best["recall"].get("10"),source_count,len(queries),json.dumps(summary["spaces"][representation])));conn.commit()
  table=pd.DataFrame(table_rows);table.to_csv(run/"tables/ann-benchmark.csv",index=False)
  (run/"metrics/ann.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
  lines=["# Exact versus SQL DiskANN", "", "Exact search is the scientific ground truth. Approximate latency is claimable only where the captured executed plan names the vector index.", ""]
  for name,value in summary["spaces"].items():lines.extend([f"## {name}","",f"Disposition: `{value['disposition']}`. Achieved maximum: `{value.get('achievedMaximum',0)}` vectors.",""])
  (run/"reports/ANN_REPORT.md").write_text("\n".join(lines))
  print(json.dumps({"runId":run_id,"spaces":{key:{"maximum":value.get("achievedMaximum"),"disposition":value.get("disposition")} for key,value in summary["spaces"].items()},"rows":len(table)},indent=2))
finally:
  for space in spaces.values():
    try:
      conn.rollback();conn.autocommit(True);cursor.execute(f"IF OBJECT_ID(N'dbo.{space['scratch']}',N'U') IS NOT NULL DROP TABLE dbo.{space['scratch']};");conn.autocommit(False)
    except Exception:
      try:conn.autocommit(False);conn.rollback()
      except Exception:pass
  conn.close()
