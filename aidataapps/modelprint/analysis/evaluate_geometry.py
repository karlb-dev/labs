#!/usr/bin/env python3
"""Compute sampled pairwise vector geometry inside SQL Server."""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--sample",type=int,default=2000);parser.add_argument("--regression-sample",type=int,default=200000);args=parser.parse_args()
if not 100<=args.sample<=5000:raise SystemExit("--sample must be 100..5000")
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
campaign=int(freeze["campaignId"]);run_id=run.name;conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",
 password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=7200);cursor=conn.cursor()
eligibility="g.campaign_id=@campaign AND g.split='train' AND g.truncated=0 AND g.reference_token_count>=16 AND v.carrier_id<>'structured-v1' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')"
sources={
 "semantic1024-qwen-raw-final-v1":f"""SELECT g.generation_id,g.model_profile_id,v.prompt_group_id,p.family,v.carrier_id,JSON_VALUE(d.config_json,'$.key') decode_key,g.reference_token_count,s.embedding
 FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id
 JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
 JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1' WHERE {eligibility}""",
 "style512-raw-final-v1":f"""SELECT g.generation_id,g.model_profile_id,v.prompt_group_id,p.family,v.carrier_id,JSON_VALUE(d.config_json,'$.key') decode_key,g.reference_token_count,s.embedding
 FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id
 JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1'
 JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE {eligibility}""",
 "fingerprint64-v1":f"""SELECT g.generation_id,g.model_profile_id,v.prompt_group_id,p.family,v.carrier_id,JSON_VALUE(d.config_json,'$.key') decode_key,g.reference_token_count,s.embedding
 FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id
 JOIN dbo.fingerprint_vectors s ON s.generation_id=g.generation_id AND s.representation_id='fingerprint64-v1' WHERE {eligibility}""",
}
summary=[]
for representation,source in sources.items():
 cursor.execute("SELECT COUNT(*) FROM dbo.geometry_pairs WHERE run_id=%s AND representation_id=%s",(run_id,representation));existing=int(cursor.fetchone()[0])
 started=time.perf_counter()
 if not existing:
  cursor.execute(f"""DECLARE @campaign bigint=%s,@sample int=%s,@run varchar(120)=%s,@representation varchar(80)=%s;
   WITH source AS ({source}), sampled AS (SELECT TOP (@sample) * FROM source ORDER BY HASHBYTES('SHA2_256',CONVERT(varbinary(8),generation_id)),generation_id)
   INSERT dbo.geometry_pairs(run_id,representation_id,left_generation_id,right_generation_id,distance,same_model,same_prompt_group,same_family,same_carrier,same_decode,length_difference)
   SELECT @run,@representation,a.generation_id,b.generation_id,VECTOR_DISTANCE('cosine',a.embedding,b.embedding),
    IIF(a.model_profile_id=b.model_profile_id,1,0),IIF(a.prompt_group_id=b.prompt_group_id,1,0),IIF(a.family=b.family,1,0),IIF(a.carrier_id=b.carrier_id,1,0),IIF(a.decode_key=b.decode_key,1,0),ABS(a.reference_token_count-b.reference_token_count)
   FROM sampled a JOIN sampled b ON a.generation_id<b.generation_id;""",(campaign,args.sample,run_id,representation));conn.commit()
 wall=time.perf_counter()-started
 aggregate=pd.read_sql("""SELECT CASE WHEN same_prompt_group=1 AND same_model=0 THEN 'same-prompt-different-model'
   WHEN same_prompt_group=0 AND same_model=1 THEN 'different-prompt-same-model' ELSE 'different-prompt-different-model' END contrast,
   COUNT(*) rows,AVG(distance) mean_distance,STDEV(distance) sd_distance
   FROM dbo.geometry_pairs WHERE run_id=%s AND representation_id=%s GROUP BY CASE WHEN same_prompt_group=1 AND same_model=0 THEN 'same-prompt-different-model'
   WHEN same_prompt_group=0 AND same_model=1 THEN 'different-prompt-same-model' ELSE 'different-prompt-different-model' END""",conn,params=(run_id,representation))
 oriented=pd.read_sql("""WITH oriented AS (SELECT left_generation_id query_id,right_generation_id neighbor_id,distance,same_model,same_prompt_group FROM dbo.geometry_pairs WHERE run_id=%s AND representation_id=%s
   UNION ALL SELECT right_generation_id,left_generation_id,distance,same_model,same_prompt_group FROM dbo.geometry_pairs WHERE run_id=%s AND representation_id=%s), scored AS
   (SELECT query_id,MIN(CASE WHEN same_model=1 AND same_prompt_group=0 THEN distance END) same_model_distance,MIN(CASE WHEN same_model=0 AND same_prompt_group=1 THEN distance END) same_prompt_distance FROM oriented GROUP BY query_id)
   SELECT COUNT(*) queries,AVG(IIF(same_model_distance<same_prompt_distance,1.0,0.0)) retrieval_win_rate,AVG(same_model_distance) mean_same_model,AVG(same_prompt_distance) mean_same_prompt
   FROM scored WHERE same_model_distance IS NOT NULL AND same_prompt_distance IS NOT NULL""",conn,params=(run_id,representation,run_id,representation)).iloc[0].to_dict()
 regression=pd.read_sql("""SELECT TOP (%s) distance,same_model,same_prompt_group,same_family,same_carrier,same_decode,length_difference FROM dbo.geometry_pairs
   WHERE run_id=%s AND representation_id=%s ORDER BY geometry_pair_id""",conn,params=(args.regression_sample,run_id,representation))
 features=["same_model","same_prompt_group","same_family","same_carrier","same_decode","length_difference"];scaled=StandardScaler().fit_transform(regression[features]);fit=LinearRegression().fit(scaled,regression.distance)
 coefficients={feature:float(value) for feature,value in zip(features,fit.coef_)};detail={"representation":representation,"sampleVectors":args.sample,"pairRows":existing or args.sample*(args.sample-1)//2,
   "sqlWallSeconds":wall,"contrasts":aggregate.to_dict(orient="records"),"retrievalWin":oriented,"standardizedRegressionCoefficients":coefficients,"regressionR2":float(fit.score(scaled,regression.distance))}
 (run/f"tables/geometry-contrasts-{representation}.csv").write_text(aggregate.to_csv(index=False));summary.append(detail)
 cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
   VALUES(%s,'RETRIEVE',%s,'sql-pairwise-geometry','train-sample','retrieval_win_rate',%s,NULL,NULL,NULL,NULL,%s,%s,%s)""",
   (run_id,representation,float(oriented.get("retrieval_win_rate") or 0),detail["pairRows"],int(oriented.get("queries") or 0),json.dumps(detail)));conn.commit()
pd.DataFrame(summary).to_json(run/"tables/geometry-summary.json",orient="records",indent=2);(run/"metrics/geometry.json").write_text(json.dumps({"schemaVersion":1,"runId":run_id,"sample":args.sample,"representations":summary},indent=2)+"\n")
(run/"reports/PROMPT_DOMINANCE.md").write_text("# Prompt-dominance Geometry\n\nThe retained SQL pair sample compares same-prompt/different-profile distances with different-prompt/same-profile distances. Positive retrieval win rate means model proximity beats prompt proximity for that query. See `metrics/geometry.json`.\n")
print(json.dumps({"runId":run_id,"representations":[row["representation"] for row in summary],"sample":args.sample},indent=2));conn.close()
