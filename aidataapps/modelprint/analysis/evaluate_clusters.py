#!/usr/bin/env python3
"""Label-hidden clustering, nuisance alignment, and seed-diversity analysis."""
import argparse, json, os, pickle
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import pymssql
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, normalize

SEED=20260822
parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--bootstrap",type=int,default=1000);parser.add_argument("--permutations",type=int,default=200);parser.add_argument("--stability",type=int,default=50);args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
campaign=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"];run_id=run.name;conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",
 password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=3600)
meta=pd.read_sql("""SELECT g.generation_id,g.model_profile_id,g.prompt_variant_id,v.prompt_group_id,v.carrier_id,p.family,p.domain,p.stratum,p.split,
 JSON_VALUE(d.config_json,'$.key') decode_key,g.length_band,g.reference_token_count,g.truncated FROM dbo.generations g
 JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
 JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id=%s""",conn,params=(campaign,))
test=meta[(meta.split=="test_id")&meta.decode_key.isin(["det","nat-0","nat-1"])&meta.carrier_id.ne("structured-v1")&(meta.truncated==0)&meta.reference_token_count.fillna(0).ge(16)].copy().sort_values("generation_id").reset_index(drop=True)
def parse(value):return np.asarray(json.loads(value),dtype=np.float32)
def load(query):
 rows=pd.read_sql(query,conn);mapping=dict(zip(rows.generation_id,rows.vector));frame=test[test.generation_id.isin(mapping)].copy().reset_index(drop=True);return frame,np.stack([parse(mapping[value]) for value in frame.generation_id])
spaces={}
spaces["semantic1024-qwen-raw-final-v1"]=load("""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
 JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id
 AND s.embedding_profile_id='qwen3-embedding-0.6b' AND s.representation_id='whole-raw-final-v1' WHERE g.campaign_id=%s"""%campaign)
spaces["style512-raw-final-v1"]=load("""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
 JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE g.campaign_id=%s"""%campaign)
for name in ["residual-ridge-v1","prompt-centered-v1"]:
 rows=pd.read_sql(f"SELECT COUNT(*) count FROM dbo.residual_vectors WHERE representation_id='{name}'",conn)
 if int(rows.iloc[0]["count"]):spaces[name]=load(f"SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.residual_vectors WHERE representation_id='{name}'")
rows=pd.read_sql("SELECT COUNT(*) count FROM dbo.fingerprint_vectors WHERE representation_id='fingerprint64-v1'",conn)
if int(rows.iloc[0]["count"]):spaces["fingerprint64-v1"]=load("SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.fingerprint_vectors WHERE representation_id='fingerprint64-v1'")

def fit_algorithm(name,X,seed=SEED):
 if name=="kmeans-4":return KMeans(n_clusters=4,n_init=20,random_state=seed).fit_predict(X)
 if name=="agglomerative-4":return AgglomerativeClustering(n_clusters=4,metric="cosine",linkage="average").fit_predict(X)
 return hdbscan.HDBSCAN(min_cluster_size=max(20,len(X)//100),metric="euclidean",core_dist_n_jobs=1).fit_predict(X)
def purity(truth,cluster):
 return float(sum(pd.Series(truth[cluster==value]).value_counts().max() for value in np.unique(cluster))/len(truth))
def grouped_bootstrap(frame,cluster,truth,n,rng):
 indices={group:np.flatnonzero(frame.prompt_group_id.to_numpy()==group) for group in frame.prompt_group_id.unique()};groups=list(indices);values=[]
 for _ in range(n):
  take=np.concatenate([indices[group] for group in rng.choice(groups,len(groups),replace=True)]);values.append(normalized_mutual_info_score(truth[take],cluster[take]))
 return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]
def permuted_model_null(frame,cluster,truth,n,rng):
 group_indices=[np.asarray(values,dtype=int) for values in frame.groupby("prompt_group_id").indices.values()];values=[]
 for _ in range(n):
  shuffled=truth.copy()
  for idx in group_indices:
   part=shuffled[idx].copy();rng.shuffle(part);shuffled[idx]=part
  values.append(normalized_mutual_info_score(shuffled,cluster))
 return values

result={"schemaVersion":1,"runId":run_id,"campaignId":campaign,"labelsHiddenDuringFit":True,"representations":{}};table_rows=[];cursor=conn.cursor()
for rep_index,(representation,(frame,X)) in enumerate(spaces.items()):
 X=normalize(X);truth=LabelEncoder().fit_transform(frame.model_profile_id);nuisance={name:LabelEncoder().fit_transform(frame[name].astype(str)) for name in ["family","domain","carrier_id","length_band","decode_key"]}
 projection=PCA(n_components=2,random_state=SEED).fit_transform(X);rep={}
 for algorithm in ["kmeans-4","agglomerative-4","hdbscan"]:
  labels=fit_algorithm(algorithm,X);rng=np.random.default_rng(SEED+rep_index*100+len(rep));model_nmi=float(normalized_mutual_info_score(truth,labels));null=permuted_model_null(frame,labels,truth,args.permutations,rng)
  boot=grouped_bootstrap(frame,labels,truth,args.bootstrap,rng);alignment={key:float(normalized_mutual_info_score(value,labels)) for key,value in nuisance.items()}
  stability=[]
  for repeat in range(args.stability):
   chosen=rng.choice(len(X),max(20,int(len(X)*.8)),replace=False);sub=fit_algorithm(algorithm,X[chosen],SEED+repeat+1);stability.append(adjusted_rand_score(labels[chosen],sub))
  metric={"rows":len(frame),"clusters":int(len(set(labels))-(1 if -1 in labels else 0)),"noiseRate":float((labels==-1).mean()),"modelNmi":model_nmi,
   "modelNmiCi95":boot,"modelAri":float(adjusted_rand_score(truth,labels)),"purity":purity(truth,labels),"silhouette":float(silhouette_score(X,labels,metric="cosine")) if len(set(labels))>1 else None,
   "nuisanceNmi":alignment,"permutationNull":{"mean":float(np.mean(null)),"p95":float(np.quantile(null,.95))},"bootstrapStabilityAri":{"mean":float(np.mean(stability)),"p05":float(np.quantile(stability,.05))}}
  supported=boot[0]>metric["permutationNull"]["p95"] and model_nmi>max(alignment.values()) and metric["bootstrapStabilityAri"]["p05"]>=.5
  metric["taxonomy"]="MODEL_ALIGNED_CLUSTERS" if supported else "CLUSTER_VISUAL_ONLY";rep[algorithm]=metric
  cursor.execute("INSERT dbo.cluster_runs(run_id,representation_id,algorithm,labels_hidden,config_json) VALUES(%s,%s,%s,1,%s); SELECT SCOPE_IDENTITY()",(run_id,representation,algorithm,json.dumps({"seed":SEED,"semiSupervisedProjection":representation=="fingerprint64-v1"})));cluster_run=int(cursor.fetchone()[0])
  cursor.executemany("INSERT dbo.cluster_assignments(cluster_run_id,generation_id,cluster_label) VALUES(%s,%s,%s)",[(cluster_run,int(gid),int(label)) for gid,label in zip(frame.generation_id,labels)])
  cursor.executemany("INSERT dbo.projection_coordinates(cluster_run_id,generation_id,x,y,method) VALUES(%s,%s,%s,%s,'pca-2')",[(cluster_run,int(gid),float(point[0]),float(point[1])) for gid,point in zip(frame.generation_id,projection)])
  cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
   VALUES(%s,'CLUSTER',%s,%s,'test_id','model_nmi',%s,%s,%s,%s,%s,%s,%s,%s); SELECT SCOPE_IDENTITY()""",(run_id,representation,algorithm,model_nmi,boot[0],boot[1],metric["permutationNull"]["mean"],metric["permutationNull"]["p95"],len(frame),frame.prompt_group_id.nunique(),json.dumps(metric)));metric_id=int(cursor.fetchone()[0])
  cursor.execute("INSERT dbo.claims(run_id,research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale,metric_result_id) VALUES(%s,'RQ7','CLUSTER',%s,%s,'test_id',%s,%s,%s,%s)",
   (run_id,representation,algorithm,metric["taxonomy"],supported,f"Model NMI {model_nmi:.4f}; maximum nuisance NMI {max(alignment.values()):.4f}; permutation p95 {metric['permutationNull']['p95']:.4f}.",metric_id));conn.commit()
  pd.DataFrame({"value":null}).to_csv(run/f"tables/permutation-cluster-{representation}-{algorithm}.csv",index=False)
  table_rows.append({"representation":representation,"algorithm":algorithm,**{key:value for key,value in metric.items() if key in ["rows","clusters","noiseRate","modelNmi","modelAri","purity","silhouette","taxonomy"]},**{f"nmi_{key}":value for key,value in alignment.items()}})
 result["representations"][representation]=rep
pd.DataFrame(table_rows).to_csv(run/"tables/cluster-alignment.csv",index=False);(run/"metrics/clusters.json").write_text(json.dumps(result,indent=2)+"\n")

# Natural-seed diversity uses exactly matched prompt variants and served profiles.
seed_meta=meta[meta.decode_key.isin(["nat-0","nat-1"])].copy();wide=seed_meta.pivot_table(index=["model_profile_id","prompt_variant_id","prompt_group_id","carrier_id"],columns="decode_key",values="generation_id",aggfunc="first").dropna().reset_index()
diversity=[]
for representation,(frame,X) in {key:value for key,value in spaces.items() if key in ["semantic1024-qwen-raw-final-v1","style512-raw-final-v1"]}.items():
 mapping={int(gid):normalize(X[index:index+1])[0] for index,gid in enumerate(frame.generation_id)}
 for _,row in wide.iterrows():
  left=mapping.get(int(row["nat-0"]));right=mapping.get(int(row["nat-1"]))
  if left is not None and right is not None:diversity.append({"representation":representation,"model_profile_id":row.model_profile_id,"prompt_variant_id":row.prompt_variant_id,"prompt_group_id":row.prompt_group_id,"carrier_id":row.carrier_id,"distance":float(1-left@right)})
pd.DataFrame(diversity).to_csv(run/"tables/seed-diversity.csv",index=False)
(run/"reports/CLUSTER_REPORT.md").write_text("# Label-hidden Cluster Report\n\nEvery fit hid model and nuisance labels. PCA coordinates are visualization-only. See `tables/cluster-alignment.csv` and `metrics/clusters.json` for retained metrics, nulls, and stability.\n")
print(json.dumps({"runId":run_id,"representations":list(result["representations"]),"clusterRows":len(table_rows),"seedPairs":len(diversity)},indent=2));conn.close()
