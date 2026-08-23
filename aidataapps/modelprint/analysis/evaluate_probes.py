#!/usr/bin/env python3
"""Fit calibrated text-only probes with grouped nulls and LOFO rotations."""
import argparse, hashlib, json, math, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pymssql
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support, top_k_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

SEED=20260822
parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--tier",default="full",choices=["smoke","dev","standard","full"])
parser.add_argument("--permutations",type=int,default=200);parser.add_argument("--bootstrap",type=int,default=1000);parser.add_argument("--jobs",type=int,default=12)
parser.add_argument("--resume-completed",action="store_true",help="Reuse a representation only when its fitted model and complete null tables exist")
parser.add_argument("--representations",help="Comma-separated IDs; default all available");args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve()
freeze=json.loads((run/"manifests/campaign-freeze.json").read_text());primary=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"];run_id=run.name
robust_path=run/"manifests/robustness-freeze.json";robust=int(json.loads(robust_path.read_text())["campaignId"]) if robust_path.exists() else None
campaign_ids=[primary]+([robust] if robust else []);campaign_sql=",".join(map(str,campaign_ids));tier=json.loads((lab/f"data/manifests/tier-{args.tier}.json").read_text());tier_ids=set(tier["variantIds"])
conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",password=os.environ["MSSQL_SA_PASSWORD"],
                     database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=3600)
base=pd.read_sql(f"""SELECT g.generation_id,g.campaign_id,g.model_profile_id,g.prompt_variant_id,v.prompt_group_id,v.carrier_id,
 p.prompt_source_id source_id,p.family,p.domain,p.stratum,p.split,JSON_VALUE(d.config_json,'$.key') decode_key,JSON_VALUE(v.metadata_json,'$.suite') variant_suite,
 g.reference_token_count,g.output_word_count,g.length_band,g.truncated,g.final_text
FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id
JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id IN ({campaign_sql})""",conn)
base=base[((base.campaign_id==primary)&base.prompt_variant_id.isin(tier_ids))|(base.campaign_id!=primary)].copy()
base=base[(base.truncated==0)&base.final_text.str.len().gt(0)&base.reference_token_count.fillna(0).ge(16)].sort_values("generation_id").reset_index(drop=True)
label_id={model:index for index,model in enumerate(models)};base["label"]=base.model_profile_id.map(label_id).astype(int)
base["family_bucket"]=base.apply(lambda r:(f"dolly:{r.family}" if r.source_id=="dolly" else
  "relation" if r.source_id in {"relation-probes","advanced-relations"} else "sae" if r.source_id=="sae-corpus" else
  "certainty" if r.source_id=="certainty" else "steering" if r.source_id=="steering" else "persona-content" if r.source_id=="persona" else
  "sycophancy" if r.source_id=="sycophancy" else "belief" if r.source_id=="belief-revision" else "g1" if r.source_id=="g1" else
  "oasst1" if r.source_id=="oasst1" else "rag-grounded" if r.source_id=="rag-lab1" else str(r.family)),axis=1)

def assign_suite(row):
  if int(row.campaign_id)==primary:
    if row.split=="test_id" and row.decode_key in {"det","nat-0","nat-1"} and row.carrier_id!="structured-v1":return "test_id"
    if row.split=="test_source_holdout" and row.decode_key in {"det","nat-0","nat-1"} and row.carrier_id!="structured-v1":return "test_source_holdout"
    if row.split=="test_id" and row.decode_key in {"det","nat-0","nat-1"} and row.carrier_id=="structured-v1":return "test_carrier_holdout"
    if row.split=="test_id" and row.decode_key=="hv" and row.carrier_id!="structured-v1":return "test_decode_shift"
  elif row.carrier_id=="persona-v1":return "persona"
  elif row.carrier_id=="rag-grounded-v1":return "rag_grounded"
  elif row.variant_suite=="pressure":return "pressure"
  return None
base["suite"]=base.apply(assign_suite,axis=1)
train=((base.campaign_id==primary)&base.split.eq("train")&base.decode_key.ne("hv")&base.carrier_id.ne("structured-v1")).to_numpy()
cal=((base.campaign_id==primary)&base.split.eq("calibration")&base.decode_key.ne("hv")&base.carrier_id.ne("structured-v1")).to_numpy()
suite_masks={suite:(base.suite==suite).to_numpy() for suite in sorted(base.suite.dropna().unique())}

def parse(value):return np.asarray(json.loads(value),dtype=np.float32)
def aligned(query):
  rows=pd.read_sql(query,conn);mapping=dict(zip(rows.generation_id,rows.vector));mask=base.generation_id.isin(mapping).to_numpy();matrix=np.zeros((len(base),len(parse(next(iter(mapping.values()))))),dtype=np.float32)
  for index,gid in enumerate(base.generation_id):
    if gid in mapping:matrix[index]=parse(mapping[gid])
  return matrix,mask

representations={}
for profile,key in [("qwen3-embedding-0.6b","qwen"),("bge-large-en-v1.5","bge")]:
  for view in ["raw-final-v1","name-masked-v1"]:
    representations[f"semantic1024-{key}-{view}"]=aligned(f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
      JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='{view}'
      JOIN dbo.semantic_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.embedding_profile_id='{profile}' AND s.representation_id='whole-{view}'
      WHERE g.campaign_id IN ({campaign_sql})""")
for view in ["raw-final-v1","name-masked-v1"]:
  representations[f"style512-{view}"]=aligned(f"""SELECT g.generation_id,CAST(s.embedding AS nvarchar(max)) vector FROM dbo.generations g
    JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='{view}'
    JOIN dbo.style_vectors s ON s.text_artifact_id=t.text_artifact_id AND s.representation_id='style512-v1' WHERE g.campaign_id IN ({campaign_sql})""")
for name in ["residual-diff-v1","residual-reject-v1","residual-ridge-v1","prompt-centered-v1"]:
  rows=pd.read_sql(f"SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.residual_vectors WHERE representation_id='{name}'",conn)
  if len(rows):representations[name]=aligned(f"SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.residual_vectors WHERE representation_id='{name}'")
rows=pd.read_sql("SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.likelihood_profile_vectors WHERE representation_id='likelihood-profile8-v1'",conn)
if len(rows):representations["likelihood-profile8-v1"]=aligned("SELECT generation_id,CAST(embedding AS nvarchar(max)) vector FROM dbo.likelihood_profile_vectors WHERE representation_id='likelihood-profile8-v1'")
paragraphs=base.final_text.map(lambda text:max(1,len([part for part in str(text).split("\n\n") if part.strip()]))).to_numpy(dtype=float)
representations["verbosity-only-v1"]=(np.column_stack([np.log1p(base.reference_token_count.to_numpy(dtype=float)),np.log1p(paragraphs)]).astype(np.float32),np.ones(len(base),dtype=bool))

# Frozen floor for fingerprint64: training-only scaling + PCA128 + LDA3, padded and normalized.
if "semantic1024-qwen-raw-final-v1" in representations and "style512-raw-final-v1" in representations:
  semantic,smask=representations["semantic1024-qwen-raw-final-v1"];style,stmask=representations["style512-raw-final-v1"];available=smask&stmask
  combined=np.concatenate([semantic,style],axis=1);scaler=StandardScaler();scaled_train=scaler.fit_transform(combined[train&available]);scaled_all=scaler.transform(combined)
  pca=PCA(n_components=min(128,scaled_train.shape[0]-1,scaled_train.shape[1]),svd_solver="randomized",random_state=SEED);train_pca=pca.fit_transform(scaled_train);all_pca=pca.transform(scaled_all)
  lda=LinearDiscriminantAnalysis(n_components=min(3,len(models)-1));lda.fit(train_pca,base.label.to_numpy()[train&available]);projected=lda.transform(all_pca).astype(np.float32)
  fingerprint=np.zeros((len(base),64),dtype=np.float32);fingerprint[:,:projected.shape[1]]=projected;fingerprint/=np.maximum(np.linalg.norm(fingerprint,axis=1,keepdims=True),1e-12)
  projection_path=run/"manifests/fingerprint64-projection.pkl";projection_path.write_bytes(pickle.dumps({"scaler":scaler,"pca":pca,"lda":lda,"method":"pca128-lda3-padded64"}))
  projection_hash=hashlib.sha256(projection_path.read_bytes()).hexdigest();representations["fingerprint64-v1"]=(fingerprint,available)
  cursor=conn.cursor();records=[(int(gid),"fingerprint64-v1",projection_hash,json.dumps(vector.tolist()),run_id) for gid,vector,ok in zip(base.generation_id,fingerprint,available) if ok]
  for start in range(0,len(records),500):
    cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.fingerprint_vectors WHERE generation_id=%s AND representation_id=%s)
      INSERT dbo.fingerprint_vectors(generation_id,representation_id,projection_manifest_hash,embedding,created_by_run) VALUES(%s,%s,%s,CAST(%s AS vector(64)),%s)""",
      [(r[0],r[1],*r) for r in records[start:start+500]]);conn.commit()
else:projection_hash=None
representation_indices={name:index for index,name in enumerate(representations)}
if args.representations:
  requested=args.representations.split(",");missing=set(requested)-set(representations)
  if missing:raise SystemExit(f"Unavailable representations: {sorted(missing)}")
  representations={name:representations[name] for name in requested}

def softmax(logits,temp=1):
  values=logits/max(temp,1e-4);values-=values.max(axis=1,keepdims=True);values=np.exp(values);return values/values.sum(axis=1,keepdims=True)
def fit_temp(logits,truth):return float(math.exp(minimize_scalar(lambda t:log_loss(truth,softmax(logits,math.exp(t)),labels=np.arange(len(models))),bounds=(-4,4),method="bounded").x))
def adaptive_ece(truth,prob,bins=10):
  conf=prob.max(1);pred=prob.argmax(1);parts=np.array_split(np.argsort(conf),bins);return float(sum(len(part)/len(truth)*abs((pred[part]==truth[part]).mean()-conf[part].mean()) for part in parts if len(part)))
def classwise_ece(truth,prob,bins=10):
  output={}
  for index,model in enumerate(models):
    confidence=prob[:,index];target=(truth==index).astype(float);parts=np.array_split(np.argsort(confidence),bins)
    output[model]=float(sum(len(part)/len(truth)*abs(target[part].mean()-confidence[part].mean()) for part in parts if len(part)))
  return output
def reliability_bins(truth,prob,bins=15):
  confidence=prob.max(1);pred=prob.argmax(1);edges=np.linspace(0,1,bins+1);output=[]
  for index in range(bins):
    mask=(confidence>=edges[index])&(confidence<(edges[index+1] if index<bins-1 else edges[index+1]+1e-12))
    output.append({"lower":float(edges[index]),"upper":float(edges[index+1]),"rows":int(mask.sum()),
      "meanConfidence":float(confidence[mask].mean()) if mask.any() else None,"accuracy":float((pred[mask]==truth[mask]).mean()) if mask.any() else None})
  return output
def conformal_q(prob,truth,alpha=.1):
  score=1-prob[np.arange(len(truth)),truth];level=min(1,math.ceil((len(score)+1)*(1-alpha))/len(score));return float(np.quantile(score,level,method="higher"))
def classifier(alpha,seed):return SGDClassifier(loss="log_loss",penalty="l2",alpha=alpha,max_iter=250,tol=1e-4,class_weight="balanced",random_state=seed,early_stopping=False,average=True)
def pipeline(alpha,seed):return make_pipeline(StandardScaler(),classifier(alpha,seed))
def choose_alpha(X,y,mask,groups):
  indices=np.flatnonzero(mask);folds=list(GroupKFold(n_splits=3).split(X[indices],y[indices],groups[indices]));alphas=[1e-5,1e-4,1e-3];best=None
  def one_cv(alpha,left,right):
    with threadpool_limits(limits=1):
      model=pipeline(alpha,SEED);model.fit(X[indices[left]],y[indices[left]]);return f1_score(y[indices[right]],model.predict(X[indices[right]]),average="macro",labels=np.arange(len(models)),zero_division=0)
  values=Parallel(n_jobs=min(args.jobs,len(alphas)*len(folds)),prefer="threads")(delayed(one_cv)(alpha,left,right) for alpha in alphas for left,right in folds)
  for alpha_index,alpha in enumerate(alphas):
    scores=values[alpha_index*len(folds):(alpha_index+1)*len(folds)]
    candidate=(float(np.mean(scores)),-alpha)
    if best is None or candidate>best[0]:best=(candidate,alpha,scores)
  return best[1],best[2]
def bootstrap(frame,pred,n,rng):
  group_indices={group:np.flatnonzero(frame.prompt_group_id.to_numpy()==group) for group in frame.prompt_group_id.unique()};groups=list(group_indices);values=[]
  for _ in range(n):
    take=np.concatenate([group_indices[group] for group in rng.choice(groups,len(groups),replace=True)]);values.append(f1_score(frame.label.to_numpy()[take],pred[take],average="macro",labels=np.arange(len(models)),zero_division=0))
  return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]
def metrics(frame,prob,bootstrap_n,rng,q):
  truth=frame.label.to_numpy();pred=prob.argmax(1);confidence=prob.max(1);order=np.argsort(-confidence);half=order[:max(1,len(order)//2)];sets=prob>=(1-q);coverage85=0
  for fraction in np.linspace(.05,1,20):
    idx=order[:max(1,int(len(order)*fraction))]
    if accuracy_score(truth[idx],pred[idx])>=.85:coverage85=float(fraction)
  per=precision_recall_fscore_support(truth,pred,labels=np.arange(len(models)),zero_division=0)
  return {"rows":len(frame),"groups":int(frame.prompt_group_id.nunique()),"accuracy":float(accuracy_score(truth,pred)),"balancedAccuracy":float(balanced_accuracy_score(truth,pred)),
    "macroF1":float(f1_score(truth,pred,average="macro",labels=np.arange(len(models)),zero_division=0)),"macroF1Ci95":bootstrap(frame,pred,bootstrap_n,rng),
    "top2Accuracy":float(top_k_accuracy_score(truth,prob,k=2,labels=np.arange(len(models)))),"nll":float(log_loss(truth,prob,labels=np.arange(len(models)))),
    "brier":float(np.mean(np.sum((prob-np.eye(len(models))[truth])**2,axis=1))),"eceAdaptive10":adaptive_ece(truth,prob),"classwiseEceAdaptive10":classwise_ece(truth,prob),
    "reliabilityEqualWidth15":reliability_bins(truth,prob),
    "selectiveAccuracyAt50Coverage":float(accuracy_score(truth[half],pred[half])),"coverageAt85Accuracy":coverage85,
    "conformalCoverage":float(sets[np.arange(len(truth)),truth].mean()),"conformalMeanSetSize":float(sets.sum(1).mean()),
    "perClass":{models[i]:{"precision":float(per[0][i]),"recall":float(per[1][i]),"f1":float(per[2][i]),"support":int(per[3][i])} for i in range(len(models))},
    "confusion":confusion_matrix(truth,pred,labels=np.arange(len(models))).tolist(),"predictions":pred,"confidence":confidence,"sets":sets}
def permute_labels(y,groups,rng):
  shuffled=y.copy()
  for indices in pd.Series(np.arange(len(y))).groupby(groups).groups.values():
    idx=np.asarray(list(indices));values=shuffled[idx].copy();rng.shuffle(values);shuffled[idx]=values
  return shuffled

y=base.label.to_numpy();groups=base.prompt_group_id.to_numpy();results={"schemaVersion":2,"runId":run_id,"campaignIds":campaign_ids,"campaignHash":freeze["campaignHash"],
 "tier":args.tier,"models":models,"rows":len(base),"dataRoles":{"train":int(train.sum()),"calibration":int(cal.sum()),**{key:int(mask.sum()) for key,mask in suite_masks.items()}},
 "classifier":"standardized-sgd-multinomial-logistic","permutations":args.permutations,"bootstrap":args.bootstrap,
 "execution":{"jobs":args.jobs,"resumeCompleted":args.resume_completed,"resumedRepresentations":[],"resumedResultCheckpoints":[],"resumedSuiteNulls":[],"resumedLofoNulls":[],"resumedFamilyNulls":[],"skippedIneligibleFamilyNullFits":0},"representations":{}}
prediction_frames=[];cursor=conn.cursor();artifact_paths=[]
for name,(X,available) in representations.items():
  rep_index=representation_indices[name];prediction_checkpoint_start=len(prediction_frames)
  fit_mask=train&available;cal_mask=cal&available
  if len(set(y[fit_mask]))<len(models) or len(set(y[cal_mask]))<len(models):continue
  active_suites={key:(mask&available) for key,mask in suite_masks.items() if (mask&available).sum()};model_path=run/f"manifests/probe-{name}.pkl"
  suite_null_paths={suite:run/f"tables/permutation-probe-{name}-{suite}.csv" for suite in active_suites};lofo_null_path=run/f"tables/permutation-probe-{name}-lofo.csv";result_checkpoint_path=run/f"manifests/probe-result-{name}.pkl"
  resume_ready=bool(args.resume_completed and model_path.exists() and lofo_null_path.exists() and all(path.exists() for path in suite_null_paths.values()))
  if resume_ready and result_checkpoint_path.exists():
    saved=pickle.loads(result_checkpoint_path.read_bytes());artifact_hashes={"model":hashlib.sha256(model_path.read_bytes()).hexdigest(),"lofo":hashlib.sha256(lofo_null_path.read_bytes()).hexdigest(),**{f"suite:{suite}":hashlib.sha256(path.read_bytes()).hexdigest() for suite,path in suite_null_paths.items()}}
    if saved.get("schemaVersion")!=1 or saved.get("representation")!=name or saved.get("campaignHash")!=freeze["campaignHash"] or saved.get("models")!=models or saved.get("tier")!=args.tier or saved.get("permutations")!=args.permutations or saved.get("bootstrap")!=args.bootstrap or saved.get("rows")!=len(base) or saved.get("artifactHashes")!=artifact_hashes or not isinstance(saved.get("predictions"),pd.DataFrame):raise SystemExit(f"Representation result checkpoint drift for {name}")
    results["representations"][name]=saved["result"];prediction_frames.append(saved["predictions"]);artifact_paths.append(str(model_path));results["execution"]["resumedRepresentations"].append(name);results["execution"]["resumedResultCheckpoints"].append(name);continue
  if resume_ready:
    results["execution"]["resumedRepresentations"].append(name)
    model_info=pickle.loads(model_path.read_bytes())
    if model_info.get("models")!=models:raise SystemExit(f"Resume model labels drift for {name}")
    model=model_info["pipeline"];alpha=float(model_info["alpha"]);temperature=float(model_info["temperature"]);q=float(model_info["conformalQ"])
    selected_alpha,cv_scores=choose_alpha(X,y,fit_mask,groups)
    if selected_alpha!=alpha:raise SystemExit(f"Resume alpha drift for {name}: {selected_alpha} != {alpha}")
    null_columns={suite:pd.read_csv(path)["value"].tolist() for suite,path in suite_null_paths.items()}
    if any(len(values)!=args.permutations for values in null_columns.values()):raise SystemExit(f"Incomplete suite null checkpoint for {name}")
    null_rows=[{suite:null_columns[suite][index] for suite in active_suites} for index in range(args.permutations)]
  else:
    alpha,cv_scores=choose_alpha(X,y,fit_mask,groups);model=pipeline(alpha,SEED);model.fit(X[fit_mask],y[fit_mask]);temperature=fit_temp(model.decision_function(X[cal_mask]),y[cal_mask])
    cal_prob=softmax(model.decision_function(X[cal_mask]),temperature);q=conformal_q(cal_prob,y[cal_mask])
    suite_checkpoint_ready=bool(args.resume_completed and all(path.exists() for path in suite_null_paths.values()))
    if suite_checkpoint_ready:
      null_columns={suite:pd.read_csv(path)["value"].tolist() for suite,path in suite_null_paths.items()}
      if any(len(values)!=args.permutations for values in null_columns.values()):raise SystemExit(f"Incomplete suite null checkpoint for {name}")
      null_rows=[{suite:null_columns[suite][index] for suite in active_suites} for index in range(args.permutations)]
      results["execution"]["resumedSuiteNulls"].append(name)
    else:
      null_scaler=StandardScaler();null_train=null_scaler.fit_transform(X[fit_mask]);null_cal=null_scaler.transform(X[cal_mask])
      null_suites={suite:null_scaler.transform(X[mask]) for suite,mask in active_suites.items()}
      def one_null(index):
        with threadpool_limits(limits=1):
          rng=np.random.default_rng(SEED+rep_index*100003+index);shuffled=permute_labels(y,groups,rng);candidate=classifier(alpha,SEED+index+1);candidate.fit(null_train,shuffled[fit_mask]);temp=fit_temp(candidate.decision_function(null_cal),shuffled[cal_mask])
          return {suite:f1_score(shuffled[mask],softmax(candidate.decision_function(null_suites[suite]),temp).argmax(1),average="macro",labels=np.arange(len(models)),zero_division=0) for suite,mask in active_suites.items()}
      null_rows=Parallel(n_jobs=min(args.jobs,args.permutations),prefer="threads")(delayed(one_null)(index) for index in range(args.permutations)) if args.permutations else []
  rep={"alpha":alpha,"groupedCvMacroF1":cv_scores,"temperature":temperature,"conformalQ90":q,"suites":{},"lofo":{}}
  for suite,mask in active_suites.items():
    frame=base.loc[mask].reset_index(drop=True);logits=model.decision_function(X[mask]);raw_prob=softmax(logits);prob=softmax(logits,temperature);metric=metrics(frame,prob,args.bootstrap,np.random.default_rng(SEED+rep_index),q)
    null=[row[suite] for row in null_rows];metric["permutationNull"]={"mean":float(np.mean(null)),"p95":float(np.quantile(null,.95))} if null else {"mean":None,"p95":None}
    pred=metric.pop("predictions");confidence=metric.pop("confidence");sets=metric.pop("sets");rep["suites"][suite]=metric
    out=frame[["generation_id","model_profile_id","prompt_group_id","source_id","family","family_bucket","carrier_id","decode_key","length_band","split"]].copy()
    out["representation"]=name;out["method"]="linear-probe";out["suite"]=suite;out["predicted_model_profile_id"]=[models[i] for i in pred];out["confidence"]=confidence
    out["probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in prob];out["uncalibrated_probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in raw_prob]
    out["candidate_set"]=[json.dumps([models[i] for i,value in enumerate(row) if value]) for row in sets];prediction_frames.append(out)
    pd.DataFrame({"value":null}).to_csv(run/f"tables/permutation-probe-{name}-{suite}.csv",index=False);pd.DataFrame(metric["confusion"],index=models,columns=models).to_csv(run/f"tables/confusion-probe-{name}-{suite}.csv")

  # Leave-one-family-out rotations use only held-out test_id rows and never expose that family during fit/calibration.
  lofo_predictions=[];lofo_metrics={};lofo_specs=[]
  test_id=active_suites.get("test_id",np.zeros(len(base),dtype=bool));families=sorted(base.loc[test_id,"family_bucket"].unique())
  for family in families:
    family_test=test_id&base.family_bucket.eq(family).to_numpy();family_train=fit_mask&base.family_bucket.ne(family).to_numpy();family_cal=cal_mask&base.family_bucket.ne(family).to_numpy()
    if family_test.sum()==0 or len(set(y[family_train]))<len(models) or len(set(y[family_cal]))<len(models):continue
    lofo_specs.append((family,family_train,family_cal,family_test,int(family_test.sum())))
  def one_lofo(spec_index,spec):
    family,family_train,family_cal,family_test,_=spec
    with threadpool_limits(limits=1):
      candidate=pipeline(alpha,SEED);candidate.fit(X[family_train],y[family_train]);family_logits=candidate.decision_function(X[family_cal]);family_temp=fit_temp(family_logits,y[family_cal]);family_q=conformal_q(softmax(family_logits,family_temp),y[family_cal])
      frame=base.loc[family_test].reset_index(drop=True);logits=candidate.decision_function(X[family_test]);raw_prob=softmax(logits);prob=softmax(logits,family_temp);metric=metrics(frame,prob,args.bootstrap,np.random.default_rng(SEED+spec_index),family_q)
    metric["permutationNull"]={"mean":None,"p95":None,"disposition":"computed-at-aggregate-rotation"};pred=metric.pop("predictions");confidence=metric.pop("confidence");sets=metric.pop("sets")
    out=frame[["generation_id","model_profile_id","prompt_group_id","source_id","family","family_bucket","carrier_id","decode_key","length_band","split"]].copy();out["representation"]=name;out["method"]="linear-probe";out["suite"]=f"lofo:{family}"
    out["predicted_model_profile_id"]=[models[i] for i in pred];out["confidence"]=confidence;out["probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in prob]
    out["uncalibrated_probabilities"]=[json.dumps({models[i]:float(row[i]) for i in range(len(models))}) for row in raw_prob];out["candidate_set"]=[json.dumps([models[i] for i,value in enumerate(row) if value]) for row in sets];return family,metric,out
  evaluated_lofo=Parallel(n_jobs=min(args.jobs,len(lofo_specs)),prefer="threads")(delayed(one_lofo)(index,spec) for index,spec in enumerate(lofo_specs)) if lofo_specs else []
  for family,metric,out in evaluated_lofo:
    lofo_metrics[family]=metric;lofo_predictions.append(out)
  if lofo_metrics:
    eligible=[value for value in lofo_metrics.values() if value["rows"]>=100] or list(lofo_metrics.values());macro=[value["macroF1"] for value in eligible]
    lofo_checkpoint_ready=bool(args.resume_completed and lofo_null_path.exists())
    if lofo_checkpoint_ready:
      resumed_lofo=pd.read_csv(lofo_null_path)
      if len(resumed_lofo)!=args.permutations or not {"mean","minimum"}.issubset(resumed_lofo.columns):raise SystemExit(f"Incomplete LOFO null checkpoint for {name}")
      mean_null=resumed_lofo["mean"].tolist();minimum_null=resumed_lofo["minimum"].tolist()
      results["execution"]["resumedLofoNulls"].append(name)
    else:
      shuffled_nulls=[]
      for index in range(args.permutations):
        rng=np.random.default_rng(SEED+7000000+rep_index*100003+index);shuffled_nulls.append(permute_labels(y,groups,rng))
      family_nulls=[]
      null_specs=[spec for spec in lofo_specs if spec[4]>=100] or lofo_specs
      results["execution"]["skippedIneligibleFamilyNullFits"]+=args.permutations*(len(lofo_specs)-len(null_specs))
      for family,family_train,family_cal,family_test,row_count in null_specs:
        family_hash=hashlib.sha256(family.encode()).hexdigest()[:16];family_path=run/f"tables/permutation-probe-{name}-lofo-family-{family_hash}.csv";values=None
        if args.resume_completed and family_path.exists():
          saved=pd.read_csv(family_path)
          if len(saved)!=args.permutations or not {"family","rows","value"}.issubset(saved.columns) or not saved.family.eq(family).all() or not saved.rows.eq(row_count).all():raise SystemExit(f"Incomplete LOFO family null checkpoint for {name}/{family}")
          values=saved.value.tolist();results["execution"]["resumedFamilyNulls"].append(f"{name}:{family}")
        if values is None:
          family_scaler=StandardScaler();family_train_x=family_scaler.fit_transform(X[family_train]);family_cal_x=family_scaler.transform(X[family_cal]);family_test_x=family_scaler.transform(X[family_test])
          def one_family_null(index):
            with threadpool_limits(limits=1):
              shuffled=shuffled_nulls[index];candidate=classifier(alpha,SEED+index+1);candidate.fit(family_train_x,shuffled[family_train]);temp=fit_temp(candidate.decision_function(family_cal_x),shuffled[family_cal])
              return f1_score(shuffled[family_test],softmax(candidate.decision_function(family_test_x),temp).argmax(1),average="macro",labels=np.arange(len(models)),zero_division=0)
          values=Parallel(n_jobs=min(args.jobs,args.permutations),prefer="threads")(delayed(one_family_null)(index) for index in range(args.permutations)) if args.permutations else []
          family_tmp=family_path.with_suffix(".csv.tmp");pd.DataFrame({"family":[family]*len(values),"rows":[row_count]*len(values),"value":values}).to_csv(family_tmp,index=False);os.replace(family_tmp,family_path)
        family_nulls.append((row_count,values))
      mean_null=[];minimum_null=[]
      for index in range(args.permutations):
        scores=[values[index] for _,values in family_nulls]
        mean_null.append(float(np.mean(scores)));minimum_null.append(float(np.min(scores)))
    lowest=min(eligible,key=lambda value:value["macroF1"])
    rep["lofo"]={"families":lofo_metrics,"nullEligibleFamilies":[family for family,_,_,_,_ in ([spec for spec in lofo_specs if spec[4]>=100] or lofo_specs)],
      "mean":{"macroF1":float(np.mean(macro)),"macroF1Ci95":[float(np.mean([v["macroF1Ci95"][0] for v in eligible])),float(np.mean([v["macroF1Ci95"][1] for v in eligible]))],"families":len(eligible),"permutationNull":{"mean":float(np.mean(mean_null)),"p95":float(np.quantile(mean_null,.95))}},
      "minimum":{"macroF1":float(np.min(macro)),"macroF1Ci95":lowest["macroF1Ci95"],"families":len(eligible),"permutationNull":{"mean":float(np.mean(minimum_null)),"p95":float(np.quantile(minimum_null,.95))}}}
    pd.DataFrame({"mean":mean_null,"minimum":minimum_null}).to_csv(lofo_null_path,index=False)
    prediction_frames.extend(lofo_predictions)
  results["representations"][name]=rep
  model_path.write_bytes(pickle.dumps({"pipeline":model,"temperature":temperature,"conformalQ":q,"models":models,"alpha":alpha}));artifact_paths.append(str(model_path))
  artifact_hashes={"model":hashlib.sha256(model_path.read_bytes()).hexdigest(),"lofo":hashlib.sha256(lofo_null_path.read_bytes()).hexdigest(),**{f"suite:{suite}":hashlib.sha256(path.read_bytes()).hexdigest() for suite,path in suite_null_paths.items()}}
  checkpoint_frames=prediction_frames[prediction_checkpoint_start:];checkpoint_predictions=pd.concat(checkpoint_frames,ignore_index=True) if checkpoint_frames else pd.DataFrame()
  result_checkpoint_tmp=result_checkpoint_path.with_suffix(".pkl.tmp");result_checkpoint_tmp.write_bytes(pickle.dumps({"schemaVersion":1,"representation":name,"campaignHash":freeze["campaignHash"],"models":models,"tier":args.tier,"permutations":args.permutations,"bootstrap":args.bootstrap,"rows":len(base),"artifactHashes":artifact_hashes,"result":rep,"predictions":checkpoint_predictions}));os.replace(result_checkpoint_tmp,result_checkpoint_path)

predictions=pd.concat(prediction_frames,ignore_index=True) if prediction_frames else pd.DataFrame();predictions.to_parquet(run/"tables/predictions.parquet",index=False);predictions.to_csv(run/"tables/predictions.csv",index=False)
cursor.execute("INSERT dbo.attribution_models(run_id,model_kind,training_manifest_hash,artifact_json) VALUES(%s,'sgd-multinomial-probes',%s,%s); SELECT SCOPE_IDENTITY()",
 (run_id,freeze["campaignHash"],json.dumps({"metrics":"metrics/attribution.json","models":models,"artifacts":artifact_paths})));attribution_id=int(cursor.fetchone()[0])
cursor.execute("INSERT dbo.calibration_models(attribution_model_id,method,calibration_manifest_hash,artifact_json) VALUES(%s,'temperature-scaling+split-conformal',%s,%s)",
 (attribution_id,freeze["campaignHash"],json.dumps({name:{"temperature":value["temperature"],"conformalQ90":value["conformalQ90"]} for name,value in results["representations"].items()})))
for (representation,method,suite),frame in predictions.groupby(["representation","method","suite"]):
  cursor.execute("INSERT dbo.prediction_runs(run_id,suite,representation_id,method,config_json) VALUES(%s,%s,%s,%s,%s); SELECT SCOPE_IDENTITY()",(run_id,suite,representation,method,json.dumps({"calibrated":True,"tier":args.tier})));prediction_run=int(cursor.fetchone()[0]);rows=[];candidates=[]
  for row in frame.itertuples():
    probabilities=json.loads(row.probabilities);candidate_set=json.loads(row.candidate_set);decision="attributed" if len(candidate_set)==1 else "ambiguous" if candidate_set else "abstain"
    rows.append((prediction_run,int(row.generation_id),row.model_profile_id,row.predicted_model_profile_id,decision,float(row.confidence),row.candidate_set,1-float(row.confidence)))
    candidates.extend((prediction_run,int(row.generation_id),model,float(probability),"{}") for model,probability in probabilities.items())
  cursor.executemany("INSERT dbo.predictions(prediction_run_id,generation_id,true_model_profile_id,predicted_model_profile_id,decision,calibrated_probability,candidate_set_json,novelty_score) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",rows)
  cursor.executemany("INSERT dbo.prediction_candidates(prediction_run_id,generation_id,model_profile_id,probability,features_json) VALUES(%s,%s,%s,%s,%s)",candidates)
conn.commit()
for name,rep in results["representations"].items():
  test_gate=("test_id" in rep["suites"] and rep["suites"]["test_id"]["macroF1Ci95"][0]>rep["suites"]["test_id"]["permutationNull"]["p95"])
  lofo_gate=(bool(rep["lofo"]) and rep["lofo"]["minimum"]["macroF1Ci95"][0]>rep["lofo"]["minimum"]["permutationNull"]["p95"])
  for suite,metric in rep["suites"].items():
    null=metric["permutationNull"];taxonomy="CLOSED_SET_SIGNAL" if suite=="test_id" and test_gate and lofo_gate else "NO_SUPPORTED_SIGNAL"
    cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
      VALUES(%s,'DECODE',%s,'linear-probe',%s,'macro_f1',%s,%s,%s,%s,%s,%s,%s,%s); SELECT SCOPE_IDENTITY()""",
      (run_id,name,suite,metric["macroF1"],metric["macroF1Ci95"][0],metric["macroF1Ci95"][1],null["mean"],null["p95"],metric["rows"],metric["groups"],json.dumps(metric)));metric_id=int(cursor.fetchone()[0])
    cursor.execute("INSERT dbo.claims(run_id,research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale,metric_result_id) VALUES(%s,'RQ1','DECODE',%s,'linear-probe',%s,%s,%s,%s,%s)",
      (run_id,name,suite,taxonomy,taxonomy=="CLOSED_SET_SIGNAL",f"Grouped bootstrap lower bound {metric['macroF1Ci95'][0]:.4f}; within-group permutation p95 {null['p95']:.4f}.",metric_id))
  if rep["lofo"]:
    for aggregate in ["mean","minimum"]:
      metric=rep["lofo"][aggregate];null=metric["permutationNull"];suite=f"lofo_{aggregate}";taxonomy="CLOSED_SET_SIGNAL" if aggregate=="minimum" and test_gate and lofo_gate else "NO_SUPPORTED_SIGNAL"
      cursor.execute("""INSERT dbo.metric_results(run_id,evidence_tag,representation_id,method,suite,metric_name,metric_value,ci_low,ci_high,null_mean,null_p95,rows_count,groups_count,detail_json)
        VALUES(%s,'DECODE',%s,'linear-probe',%s,'macro_f1',%s,%s,%s,%s,%s,NULL,%s,%s); SELECT SCOPE_IDENTITY()""",
        (run_id,name,suite,metric["macroF1"],metric["macroF1Ci95"][0],metric["macroF1Ci95"][1],null["mean"],null["p95"],metric["families"],json.dumps(metric)));metric_id=int(cursor.fetchone()[0])
      cursor.execute("INSERT dbo.claims(run_id,research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale,metric_result_id) VALUES(%s,'RQ1','DECODE',%s,'linear-probe',%s,%s,%s,%s,%s)",
        (run_id,name,suite,taxonomy,taxonomy=="CLOSED_SET_SIGNAL",f"LOFO {aggregate} {metric['macroF1']:.4f}; grouped-null p95 {null['p95']:.4f}.",metric_id))
conn.commit();(run/"metrics/attribution.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps({"runId":run_id,"rows":len(base),"representations":list(results["representations"]),"predictionRows":len(predictions),"projectionHash":projection_hash},indent=2));conn.close()
