#!/usr/bin/env python3
"""Reconstruct the complete, ceiling-aware ModelPrint results pack."""
import argparse,itertools,json,os,re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymssql
import seaborn as sns
from sklearn.metrics import roc_curve

parser=argparse.ArgumentParser();parser.add_argument("--run");args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
campaign=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"];run_id=run.name;conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),login_timeout=60,timeout=3600)
sns.set_theme(style="whitegrid");captions={}
def md(frame,columns=None):
 if frame.empty:return "_No completed rows._"
 frame=frame if columns is None else frame[[column for column in columns if column in frame]];lines=["| "+" | ".join(frame.columns)+" |","|"+"|".join(["---"]*len(frame.columns))+"|"]
 for row in frame.itertuples(index=False):lines.append("| "+" | ".join("—" if pd.isna(value) else f"{value:.4f}" if isinstance(value,(float,np.floating)) else str(value) for value in row)+" |")
 return "\n".join(lines)
def save(code,name,frame,draw,caption):
 frame.to_csv(run/f"tables/{code}_{name}.csv",index=False);plt.figure(figsize=(11,6))
 if frame.empty:plt.axis("off");plt.text(.5,.5,"NOT RUN / NO RETAINED ROWS",ha="center",va="center",fontsize=15)
 else:draw(frame)
 plt.tight_layout();plt.savefig(run/f"figures/{code}_{name}.png",dpi=160);plt.close();captions[code]=caption

quality=pd.read_sql("""SELECT g.generation_id,g.model_profile_id,JSON_VALUE(d.config_json,'$.key') decode_key,v.carrier_id,p.split,p.prompt_source_id source_id,p.family,
 g.reference_token_count,g.output_token_count,g.output_char_count,g.output_word_count,g.length_band,g.truncated,g.template_residue_found,g.self_name_found,g.finish_reason,g.latency_ms,g.output_sha256
FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id WHERE g.campaign_id=%s""",conn,params=(campaign,))
jobs=pd.read_sql("SELECT model_profile_id,status,COUNT(*) count FROM dbo.generation_jobs WHERE campaign_id=%s GROUP BY model_profile_id,status",conn,params=(campaign,));quality.to_parquet(run/"tables/generation_rows.parquet",index=False);jobs.to_csv(run/"tables/generation_completeness.csv",index=False)
expected=int(freeze["expectedJobs"]);complete=int(jobs.loc[jobs.status=="complete","count"].sum()) if len(jobs) else 0;quality_summary={"expectedJobs":expected,"complete":complete,"completionRate":complete/expected if expected else 0,"rows":len(quality),"truncated":int(quality.truncated.sum()) if len(quality) else 0,"templateResidue":int(quality.template_residue_found.sum()) if len(quality) else 0,"selfName":int(quality.self_name_found.sum()) if len(quality) else 0,"exactCollisionHashes":int((quality.groupby("output_sha256").model_profile_id.nunique()>1).sum()) if len(quality) else 0}
(run/"metrics/data-quality.json").write_text(json.dumps(quality_summary,indent=2)+"\n");(run/"reports/DATA_QUALITY.md").write_text("# Data Quality\n\n"+md(jobs)+"\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in quality_summary.items())+"\n")

headline=[];attribution=json.loads((run/"metrics/attribution.json").read_text()) if (run/"metrics/attribution.json").exists() else None
if attribution:
 for representation,block in attribution["representations"].items():
  rows=list(block.get("suites",{}).items())+[(f"lofo_{key}",value) for key,value in block.get("lofo",{}).items() if key in {"mean","minimum"}];test=block.get("suites",{}).get("test_id");lofo=block.get("lofo",{}).get("minimum");gate=bool(test and lofo and test["macroF1Ci95"][0]>test["permutationNull"]["p95"] and lofo["macroF1Ci95"][0]>lofo["permutationNull"]["p95"])
  for suite,result in rows:
   null=result.get("permutationNull",{});headline.append({"representation":representation,"method":"linear-probe","suite":suite,"rows":result.get("rows"),"macro_f1":result.get("macroF1"),"ci_low":result.get("macroF1Ci95",[None,None])[0],"ci_high":result.get("macroF1Ci95",[None,None])[1],"permutation_null_p95":null.get("p95"),"accuracy":result.get("accuracy"),"top2_accuracy":result.get("top2Accuracy"),"ece":result.get("eceAdaptive10"),"selective_accuracy_50":result.get("selectiveAccuracyAt50Coverage"),"coverage_at_85_accuracy":result.get("coverageAt85Accuracy"),"taxonomy":"CLOSED_SET_SIGNAL" if suite in {"test_id","lofo_minimum"} and gate else "NO_SUPPORTED_SIGNAL"})
for path,method,key in [(run/"metrics/retrieval.json","knn-vote","representations"),(run/"metrics/chunk-retrieval.json","chunk-vote","segmenters")]:
 if path.exists():
  for representation,block in json.loads(path.read_text()).get(key,{}).items():
   for suite,result in block.get("suites",{}).items():
    headline.append({"representation":representation if method=="knn-vote" else f"chunk:{representation}","method":method,"suite":suite,"rows":result.get("rows"),"macro_f1":result.get("macroF1"),"ci_low":result.get("macroF1Ci95",[None,None])[0],"ci_high":result.get("macroF1Ci95",[None,None])[1],"permutation_null_p95":result.get("permutationNull",{}).get("p95"),"accuracy":result.get("accuracy"),"top2_accuracy":result.get("top2Accuracy"),"ece":None,"selective_accuracy_50":None,"coverage_at_85_accuracy":None,"taxonomy":"RETRIEVAL_SIGNAL" if result.get("macroF1Ci95",[0])[0]>result.get("permutationNull",{}).get("p95",1) else "NO_SUPPORTED_SIGNAL"})
headline_columns=["representation","method","suite","rows","macro_f1","ci_low","ci_high","permutation_null_p95","accuracy","top2_accuracy","ece","selective_accuracy_50","coverage_at_85_accuracy","taxonomy"]
headline=pd.DataFrame(headline,columns=headline_columns);headline.to_csv(run/"tables/headline.csv",index=False);headline_md="# ModelPrint Headline Results\n\nRows are reconstructed from retained prediction and metric artifacts. `DECODE` is supervised; `RETRIEVE` is exact SQL evidence and is not a calibrated probability. Missing cells are not imputed.\n\n"+md(headline,["representation","method","suite","macro_f1","ci_low","ci_high","permutation_null_p95","accuracy","top2_accuracy","ece","selective_accuracy_50","coverage_at_85_accuracy","taxonomy"])+"\n";(run/"reports/HEADLINE.md").write_text(headline_md)

source=quality.copy();source["reference_token_count"]=source.reference_token_count.fillna(source.output_token_count)
save("F01","length_by_model_carrier",source,lambda f:(sns.boxplot(data=f,x="model_profile_id",y="reference_token_count",hue="carrier_id",showfliers=False),plt.xticks(rotation=20,ha="right"),plt.title("Output length by served profile and carrier\nTruncation is retained in the source table")),"Frozen reference-token length distributions; descriptive only, with truncation explicit.")
collision=[]
for (digest,band),frame in quality.groupby(["output_sha256","length_band"]):
 for left,right in itertools.combinations(sorted(frame.model_profile_id.unique()),2):collision.append({"left_model":left,"right_model":right,"length_band":band,"output_sha256":digest})
collision=pd.DataFrame(collision);collision=collision.groupby(["left_model","right_model","length_band"]).size().rename("collisions").reset_index() if len(collision) else pd.DataFrame(columns=["left_model","right_model","length_band","collisions"])
save("F02","cross_model_collisions",collision,lambda f:(sns.barplot(data=f,x="left_model",y="collisions",hue="right_model"),plt.xticks(rotation=20,ha="right"),plt.title("Cross-profile exact-output collisions\nCounts are not authorship evidence")),"Byte-identical collisions by profile pair and length; a data-quality observation, not a signature.")
files=sorted((run/"tables").glob("geometry-contrasts-*.csv"));geometry=pd.concat([pd.read_csv(path).assign(representation=path.stem.replace("geometry-contrasts-","")) for path in files],ignore_index=True) if files else pd.DataFrame()
save("F03","geometry_contrasts",geometry,lambda f:(sns.barplot(data=f,x="representation",y="mean_distance",hue="contrast"),plt.xticks(rotation=20,ha="right"),plt.title("SQL sampled geometry contrasts\nPrompt and model are controlled labels")),"SQL pair distances compare prompt and model alignment; geometry is not a calibrated decision.")
gm=json.loads((run/"metrics/geometry.json").read_text()) if (run/"metrics/geometry.json").exists() else {"representations":[]};win=pd.DataFrame([{"representation":v["representation"],"retrieval_win_rate":v.get("retrievalWin",{}).get("retrieval_win_rate"),"queries":v.get("retrievalWin",{}).get("queries"),"length_band":"all"} for v in gm.get("representations",[])])
save("F04","retrieval_win_rate",win,lambda f:(sns.barplot(data=f,x="representation",y="retrieval_win_rate"),plt.axhline(.5,color="black",ls="--"),plt.xticks(rotation=20,ha="right"),plt.ylim(0,1),plt.title("Model-proximity retrieval win rate\nAggregate retained SQL pair sample")),"Fraction where same-model proximity beats same-prompt proximity. The retained sample is aggregate when no length label exists.")
save("F05","attribution_by_suite",headline,lambda f:(sns.barplot(data=f,x="representation",y="macro_f1",hue="suite"),plt.axhline(.25,color="black",ls="--"),plt.xticks(rotation=25,ha="right"),plt.ylim(0,1),plt.title("Attribution by suite\nCI and grouped-null p95 are in the source CSV")),"Closed-set macro-F1; only bootstrap and LOFO results above the frozen permutation null support a signal claim.")

conf=[]
for path in sorted((run/"tables").glob("confusion-probe-*-test_id.csv")):
 frame=pd.read_csv(path,index_col=0);rep=path.stem[len("confusion-probe-"):-len("-test_id")]
 for truth in frame.index:
  for predicted in frame.columns:conf.append({"representation":rep,"true_model":truth,"predicted_model":predicted,"rows":frame.loc[truth,predicted]})
conf=pd.DataFrame(conf)
def draw_conf(f):
 best=headline[(headline.method=="linear-probe")&(headline.suite=="test_id")].sort_values("macro_f1",ascending=False).head(1);rep=best.iloc[0].representation if len(best) else f.iloc[0].representation;sns.heatmap(f[f.representation==rep].pivot(index="true_model",columns="predicted_model",values="rows"),annot=True,fmt=".0f",cmap="Blues");plt.title(f"Test-ID confusion: {rep}\nAll matrices remain in CSV")
save("F06","confusion_grid",conf,draw_conf,"Confusion counts for completed test-ID probes; the panel shows the best completed probe and remains closed-set.")

predictions=pd.read_parquet(run/"tables/predictions.parquet") if (run/"tables/predictions.parquet").exists() else pd.DataFrame();by_length=pd.DataFrame();coverage=pd.DataFrame();reliability=pd.DataFrame()
if len(predictions):
 merged=predictions.merge(quality[["generation_id","length_band"]],on="generation_id",how="left");merged["correct"]=merged.model_profile_id==merged.predicted_model_profile_id;by_length=merged.groupby(["representation","suite","length_band"]).agg(rows=("correct","size"),accuracy=("correct","mean")).reset_index();curves=[]
 for (representation,suite),frame in merged.groupby(["representation","suite"]):
  frame=frame.sort_values("confidence",ascending=False)
  for fraction in np.linspace(.05,1,20):curves.append({"representation":representation,"suite":suite,"coverage":fraction,"selective_accuracy":float(frame.head(max(1,int(len(frame)*fraction))).correct.mean())})
 coverage=pd.DataFrame(curves);best=headline[(headline.method=="linear-probe")&(headline.suite=="test_id")].sort_values("macro_f1",ascending=False).head(1);raw=[]
 if len(best):
  frame=predictions[(predictions.representation==best.iloc[0].representation)&(predictions.suite=="test_id")]
  for state,column in [("calibrated","probabilities"),("uncalibrated","uncalibrated_probabilities")]:
   if column not in frame:continue
   for row in frame.itertuples():
    values=json.loads(getattr(row,column));confidence=max(values.values());raw.append({"state":state,"confidence":confidence,"correct":max(values,key=values.get)==row.model_profile_id})
 if raw:
  reliability=pd.DataFrame(raw);reliability["bin"]=pd.cut(reliability.confidence,np.linspace(0,1,16),include_lowest=True);reliability=reliability.groupby(["state","bin"],observed=True).agg(rows=("correct","size"),mean_confidence=("confidence","mean"),accuracy=("correct","mean")).reset_index();reliability["bin"]=reliability.bin.astype(str)
save("F07","accuracy_by_length_band",by_length,lambda f:(sns.barplot(data=f[f.suite=="test_id"],x="length_band",y="accuracy",hue="representation"),plt.axhline(.25,color="black",ls="--"),plt.ylim(0,1),plt.title("Test-ID accuracy by reference-token length\nVerbosity-only is a nuisance baseline")),"Accuracy by frozen length band, with verbosity retained as nuisance baseline; short-text failures do not imply long-text failure.")
save("F08","selective_accuracy_coverage",coverage,lambda f:(sns.lineplot(data=f[f.suite.astype(str).str.contains("test_id|lofo")],x="coverage",y="selective_accuracy",hue="representation",style="suite"),plt.axhline(.85,color="black",ls="--"),plt.ylim(0,1),plt.title("Selective accuracy versus coverage\nOnly calibrated probe confidence is used")),"Confidence-ordered accuracy; the 0.85 line matters only together with frozen coverage, ECE, and LOFO gates.")
save("F09","reliability",reliability,lambda f:(sns.lineplot(data=f,x="mean_confidence",y="accuracy",hue="state",marker="o"),plt.plot([0,1],[0,1],"k--"),plt.xlim(0,1),plt.ylim(0,1),plt.title("Reliability before/after temperature scaling\n15 equal-width bins")),"Reliability for the best test-ID probe; closed-set calibration does not guarantee OOD calibration.")
ood=pd.read_csv(run/"tables/ood-by-source.csv") if (run/"tables/ood-by-source.csv").exists() else pd.DataFrame();save("F10","ood_novelty",ood,lambda f:(sns.barplot(data=f,x="source",y="unknown_acceptance_rate",hue="split_role"),plt.xticks(rotation=20,ha="right"),plt.ylim(0,1),plt.title("Unknown-source acceptance\nThreshold frozen on development controls")),"Unknown-acceptance by disjoint source; human geometry is descriptive and this is not a human detector.")

pairs=pd.read_parquet(run/"tables/pairs.parquet") if (run/"tables/pairs.parquet").exists() else pd.DataFrame();pair_roc=[]
if len(pairs) and "probability" in pairs:
 for cell,frame in pairs[pairs.split.astype(str).str.startswith("test")].groupby("pair_cell"):
  if frame.same_source.nunique()<2:continue
  fpr,tpr,thresholds=roc_curve(frame.same_source,frame.probability);pair_roc.extend({"pair_cell":cell,"false_positive_rate":float(x),"true_positive_rate":float(y),"threshold":float(z)} for x,y,z in zip(fpr,tpr,thresholds))
pair_roc=pd.DataFrame(pair_roc);save("F11","pairwise_roc",pair_roc,lambda f:(sns.lineplot(data=f,x="false_positive_rate",y="true_positive_rate",hue="pair_cell"),plt.plot([0,1],[0,1],"k--"),plt.xlim(0,1),plt.ylim(0,1),plt.title("Same-source ROC by pair cell\nHard cells govern the claim")),"ROC by frozen pair cell; hard-pair bootstrap and permutation gates govern same-source claims.")
cluster=pd.read_csv(run/"tables/cluster-alignment.csv") if (run/"tables/cluster-alignment.csv").exists() else pd.DataFrame();save("F12","cluster_alignment",cluster,lambda f:(sns.scatterplot(data=f,x="nmi_family",y="modelNmi",hue="representation",style="algorithm",s=100),plt.plot([0,1],[0,1],"k--"),plt.xlim(0,1),plt.ylim(0,1),plt.title("Label-hidden cluster alignment\nAbove diagonal favors model over family")),"Model NMI versus nuisance NMI after label-hidden fitting; fingerprint is semi-supervised and projections alone prove nothing.")
ann=pd.read_csv(run/"tables/ann-benchmark.csv") if (run/"tables/ann-benchmark.csv").exists() else pd.DataFrame();save("F13","ann_recall_latency",ann,lambda f:(sns.lineplot(data=f[f.k==10],x="corpus_size",y="recall",hue="representation",style="oversampling",marker="o"),plt.ylim(0,1.02),plt.title("DiskANN recall@10 by prefix\nLatency requires captured index-plan evidence")),"Recall against exact SQL neighbors; unknown-plan rows are excluded from latency claims.")

likelihood=pd.read_sql("""SELECT l.scoring_model_profile_id scorer,g.model_profile_id source,l.prompted,COUNT(*) rows,AVG(l.bits_per_char) mean_bits_per_char,STDEV(l.bits_per_char) sd_bits_per_char FROM dbo.likelihood_scores l JOIN dbo.generations g ON g.generation_id=l.generation_id WHERE g.campaign_id=%s GROUP BY l.scoring_model_profile_id,g.model_profile_id,l.prompted""",conn,params=(campaign,));likelihood.to_csv(run/"tables/cross_likelihood.csv",index=False);matrix=likelihood[likelihood.prompted==1].pivot(index="scorer",columns="source",values="mean_bits_per_char").reindex(index=models,columns=models).reset_index() if len(likelihood) else pd.DataFrame()
save("F14","cross_likelihood_matrix",matrix,lambda f:(sns.heatmap(f.set_index("scorer").reindex(index=models,columns=models),annot=True,fmt=".3f",cmap="mako_r"),plt.title("Prompted cross-likelihood (bits/character)\nIncomplete cells remain blank")),"Model-access likelihood ceiling; blank cells mean partial likelihood and lower is better.")
comparison=[]
if len(headline):
 for suite,frame in headline.groupby("suite"):
  like=frame[frame.representation=="likelihood-profile8-v1"];text=frame[(frame.method=="linear-probe")&~frame.representation.str.contains("likelihood")]
  if len(like) and len(text):
   ceiling=float(like.macro_f1.max());best=float(text.macro_f1.max());comparison.append({"suite":suite,"best_text_macro_f1":best,"likelihood_macro_f1":ceiling,"text_fraction_of_likelihood":best/ceiling if ceiling else None})
comparison=pd.DataFrame(comparison);save("F15","likelihood_vs_text_only",comparison,lambda f:(sns.barplot(data=f,x="suite",y="text_fraction_of_likelihood"),plt.axhline(1,color="black",ls="--"),plt.xticks(rotation=20,ha="right"),plt.title("Best text-only as fraction of likelihood profile\nLikelihood requires model access")),"Matched-suite text macro-F1 divided by likelihood-profile performance; only complete likelihood profiles appear.")

phrase=pd.read_csv(run/"tables/distinctive-phrases.csv") if (run/"tables/distinctive-phrases.csv").exists() else pd.DataFrame();scalar=pd.read_sql("""SELECT g.model_profile_id,g.final_text,f.features_json FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.output_scalar_features f ON f.text_artifact_id=t.text_artifact_id WHERE g.campaign_id=%s""",conn,params=(campaign,));field_rows=[];field="# ModelPrint Field Guide\n\nEvery item is a campaign frequency statement, never a signature or proof of authorship.\n"
if len(scalar):
 values=pd.json_normalize(scalar.features_json.map(json.loads));features=list(values.columns);values["model"]=scalar.model_profile_id.to_numpy()
 for model in models:
  own=values[values.model==model][features];other=values[values.model!=model][features];effect=(own.mean()-other.mean())/np.sqrt((own.var(ddof=1)+other.var(ddof=1))/2).replace(0,np.nan);top=effect.reindex(effect.abs().sort_values(ascending=False).index).head(10);mp=phrase[phrase.model_profile_id==model].reindex(phrase[phrase.model_profile_id==model].z_score.abs().sort_values(ascending=False).index).head(15) if len(phrase) else pd.DataFrame()
  field+=f"\n## {model}\n\n### Distinctive phrases\n\n"+("\n".join(f"- `{row.phrase}`: log-odds `{row.log_odds:+.3f}`, z `{row.z_score:+.2f}`" for row in mp.itertuples()) or "_Phrase stage unavailable._")+"\n\n### Style fields\n\n"+"\n".join(f"- `{name}`: standardized mean difference `{value:+.3f}`" for name,value in top.items())+"\n\n### Example openers / closers\n\n"
  for text in scalar[scalar.model_profile_id==model].sort_values("final_text").head(3).final_text:
   compact=" ".join(str(text).split());field+=f"- opener: “{compact[:120]}” · closer: “{compact[-120:]}”\n"
  field_rows.extend({"model_profile_id":model,"evidence_kind":"style","item":name,"effect":value} for name,value in top.items());field_rows.extend({"model_profile_id":model,"evidence_kind":"phrase","item":row.phrase,"effect":row.z_score} for row in mp.itertuples())
(run/"reports/MODELPRINT_FIELD_GUIDE.md").write_text(field);field_table=pd.DataFrame(field_rows);save("F16","field_guide",field_table,lambda f:(sns.barplot(data=f.groupby(["model_profile_id","evidence_kind"]).head(5),x="model_profile_id",y="effect",hue="evidence_kind"),plt.axhline(0,color="black"),plt.xticks(rotation=20,ha="right"),plt.title("Largest phrase/style frequency effects\nEffects are not signatures")),"Phrase z-scores and standardized style effects; frequency differences cannot prove authorship.")
seed=pd.read_csv(run/"tables/seed-diversity.csv") if (run/"tables/seed-diversity.csv").exists() else pd.DataFrame();save("F17","seed_diversity",seed,lambda f:(sns.boxplot(data=f,x="model_profile_id",y="distance",hue="representation",showfliers=False),plt.xticks(rotation=20,ha="right"),plt.title("Natural-seed within-profile diversity\nMatched variants only")),"Cosine distance between matched natural seeds; decode diversity is not attribution accuracy.")
nulls=[]
for path in sorted((run/"tables").glob("permutation-*.csv")):
 frame=pd.read_csv(path);numeric=[column for column in frame.select_dtypes(include="number").columns]
 for column in numeric:nulls.append(pd.DataFrame({"source":path.stem.replace("permutation-","")+":"+column,"value":frame[column]}))
null=pd.concat(nulls,ignore_index=True).dropna() if nulls else pd.DataFrame();save("F18","permutation_nulls",null,lambda f:(sns.violinplot(data=f[f.source.isin(f.source.unique()[:20])],x="source",y="value",inner="quart"),plt.xticks(rotation=30,ha="right"),plt.title("Within-prompt-group full-pipeline nulls\nFirst 20 retained sources shown")),"Full-pipeline grouped nulls; nominal chance is orientation only and each claim uses its null p95.")
mixed_json=json.loads((run/"tables/mixed-source-localization.json").read_text()) if (run/"tables/mixed-source-localization.json").exists() else {};mixed=pd.DataFrame([{"metric":key,"value":value} for key,value in mixed_json.items() if isinstance(value,(int,float))]);save("F19","mixed_source_localization",mixed,lambda f:(sns.barplot(data=f,x="metric",y="value"),plt.xticks(rotation=25,ha="right"),plt.title("Mixed-source paragraph localization\nSame-prompt changes by construction")),"Constructed same-prompt localization; this does not imply open-world authorship localization.")
(run/"reports/FIGURE_CAPTIONS.md").write_text("# Figure Captions and Claim Ceilings\n\n"+"\n".join(f"- **{key}** — {value}" for key,value in sorted(captions.items()))+"\n")

claims=pd.read_sql("SELECT research_question,evidence_tag,representation_id,method,suite,taxonomy,supported,rationale FROM dbo.claims WHERE run_id=%s ORDER BY research_question,evidence_tag,representation_id,method,suite",conn,params=(run_id,))
if claims.empty and len(headline):claims=headline.assign(research_question="RQ1",evidence_tag=np.where(headline.method=="linear-probe","DECODE","RETRIEVE"),supported=headline.taxonomy.str.contains("SIGNAL"),rationale="See retained headline metric").rename(columns={"representation":"representation_id"})
claims.to_csv(run/"tables/claims.csv",index=False);rq=[]
for question,frame in claims.groupby("research_question"):
 supported=frame[frame.supported==1];rq.append({"research_question":question,"adjudication":supported.iloc[0].taxonomy if len(supported) else "NO_SUPPORTED_SIGNAL","supported_rows":len(supported),"tested_rows":len(frame)})
rq=pd.DataFrame(rq);rq.to_csv(run/"tables/claims-by-research-question.csv",index=False);claims_md="# ModelPrint Claims Table\n\nEvery row carries an evidence tag; missing stages are absent rather than inferred.\n\n"+md(claims)+"\n\n## Research-question adjudication\n\n"+md(rq)+"\n";(run/"reports/MODELPRINT_CLAIMS_TABLE.md").write_text(claims_md)
attribution_report="# Attribution Report\n\n"+headline_md.split("\n\n",1)[1]+"\n\nA probe shows decodability; exact SQL kNN shows retrievability; neither alone is an open-world authorship claim.\n";(run/"reports/ATTRIBUTION_REPORT.md").write_text(attribution_report)
for name,title in [("SAME_SOURCE_REPORT.md","Same-source verification"),("CLUSTER_REPORT.md","Label-hidden clustering"),("ANN_REPORT.md","Exact versus approximate SQL retrieval"),("OOD_REPORT.md","Out-of-distribution abstention")]:
 path=run/"reports"/name
 if not path.exists():path.write_text(f"# {title}\n\nDisposition: `NOT_RUN`. No result is claimed.\n")
status="complete" if complete==expected else f"in progress ({complete}/{expected})";capability=json.loads((run/"environment/sql-server-capabilities.json").read_text()).get("syntaxSelection","unknown");summary=f"# ModelPrint Run Summary\n\n- Run: `{run_id}`\n- Campaign: `{freeze['campaignHash']}`\n- Generation: {status}\n- SQL vector mode: `{capability}`\n- Profiles with rows: `{quality.model_profile_id.nunique() if len(quality) else 0}`\n- Headline rows: `{len(headline)}`\n\nMissing stages are never imputed.\n";(run/"reports/RUN_SUMMARY.md").write_text(summary)
root_docs={"MODELPRINT_STATE_OF_RECORD.md":summary+"\n"+headline_md,"MODELPRINT_CLAIMS_TABLE.md":claims_md,"MODELPRINT_ATTRIBUTION_REPORT.md":attribution_report,"MODELPRINT_SAME_SOURCE_REPORT.md":(run/"reports/SAME_SOURCE_REPORT.md").read_text(),"MODELPRINT_CLUSTER_REPORT.md":(run/"reports/CLUSTER_REPORT.md").read_text(),"MODELPRINT_OOD_REPORT.md":(run/"reports/OOD_REPORT.md").read_text(),"MODELPRINT_SQL_VECTOR_REPORT.md":(run/"reports/ANN_REPORT.md").read_text()}
for name,text in root_docs.items():(lab/name).write_text(text)

best_lofo=headline[(headline.method=="linear-probe")&(headline.suite=="lofo_minimum")].sort_values("macro_f1",ascending=False).head(1) if len(headline) else pd.DataFrame();ann_dispositions=[]
if (run/"metrics/ann.json").exists():ann_dispositions=[f"{key}: {value.get('disposition')}" for key,value in json.loads((run/"metrics/ann.json").read_text()).get("spaces",{}).items()]
findings=["## What we found","","<!-- generated by analysis/build_reports.py -->",f"The strongest held-family minimum was `{best_lofo.iloc[0].representation}` at macro-F1 `{best_lofo.iloc[0].macro_f1:.3f}` with taxonomy `{best_lofo.iloc[0].taxonomy}`." if len(best_lofo) else "Held-family attribution has not completed, so no winner is named.",f"Across matched suites, the best text-only probe reached `{comparison.text_fraction_of_likelihood.mean():.3f}` of the likelihood-profile macro-F1 on average." if len(comparison) else "The likelihood ceiling is incomplete and no text-to-likelihood ratio is asserted.","ANN disposition: "+("; ".join(ann_dispositions) if ann_dispositions else "not yet run")+".","Legitimate app statement: for sufficiently long text from the frozen four served profiles, the app can return scoped witnesses and a calibrated closed-set candidate or abstain; it cannot prove authorship of arbitrary prose."]
readme=(lab/"README.md").read_text();block="\n".join(findings)+"\n\n## What we cannot say";readme=re.sub(r"\n## What we found.*?\n## What we cannot say","\n"+block,readme,flags=re.S) if "## What we found" in readme else readme.replace("\n## What we cannot say","\n\n"+block);(lab/"README.md").write_text(readme)
print(json.dumps({"run":str(run),"generationRows":len(quality),"headlineRows":len(headline),"claims":len(claims),"figures":len(captions)},indent=2));conn.close()
