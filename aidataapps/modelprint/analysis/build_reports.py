#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymssql
import seaborn as sns
from sklearn.metrics import accuracy_score

parser=argparse.ArgumentParser();parser.add_argument("--run");args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve()
freeze=json.loads((run/"manifests/campaign-freeze.json").read_text());campaign=int(freeze["campaignId"]);models=freeze["campaign"]["profiles"]
conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",password=os.environ["MSSQL_SA_PASSWORD"],
                     database=os.getenv("MSSQL_DATABASE","ModelPrint"),login_timeout=60,timeout=3600)
sns.set_theme(style="whitegrid")

quality=pd.read_sql("""SELECT g.generation_id,g.model_profile_id,JSON_VALUE(d.config_json,'$.key') decode_key,v.carrier_id,p.split,p.prompt_source_id source_id,p.family,
 g.reference_token_count,g.output_token_count,g.output_char_count,g.output_word_count,g.length_band,g.truncated,g.template_residue_found,g.self_name_found,g.finish_reason,g.latency_ms,g.output_sha256
FROM dbo.generations g JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
JOIN dbo.prompt_groups p ON p.prompt_group_id=v.prompt_group_id WHERE g.campaign_id=%s""",conn,params=(campaign,))
jobs=pd.read_sql("SELECT model_profile_id,status,COUNT(*) count FROM dbo.generation_jobs WHERE campaign_id=%s GROUP BY model_profile_id,status",conn,params=(campaign,))
quality.to_parquet(run/"tables/generation_rows.parquet",index=False);jobs.to_csv(run/"tables/generation_completeness.csv",index=False)

headline_rows=[]
metric_path=run/"metrics/attribution.json"
metrics=json.loads(metric_path.read_text()) if metric_path.exists() else None
if metrics:
  for representation,block in metrics["representations"].items():
    for suite,result in block.get("suites",{}).items():
      lower=result["macroF1Ci95"][0];null95=result["permutationNull"]["p95"]
      label="CLOSED_SET_SIGNAL" if lower>null95 else "NO_SUPPORTED_SIGNAL"
      headline_rows.append({"representation":representation,"method":"linear-probe","suite":suite,"rows":result["rows"],"macro_f1":result["macroF1"],
        "ci_low":lower,"ci_high":result["macroF1Ci95"][1],"permutation_null_p95":null95,"accuracy":result["accuracy"],"top2_accuracy":result["top2Accuracy"],
        "ece":result["eceAdaptive10"],"selective_accuracy_50":result["selectiveAccuracyAt50Coverage"],"coverage_at_85_accuracy":result["coverageAt85Accuracy"],"taxonomy":label})
headline=pd.DataFrame(headline_rows);headline.to_csv(run/"tables/headline.csv",index=False)

def markdown_table(frame,columns=None):
  if frame.empty:return "_No completed rows._"
  selected=frame if columns is None else frame[columns]
  headers=list(selected.columns);lines=["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]
  for row in selected.itertuples(index=False):
    values=[]
    for value in row:
      if isinstance(value,float): values.append(f"{value:.4f}")
      else: values.append(str(value))
    lines.append("| "+" | ".join(values)+" |")
  return "\n".join(lines)

headline_md="# ModelPrint Headline Results\n\nAll rows below are reconstructed from retained predictions. A probe result is supervised decoding evidence, not nearest-neighbor attribution.\n\n"+markdown_table(headline,
 ["representation","method","suite","macro_f1","ci_low","ci_high","permutation_null_p95","accuracy","ece","taxonomy"])+"\n"
(run/"reports/HEADLINE.md").write_text(headline_md)

expected=freeze["expectedJobs"];complete=int((jobs.loc[jobs.status=="complete","count"]).sum()) if not jobs.empty else 0
quality_summary={"expectedJobs":expected,"complete":complete,"completionRate":complete/expected if expected else 0,"rows":len(quality),
 "truncated":int(quality.truncated.sum()) if len(quality) else 0,"templateResidue":int(quality.template_residue_found.sum()) if len(quality) else 0,
 "selfName":int(quality.self_name_found.sum()) if len(quality) else 0,"exactCollisionHashes":int((quality.groupby("output_sha256").model_profile_id.nunique()>1).sum()) if len(quality) else 0}
(run/"metrics/data-quality.json").write_text(json.dumps(quality_summary,indent=2))
data_quality="# Data Quality\n\n"+markdown_table(jobs)+"\n\n"+"\n".join(f"- {key}: `{value}`" for key,value in quality_summary.items())+"\n"
(run/"reports/DATA_QUALITY.md").write_text(data_quality)

likelihood=pd.read_sql("""SELECT l.scoring_model_profile_id scorer,g.model_profile_id source,l.prompted,COUNT(*) rows,AVG(l.bits_per_char) mean_bits_per_char,STDEV(l.bits_per_char) sd_bits_per_char
FROM dbo.likelihood_scores l JOIN dbo.generations g ON g.generation_id=l.generation_id WHERE g.campaign_id=%s GROUP BY l.scoring_model_profile_id,g.model_profile_id,l.prompted""",conn,params=(campaign,))
likelihood.to_csv(run/"tables/cross_likelihood.csv",index=False)

if len(quality):
  source=quality.copy();source.reference_token_count=source.reference_token_count.fillna(source.output_token_count)
  source.to_csv(run/"tables/F01_length_by_model_carrier.csv",index=False)
  plt.figure(figsize=(11,6));sns.boxplot(data=source,x="model_profile_id",y="reference_token_count",hue="carrier_id",showfliers=False)
  plt.xticks(rotation=20,ha="right");plt.title("Output length by served profile and carrier\nReference-token counts; truncated rows retained");plt.tight_layout();plt.savefig(run/"figures/F01_length_by_model_carrier.png",dpi=160);plt.close()
  collisions=(quality.groupby(["output_sha256","length_band"]).filter(lambda x:x.model_profile_id.nunique()>1).groupby("length_band").size().rename("rows").reset_index())
  collisions.to_csv(run/"tables/F02_cross_model_collisions.csv",index=False)
  plt.figure(figsize=(7,4));sns.barplot(data=collisions,x="length_band",y="rows");plt.title("Rows participating in cross-profile exact collisions");plt.tight_layout();plt.savefig(run/"figures/F02_cross_model_collisions.png",dpi=160);plt.close()
if len(headline):
  headline.to_csv(run/"tables/F05_attribution_by_suite.csv",index=False)
  plt.figure(figsize=(11,6));sns.barplot(data=headline,x="representation",y="macro_f1",hue="suite")
  plt.axhline(.25,color="black",ls="--",label="nominal chance");plt.xticks(rotation=25,ha="right");plt.ylim(0,1);plt.title("Supervised attribution by held-out suite\nError intervals and permutation nulls are in the source table");plt.tight_layout();plt.savefig(run/"figures/F05_attribution_by_suite.png",dpi=160);plt.close()
  null_files=list((run/"tables").glob("permutation-*.csv"))
  if null_files:
    null=pd.concat([pd.read_csv(path).assign(source=path.stem.replace("permutation-","")) for path in null_files],ignore_index=True)
    null.to_csv(run/"tables/F18_permutation_nulls.csv",index=False);plt.figure(figsize=(11,6));sns.violinplot(data=null,x="source",y="value",inner="quart")
    plt.xticks(rotation=25,ha="right");plt.title("Within-prompt-group permutation nulls");plt.tight_layout();plt.savefig(run/"figures/F18_permutation_nulls.png",dpi=160);plt.close()
if len(likelihood):
  prompt_like=likelihood[likelihood.prompted==1];matrix=prompt_like.pivot(index="scorer",columns="source",values="mean_bits_per_char").reindex(index=models,columns=models)
  matrix.to_csv(run/"tables/F14_cross_likelihood_matrix.csv");plt.figure(figsize=(8,7));sns.heatmap(matrix,annot=True,fmt=".3f",cmap="mako_r")
  plt.title("Prompted cross-likelihood (bits/character; lower is better)");plt.tight_layout();plt.savefig(run/"figures/F14_cross_likelihood_matrix.png",dpi=160);plt.close()

claims=headline[["representation","method","suite","taxonomy","macro_f1","permutation_null_p95"]].copy() if len(headline) else pd.DataFrame(columns=["representation","method","suite","taxonomy","macro_f1","permutation_null_p95"])
claims.to_csv(run/"tables/claims.csv",index=False)
claims_md="# ModelPrint Claims Table\n\nEvidence tag: `DECODE` for the supervised rows below. Retrieval, calibration, likelihood, clustering, and engineering claims appear only after their own stages complete.\n\n"+markdown_table(claims)+"\n"
(run/"reports/MODELPRINT_CLAIMS_TABLE.md").write_text(claims_md)

attribution_report="# Attribution Report\n\n"+headline_md.split("\n\n",1)[1]+"\n\n## Interpretation rules\n\n- Semantic probe performance means profile information is decodable; it does not mean semantic nearest neighbors retrieve the profile.\n- A positive label requires the grouped-bootstrap lower bound to exceed the frozen permutation null.\n- Source holdout is distribution shift and is reported independently from in-distribution test groups.\n"
(run/"reports/ATTRIBUTION_REPORT.md").write_text(attribution_report)

field="# ModelPrint Field Guide\n\nThis field guide is a frequency summary, never a signature. Feature rankings are generated only after scalar features exist.\n"
try:
  scalar=pd.read_sql("""SELECT g.model_profile_id,f.features_json FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
  JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id AND t.text_view_id='raw-final-v1' JOIN dbo.output_scalar_features f ON f.text_artifact_id=t.text_artifact_id
  WHERE g.campaign_id=%s""",conn,params=(campaign,))
  if len(scalar):
    values=pd.json_normalize(scalar.features_json.map(json.loads));values["model"]=scalar.model_profile_id.to_numpy();means=values.groupby("model").mean(numeric_only=True)
    for model in means.index:
      other=means.drop(index=model).mean();delta=(means.loc[model]-other).sort_values(key=abs,ascending=False).head(10)
      field+=f"\n## {model}\n\n"+"\n".join(f"- `{name}`: mean difference `{value:+.4f}`" for name,value in delta.items())+"\n"
except Exception as error: field+=f"\nScalar field guide unavailable: `{error}`\n"
(run/"reports/MODELPRINT_FIELD_GUIDE.md").write_text(field)

status_note="complete" if complete==expected else f"in progress ({complete}/{expected})"
summary=f"""# ModelPrint Run Summary

- Run: `{run.name}`
- Campaign: `{freeze['campaignHash']}`
- Generation: {status_note}
- SQL capability: `{json.loads((run/'environment/sql-server-capabilities.json').read_text())['syntaxSelection']}`
- Profiles with rows: {quality.model_profile_id.nunique() if len(quality) else 0}
- Scientific results: {'available in HEADLINE.md' if len(headline) else 'not yet evaluated'}

This run separates row retention, supervised decoding, SQL retrieval, likelihood, calibration, OOD, grouping, clustering, and ANN evidence. Missing stages are not imputed.
"""
(run/"reports/RUN_SUMMARY.md").write_text(summary)
for name,title in [("SAME_SOURCE_REPORT.md","Same-source verification"),("CLUSTER_REPORT.md","Label-hidden clustering"),("ANN_REPORT.md","Exact versus approximate SQL retrieval"),("OOD_REPORT.md","Out-of-distribution abstention")]:
  path=run/"reports"/name
  if not path.exists(): path.write_text(f"# {title}\n\nDisposition: `NOT_YET_RUN`. No result is claimed until retained row artifacts and metrics exist.\n")

root_docs={"MODELPRINT_STATE_OF_RECORD.md":summary+"\n"+headline_md,"MODELPRINT_CLAIMS_TABLE.md":claims_md,"MODELPRINT_ATTRIBUTION_REPORT.md":attribution_report,
 "MODELPRINT_SAME_SOURCE_REPORT.md":(run/"reports/SAME_SOURCE_REPORT.md").read_text(),"MODELPRINT_CLUSTER_REPORT.md":(run/"reports/CLUSTER_REPORT.md").read_text(),
 "MODELPRINT_OOD_REPORT.md":(run/"reports/OOD_REPORT.md").read_text(),"MODELPRINT_SQL_VECTOR_REPORT.md":(run/"reports/ANN_REPORT.md").read_text()}
for name,text in root_docs.items():(lab/name).write_text(text)

inventory=[]
for path in sorted(run.rglob("*")):
  if path.is_file() and path.name!="ARTIFACT_INVENTORY.json":inventory.append({"path":str(path.relative_to(run)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
(run/"ARTIFACT_INVENTORY.json").write_text(json.dumps({"schemaVersion":1,"runId":run.name,"artifacts":inventory},indent=2))
print(json.dumps({"run":str(run),"generationRows":len(quality),"headlineRows":len(headline),"artifacts":len(inventory)},indent=2));conn.close()
