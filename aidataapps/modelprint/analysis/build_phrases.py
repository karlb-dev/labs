#!/usr/bin/env python3
"""Extract deterministic document n-grams and compute SQL-native log odds."""
import argparse, collections, hashlib, json, math, os, re
from pathlib import Path

import pandas as pd
import pymssql

parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--minimum-df",type=int,default=20);parser.add_argument("--maximum-phrases",type=int,default=10000);args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();freeze=json.loads((run/"manifests/campaign-freeze.json").read_text())
campaign=int(freeze["campaignId"]);run_id=run.name;conn=pymssql.connect(server=os.getenv("SQLSERVER_HOST","127.0.0.1"),port=int(os.getenv("SQLSERVER_PORT","1433")),user="sa",
 password=os.environ["MSSQL_SA_PASSWORD"],database=os.getenv("MSSQL_DATABASE","ModelPrint"),autocommit=False,login_timeout=60,timeout=3600)
rows=pd.read_sql("""SELECT g.generation_id,g.model_profile_id,g.final_text FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id=%s AND g.split='train' AND g.truncated=0 AND g.reference_token_count>=16
AND v.carrier_id<>'structured-v1' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')""",conn,params=(campaign,))
word_re=re.compile(r"[a-z0-9]+(?:['’][a-z]+)?",re.I);stop={"the","a","an","and","or","but","of","to","in","on","for","with","is","are","was","were","be","been","it","this","that"}
def phrases(text):
  words=[value.lower().replace("’","'") for value in word_re.findall(str(text))][:768];output=set()
  for size in (2,3,4):
    for index in range(len(words)-size+1):
      values=words[index:index+size]
      if all(value in stop for value in values):continue
      phrase=" ".join(values)
      if len(phrase)<=200:output.add(phrase)
  return output
documents=[];frequency=collections.Counter()
for row in rows.itertuples():
  values=phrases(row.final_text);documents.append((int(row.generation_id),values));frequency.update(values)
maximum_df=max(1,int(len(rows)*.50));selected={phrase for phrase,count in frequency.most_common() if count>=args.minimum_df and count<=maximum_df}
selected=set(sorted(selected,key=lambda value:(-frequency[value],value))[:args.maximum_phrases]);cursor=conn.cursor()
dictionary=[("word-ngram-2-4-v1",phrase,hashlib.sha256(phrase.encode()).hexdigest()) for phrase in sorted(selected)]
for start in range(0,len(dictionary),1000):
  cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.phrase_dictionary WHERE phrase_kind=%s AND phrase_sha256=%s)
    INSERT dbo.phrase_dictionary(phrase_kind,phrase,phrase_sha256) VALUES(%s,%s,%s)""",[(row[0],row[2],*row) for row in dictionary[start:start+1000]]);conn.commit()
cursor.execute("SELECT phrase_id,phrase FROM dbo.phrase_dictionary WHERE phrase_kind='word-ngram-2-4-v1'");phrase_ids={phrase:int(identifier) for identifier,phrase in cursor.fetchall() if phrase in selected}
occurrences=[(phrase_ids[phrase],generation_id) for generation_id,values in documents for phrase in values if phrase in phrase_ids]
for start in range(0,len(occurrences),5000):
  cursor.executemany("IF NOT EXISTS(SELECT 1 FROM dbo.phrase_occurrences WHERE phrase_id=%s AND generation_id=%s) INSERT dbo.phrase_occurrences(phrase_id,generation_id) VALUES(%s,%s)",
    [(row[0],row[1],*row) for row in occurrences[start:start+5000]]);conn.commit()
criteria={"campaignId":campaign,"split":"train","decode":["det","nat-0","nat-1"],"minimumReferenceTokens":16,"maximumDocumentFraction":.5,"minimumDf":args.minimum_df,"maximumPhrases":args.maximum_phrases}
manifest_hash=hashlib.sha256((freeze["campaignHash"]+json.dumps(criteria,sort_keys=True)).encode()).hexdigest()
cursor.execute("""DECLARE @alpha0 float=500.0,@campaign bigint=%s,@manifest char(64)=%s;
WITH eligible AS (SELECT g.generation_id,g.model_profile_id FROM dbo.generations g JOIN dbo.prompt_variants v ON v.prompt_variant_id=g.prompt_variant_id
 JOIN dbo.decode_configs d ON d.decode_config_id=g.decode_config_id WHERE g.campaign_id=@campaign AND g.split='train' AND g.truncated=0 AND g.reference_token_count>=16
 AND v.carrier_id<>'structured-v1' AND JSON_VALUE(d.config_json,'$.key') IN('det','nat-0','nat-1')),
docs AS (SELECT model_profile_id,COUNT(*) n_docs FROM eligible GROUP BY model_profile_id),
observed AS (SELECT o.phrase_id,e.model_profile_id,COUNT(DISTINCT o.generation_id) df_model FROM dbo.phrase_occurrences o JOIN eligible e ON e.generation_id=o.generation_id GROUP BY o.phrase_id,e.model_profile_id),
tot AS (SELECT phrase_id,SUM(df_model) df_all FROM observed GROUP BY phrase_id),
df AS (SELECT tot.phrase_id,d.model_profile_id,COALESCE(o.df_model,0) df_model,tot.df_all FROM tot CROSS JOIN docs d LEFT JOIN observed o ON o.phrase_id=tot.phrase_id AND o.model_profile_id=d.model_profile_id),
n AS (SELECT SUM(n_docs) n_all FROM docs),
scored AS (SELECT df.phrase_id,df.model_profile_id,df.df_model,df.df_all-df.df_model df_other,d.n_docs n_model,n.n_all-d.n_docs n_other,@alpha0*df.df_all/n.n_all alpha_p
 FROM df JOIN docs d ON d.model_profile_id=df.model_profile_id CROSS JOIN n),
final AS (SELECT *,LOG((df_model+alpha_p)/(n_model-df_model+@alpha0-alpha_p))-LOG((df_other+alpha_p)/(n_other-df_other+@alpha0-alpha_p)) log_odds FROM scored)
INSERT dbo.model_phrase_stats(phrase_id,model_profile_id,document_frequency_model,document_frequency_other,log_odds,z_score,training_manifest_hash)
SELECT phrase_id,model_profile_id,df_model,df_other,log_odds,log_odds/SQRT(1.0/(df_model+alpha_p)+1.0/(df_other+alpha_p)),@manifest FROM final
WHERE NOT EXISTS(SELECT 1 FROM dbo.model_phrase_stats s WHERE s.phrase_id=final.phrase_id AND s.model_profile_id=final.model_profile_id AND s.training_manifest_hash=@manifest);""",(campaign,manifest_hash));conn.commit()
stats=pd.read_sql("""SELECT s.model_profile_id,d.phrase,s.document_frequency_model,s.document_frequency_other,s.log_odds,s.z_score
FROM dbo.model_phrase_stats s JOIN dbo.phrase_dictionary d ON d.phrase_id=s.phrase_id WHERE s.training_manifest_hash=%s""",conn,params=(manifest_hash,))
doc_counts=rows.groupby("model_profile_id").size().to_dict();n_all=len(rows);max_error=0.0
for row in stats.itertuples():
  alpha=500*(row.document_frequency_model+row.document_frequency_other)/n_all;n_model=doc_counts[row.model_profile_id];n_other=n_all-n_model
  log_odds=(math.log((row.document_frequency_model+alpha)/(n_model-row.document_frequency_model+500-alpha))-
            math.log((row.document_frequency_other+alpha)/(n_other-row.document_frequency_other+500-alpha)))
  z=log_odds/(1/(row.document_frequency_model+alpha)+1/(row.document_frequency_other+alpha))**.5
  max_error=max(max_error,abs(log_odds-row.log_odds),abs(z-row.z_score))
if max_error>1e-9:raise RuntimeError(f"SQL/Python phrase log-odds parity failed: {max_error}")
top=pd.concat([part.reindex(part.z_score.abs().sort_values(ascending=False).index).head(15) for _,part in stats.groupby("model_profile_id")],ignore_index=True) if len(stats) else stats
top.to_csv(run/"tables/distinctive-phrases.csv",index=False)
manifest={"schemaVersion":1,"campaignId":campaign,"trainingManifestHash":manifest_hash,"documents":len(rows),"candidatePhrases":len(frequency),"selectedPhrases":len(selected),"occurrences":len(occurrences),"statsRows":len(stats),"sqlPythonMaximumAbsoluteError":max_error,"criteria":criteria}
(run/"manifests/phrases.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest,indent=2));conn.close()
