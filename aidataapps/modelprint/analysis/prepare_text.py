#!/usr/bin/env python3
import argparse, hashlib, json, os, re
from pathlib import Path

import pandas as pd
import pymssql
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
run_dir = Path(args.run).resolve()
freeze = json.loads((run_dir / "manifests/campaign-freeze.json").read_text())
campaign_id = int(freeze["campaignId"])
robustness_path = run_dir / "manifests/robustness-freeze.json"
campaign_ids = [campaign_id] + ([int(json.loads(robustness_path.read_text())["campaignId"])] if robustness_path.exists() else [])
campaign_placeholders = ",".join(["%s"] * len(campaign_ids))
lab_dir = Path(__file__).resolve().parents[1]
registry = json.loads((lab_dir / "data/manifests/model-registry-snapshot.json").read_text())
ref = registry["embeddings"]["qwen3-embedding-0.6b"]
tokenizer = AutoTokenizer.from_pretrained(ref["modelId"], revision=ref["revision"], token=os.getenv("HF_TOKEN") or None,
                                            cache_dir=lab_dir / "data/work/tokenizers", trust_remote_code=False)
conn = pymssql.connect(server=os.getenv("SQLSERVER_HOST", "127.0.0.1"), port=int(os.getenv("SQLSERVER_PORT", "1433")),
                       user="sa", password=os.environ["MSSQL_SA_PASSWORD"], database=os.getenv("MSSQL_DATABASE", "ModelPrint"),
                       autocommit=False, login_timeout=60, timeout=3600)
cursor = conn.cursor(as_dict=True)
cursor.execute(f"""SELECT g.generation_id,g.final_text,g.reasoning_text,t.text_artifact_id,t.artifact_text
FROM dbo.generations g JOIN dbo.generation_text_artifacts m ON m.generation_id=g.generation_id
JOIN dbo.text_artifacts t ON t.text_artifact_id=m.text_artifact_id
WHERE g.campaign_id IN ({campaign_placeholders}) AND t.text_view_id='raw-final-v1' ORDER BY g.generation_id""", tuple(campaign_ids))
rows = list(cursor)

def band(count):
    if count < 16: return "micro"
    if count < 64: return "short"
    if count < 256: return "medium"
    return "long"

updates, unique_artifacts = [], {}
for row in rows:
    count = len(tokenizer.encode(row["final_text"], add_special_tokens=False))
    reasoning_count = len(tokenizer.encode(row["reasoning_text"], add_special_tokens=False)) if row["reasoning_text"] else None
    updates.append((count, reasoning_count, band(count), row["generation_id"]))
    unique_artifacts[row["text_artifact_id"]] = row["artifact_text"]
cursor.executemany("UPDATE dbo.generations SET reference_token_count=%s,reasoning_token_count=%s,length_band=%s WHERE generation_id=%s", updates)

segments = []
def add_segment(artifact_id, segmenter, ordinal, start, end, token_start=None, token_end=None):
    text = unique_artifacts[artifact_id][start:end]
    if not text.strip(): return
    segments.append((artifact_id, segmenter, ordinal, start, end, token_start, token_end, text,
                     hashlib.sha256(text.encode()).hexdigest(), 1 if len(tokenizer.encode(text, add_special_tokens=False)) >= 16 else 0))

sentence_re = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
for artifact_id, text in unique_artifacts.items():
    tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    add_segment(artifact_id, "whole-v1", 0, 0, len(text), 0, len(offsets))
    for ordinal, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.DOTALL)):
        add_segment(artifact_id, "paragraph-v1", ordinal, match.start(), match.end())
    packs, current = [], []
    for match in sentence_re.finditer(text):
        if not match.group().strip(): continue
        proposed = current + [match]
        if current and len(tokenizer.encode(text[current[0].start():match.end()], add_special_tokens=False)) > 128:
            packs.append(current); current = [match]
        else: current = proposed
    if current: packs.append(current)
    for ordinal, pack in enumerate(packs): add_segment(artifact_id, "sentence-pack-128-v1", ordinal, pack[0].start(), pack[-1].end())
    for ordinal, token_start in enumerate(range(0, len(offsets), 72)):
        token_end = min(token_start + 96, len(offsets))
        if token_start >= token_end: break
        start, end = offsets[token_start][0], offsets[token_end - 1][1]
        add_segment(artifact_id, "token-window-96-24-v1", ordinal, start, end, token_start, token_end)
        if token_end == len(offsets): break

for chunk in [segments[i:i+1000] for i in range(0, len(segments), 1000)]:
    cursor.executemany("""IF NOT EXISTS(SELECT 1 FROM dbo.output_segments WHERE text_artifact_id=%s AND segmenter_id=%s AND ordinal=%s)
INSERT dbo.output_segments(text_artifact_id,segmenter_id,ordinal,char_start,char_end,token_start,token_end,segment_text,segment_sha256,is_primary_eligible)
VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", [(r[0],r[1],r[2],*r) for r in chunk])
conn.commit()

cursor.execute(f""";WITH artifacts AS
(SELECT DISTINCT t.text_artifact_id,t.artifact_text FROM dbo.text_artifacts t JOIN dbo.generation_text_artifacts m ON m.text_artifact_id=t.text_artifact_id
 JOIN dbo.generations g ON g.generation_id=m.generation_id WHERE g.campaign_id IN ({campaign_placeholders}) AND t.text_view_id='raw-final-v1')
INSERT dbo.output_segments(text_artifact_id,segmenter_id,ordinal,char_start,char_end,token_start,token_end,segment_text,segment_sha256,is_primary_eligible)
SELECT a.text_artifact_id,'sql-chunks-v1',CONVERT(int,c.chunk_order),CONVERT(int,c.chunk_offset),CONVERT(int,c.chunk_offset+c.chunk_length),NULL,NULL,c.chunk,
 LOWER(CONVERT(varchar(64),HASHBYTES('SHA2_256',CONVERT(varbinary(max),c.chunk)),2)),CASE WHEN c.chunk_length>=64 THEN 1 ELSE 0 END
FROM artifacts a CROSS APPLY AI_GENERATE_CHUNKS(SOURCE=a.artifact_text,CHUNK_TYPE=FIXED,CHUNK_SIZE=600,OVERLAP=20,ENABLE_CHUNK_SET_ID=1) c
WHERE NOT EXISTS(SELECT 1 FROM dbo.output_segments s WHERE s.text_artifact_id=a.text_artifact_id AND s.segmenter_id='sql-chunks-v1' AND s.ordinal=CONVERT(int,c.chunk_order));""", tuple(campaign_ids))
conn.commit()

df = pd.DataFrame(updates, columns=["reference_token_count","reasoning_token_count","length_band","generation_id"])
table_path = run_dir / "tables/reference_token_counts.parquet"
df.to_parquet(table_path, index=False)
manifest = {"schemaVersion":1,"campaignIds":campaign_ids,"generations":len(rows),"uniqueArtifacts":len(unique_artifacts),"appSegments":len(segments),
            "lengthBands":df["length_band"].value_counts().to_dict(),"referenceTokenizer":{"model":ref["modelId"],"revision":ref["revision"]},
            "table":str(table_path),"sqlChunks":{"chunkSize":600,"overlapPercent":20}}
print(json.dumps(manifest))
conn.close()
