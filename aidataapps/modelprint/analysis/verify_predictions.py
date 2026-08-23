#!/usr/bin/env python3
"""Recompute retained probe metrics without fitting or database access."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

parser=argparse.ArgumentParser();parser.add_argument("--run");parser.add_argument("--tolerance",type=float,default=1e-12);args=parser.parse_args()
lab=Path(__file__).resolve().parents[1];run=Path(args.run or lab/(lab/".current-run").read_text().strip()).resolve();metrics=json.loads((run/"metrics/attribution.json").read_text());models=metrics["models"]
rows=pd.read_parquet(run/"tables/predictions.parquet");checked=[];failures=[]
for (representation,suite),frame in rows.groupby(["representation","suite"]):
 truth=frame.model_profile_id.to_numpy();pred=frame.predicted_model_profile_id.to_numpy();macro=float(f1_score(truth,pred,average="macro",labels=models,zero_division=0));accuracy=float(accuracy_score(truth,pred))
 block=metrics["representations"].get(representation,{});expected=None
 if str(suite).startswith("lofo:"):expected=block.get("lofo",{}).get("families",{}).get(str(suite).split(":",1)[1])
 else:expected=block.get("suites",{}).get(suite)
 if expected is None:failures.append({"representation":representation,"suite":suite,"error":"missing metric block"});continue
 error=max(abs(macro-float(expected["macroF1"])),abs(accuracy-float(expected["accuracy"])));checked.append({"representation":representation,"suite":suite,"rows":len(frame),"macroF1":macro,"accuracy":accuracy,"maximumError":error})
 if error>args.tolerance:failures.append(checked[-1])
output={"schemaVersion":1,"runId":run.name,"checked":checked,"maximumError":max([row["maximumError"] for row in checked] or [0]),"tolerance":args.tolerance,"failures":failures}
(run/"metrics/prediction-reconstruction.json").write_text(json.dumps(output,indent=2)+"\n");print(json.dumps(output,indent=2))
if failures:raise SystemExit(2)
