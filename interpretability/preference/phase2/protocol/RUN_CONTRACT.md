# Phase 2 run contract

Every producing stage: immutable per-item JSONL (`results.jsonl`,
canonical JSON, fsync per batch), atomic resume cursor
(`state/resume_cursor.json`), same-command resume, resume REFUSED on any
config/bank/model hash change, checkpoint/mirror cadence <= 10 minutes,
run dirs gitignored under `interpretability/runs/pref2_*` and mirrored to
`MyDrive/preference/phase2/runs/`. Stages: `behavioral_dev` (B-DEV;
port/format gates), `surface_frozen` (B-SURF), `calibration_dev`
(B-PC-MECH train+val, E2), `behavioral_frozen` (frozen battery;
exactly-once per model; requires the freeze tag + E2 amendment),
`capture_pass` (E9, cross-model mechanism only). Captures: single-row,
native bf16, sharded per site with per-shard SHA256 in
`state/captures_manifest.json`; readers verify hashes. Diagnostics run
once per run dir and hard-gate: render parity, parser adversarial matrix,
hook no-op, capture replay parity, single-row replay determinism, batched
generation equality.
