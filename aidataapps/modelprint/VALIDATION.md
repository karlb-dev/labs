# ModelPrint Validation Record

This record is populated from retained artifacts. A passing foundation does not imply a positive scientific result.

## Foundation gates

| Gate | Current disposition | Evidence |
|---|---|---|
| Branch and isolated Lab 02 path | PASS | `aidataapps-modelprint`, `aidataapps/modelprint/` |
| SQL Server 2025 compatibility 170 | PASS | current run capability artifact |
| Exact SQL vector functions | PASS | current run capability artifact |
| Approximate index execution | PASS, legacy syntax | captured vector-index-seek plan |
| Stale-index behavior disabled | PASS | capability artifact |
| Model revisions and HF access | PASS | model-registry snapshot |
| Serving images resolved to digests | PASS | model-registry snapshot |
| Prompt exact duplicates | PASS, zero | `data/audits/canonical_text_duplicates.csv` |
| Prompt groups crossing splits | PASS, zero | `data/audits/prompt_group_overlap.csv` |
| Near-duplicates crossing splits | PASS, zero | build-time hard gate |
| Relation templates crossing splits | PASS, zero | build-time hard gate |
| Prompt self-name leakage | PASS, zero | `data/audits/prompt_name_leakage.csv` |
| Unit/type checks | PASS | 12 tests at foundation milestone |

## Environment notes

- The Colab VM uses a rootless Docker daemon at `/run/user/1000/docker.sock`.
- The run reuses Lab 01's healthy SQL Server container while using the isolated `ModelPrint` database. Lab 01 chat/embedding containers are stopped; their volumes and data are retained.
- Qwen3 and BGE embedding services remain resident. One target chat model is loaded at a time.
- The local SQL Server 2025 CU8 build uses an unversioned legacy vector-index format. Its observed minimum-row and maintenance-DMV behavior differs from current product documentation and is reported as observed capability evidence, not generalized.

## Scientific gates

Per-profile port gates, generation completeness, feature manifests, calibration, OOD, exact/ANN comparison, reconstruction, database export, and final claims are pending the experiment run. They will be filled by report generation rather than manually asserted.
