"""Stage the exact OLMo control snapshot to local NVMe and hash every file."""
from __future__ import annotations

import json
from pathlib import Path

from jspace_gemma.gpu import require_cuda
from jspace_gemma.manifests import file_sha256, require_clean_tree
from jspace_gemma.paths import directory
from jspace_gemma.staging import stage_snapshot

REPO_ID = "allenai/Olmo-3-32B-Think"
REVISION = "ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7"
SEED = Path("/content/drive/MyDrive/hf_cache/models--allenai--Olmo-3-32B-Think")
CACHE = Path("/content/hf_olmo_control")


def main() -> None:
    git = require_clean_tree()
    gpu = require_cuda()
    output = directory("manifests") / "olmo_control_local_snapshot_v1.json"
    result = stage_snapshot(
        repo_id=REPO_ID,
        revision=REVISION,
        cache_root=CACHE,
        seed_model_root=SEED,
        output_manifest=output,
    )
    print(
        json.dumps(
            {
                "snapshot": result["snapshot"],
                "manifest": str(output),
                "manifest_sha256": file_sha256(output),
                "weight_shards": len(result["weight_shards"]),
                "all_content_hashes_verified": result["all_content_hashes_verified"],
                "code_commit": git["code_commit"],
                "gpu": gpu,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
