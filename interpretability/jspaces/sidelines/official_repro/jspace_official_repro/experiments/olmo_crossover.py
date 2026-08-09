"""OLMo instrument cross-over: new merged OR1 lens vs frozen campaign
lens under paper swap and campaign protected ablation (plan §11.3)."""
from __future__ import annotations

from ..paths import DRIVE_ROOT
from ..readout import lens_to_device
from . import admission as admission_module
from .crossover import run_lane
from .olmo_lane import load_campaign_lens, load_or1_lens


def main() -> None:
    if (DRIVE_ROOT / "crossover_olmo" / "crossover_olmo.json").exists():
        print("olmo crossover already banked")
        return
    model, hf_model, tokenizer = admission_module.load_olmo()
    merged = load_or1_lens("merged")
    campaign = load_campaign_lens()
    lens_to_device(merged, "cuda:0")
    lens_to_device(campaign, "cuda:0")
    summary = run_lane(model, hf_model, merged, lane="olmo",
                       extra_lenses={"campaign": campaign})
    print("crossover olmo:", summary["paper_swap_top1_primary"])


if __name__ == "__main__":
    main()
