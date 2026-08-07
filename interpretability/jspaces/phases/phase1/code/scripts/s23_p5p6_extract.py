# CPU extractor: P5 (cot_rescue) + P6 (robustness_seed1) aggregates in
# report-ready form. Run after the s21 driver completes; prints the
# numbers the handout/report fill-in pass needs, in one block.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import RUN_DIR_V2, read_json

M = RUN_DIR_V2 / "metrics"


def p5():
    d = read_json(M / "cot_rescue.json")
    items = d["items"]
    print("=== P5 cot_rescue ===")
    ref = d.get("nothink_reference", {})
    print("no-think refs:", ref)
    for kind in ("twohop", "onehop", "arithmetic"):
        ks = [r for r in items.values() if r["kind"] == kind]
        for cond in ("frozen_j10", "frozen_rand10", "none", "jspace_k20"):
            rows = [r[cond] for r in ks if cond in r]
            if not rows:
                continue
            n = len(rows)
            f = lambda key: sum(r[key] for r in rows) / n
            print(f"{kind:>10} {cond:>14} n={n:>2} post={f('post'):.2f} "
                  f"any={f('any'):.2f} think={f('think'):.2f} "
                  f"closed={f('closed_think'):.2f}")
    # paired rescue contrast on twohop: frozen_j10 'any' vs no-think recall
    ks = [r for r in items.values()
          if r["kind"] == "twohop" and "frozen_j10" in r]
    anyr = sum(r["frozen_j10"]["any"] for r in ks) / max(len(ks), 1)
    print(f"headline: frozen_j10 twohop any-rate {anyr:.2f} (n={len(ks)}) vs "
          f"no-think frozen_j10 recall {ref.get('frozen_j10_twohop')} vs "
          f"no-think baseline {ref.get('none_twohop')}")


def p6():
    p = M / "robustness_seed1.json"
    if not p.exists():
        print("=== P6 missing ===")
        return
    d = read_json(p)
    print("=== P6 robustness_seed1 ===")
    print("notes:", d.get("notes"))
    for cond, tasks in d["conditions"].items():
        for t, e in tasks.items():
            print(f"{cond:>16} {t:>10}: {e['mean']:.3f} "
                  f"[{e['ci_lo']:.3f},{e['ci_hi']:.3f}] n={e['n']}")
    print("seed comparison:", json.dumps(d.get("seed_comparison", {})))


if __name__ == "__main__":
    p5()
    p6()
