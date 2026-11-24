#!/usr/bin/env python
"""Real Meddies-PII v2 evaluation: entity-level P/R/F1 vs gold spans.

Downloads nothing itself — expects the parquet shards from
scripts/download_meddies.py (or hf_hub_download) under ./data/meddies-pii.
Samples N documents per language, runs the integrated pipeline's stages,
and scores predicted spans against gold {start,end,category} annotations.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipelines.hybrid_pipeline_integrated import IntegratedPipeline
from src.evaluation.metrics import compute_entity_metrics, compute_micro_macro_f1

CACHE = Path.home() / ".cache/huggingface/hub/datasets--Meddies--meddies-pii-v2/snapshots"
N_PER_LANG = 200
SEED = 7


def find_parquet(lang_dir: str) -> Path:
    hits = list(CACHE.glob(f"*/{lang_dir}/train-*.parquet"))
    if not hits:
        sys.exit(f"parquet for {lang_dir} not found — run scripts/download_meddies.py")
    return hits[0]


def gold_spans(labels) -> list:
    out = []
    for lab in labels:
        try:
            out.append((int(lab["start"]), int(lab["end"]), lab["category"]))
        except Exception:
            continue
    return out


def predicted_spans(pipeline, text: str, lang: str) -> list:
    rule_entities = pipeline._run_rules_stage(text)
    ner_entities = pipeline._run_ner_stage(text, lang)
    merged = pipeline._run_merge_stage(rule_entities, ner_entities)
    verified = pipeline._run_verification_stage(merged, text)
    return [(e["start"], e["end"], e["type"]) for e in verified]


def main() -> int:
    use_presidio = "--presidio" in sys.argv
    config = IntegratedPipeline.Config(
        use_giner=False, use_llm=False, enable_timing=False, use_presidio=use_presidio
    )
    pipeline = IntegratedPipeline(config)

    summary = {}
    for lang_dir, lang in [("english", "en"), ("tamil", "ta")]:
        df = pd.read_parquet(find_parquet(lang_dir))
        sample = df.sample(n=min(N_PER_LANG, len(df)), random_state=SEED)

        all_gold, all_pred = [], []
        for _, row in sample.iterrows():
            text = str(row["text"])
            try:
                pred = predicted_spans(pipeline, text, lang)
            except Exception:
                pred = []
            all_gold.append(gold_spans(row["label"]))
            all_pred.append(pred)

        pooled = compute_entity_metrics(
            [s for g in all_gold for s in g],
            [s for p in all_pred for s in p],
        )
        mm = compute_micro_macro_f1(all_gold, all_pred, "entity")
        summary[lang] = {"pooled": pooled, "micro_macro": mm, "n": len(sample)}

        print(f"\n=== {lang_dir} (n={len(sample)}) ===")
        print(f"  entity-level  P={pooled['precision']:.3f} R={pooled['recall']:.3f} F1={pooled['f1']:.3f}  (tp={pooled['tp']} fp={pooled['fp']} fn={pooled['fn']})")
        print(f"  micro F1={mm['micro_f1']:.3f}  macro F1={mm['macro_f1']:.3f}")
        worst = sorted(mm["category_metrics"].items(), key=lambda kv: kv[1]["f1"])[:5]
        print("  weakest categories:", ", ".join(f"{k}={v['f1']:.2f}" for k, v in worst))

    print("\nDone. Numbers above are real gold-span F1 on Meddies-PII v2 train samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
