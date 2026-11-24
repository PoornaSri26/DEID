#!/usr/bin/env python
"""
Main demo script for the Privacy-Preserving De-identification project.

 Demonstrates the full 5-stage hybrid pipeline on sample clinical text,
 with fallback to rule-based-only when LLM/models not fully configured.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from pipelines.hybrid_pipeline import HybridPipeline
from src.evaluation.metrics import compute_micro_macro_f1, compute_clinical_utility_survival, compute_per_category_f1


def main():
    print("=" * 70)
    print("PRIVACY-PRESERVING DE-IDENTIFICATION: HYBRID PIPELINE DEMO")
    print("=" * 70)
    print()
    print("Project: Hybrid rules + NER + local LLM verification")
    print("Evaluated on: Public English and Tamil datasets (Meddies-PII)")
    print("i2b2 2014: Secondary gold-standard (access pending via DUA)")
    print()

    # Initialize pipeline
    # Note: LLM verification is simulated; rules + NER will work standalone
    pipeline = HybridPipeline(
        giner_threshold=0.5,
        # LLM will use heuristic verification (see pipeline stage 4)
    )

    # Test cases covering different scenarios
    test_cases = [
        # Case 1: Original from presentation slides
        {
            "name": "Case 1: Full clinical note (from Slide 5)",
            "text": "Ravi Teja, 52, admitted to Yashoda Hospital, Hyderabad on 12 March 2026. Ph 98480 12345.",
            "expected_entities": ["human_name", "date", "phone_number", "organization"]
        },
        # Case 2: With email and ID
        {
            "name": "Case 2: With contact and MRN",
            "text": "Patient Maria Santos, MRN 12345678, diagnosed with Type 2 diabetes. Contact: maria@clinic.edu",
            "expected_entities": ["human_name", "date", "phone_number", "email_address", "id_number"]
        },
        # Case 3: Longer note with multiple entities
        {
            "name": "Case 3: Extended clinical note",
            "text": "John Doe was seen in clinic on 01/15/2024. He has hypertension and takes Lisinopril 10mg. "
                   "Follow up in 3 months. Phone: 555-0199. Email: john.doe@health.org. MRN: 987654321.",
            "expected_entities": ["human_name", "date", "phone_number", "email_address", "id_number"]
        }
    ]

    # Run pipeline on each test case
    all_results = []

    print("\n\nRunning 5-stage hybrid pipeline on test cases:\n")
    print("-" * 70)

    for i, case in enumerate(test_cases, 1):
        print("CASE " + str(i) + " - " + case["name"])
        print("-" * 70)

        result = pipeline.run_full_pipeline(case["text"])
        all_results.append(result)

        # Print summary
        summary = result["summary"]
        print()
        print("  Total entities detected: " + str(summary["total_entities"]))
        print("  By type: " + str(summary["by_type"]))
        print("  Replacements made: " + str(pipeline.stats["stage5_replaced"]))
        print("  Privacy risk assessment: " + summary["privacy_risk"].upper())

        # Show the de-identified output
        print("  De-identified text: " + result["deidentified_text"])

    # Compute aggregate metrics if we have multiple results
    print()
    print("AGGREGATE RESULTS")
    print("=" * 70)

    total_entities = sum(r["summary"]["total_entities"] for r in all_results)
    total_replacements = sum(r["stats"]["stage5_replaced"] for r in all_results)

    print()
    print("Total entities across all cases: " + str(total_entities))
    print("Total replacements made: " + str(total_replacements))
    print("Average replacements per document: " + str(round(total_replacements/len(all_results), 1)))

    # Check clinical utility survival for case 3
    print()
    print("Clinical utility check (diagnosis preservation):")
    case3 = all_results[2]  # Case 3
    gold = case3["original_text"]
    deid = case3["deidentified_text"]

    medical_terms = ["Type 2 diabetes", "Lisinopril", "50mg"]  # Key medical terms
    utility = compute_clinical_utility_survival(gold, deid, medical_terms)

    print("  Terms checked: " + str(utility["total_medical_terms"]))
    print("  Survived: " + str(utility["survived"]))
    print("  Lost: " + str(utility["lost"]))
    print("  Survival rate: " + str(round(utility["survival_rate"], 1) * 100) + "%")

    if utility["survival_rate"] > 0.5:
        print("  Good: Core clinical content preserved")
    else:
        print("  Warning: Significant clinical content lost")

    # Per-category F1 analysis (synthetic example)
    print()
    print("Per-category entity performance (synthetic example):")
    # Create synthetic gold/pred for demo
    all_gold = [
        [(0, 8, "human_name"), (12, 22, "date"), (30, 38, "phone_number")],
        [(0, 6, "human_name"), (25, 31, "date"), (45, 52, "id_number")]
    ]
    all_pred = [
        [(0, 8, "human_name"), (12, 18, "date")],
        [(0, 6, "human_name"), (25, 31, "date"), (45, 52, "id_number"), (60, 68, "extra")]
    ]

    cat_metrics = compute_micro_macro_f1(all_gold, all_pred, "entity")

    print("  Micro F1: P=" + str(round(cat_metrics['micro_precision'], 3)) + ", R=" + str(round(cat_metrics['micro_recall'], 3)) + ", F1=" + str(round(cat_metrics['micro_f1'], 3)))
    print("  Macro F1: P=" + str(round(cat_metrics['macro_precision'], 3)) + ", R=" + str(round(cat_metrics['macro_recall'], 3)) + ", F1=" + str(round(cat_metrics['macro_f1'], 3)))

    per_cat = compute_per_category_f1(all_gold, all_pred)
    print()
    print("  Per-category F1:")
    for cat, m in sorted(per_cat.items()):
        print("    " + cat + ": F1=" + str(round(m['f1'], 3)) + " (P=" + str(round(m['precision'], 3)) + ", R=" + str(round(m['recall'], 3)) + ", TP=" + str(m['tp']) + ", FP=" + str(m['fp']) + ", FN=" + str(m['fn']) + ")")

    # Emphasize macro F1 importance
    print()
    print("  Note: Macro F1 averages per-category F1, so rare categories")
    print("        (like IDs) are not hidden by common ones (like names).")

    print()
    print("NEXT STEPS")
    print("=" * 70)
    print("  1. Download Meddies-PII: python -m scripts.download_meddies")
    print("  2. Fine-tune GLiNER: python -m src.evaluation.baselines --model giner")
    print("  3. Set up Llama.cpp for local LLM verification")
    print("  4. Apply for i2b2 DUA: portal.dbmi.hms.harvard.edu")
    print("  5. Run full experiments: python -m src.evaluation.ablations")
    print("=" * 70)

    print()
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()