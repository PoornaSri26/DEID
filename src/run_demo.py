#!/usr/bin/env python
"""
Demo script for the Privacy-Preserving De-identification project.

Demonstrates the canonical 5-stage hybrid pipeline
(pipelines.hybrid_pipeline_integrated.IntegratedPipeline) on sample
clinical text, in rule-based fallback mode (no GlNER/LLM required).
"""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.hybrid_pipeline_integrated import IntegratedPipeline
from src.evaluation.metrics import compute_clinical_utility_survival


def main():
    print("=" * 70)
    print("PRIVACY-PRESERVING DE-IDENTIFICATION: HYBRID PIPELINE DEMO")
    print("=" * 70)
    print()
    print("Project: Hybrid rules + NER + local LLM verification")
    print("Evaluated on: Public English and Tamil datasets (Meddies-PII)")
    print("i2b2 2014: Secondary gold-standard (access pending via DUA)")
    print()

    config = IntegratedPipeline.Config(
        use_giner=False,   # GlNER optional; falls back to rules
        use_llm=False,     # heuristic verification only
        enable_timing=True,
    )
    pipeline = IntegratedPipeline(config)

    test_cases = [
        {
            "name": "Case 1: Full clinical note (from Slide 5)",
            "text": "Ravi Teja, 52, admitted to Yashoda Hospital, Hyderabad on 12 March 2026. Ph 98480 12345.",
        },
        {
            "name": "Case 2: With contact and MRN",
            "text": "Patient Maria Santos, MRN 12345678, diagnosed with Type 2 diabetes. Contact: maria@clinic.edu",
        },
        {
            "name": "Case 3: Extended clinical note",
            "text": "John Doe was seen in clinic on 01/15/2024. He has hypertension and takes Lisinopril 10mg. "
                    "Follow up in 3 months. Phone: 555-0199. Email: john.doe@health.org. MRN: 987654321.",
        },
    ]

    all_results = []

    print("\nRunning 5-stage hybrid pipeline on test cases:\n")
    print("-" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"CASE {i} - {case['name']}")
        print("-" * 70)

        result = pipeline.run_full_pipeline(case["text"], language="en")
        all_results.append(result)

        summary = result["summary"]
        print()
        print(f"  Total entities detected: {summary['total_entities']}")
        print(f"  By type: {summary['by_type']}")
        print(f"  Replacements made: {summary['replacements_made']}")
        print(f"  Pipeline time: {result.get('timing_total', 0):.3f}s")
        print(f"  De-identified text: {result['deidentified_text']}")

    print()
    print("AGGREGATE RESULTS")
    print("=" * 70)

    total_entities = sum(r["summary"]["total_entities"] for r in all_results)
    total_replacements = sum(r["summary"]["replacements_made"] for r in all_results)

    print()
    print(f"Total entities across all cases: {total_entities}")
    print(f"Total replacements made: {total_replacements}")
    print(f"Average replacements per document: {total_replacements / len(all_results):.1f}")

    # Clinical utility survival for case 3
    print()
    print("Clinical utility check (diagnosis preservation):")
    case3 = test_cases[2]
    deid3 = all_results[2]["deidentified_text"]

    medical_terms = ["hypertension", "Lisinopril", "10mg"]
    utility = compute_clinical_utility_survival(case3["text"], deid3, medical_terms)

    print(f"  Terms checked: {utility['total_medical_terms']}")
    print(f"  Survived: {utility['survived']}")
    print(f"  Lost: {utility['lost']}")
    print(f"  Survival rate: {utility['survival_rate'] * 100:.0f}%")

    if utility["survival_rate"] > 0.5:
        print("  Good: Core clinical content preserved")
    else:
        print("  Warning: Significant clinical content lost")

    print()
    print("NEXT STEPS")
    print("=" * 70)
    print("  1. Download Meddies-PII: python scripts/download_meddies.py")
    print("  2. De-identify text:   python -m deid \"clinical note text\"")
    print("  3. Run evaluation:     python src/evaluation/run_evaluation.py")
    print("  4. Set up GlNER/LLM for higher recall (optional extras in requirements.txt)")
    print("  5. Apply for i2b2 DUA: portal.dbmi.hms.harvard.edu")
    print("=" * 70)

    print()
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
