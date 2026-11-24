"""
Evaluation metrics for de-identification systems.

Entity-level P, R, F1, micro & macro, precision/recall/F1 per category,
false redaction rate, and clinical utility proxy.
"""

from typing import Dict, List, Tuple, Any, Set, Optional
from collections import defaultdict


def span_matches(
    gold_span: Tuple[int, int, str],  # (start, end, type)
    pred_span: Tuple[int, int, str],  # (start, end, type)
    match_type: str = "entity",
) -> bool:
    """
    Determine if a predicted span matches a gold span.
    
    Entity-level: full span AND type must match
    Token-level: token-level overlap
    Character-level: character overlap (not implemented here)
    
    Args:
        gold_span: (start, end, type) from ground truth
        pred_span: (start, end, type) from prediction
        match_type: "entity", "token", or "character"
    
    Returns:
        True if spans match according to match_type
    """
    g_start, g_end, g_type = gold_span
    p_start, p_end, p_type = pred_span
    
    if match_type == "entity":
        # Entity-level: exact span AND type match
        return g_type == p_type and g_start == p_start and g_end == p_end
    
    elif match_type == "token":
        # Token-level: overlapping tokens (simplified)
        # For this demo, consider token overlap if spans overlap at all
        overlap = max(0, min(g_end, p_end) - max(g_start, p_start))
        return overlap > 0 and g_type == p_type
    
    return False


def compute_entity_metrics(
    gold_entities: List[Tuple[int, int, str]],
    pred_entities: List[Tuple[int, int, str]],
    match_type: str = "entity",
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 for entity detection.
    
    Args:
        gold_entities: List of (start, end, type) from ground truth
        pred_entities: List of (start, end, type) from prediction
        match_type: "entity", "token", or "character"
    
    Returns:
        Dict with precision, recall, f1, tp, fp, fn
    """
    # Convert to sets for matching
    gold_set = set((s, e, t) for s, e, t in gold_entities)
    pred_set = set((s, e, t) for s, e, t in pred_entities)
    
    # True Positives: predicted entities that match a gold entity
    tp = 0
    matched_gold = set()
    
    for p_entity in pred_entities:
        for g_entity in gold_set:
            if span_matches(g_entity, p_entity, match_type):
                tp += 1
                matched_gold.add(g_entity)
                break
    
    # False Positives: predicted entities not matching any gold
    fp = len(pred_entities) - tp
    
    # False Negatives: gold entities not matched by any prediction
    fn = len(gold_set) - len(matched_gold)
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_micro_macro_f1(
    all_gold: List[List[Tuple[int, int, str]]],
    all_pred: List[List[Tuple[int, int, str]]],
    match_type: str = "entity",
) -> Dict[str, float]:
    """
    Compute micro-averaged and macro-averaged F1 across all categories.
    
    Micro F1: pool all TP, FP, FN across categories, then compute F1
    Macro F1: compute F1 per category, then average (rare categories not hidden)
    
    Args:
        all_gold: List of gold entity lists (one per document)
        all_pred: List of predicted entity lists (one per document)
        match_type: "entity", "token", or "character"
    
    Returns:
        Dict with micro_precision, micro_recall, micro_f1,
               macro_precision, macro_recall, macro_f1
    """
    # --- Micro-averaged ---
    # Pool all counts across all documents
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for gold, pred in zip(all_gold, all_pred):
        metrics = compute_entity_metrics(gold, pred, match_type)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # --- Macro-averaged ---
    # Compute F1 per category, then average
    all_categories = set()

    for gold, pred in zip(all_gold, all_pred):
        for s, e, t in gold:
            all_categories.add(t)
        for s, e, t in pred:
            all_categories.add(t)

    # Per-category computation
    category_metrics = {}
    
    for cat in all_categories:
        cat_gold = []
        cat_pred = []
        
        for gold, pred in zip(all_gold, all_pred):
            for s, e, t in gold:
                if t == cat:
                    cat_gold.append((s, e, t))
            for s, e, t in pred:
                if t == cat:
                    cat_pred.append((s, e, t))
        
        metrics = compute_entity_metrics(cat_gold, cat_pred, match_type)
        category_metrics[cat] = metrics
    
    # Average F1 across categories
    f1_scores = [metrics["f1"] for metrics in category_metrics.values()]
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # Also average precision and recall for macro
    macro_precision = sum(m["precision"] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    macro_recall = sum(m["recall"] for m in category_metrics.values()) / len(category_metrics) if category_metrics else 0.0
    
    return {
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "category_metrics": {cat: {"precision": round(m["precision"], 4),
                                    "recall": round(m["recall"], 4),
                                    "f1": round(m["f1"], 4)}
                            for cat, m in category_metrics.items()},
        "categories": sorted(all_categories),
    }


def compute_false_redaction_rate(
    gold_entities: List[Tuple[int, int, str]],
    pred_entities: List[Tuple[int, int, str]],
    match_type: str = "entity",
) -> Dict[str, Any]:
    """
    Compute false redaction rate.

    A predicted entity is a false redaction when it does not match any gold
    entity — i.e. non-sensitive content was wrongly removed.

    FRR = FP / (TP + FP): fraction of redacted spans that were not PHI.
    This is equivalent to 1 - precision at the span level.

    Args:
        gold_entities: List of (start, end, type) from ground truth
        pred_entities: List of (start, end, type) from prediction
        match_type: "entity" or "token"

    Returns:
        Dict with false_redaction_rate, tp, fp, fn, total_predicted
    """
    metrics = compute_entity_metrics(gold_entities, pred_entities, match_type)
    tp, fp = metrics["tp"], metrics["fp"]
    total_predicted = tp + fp

    frr = fp / total_predicted if total_predicted > 0 else 0.0

    return {
        "false_redaction_rate": round(frr, 4),
        "tp": tp,
        "fp": fp,
        "fn": metrics["fn"],
        "total_predicted": total_predicted,
    }


def compute_false_negative_rate(
    gold_entities: List[Tuple[int, int, str]],
    pred_entities: List[Tuple[int, int, str]],
    match_type: str = "entity",
) -> Dict[str, Any]:
    """
    Compute false negative rate: fraction of gold PHI spans missed.

    FNR = FN / (TP + FN) = 1 - recall at the span level.
    """
    metrics = compute_entity_metrics(gold_entities, pred_entities, match_type)
    tp, fn = metrics["tp"], metrics["fn"]
    total_gold = tp + fn

    return {
        "false_negative_rate": round(fn / total_gold, 4) if total_gold > 0 else 0.0,
        "tp": tp,
        "fp": metrics["fp"],
        "fn": fn,
        "total_gold": total_gold,
    }


def compute_clinical_utility_survival(
    gold_text: str,
    deidentified_text: str,
    medical_terms: List[str],
) -> Dict[str, Any]:
    """
    Compute clinical utility proxy: share of medical terms that survive de-identification.
    
    Args:
        gold_text: Original clinical text
        deidentified_text: De-identified output
        medical_terms: List of known medical terms/diagnoses/drugs to check
    
    Returns:
        Dict with survival rates per term category
    """
    survival_counts = {"total": 0, "survived": 0, "lost": 0}
    term_found = {}
    
    for term in medical_terms:
        # Case-insensitive check
        term_lower = term.lower()
        in_gold = term_lower in gold_text.lower()
        in_deid = term_lower in deidentified_text.lower()
        
        if in_gold:
            survival_counts["total"] += 1
            if in_deid:
                survival_counts["survived"] += 1
                term_found[term] = "survived"
            else:
                survival_counts["lost"] += 1
                term_found[term] = "lost"
    
    survival_rate = survival_counts["survived"] / survival_counts["total"] if survival_counts["total"] > 0 else 0.0
    
    return {
        "total_medical_terms": survival_counts["total"],
        "survived": survival_counts["survived"],
        "lost": survival_counts["lost"],
        "survival_rate": round(survival_rate, 4),
        "term_details": term_found,
    }


def compute_per_category_f1(
    all_gold: List[List[Tuple[int, int, str]]],
    all_pred: List[List[Tuple[int, int, str]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-category precision, recall, F1.
    
    Useful for seeing how rare categories (like IDs) perform
    versus common ones (like names).
    """
    # Collect all entities by category across all documents
    gold_by_cat = defaultdict(list)
    pred_by_cat = defaultdict(list)
    all_cats = set()
    
    for gold, pred in zip(all_gold, all_pred):
        for s, e, t in gold:
            gold_by_cat[t].append((s, e))
            all_cats.add(t)
        for s, e, t in pred:
            pred_by_cat[t].append((s, e))
            all_cats.add(t)
    
    per_category = {}
    for cat in all_cats:
        # Build gold/pred with type info for entity-level matching
        gold_entities = [(s, e, cat) for s, e in gold_by_cat[cat]]
        pred_entities = [(s, e, cat) for s, e in pred_by_cat[cat]]
        
        metrics = compute_entity_metrics(gold_entities, pred_entities, "entity")
        per_category[cat] = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
    
    return per_category


# ============================================================
# Example usage / demo
# ============================================================

def demo_metrics():
    """Demo the metrics computation."""
    print("=" * 60)
    print("EVALUATION METRICS DEMO")
    print("=" * 60)
    
    # Example: 2 documents
    all_gold = [
        # Doc 1: ground truth entities
        [(0, 8, "human_name"), (12, 22, "date"), (30, 38, "phone_number")],
        # Doc 2
        [(0, 6, "human_name"), (25, 31, "date"), (45, 52, "id_number")],
    ]
    
    all_pred = [
        # Doc 1: predictions
        [(0, 8, "human_name"), (12, 18, "date")],  # Missed part of date
        # Doc 2
        [(0, 6, "human_name"), (25, 31, "date"), (45, 52, "id_number"), (60, 68, "extra")],  # FP
    ]
    
    # Entity-level metrics micro/macro
    metrics = compute_micro_macro_f1(all_gold, all_pred, "entity")
    
    print("\nMicro-averaged metrics:")
    print(f"  P: {metrics['micro_precision']:.3f}, R: {metrics['micro_recall']:.3f}, F1: {metrics['micro_f1']:.3f}")
    
    print("\nMacro-averaged metrics:")
    print(f"  P: {metrics['macro_precision']:.3f}, R: {metrics['macro_recall']:.3f}, F1: {metrics['macro_f1']:.3f}")
    
    print("\nPer-category F1:")
    per_cat = compute_per_category_f1(all_gold, all_pred)
    for cat, m in per_cat.items():
        print(f"  {cat}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f} (TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    
    # Clinical utility demo
    print("\nClinical utility survival check:")
    gold = "Patient John Smith was diagnosed with Type 2 diabetes and prescribed Metformin 500mg."
    deid = "Patient [NAME] was diagnosed with Type 2 diabetes and prescribed Metformin 500mg."
    
    result = compute_clinical_utility_survival(gold, deid, ["Type 2 diabetes", "Metformin", "Patient name"])
    print(f"  Terms total: {result['total_medical_terms']}")
    print(f"  Survived: {result['survived']}")
    print(f"  Lost: {result['lost']}")
    print(f"  Survival rate: {result['survival_rate']:.3f}")
    print(f"  Details: {result['term_details']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_metrics()