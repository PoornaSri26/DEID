"""Tests for evaluation metrics."""
from src.evaluation.metrics import (
    span_matches,
    compute_entity_metrics,
    compute_micro_macro_f1,
    compute_per_category_f1,
    compute_clinical_utility_survival,
    compute_false_redaction_rate,
    compute_false_negative_rate,
)


def test_span_matches_entity_level():
    assert span_matches((0, 4, "name"), (0, 4, "name"), "entity")
    assert not span_matches((0, 4, "name"), (0, 4, "date"), "entity")
    assert not span_matches((0, 4, "name"), (1, 4, "name"), "entity")


def test_span_matches_token_overlap():
    assert span_matches((0, 10, "name"), (5, 15, "name"), "token")
    assert not span_matches((0, 10, "name"), (5, 15, "date"), "token")


def test_compute_entity_metrics_perfect():
    gold = [(0, 4, "name"), (10, 14, "date")]
    pred = [(0, 4, "name"), (10, 14, "date")]
    m = compute_entity_metrics(gold, pred)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0
    assert m["f1"] == 1.0


def test_compute_entity_metrics_partial():
    gold = [(0, 4, "name"), (10, 14, "date")]
    pred = [(0, 4, "name"), (20, 24, "date")]  # one wrong span
    m = compute_entity_metrics(gold, pred)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert 0 < m["f1"] < 1


def test_micro_macro_f1():
    all_gold = [[(0, 4, "name"), (10, 14, "date")]]
    all_pred = [[(0, 4, "name"), (10, 14, "date")]]
    m = compute_micro_macro_f1(all_gold, all_pred)
    assert m["micro_f1"] == 1.0
    assert m["macro_f1"] == 1.0


def test_per_category_f1():
    all_gold = [[(0, 4, "name")], [(0, 4, "date")]]
    all_pred = [[(0, 4, "name")], [(0, 4, "date")]]
    per_cat = compute_per_category_f1(all_gold, all_pred)
    assert per_cat["name"]["f1"] == 1.0
    assert per_cat["date"]["f1"] == 1.0


def test_false_redaction_rate():
    gold = [(0, 4, "name"), (10, 14, "date")]
    pred = [(0, 4, "name"), (10, 14, "date"), (20, 25, "name")]  # one FP
    res = compute_false_redaction_rate(gold, pred)
    assert res["tp"] == 2 and res["fp"] == 1
    assert res["false_redaction_rate"] == round(1 / 3, 4)

    res_none = compute_false_redaction_rate([], [])
    assert res_none["false_redaction_rate"] == 0.0


def test_false_negative_rate():
    gold = [(0, 4, "name"), (10, 14, "date")]
    pred = [(0, 4, "name")]  # missed the date
    res = compute_false_negative_rate(gold, pred)
    assert res["tp"] == 1 and res["fn"] == 1
    assert res["false_negative_rate"] == 0.5


def test_clinical_utility_survival():
    gold = "Patient diagnosed with Type 2 diabetes, prescribed Metformin."
    deid = "Patient [NAME] diagnosed with Type 2 diabetes, prescribed Metformin."
    res = compute_clinical_utility_survival(gold, deid, ["Type 2 diabetes", "Metformin"])
    assert res["survival_rate"] == 1.0

    deid_lost = "Patient [NAME] diagnosed with [CONDITION], prescribed [DRUG]."
    res2 = compute_clinical_utility_survival(gold, deid_lost, ["Type 2 diabetes", "Metformin"])
    assert res2["survival_rate"] == 0.0
