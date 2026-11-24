"""Tests for rule-based detection patterns."""
import re
import pytest

from pipelines.hybrid_pipeline_simple import apply_rules_simple


def types_of(detected):
    return {d["type"] for d in detected}


def test_phone_10_digit():
    _, detected = apply_rules_simple("Call 9848012345 for results.")
    phones = [d for d in detected if d["type"] == "phone_number"]
    assert len(phones) == 1
    assert phones[0]["text"] == "9848012345"


def test_phone_spaced():
    _, detected = apply_rules_simple("Ph 98480 12345.")
    phones = [d for d in detected if d["type"] == "phone_number"]
    assert len(phones) == 1


def test_phone_country_code():
    _, detected = apply_rules_simple("Call +91-98480-12345 now.")
    phones = [d for d in detected if d["type"] == "phone_number"]
    assert len(phones) >= 1


def test_date_formats():
    for text in ["on 01/15/2026", "on 2026-03-12", "on 14-02-2026", "on 12 March 2026", "on March 12, 2026"]:
        _, detected = apply_rules_simple(text)
        assert "date" in types_of(detected), f"no date detected in: {text}"


def test_email():
    _, detected = apply_rules_simple("Email john@clinic.edu please.")
    assert "email_address" in types_of(detected)


def test_id_number():
    _, detected = apply_rules_simple("MRN 987654321 noted.")
    assert "id_number" in types_of(detected)


def test_human_name():
    _, detected = apply_rules_simple("John Doe was admitted.")
    names = [d for d in detected if d["type"] == "human_name"]
    assert len(names) == 1
    assert names[0]["text"] == "John Doe"


def test_honorific_does_not_swallow_first_name():
    _, detected = apply_rules_simple("Patient Maria Santos was admitted.")
    names = [d for d in detected if d["type"] == "human_name"]
    assert any(n["text"] == "Maria Santos" for n in names)


def test_dr_honorific():
    _, detected = apply_rules_simple("Dr John Smith saw the patient.")
    names = [d for d in detected if d["type"] == "human_name"]
    assert any(n["text"] == "John Smith" for n in names)


def test_phone_7_digit():
    _, detected = apply_rules_simple("Phone: 555-0199 for follow-up.")
    phones = [d for d in detected if d["type"] == "phone_number"]
    assert len(phones) == 1
    assert phones[0]["text"] == "555-0199"


def test_organization_name():
    _, detected = apply_rules_simple("Admitted to Yashoda Hospital yesterday.")
    orgs = [d for d in detected if d["type"] == "company_name"]
    assert len(orgs) == 1
    assert orgs[0]["text"] == "Yashoda Hospital"


def test_org_beats_name_on_same_span():
    _, detected = apply_rules_simple("Admitted to Yashoda Hospital, Hyderabad.")
    spans = {(d["start"], d["end"]): d["type"] for d in detected}
    # The hospital span must be company_name, not human_name
    assert any(t == "company_name" for t in spans.values())
    assert not any(t == "human_name" and "Hospital" in str(s) for s, t in spans.items())


def test_city_detection():
    _, detected = apply_rules_simple("Patient from Chennai visited.")
    assert "address" in types_of(detected)


def test_age_detection():
    _, detected = apply_rules_simple("Ravi Teja, 52 years old, admitted.")
    ages = [d for d in detected if d["type"] == "age"]
    assert len(ages) == 1


def test_tamil_month_date():
    _, detected = apply_rules_simple("நோயாளி 12 மார்ச் 2026-இல் வந்தார்.")
    assert "date" in types_of(detected)


def test_tamil_name_after_marker():
    _, detected = apply_rules_simple("நோயாளி ஜான் ட Whitehead வந்தார்.")
    names = [d for d in detected if d["type"] == "human_name"]
    assert len(names) >= 1


def test_replacement_preserves_positions():
    text = "John Doe admitted on 01/15/2026, MRN 987654321."
    final, detected = apply_rules_simple(text)
    # No original PII digits should remain for detected spans
    for d in detected:
        assert d["text"] not in final or d["type"] == "human_name"
    # Detected spans must reference valid original-text slices
    for d in detected:
        assert text[d["start"]:d["end"]] == d["text"]


def test_no_false_positive_on_clean_text():
    _, detected = apply_rules_simple("The patient took 500mg of medication daily.")
    # 500mg is not a phone/id; dose numbers shouldn't be flagged as phone
    assert not any(d["type"] == "phone_number" for d in detected)
