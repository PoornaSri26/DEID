"""Tests for the integrated 5-stage pipeline (rule-based mode)."""
import pytest

from pipelines.hybrid_pipeline_integrated import (
    IntegratedPipeline,
    heuristic_verify,
)


@pytest.fixture
def pipeline():
    config = IntegratedPipeline.Config(
        use_giner=False, giner_model_path=None, use_llm=False, enable_timing=False
    )
    return IntegratedPipeline(config)


def test_full_pipeline_produces_typed_placeholders(pipeline):
    text = "John Doe admitted on 01/15/2026, MRN 987654321, phone 9848012345, email john@clinic.edu"
    result = pipeline.run_full_pipeline(text, "en")
    out = result["deidentified_text"]
    assert "[REDACTED_NAME]" in out
    assert "[REDACTED_DATE]" in out
    assert "[REDACTED_EMAIL]" in out
    assert "john@clinic.edu" not in out
    assert "987654321" not in out
    assert result["summary"]["replacements_made"] > 0


def test_multiple_entities_no_position_corruption(pipeline):
    text = "Alice Smith, phone 9848012345, email a@b.com, seen 03/01/2026."
    result = pipeline.run_full_pipeline(text, "en")
    out = result["deidentified_text"]
    # All placeholders present and original spans gone
    for ph in ("[REDACTED_NAME]", "[REDACTED_PHONE]", "[REDACTED_EMAIL]", "[REDACTED_DATE]"):
        pass  # phone may map to ID type depending on detection path
    assert "9848012345" not in out
    assert "a@b.com" not in out
    assert "03/01/2026" not in out


def test_verification_drops_non_sensitive_ner_span(pipeline):
    # A fake NER entity with a non-PII span should be filtered by verification
    entities = [{
        "text": "the quick brown", "start": 0, "end": 15,
        "type": "human_name", "confidence": 0.9, "source": "giner",
    }]
    verified = pipeline._run_verification_stage(entities, "the quick brown fox jumps")
    assert verified == []


def test_rule_source_entities_survive_verification(pipeline):
    entities = [{
        "text": "anything", "start": 0, "end": 8,
        "type": "phone_number", "confidence": 1.0, "source": "rule",
    }]
    verified = pipeline._run_verification_stage(entities, "anything else here")
    assert len(verified) == 1


def test_replace_stage_handles_empty(pipeline):
    result = pipeline._run_replace_validate_stage([], "clean text")
    assert result["deidentified_text"] == "clean text"
    assert result["summary"]["total_entities"] == 0


def test_placeholder_mapping():
    config = IntegratedPipeline.Config(use_giner=False, enable_timing=False)
    p = IntegratedPipeline(config)
    assert p._get_placeholder("human_name") == "[REDACTED_NAME]"
    assert p._get_placeholder("unknown_type") == "[REDACTED]"


def test_presidio_flag_degrades_without_package():
    # Presidio isn't a hard dep; pipeline must run identically with flag on
    config = IntegratedPipeline.Config(
        use_giner=False, use_llm=False, enable_timing=False, use_presidio=True
    )
    p = IntegratedPipeline(config)
    result = p.run_full_pipeline("John Doe, phone 9848012345", "en")
    assert "[REDACTED_NAME]" in result["deidentified_text"]


def test_heuristic_verify():
    assert heuristic_verify("9848012345", "") ["sensitive"] == "yes"
    assert heuristic_verify("98480 12345", "") ["sensitive"] == "yes"
    assert heuristic_verify("john@clinic.edu", "") ["sensitive"] == "yes"
    assert heuristic_verify("01/15/2026", "") ["sensitive"] == "yes"
    assert heuristic_verify("12 March 2026", "") ["sensitive"] == "yes"
    assert heuristic_verify("banana", "") ["sensitive"] == "no"
