"""Tests for i2b2 <-> Meddies-PII label mapping."""
from src.utils.label_mapping import LabelMapper


def test_i2b2_to_meddies_family():
    assert LabelMapper.i2b2_to_meddies_label("NAME") == "human_name"
    assert LabelMapper.i2b2_to_meddies_label("DATE") == "date"
    assert LabelMapper.i2b2_to_meddies_label("ID") == "id_number"
    assert LabelMapper.i2b2_to_meddies_label("BOGUS") == "unknown"


def test_meddies_to_i2b2_label_returns_category():
    # Regression: this used to always return "ID"
    assert LabelMapper.meddies_to_i2b2_label("human_name") == "NAME"
    assert LabelMapper.meddies_to_i2b2_label("date") == "DATE"
    assert LabelMapper.meddies_to_i2b2_label("phone_number") == "CONTACT"
    assert LabelMapper.meddies_to_i2b2_label("email_address") == "CONTACT"
    assert LabelMapper.meddies_to_i2b2_label("address") == "LOCATION"
    assert LabelMapper.meddies_to_i2b2_label("company_name") == "PROFESSION"
    assert LabelMapper.meddies_to_i2b2_label("id_number") == "ID"


def test_get_mapped_entities_i2b2_to_meddies():
    entities = [{"text": "John Smith", "start": 0, "end": 10, "type": "NAME"}]
    mapped = LabelMapper.get_mapped_entities(entities, "i2b2-to-meddies")
    assert mapped[0]["type"] == "human_name"
    assert mapped[0]["source"] == "i2b2"


def test_get_mapped_entities_unmapped_keeps_type():
    entities = [{"text": "X", "start": 0, "end": 1, "type": "WEIRD"}]
    mapped = LabelMapper.get_mapped_entities(entities, "i2b2-to-meddies")
    assert mapped[0]["type"] == "WEIRD"
    assert mapped[0]["source"] == "i2b2-unmapped"


def test_get_mapped_entities_meddies_to_i2b2():
    entities = [{"text": "John", "start": 0, "end": 4, "type": "human_name"}]
    mapped = LabelMapper.get_mapped_entities(entities, "meddies-to-i2b2")
    assert mapped[0]["type"] == "NAME"
