"""
Label mapping between i2b2/n2c2 and Meddies-PII entity schemas.

i2b2 categories (HIPAA-style):
- NAME
- PROFESSION
- LOCATION
- AGE
- DATE
- CONTACT (email, phone)
- ID (MRN, etc.)

Meddies-PII labels (9 fixed):
- address
- company_name
- date
- email_address
- human_name
- id_number
- phone_number
- private_url
- secret

Seven entity families (stable across languages):
- human_name
- date
- id_number
- phone_number
- email_address
- address
- company_name
"""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class LabelMapper:
    """Mapper between different NER label schemas."""
    
    # i2b2 7 categories (commonly used)
    I2B2_CATEGORIES = [
        "NAME",
        "PROFESSION", 
        "LOCATION",
        "AGE",
        "DATE",
        "CONTACT",
        "ID"
    ]
    
    # Meddies-PII 9 labels
    MEDDIES_LABELS = [
        "address",
        "company_name",
        "date",
        "email_address",
        "human_name",
        "id_number",
        "phone_number",
        "private_url",
        "secret"
    ]
    
    # Seven stable entity families
    FAMILY_LABELS = [
        "human_name",
        "date",
        "id_number",
        "phone_number",
        "email_address",
        "address",
        "company_name"
    ]
    
    # Direct mapping: i2b2 → Meddies family
    I2B2_TO_FAMILY = {
        "NAME": "human_name",
        "PROFESSION": "company_name",
        "LOCATION": "address",
        "AGE": "date",  # AGE often maps to date patterns
        "DATE": "date",
        "CONTACT": "phone_number",  # Simplification; CONTACT has email+phone
        "ID": "id_number"
    }
    
    # Reverse: Meddies family → i2b2
    FAMILY_TO_I2B2 = {
        "human_name": "NAME",
        "company_name": "PROFESSION",
        "address": "LOCATION",
        "date": "DATE",
        "phone_number": "CONTACT",
        "id_number": "ID",
        "email_address": "CONTACT"
    }
    
    # Meddies-specific labels not in i2b2 direct mapping
    MEDDIES_EXTRA = {
        "private_url": "contact",  # Map to CONTACT category
        "secret": "id_number"  # Map to ID category
    }
    
    @classmethod
    def i2b2_to_meddies_label(cls, i2b2_cat: str) -> str:
        """Convert i2b2 category to Meddies-PII label (family)."""
        return cls.I2B2_TO_FAMILY.get(i2b2_cat, "unknown")
    
    @classmethod
    def meddies_to_i2b2_label(cls, meddies_label: str) -> str:
        """Convert Meddies-PII label to i2b2 category."""
        # First check direct family mapping
        family = cls.meddies_to_i2b2_family(meddies_label)
        return cls.FAMILY_TO_I2B2.get(family, "ID")  # Default
    
    @classmethod
    def meddies_to_i2b2_family(cls, meddies_label: str) -> str:
        """Convert Meddies-PII label to i2b2 category name."""
        # Direct mapping for the 7 family labels
        family_map = {
            "human_name": "NAME",
            "date": "DATE",
            "id_number": "ID",
            "phone_number": "CONTACT",
            "email_address": "CONTACT",
            "address": "LOCATION",
            "company_name": "PROFESSION"
        }
        return family_map.get(meddies_label, "ID")
    
    @classmethod
    def get_mapped_entities(
        i2b2_entities: List[Dict[str, Any]],
        mapping: str = "i2b2-to-meddies"
    ) -> List[Dict[str, Any]]:
        """
        Map i2b2 entities to Meddies-PII label format.
        
        Each entity dict should have: {"text": ..., "start": ..., "end": ..., "type": ...}
        
        Args:
            i2b2_entities: List of i2b2-style entity dicts
            mapping: "i2b2-to-meddies" or "meddies-to-i2b2"
        
        Returns:
            List of entities in target schema
        """
        mapped = []
        
        for entity in i2b2_entities:
            ent_type = entity.get("type", "").upper()
            ent_text = entity.get("text", "")
            ent_start = entity.get("start", 0)
            ent_end = entity.get("end", 0)
            
            if mapping == "i2b2-to-meddies":
                # Map i2b2 category to Meddies family
                meddies_label = cls.i2b2_to_meddies_label(ent_type)
                if meddies_label != "unknown":
                    mapped.append({
                        "text": ent_text,
                        "start": ent_start,
                        "end": ent_end,
                        "type": meddies_label,
                        "source": "i2b2"
                    })
                else:
                    # Unknown i2b2 category - keep as-is with note
                    mapped.append({
                        "text": ent_text,
                        "start": ent_start,
                        "end": ent_end,
                        "type": ent_type,
                        "source": "i2b2-unmapped",
                        "note": f"i2b2 category '{ent_type}' not directly mapped"
                    })
            
            elif mapping == "meddies-to-i2b2":
                # Map Meddies label to i2b2 category
                i2b2_cat = cls.meddies_to_i2b2_family(ent_type)
                mapped.append({
                    "text": ent_text,
                    "start": ent_start,
                    "end": ent_end,
                    "type": i2b2_cat,
                    "source": "meddies"
                })
        
        return mapped
    
    @classmethod
    def compute_label_coverage(
        gold_entities: List[Dict[str, Any]],
        source_schema: str = "i2b2",
        target_schema: str = "meddies"
    ) -> Dict[str, Any]:
        """
        Compute how well the label schemas cover each other.
        
        Useful for reporting: "Which i2b2 categories could not be mapped to Meddies?"
        """
        i2b2_cats_found = set()
        meddies_labels_found = set()
        unmapped_i2b2 = []
        unmapped_meddies = []
        
        # Collect what's in the gold data
        for entity in gold_entities:
            src_type = entity.get("type", "").upper()
            
            if source_schema == "i2b2":
                i2b2_cats_found.add(src_type)
                # Try to map to Meddies
                meddies_label = cls.i2b2_to_meddies_label(src_type)
                if meddies_label != "unknown":
                    meddies_labels_found.add(meddies_label)
                else:
                    unmapped_i2b2.append(src_type)
            
            elif source_schema == "meddies":
                meddies_labels_found.add(src_type)
                # Try to map to i2b2
                i2b2_cat = cls.meddies_to_i2b2_family(src_type)
                if i2b2_cat != "ID":  # ID is default/fallback
                    # Actually always add
                    pass
                # Check if i2b2 category exists
                # We'll just track what was found
        
        # All possible i2b2 categories
        all_i2b2 = set(cls.I2B2_CATEGORIES)
        all_meddies = set(cls.MEDDIES_LABELS)
        
        # Coverage analysis
        coverage = {
            "i2b2_categories_found": len(i2b2_cats_found),
            "i2b2_categories_total": len(all_i2b2),
            "i2b2_coverage_pct": round(len(i2b2_cats_found) / len(all_i2b2) * 100, 1) if all_i2b2 else 0,
            "meddies_labels_found": len(meddies_labels_found),
            "meddies_labels_total": len(all_meddies),
            "meddies_coverage_pct": round(len(meddies_labels_found) / len(all_meddies) * 100, 1) if all_meddies else 0,
            "unmapped_i2b2_categories": list(set(unmapped_i2b2)),
            "mapping_summary": {
                "i2b2_to_meddies_covered": [
                    cat for cat in all_i2b2 
                    if cls.i2b2_to_meddies_label(cat) != "unknown"
                ],
                "meddies_to_i2b2_covered": [
                    label for label in all_meddies 
                    if cls.meddies_to_i2b2_family(label) in cls.FAMILY_TO_I2B2
                ]
            }
        }
        
        return coverage
    
    @classmethod
    def example_mapping_data():
        """Example entity mapping demonstration."""
        print("=" * 60)
        print("LABEL MAPPING EXAMPLES")
        print("=" * 60)
        
        # Example i2b2 entities
        i2b2_entities = [
            {"text": "John Smith", "start": 0, "end": 10, "type": "NAME"},
            {"text": "Dr. Yashoda Hospital", "start": 20, "end": 38, "type": "LOCATION"},
            {"text": "12 March 2026", "start": 40, "end": 52, "type": "DATE"},
            {"text": "98480 12345", "start": 60, "end": 68, "type": "CONTACT"},
            {"text": "MRN 12345678", "start": 70, "end": 79, "type": "ID"},
        ]
        
        print("\n1. i2b2 → Meddies-PII mapping:")
        mapped = cls.get_mapped_entities(i2b2_entities, "i2b2-to-meddies")
        for e in mapped:
            print(f"  i2b2 '{e['text']}' → Meddies '{e['type']}'")
        
        print("\n2. Coverage analysis:")
        # Create synthetic gold dataset
        gold = [
            {"text": "John Smith", "type": "NAME"},
            {"text": "Yashoda Hospital", "type": "LOCATION"},
            {"text": "12 March 2026", "type": "DATE"},
            {"text": "98480 12345", "type": "CONTACT"},
            {"text": "12345678", "type": "ID"},
        ]
        
        coverage = cls.compute_label_coverage(gold, "i2b2", "meddies")
        print(f"  i2b2 coverage of Meddies: {coverage['i2b2_coverage_pct']}%")
        print(f"  Meddies coverage of i2b2: {coverage['meddies_coverage_pct']}%")
        if coverage['unmapped_i2b2_categories']:
            print(f"  Unmapped i2b2 categories: {coverage['unmapped_i2b2_categories']}")
        
        print("\n3. Meddies → i2b2 mapping:")
        meddies_entities = [
            {"text": "John Smith", "start": 0, "end": 10, "type": "human_name"},
            {"text": "Yashoda Hospital", "start": 20, "end": 38, "type": "company_name"},
            {"text": "12 March 2026", "start": 40, "end": 52, "type": "date"},
            {"text": "98480 12345", "start": 60, "end": 68, "type": "phone_number"},
        ]
        mapped_back = cls.get_mapped_entities(meddies_entities, "meddies-to-i2b2")
        for e in mapped_back:
            print(f"  Meddies '{e['text']}' → i2b2 '{e['type']}'")
        
        print("\n" + "=" * 60)


# Run example when module is imported
if __name__ == "__main__":
    LabelMapper.example_mapping_data()