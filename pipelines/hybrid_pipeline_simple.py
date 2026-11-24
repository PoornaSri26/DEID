"""
Simplified 5-stage hybrid de-identification pipeline.

Key simplifications over original hybrid_pipeline.py:
- Stage 3 merge: Simple deduplication instead of complex overlap resolution
- Stage 4 L
"""

import re
from typing import List, Dict, Tuple


def apply_rules_simple(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Stage 1 (simplified): Rule-based detection of structured identifiers.

    Detects: phone numbers (various formats), dates (various formats), emails, ID numbers.
    Returns text with replacements and detected entities.
    """
    detected = []

    # Phone numbers: multiple formats
    # 1. 10 digits without separator: 9848012345
    phone_pattern1 = r'(\b[6-9]\d{9}\b)'
    for m in re.finditer(phone_pattern1, text):
        detected.append({
            "text": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "type": "phone_number",
            "source": "rule"
        })

    # 2. Digits with space/hyphen separator: 98480 12345, 98480-12345 (5+5 format)
    phone_pattern2 = r'(\b[6-9]\d{4}[- ]\d{5}\b)'
    for m in re.finditer(phone_pattern2, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "phone_number",
                "source": "rule"
            })

    # 3. Phone with country code: +1-984-801-2345, +91-98480-12345, +919848012345
    phone_pattern3 = r'(\+\d{1,3}[- ]?(?:\d{5}[- ]?\d{5}|\d{3}[- ]?\d{3}[- ]?\d{4}|\d{10}))'
    for m in re.finditer(phone_pattern3, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "phone_number",
                "source": "rule"
            })

    # Date patterns
    # 1. DD/MM/YYYY: 12/03/2026
    date_pattern1 = r'(\d{2}/\d{2}/\d{4})'
    for m in re.finditer(date_pattern1, text):
        detected.append({
            "text": m.group(1),
            "start": m.start(),
            "end": m.end(),
            "type": "date",
            "source": "rule"
        })

    # 2. YYYY-MM-DD: 2026-03-12
    date_pattern2 = r'(\d{4}-\d{2}-\d{2})'
    for m in re.finditer(date_pattern2, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected if d["type"] == "date"):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "date",
                "source": "rule"
            })

    # 3. Month DD, YYYY: March 12, 2026 or 12 March 2026
    date_pattern3 = r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
    for m in re.finditer(date_pattern3, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected if d["type"] == "date"):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "date",
                "source": "rule"
            })

    # Also match "Month DD, YYYY" format: "March 12, 2026"
    date_pattern4 = r'([A-Za-z]+\s+\d{1,2},\s+\d{4})'
    for m in re.finditer(date_pattern4, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected if d["type"] == "date"):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "date",
                "source": "rule"
            })

    # Email addresses
    email_pattern = r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
    for m in re.finditer(email_pattern, text):
        if not any(m.start() >= d["start"] and m.end() <= d["end"] for d in detected):
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "email_address",
                "source": "rule"
            })

    # ID numbers: 8-12 digit sequences (likely MRN, Aadhaar, etc.)
    id_pattern = r'(\b\d{8,12}\b)'
    for m in re.finditer(id_pattern, text):
        # Filter out things already detected as dates or phones
        is_duplicate = False
        for d in detected:
            # Check significant overlap
            if m.start() >= d["start"] - 2 and m.end() <= d["end"] + 2:
                is_duplicate = True
                break
            # Check if same span
            if m.start() >= d["start"] and m.end() <= d["end"]:
                is_duplicate = True
                break
        if not is_duplicate:
            detected.append({
                "text": m.group(1),
                "start": m.start(),
                "end": m.end(),
                "type": "id_number",
                "source": "rule"
            })

    # Human names: capitalized proper nouns in clinical context
    # Pattern: two capitalized words (first/last name)
    # Handles: John Smith, Margaret R. Thompson (via NER fallback)
    name_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)'
    for m in re.finditer(name_pattern, text):
        name_text = m.group(1)
        # Filter: skip very short names, common phrases
        if len(name_text) < 3:
            continue
        # Check if already detected (by text match)
        if any(d["text"] == name_text and d["type"] == "human_name" for d in detected):
            continue
        # Filter out common non-name phrases
        name_lower = name_text.lower().split()
        if "patient" in name_lower or name_lower[0] in ["was", "been", "dr", "mr", "mrs", "ms"]:
            continue
        detected.append({
            "text": name_text,
            "start": m.start(),
            "end": m.end(),
            "type": "human_name",
            "confidence": 0.7,
            "source": "rule"
        })

    

    # Also match three-word names with middle initial: "Margaret R. Thompson"
    # Pattern: First [Middle.] Last
    mi_pattern = r'([A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+)'
    for m in re.finditer(mi_pattern, text):
        name_text = m.group(1).strip()
        # Filter: skip if any part is a common word
        name_lower = name_text.lower().split()
        if "patient" in name_lower or name_lower[0] in ["was", "been", "dr", "mr", "mrs", "ms"]:
            continue
        # Check if already detected
        if any(d["text"] == name_text and d["type"] == "human_name" for d in detected):
            continue
        # Filter: skip very short names
        if len(name_text) < 3:
            continue
        detected.append({
            "text": name_text,
            "start": m.start(),
            "end": m.end(),
            "type": "human_name",
            "confidence": 0.7,
            "source": "rule"
        })

    # Resolve overlaps across all detections: keep the earliest-starting,
    # then longest span; drop anything overlapping an already-kept entity.
    # Prevents e.g. "+91-98480-12345" and its inner "98480-12345" both
    # being replaced (which would corrupt positions).
    detected.sort(key=lambda d: (d["start"], -(d["end"] - d["start"])))
    deduped = []
    for d in detected:
        if any(d["start"] < k["end"] and d["end"] > k["start"] for k in deduped):
            continue
        deduped.append(d)
    detected = deduped

    # Apply replacements (placeholders)
    final_text = text
    # Sort by start position descending to replace right-to-left
    detected.sort(key=lambda x: -x["start"])

    # Replace entities from right to left
    # Since we process rightmost first, left positions remain valid
    for d in detected:
        placeholders = {
            "phone_number": "[PHONE]",
            "date": "[DATE]",
            "email_address": "[EMAIL]",
            "id_number": "[ID]"
        }
        placeholder = placeholders.get(d["type"], "[REDACTED]")

        start = d["start"]
        end = d["end"]

        # Replace in the current final_text
        # Positions to the left of start are unaffected by previous replacements
        if 0 <= start < len(final_text) and end <= len(final_text):
            final_text = final_text[:start] + placeholder + final_text[end:]

    return final_text, detected
