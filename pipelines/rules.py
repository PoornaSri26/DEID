"""
Stage 1: Rules-based detection.

Regular expressions for predictable formats:
- Phone numbers (with optional +91, 10 digits starting 6-9)
- 6-digit PIN codes
- Email addresses
- Date formats (MM/DD/YYYY, YYYY-MM-DD, etc.)
- ID number patterns (general, not specific Aadhaar/PAN)
"""

import re
from typing import Dict, List, Tuple, Optional, Set

# --- Phone number patterns ---
# Matches: +91 98480 12345, 9848012345, 09848012345
# Optional country code +91, then 10 digits starting with 6-9
PHONE_PATTERN = re.compile(
    r'(?:\+?91[-\s]?)?[6-9]\d{9}'
    r'|(?:\d{3}[-\s]?)?[6-9]\d{2}[-\s]?\d{4}'
    r'|\d{5}[-\s]\d{6}'
)

# --- Date patterns ---
# MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, Month DD, YYYY
DATE_PATTERNS = [
    re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),           # MM/DD/YYYY
    re.compile(r'\b\d{2}-\d{2}-\d{4}\b'),           # DD-MM-YYYY
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),           # YYYY-MM-DD (ISO)
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b', re.IGNORECASE),  # Month DD, YYYY
]

# --- Email pattern ---
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# --- 6-digit PIN code ---
PIN_PATTERN = re.compile(r'\b\d{6}\b')

# --- ID number general pattern ---
# Not specific to any country; matches sequences that look like ID numbers
ID_PATTERN = re.compile(r'\b\d{8,12}\b')

# --- Company/organization name patterns (for later use) ---
COMPANY_PATTERN = re.compile(r'\b(?:Hospital|Clinic|Medical|Center|Health)[\s]?[A-Za-z0-9\s]*\b', re.IGNORECASE)


def find_phone_numbers(text: str) -> List[Dict[str, str]]:
    """Find phone numbers in text using regex."""
    matches = []
    for m in PHONE_PATTERN.finditer(text):
        matches.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
            "type": "phone_number"
        })
    return matches


def find_dates(text: str) -> List[Dict[str, str]]:
    """Find date patterns in text."""
    matches = []
    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            # Avoid duplicate matches at same position
            if not any(existing["start"] == m.start() and existing["end"] == m.end() 
                       for existing in matches):
                matches.append({
                    "text": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                    "type": "date"
                })
    return matches


def find_emails(text: str) -> List[Dict[str, str]]:
    """Find email addresses in text."""
    matches = []
    for m in EMAIL_PATTERN.finditer(text):
        matches.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
            "type": "email_address"
        })
    return matches


def find_pincodes(text: str) -> List[Dict[str, str]]:
    """Find 6-digit PIN codes in text."""
    matches = []
    for m in PIN_PATTERN.finditer(text):
        matches.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
            "type": "id_number"
        })
    return matches


def find_id_numbers(text: str) -> List[Dict[str, str]]:
    """Find potential ID numbers (general pattern)."""
    matches = []
    for m in ID_PATTERN.finditer(text):
        # Filter out obvious non-ID numbers (very short, very long, etc.)
        if 8 <= len(m.group()) <= 12:
            matches.append({
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "type": "id_number"
            })
    return matches


def apply_rules(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Apply all rule-based patterns to text.
    
    Returns:
        - Modified text with identified PHI replaced by placeholders
        - List of detected entities with spans and types
    """
    detected = []
    # Work from right to left to preserve character positions
    replacements = []
    
    # Find and tag all matches
    phone_matches = find_phone_numbers(text)
    date_matches = find_dates(text)
    email_matches = find_emails(text)
    pincode_matches = find_pincodes(text)
    id_matches = find_id_numbers(text)
    
    all_matches = phone_matches + date_matches + email_matches + pincode_matches + id_matches
    
    # Sort by start position, then by length (longer first for same start)
    all_matches.sort(key=lambda x: (x["start"], -x["end"]))
    
    # Check for overlaps and merge
    seen_ranges = []
    for match in all_matches:
        start, end = match["start"], match["end"]
        # Check overlap with already seen ranges
        overlapped = False
        for seen_start, seen_end in seen_ranges:
            if start < seen_end and end > seen_start:
                overlapped = True
                # Keep the longer span
                if (end - start) > (seen_end - seen_start):
                    # Replace the older one
                    seen_ranges = [(s, e) for s, e in seen_ranges 
                                   if not (start <= s and end >= e)]
                    seen_ranges.append((start, end))
                else:
                    seen_ranges.append((start, end))
                overlapped = True
                break
        
        if not overlapped:
            seen_ranges.append((start, end))
    
    # Build replacement map (right to left to preserve positions)
    # Sort by start position descending
    seen_ranges.sort(key=lambda x: -x[0])
    
    # Replace in original text from right to left
    text_list = list(text)
    offset = 0
    
    for start, end in seen_ranges:
        # Adjust for previous replacements
        adj_start = start + offset
        adj_end = end + offset
        
        # Determine placeholder type based on content
        original_text = text[start:end]
        if re.match(PHONE_PATTERN, original_text):
            placeholder = "[PHONE]"
        elif any(pattern.search(original_text) for pattern in DATE_PATTERNS):
            placeholder = "[DATE]"
        elif EMAIL_PATTERN.match(original_text):
            placeholder = "[EMAIL]"
        elif PIN_PATTERN.match(original_text):
            placeholder = "[PIN]"
        else:
            placeholder = "[ID]"
        
        # Replace
        text_list[adj_start:adj_end] = list(placeholder)
        offset += len(placeholder) - (adj_end - adj_start)
    
    final_text = "".join(text_list)
    
    # Convert ranges back to detected entities
    detected = []
    for start, end in sorted(seen_ranges, key=lambda x: x[0]):
        # Find matching entity type
        original = text[start - offset:end - offset] if offset else text[start:end]
        ent_type = "unknown"
        if re.match(PHONE_PATTERN, original):
            ent_type = "phone_number"
        elif any(pattern.search(original) for pattern in DATE_PATTERNS):
            ent_type = "date"
        elif EMAIL_PATTERN.match(original):
            ent_type = "email_address"
        elif PIN_PATTERN.match(original):
            ent_type = "id_number"
        else:
            ent_type = "id_number"  # default
        
        detected.append({
            "text": original,
            "start": start - offset,
            "end": end - offset,
            "type": ent_type
        })
    
    return final_text, detected


def rule_demo():
    """Demo the rules pipeline."""
    test_cases = [
        "Ravi Teja, 52, admitted to Yashoda Hospital, Hyderabad on 12 March 2026. Ph 98480 12345.",
        "Patient visited on 03/15/2026, email: patient@test.com, MRN: 12345678",
        "Call back at +91-98480-12345 or 09848012345 for results",
    ]
    
    for i, text in enumerate(test_cases, 1):
        final, detected = apply_rules(text)
        print(f"\n--- Test Case {i} ---")
        print(f"Original:  {text}")
        print(f"De-identified: {final}")
        print(f"Detected entities ({len(detected)}):")
        for d in detected:
            print(f"  - {d['type']}: '{d['text']}' at [{d['start']},{d['end']}]")


if __name__ == "__main__":
    rule_demo()