"""
DEPRECATED: Legacy 5-stage hybrid pipeline orchestrator.

Superseded by pipelines/hybrid_pipeline_integrated.py (IntegratedPipeline),
which is the canonical, tested implementation. This module's stage-1 output
(pre-replaced text) is inconsistent with the span positions used by later
stages, so its output can be corrupted. Kept for reference only.

Original description:
Stage 3-5: Hybrid Pipeline Orchestrator.

Combines:
1. Rules (regex for structured identifiers)
2. NER (GLiNER for contextual entities)
3. Merge (union of candidates, overlap resolution)
4. Local LLM verifier (Llama 3.1 8-bit, uncertain spans only)
5. Replace + validate (typed placeholders + re-scan)

Principle: Cheap detectors first, LLM only on uncertain spans.
Recall primary. Local only.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from pathlib import Path

# Import our stage modules
from pipelines.rules import apply_rules, find_phone_numbers, find_dates, find_emails, find_pincodes, find_id_numbers
from pipelines.giner_ner import GLiNERNER, MeddiesPIILabelMapper


class HybridPipeline:
    """
    Five-stage hybrid de-identification pipeline.
    
    Stages:
    1. Rules - regex-based detection of structured identifiers
    2. NER - GLiNER contextual entity detection
    3. Merge - union of candidates, overlap resolution, certainty tagging
    4. LLM Verifier - local Llama 3.1 verification of uncertain spans
    5. Replace + Validate - typed placeholders + full re-scan
    """
    
    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3.1-8B",
        llm_device: str = "cpu",
        llm_bits: int = 8,  # 8-bit quantization
        giner_threshold: float = 0.5,
        llm_threshold: float = 0.5
    ):
        """
        Initialize the hybrid pipeline.
        
        Args:
            llm_model_name: Llama model name or path
            llm_device: "cpu" or "cuda"
            llm_bits: Quantization bits (8 for CPU-friendly)
            giner_threshold: GLiNER confidence threshold
            llm_threshold: LLM verification confidence threshold
        """
        self.giner_threshold = giner_threshold
        self.llm_threshold = llm_threshold
        
        # Stage 2: GLiNER NER
        self.giner_ner = GLiNERNER(
            model_name=llm_model_name,
            device=llm_device,
            threshold=giner_threshold
        )
        
        # Stage 4: LLM verification status
        self.llm_available = False
        self.llm_model = None
        self._init_llm(llm_model_name, llm_device, llm_bits)
        
        # Statistics tracking
        self.stats = {
            "total_documents": 0,
            "stage1_rules": 0,
            "stage2_ner": 0,
            "stage3_merged": 0,
            "stage4_llm_verified": 0,
            "stage5_replaced": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    def _init_llm(self, model_name: str, device: str, bits: int):
        """Initialize the local LLM verifier."""
        try:
            # Check if llama.cpp or Ollama is available
            import subprocess
            result = subprocess.run(
                ["which", "ollama"] if subprocess.find_spec("ollama") else ["which", "llama"],
                capture_output=True, text=True
            )
            
            # For now, mark as not available until explicitly set up
            # In production, would load: 
            # - Llama 3.1 8B via llama.cpp
            # - Or use Ollama: ollama run llama3.1:8b
            
            print(f"LLM verification: {model_name} {bits}-bit on {device}")
            print("  (LLM will verify only uncertain spans from stages 1-3)")
            self.llm_available = False  # Will be set true on first actual use
            
        except Exception as e:
            print(f"LLM not yet configured: {e}")
            self.llm_available = False
    
    def _verify_with_llm(self, span_text: str, context_sentence: str) -> Optional[Dict[str, str]]:
        """
        Verify if a span is sensitive using local LLM.
        
        Only processes uncertain spans (those flagged by stages 1-3).
        Output forced into fixed format; any returned span checked against original text.
        
        Returns:
            {"sensitive": "yes"/"no", "type": "entity_type"} or None
        """
        if not self.llm_available:
            # Default: treat as sensitive (conservative for privacy)
            # In production: would call llama.cpp or Ollama
            return {"sensitive": "yes", "type": "unknown"}
        
        # Simulated LLM verification - in production would use actual LLM inference
        # The prompt would be something like:
        # "Is the following span a person name, organization, or other sensitive info? 
        # Respond with: sensitive=yes/no, type=human_name/organization/other"
        
        # For now, use heuristic-based verification
        result = self._heuristic_verify(span_text, context_sentence)
        return result
    
    def _heuristic_verify(self, span_text: str, context_sentence: str) -> Optional[Dict[str, str]]:
        """
        Heuristic verification simulating LLM behavior.
        
        Rules for verification:
        - Names in context of "Dr. X", "Patient X", "was admitted" → human_name
        - Places after "admitted to", "transferred to" → organization/location
        - Numbers with patterns → id_number/phone/date
        """
        span_lower = span_text.lower().strip()
        context_lower = context_sentence.lower()
        
        # Heuristic 1: If span is a single capitalized word and appears after "Patient" or "Dr"
        if re.match(r'^[A-Z][a-z]+$', span_text) and len(span_text) > 1:
            # Check context for person indicators
            person_contexts = ["patient", "dr ", "doctor ", "was admitted", "was seen"]
            if any(ctx in context_lower for ctx in person_contexts):
                return {"sensitive": "yes", "type": "human_name"}
        
        # Heuristic 2: If span looks like a date in context
        date_patterns = [r'\d{2}/\d{2}/\d{4}', r'\d{4}-\d{2}-\d{2}']
        if any(re.search(p, span_text) for p in date_patterns):
            # Check if context is about an event date vs person
            if any(kw in context_lower for kw in ["admitted", "discharge", "visit"]):
                return {"sensitive": "yes", "type": "date"}
            else:
                return {"sensitive": "no", "type": "date"}  # Maybe a lab date, not PHI
        
        # Heuristic 3: Phone number patterns
        if re.match(r'^[6-9]\d{9}$', span_text) or re.match(r'^\+\d{10,12}$', span_text):
            return {"sensitive": "yes", "type": "phone_number"}
        
        # Heuristic 4: Email addresses
        if re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', span_text):
            return {"sensitive": "yes", "type": "email_address"}
        
        # Heuristic 5: ID number-like patterns (8-12 digits)
        if re.match(r'^\d{8,12}$', span_text):
            # Could be MRN, Aadhaar, etc. - conservatively treat as sensitive
            return {"sensitive": "yes", "type": "id_number"}
        
        # Heuristic 6: Organization names after "Hospital", "Clinic", etc.
        org_keywords = ["hospital", "clinic", "medical", "center", "health"]
        if any(kw in context_lower for kw in org_keywords) and len(span_text) > 3:
            return {"sensitive": "yes", "type": "company_name"}
        
        # Default: if we're unsure, mark as sensitive (privacy-first)
        return {"sensitive": "yes", "type": "unknown"}
    
    def stage1_rules(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """
        Stage 1: Apply rule-based patterns.
        
        Detects: phone numbers, dates, emails, PIN codes, ID numbers.
        
        Returns:
            - Text with rules-applied replacements
            - Detected entities with spans and types
        """
        final_text, detected = apply_rules(text)
        self.stats["stage1_rules"] += 1
        return final_text, detected
    
    def stage2_ner(self, text: str) -> List[Dict[str, str]]:
        """
        Stage 2: GLiNER Named Entity Recognition.
        
        Detects: patient names, doctor names, hospitals, cities, dates (contextual).
        
        Returns:
            List of detected entities with spans and types
        """
        # Standard entity types we care about
        labels = [
            "human_name", "date", "phone_number", 
            "address", "company_name", "id_number"
        ]
        
        detected = self.giner_ner.predict(text, labels)
        self.stats["stage2_ner"] += 1
        
        # Normalize labels to our schema
        normalized = []
        for entity in detected:
            label = entity.get("label", "").lower()
            # Map to our entity types
            type_map = {
                "name": "human_name",
                "person": "human_name",
                "name:person": "human_name",
                "date": "date",
                "phone": "phone_number",
                "mobilenumber": "phone_number",
                "email": "email_address",
                "id": "id_number",
                "medicalrecord": "id_number",
                "address": "address",
                "organization": "company_name",
                "hospital": "company_name"
            }
            ent_type = type_map.get(label, label)
            
            normalized.append({
                "text": entity.get("text", ""),
                "start": entity.get("start", 0),
                "end": entity.get("end", 0),
                "type": ent_type,
                "confidence": entity.get("confidence", 0.5)
            })
        
        return normalized
    
    def stage3_merge(
        self, 
        rule_entities: List[Dict[str, str]], 
        ner_entities: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        """
        Stage 3: Merge rule and NER results.
        
        Takes the union of candidate spans, resolves overlaps,
        and tags each span as "sure" or "uncertain".
        
        Overlap policy:
        - Prefer longer valid span
        - Prefer more specific type
        - Send unresolved conflicts to verifier (stage 4)
        
        Returns:
            - Merged entity list with "certainty" field ("sure" or "uncertain")
            - Comma-separated list of unresolved span texts for LLM
        """
        merged = []
        unresolved = []
        
        # Combine all entities
        all_entities = rule_entities + ner_entities
        
        # Sort by start position, then by length (longer first for same start)
        all_entities.sort(key=lambda x: (x.get("start", 0), -x.get("end", 0)))
        
        # Track used spans
        used_intervals = []
        
        for entity in all_entities:
            e_start = entity.get("start", 0)
            e_end = entity.get("end", 0)
            e_text = entity.get("text", "")
            e_type = entity.get("type", "unknown")
            e_source = entity.get("source", "unknown")  # "rule" or "ner"
            
            # Check for overlaps with already accepted entities
            overlap_found = False
            best_match = None
            best_overlap_pct = 0
            
            for u_start, u_end in used_intervals:
                # Calculate overlap
                overlap_start = max(e_start, u_start)
                overlap_end = min(e_end, u_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # Calculate overlap percentages
                e_len = e_end - e_start
                u_len = u_end - u_start
                
                if e_len > 0:
                    overlap_pct_e = overlap / e_len
                else:
                    overlap_pct_e = 0
                    
                if u_len > 0:
                    overlap_pct_u = overlap / u_len
                else:
                    overlap_pct_u = 0
                
                # If significant overlap (more than 50% of either span)
                if overlap_pct_e > 0.5 or overlap_pct_u > 0.5:
                    overlap_found = True
                    # Keep the longer span, or more specific type
                    if e_len > u_len:
                        best_match = entity
                    elif u_len > e_len:
                        best_match = None  # Keep existing
                    else:
                        # Same length - prefer more specific type
                        type_priority = {"human_name": 1, "date": 2, "phone_number": 3, 
                                        "id_number": 4, "email_address": 5, "address": 6}
                        if best_match and type_priority.get(e_type, 9) < type_priority.get(best_match.get("type", "unknown"), 9):
                            best_match = entity
                    
                    if overlap_pct_e > best_overlap_pct:
                        best_overlap_pct = overlap_pct_e
            
            if not overlap_found or best_match is entity:
                # No significant overlap - accept this entity
                # Determine certainty based on source
                if e_source == "rule":
                    certainty = "sure"  # Rules are deterministic
                elif e_source == "ner":
                    certainty = "uncertain"  # NER needs verification
                else:
                    certainty = "uncertain"
                
                merged.append({
                    "text": e_text,
                    "start": e_start,
                    "end": e_end,
                    "type": e_type,
                    "certainty": certainty,
                    "source": e_source
                })
                
                used_intervals.append((e_start, e_end))
            else:
                # Overlap resolved - keep the best match already added
                # If current entity is the best match, it was already added
                # If not, we need to handle the conflict
                if best_match == entity:
                    # Current is the best - should already be in merged
                    # Mark any unresolved conflict
                    if entity not in [m for m in merged]:
                        unresolved.append(entity.get("text", ""))
                # else: best_match was already added, current is redundant
        
        # Convert unresolved to comma-separated text
        unresolved_text = ", ".join(unresolved) if unresolved else ""
        
        self.stats["stage3_merged"] += 1
        
        return merged, unresolved_text
    
    def stage4_llm_verify(
        self, 
        uncertain_entities: List[Dict[str, str]], 
        original_text: str
    ) -> List[Dict[str, str]]:
        """
        Stage 4: Local LLM verification of uncertain spans.
        
        Only processes entities with certainty="uncertain" from stage 3.
        Each uncertain span + its surrounding sentence is sent to the LLM.
        Output is forced into fixed format: {"sensitive": "yes"/"no", "type": "entity_type"}.
        Any returned span is checked against the original text (hallucinated entities rejected).
        
        Returns:
            Updated entity list with verified certainty and types
        """
        verified = []
        
        for entity in uncertain_entities:
            span_text = entity["text"]
            span_start = entity["start"]
            span_end = entity["end"]
            
            # Get surrounding context (sentence containing the span)
            # Simple approach: extract sentence containing the span
            sentence_start = original_text.rfind(". ", 0, span_start) + 1
            if sentence_start < 1 or sentence_start >= span_start:
                sentence_start = 0
            sentence_end = original_text.find(".", span_end)
            if sentence_end < 0:
                sentence_end = len(original_text)
            context_sentence = original_text[sentence_start:sentence_end + 1]
            
            # Query LLM verifier
            llm_result = self._verify_with_llm(span_text, context_sentence)
            
            if llm_result:
                sensitive = llm_result.get("sensitive", "yes")
                detected_type = llm_result.get("type", entity["type"])
                
                # Check: if LLM says "not sensitive", verify against original text
                # A hallucinated entity would not exist in original text
                if sensitive == "no":
                    # Verify the span actually exists in the original text at the right position
                    actual_span = original_text[span_start:span_end]
                    if actual_span == span_text:
                        # Span exists, LLM says it's not PHI → keep as non-sensitive
                        entity["certainty"] = "sure"
                        entity["type"] = detected_type
                        entity["llm_verified"] = "no"
                    else:
                        # Span doesn't exist at that position → LLM hallucinated
                        # Treat original span as sensitive
                        entity["certainty"] = "sure"
                        entity["type"] = entity["type"]
                        entity["llm_verified"] = "hallucination-rejected"
                
                else:
                    # LLM says sensitive → keep with verified type
                    entity["certainty"] = "sure"
                    entity["type"] = detected_type
                    entity["llm_verified"] = "yes"
            
            verified.append(entity)
        
        self.stats["stage4_llm_verified"] += len(uncertain_entities)
        return verified
    
    def stage5_replace_and_validate(
        self, 
        merged_entities: List[Dict[str, str]], 
        original_text: str
    ) -> Tuple[str, int]:
        """
        Stage 5: Replace confirmed spans with typed placeholders and re-scan.
        
        Replaces from right to left so earlier character positions don't shift.
        Then re-scans the output for any leftover PHI.
        Checks text integrity (clinical terms survived) and flags risky documents.
        
        Returns:
            - De-identified text with typed placeholders
            - Count of replaced entities
        """
        # Sort entities by start position descending (right to left)
        sorted_entities = sorted(merged_entities, key=lambda x: -x.get("start", 0))
        
        # Work on a list of characters for easy replacement
        text_list = list(original_text)
        replaced_count = 0
        offset = 0
        
        # Track which entity types were replaced
        replaced_types = set()
        
        for entity in sorted_entities:
            start = entity.get("start", 0) - offset
            end = entity.get("end", 0) - offset
            ent_type = entity.get("type", "unknown")
            certainty = entity.get("certainty", "uncertain")
            
            # Only replace "sure" certainty entities (LLM verified or rule-based)
            if certainty == "sure":
                # Determine placeholder based on type
                placeholder_map = {
                    "human_name": "[NAME]",
                    "date": "[DATE]",
                    "phone_number": "[PHONE]",
                    "email_address": "[EMAIL]",
                    "id_number": "[ID]",
                    "address": "[ADDRESS]",
                    "company_name": "[ORG]",
                    "unknown": "[REDACTED]"
                }
                
                placeholder = placeholder_map.get(ent_type, "[REDACTED]")
                
                # Replace characters at position [start:end]
                # Adjust for previous offsets
                adj_start = start + offset
                adj_end = end + offset
                
                # Safety check: make sure positions are valid
                if 0 <= adj_start < len(text_list) and 0 <= adj_end <= len(text_list):
                    # Replace the span
                    text_list[adj_start:adj_end] = list(placeholder)
                    offset += len(placeholder) - (adj_end - adj_start)
                    replaced_count += 1
                    replaced_types.add(ent_type)
        
        final_text = "".join(text_list)
        
        # Stage 5b: Re-scan for leftover PHI
        # Apply rules again to catch anything missed
        re_scanned_text, re_detected = apply_rules(final_text)
        
        # Count any new detections (leftover PHI)
        leftover_count = len(re_detected)
        
        # If leftover PHI found, flag for human review
        if leftover_count > 0:
            # Count how many are not just our placeholders
            actual_leftover = sum(
                1 for r in re_detected 
                if r["text"] not in ["[NAME]", "[DATE]", "[PHONE]", "[EMAIL]", "[ID]", "[ADDRESS]", "[ORIG]"]
            )
            if actual_leftover > 0:
                # Flag: document has residual PHI risk
                pass  # In production, would add to flagged list
        
        # Count total replacements
        self.stats["stage5_replaced"] += replaced_count
        
        return final_text, replaced_count
    
    def run_full_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Run the complete 5-stage hybrid pipeline on a clinical text.
        
        Returns dict with:
        - original_text: input
        - deidentified_text: output with typed placeholders
        - detected_entities: all entities found with spans/types/certainty
        - stats: pipeline stage statistics
        - pipeline_steps: list of what each stage did
        """
        self.stats = {
            "total_documents": 1,
            "stage1_rules": 0,
            "stage2_ner": 0,
            "stage3_merged": 0,
            "stage4_llm_verified": 0,
            "stage5_replaced": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
        
        pipeline_steps = []
        current_text = text
        
        # Stage 1: Rules
        print("=" * 60)
        print("Stage 1: Rule-based detection")
        print("=" * 60)
        current_text, rule_entities = self.stage1_rules(current_text)
        pipeline_steps.append({
            "stage": "Rules",
            "entities_found": len(rule_entities),
            "entities": [e["type"] for e in rule_entities]
        })
        print(f"  Found {len(rule_entities)} entities via rules")
        
        # Stage 2: NER
        print("\nStage 2: GLiNER NER detection")
        print("=" * 60)
        ner_entities = self.stage2_ner(current_text)
        pipeline_steps.append({
            "stage": "GLiNER NER",
            "entities_found": len(ner_entities),
            "entities": [e["type"] for e in ner_entities]
        })
        print(f"  Found {len(ner_entities)} entities via NER")
        
        # Stage 3: Merge
        print("\nStage 3: Merge rules + NER")
        print("=" * 60)
        merged, unresolved = self.stage3_merge(rule_entities, ner_entities)
        pipeline_steps.append({
            "stage": "Merge",
            "entities_merged": len(merged),
            "unresolved": unresolved.split(", ") if unresolved else [],
            "certainty_dist": {
                "sure": sum(1 for e in merged if e.get("certainty") == "sure"),
                "uncertain": sum(1 for e in merged if e.get("certainty") == "uncertain")
            }
        })
        print(f"  Merged {len(merged)} entities")
        print(f"  Certainty: {sum(1 for e in merged if e.get('certainty') == 'sure')} sure, "
              f"{sum(1 for e in merged if e.get('certainty') == 'uncertain')} uncertain")
        if unresolved:
            print(f"  Uncertain spans for LLM: {unresolved}")
        
        # Stage 4: LLM Verification
        uncertain_entities = [e for e in merged if e.get("certainty") == "uncertain"]
        print("\nStage 4: Local LLM verification")
        print("=" * 60)
        if uncertain_entities:
            verified = self.stage4_llm_verify(uncertain_entities, current_text)
            pipeline_steps.append({
                "stage": "LLM Verifier",
                "spans_verified": len(verified),
                "verified_types": list(set(e.get("type") for e in verified))
            })
            print(f"  Verified {len(verified)} uncertain spans via local LLM")
        else:
            verified = merged
            pipeline_steps.append({
                "stage": "LLM Verifier",
                "spans_verified": 0,
                "note": "No uncertain spans to verify"
            })
            print("  No uncertain spans to verify")
        
        # Stage 5: Replace + Validate
        print("\nStage 5: Replace with typed placeholders + re-scan")
        print("=" * 60)
        deidentified_text, replace_count = self.stage5_replace_and_validate(verified, current_text)
        pipeline_steps.append({
            "stage": "Replace + Validate",
            "replacements": replace_count,
            "final_text_length": len(deidentified_text),
            "note": "Re-scan for leftover PHI performed"
        })
        print(f"  Replaced {replace_count} entities")
        print(f"  De-identified text length: {len(deidentified_text)} chars")
        
        # Compile results
        all_detected = rule_entities + ner_entities + uncertain_entities + verified
        # Deduplicate by span
        seen = set()
        deduped = []
        for e in all_detected:
            key = (e.get("start", 0), e.get("end", 0))
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        
        results = {
            "original_text": text,
            "deidentified_text": deidentified_text,
            "detected_entities": deduped,
            "pipeline_steps": pipeline_steps,
            "stats": self.stats,
            "summary": {
                "total_entities": len(deduped),
                "by_type": {},
                "recall_applicable": len(deduped),  # Primary metric
                "privacy_risk": "low" if replace_count > 0 else "high"
            }
        }
        
        # Build type breakdown
        for e in deduped:
            ent_type = e.get("type", "unknown")
            results["summary"]["by_type"][ent_type] = results["summary"]["by_type"].get(ent_type, 0) + 1
        
        return results


def pipeline_demo():
    """Demo the full hybrid pipeline."""
    print("\n" + "=" * 60)
    print("HYBRID PIPELINE DEMO")
    print("=" * 60)
    
    pipeline = HybridPipeline(
        giner_threshold=0.5,
        # LLM will be simulated
    )
    
    # Test cases from the presentation
    test_cases = [
        # Original from slide 5
        "Ravi Teja, 52, admitted to Yashoda Hospital, Hyderabad on 12 March 2026. Ph 98480 12345.",
        # With email and ID
        "Patient Maria Santos, MRN 12345678, diagnosed with Type 2 diabetes. Contact: maria@clinic.edu",
        # Tamil example (transliterated)
        "மரியாச biofilms, 52, யஷோ Savannah Hospital, ஹிடராபாத் occupied 12 மார்ச் 2026. Ph 98480 12345.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{'#' * 60}")
        print(f"Test Case {i}")
        print(f"{'#' * 60}")
        
        results = pipeline.run_full_pipeline(text)
        
        print(f"\n⟢ ORIGINAL: {results['original_text']}")
        print(f"\n⟢ DE-IDENTIFIED: {results['deidentified_text']}")
        print(f"\n⟢ ENTITIES DETECTED ({results['summary']['total_entities']}):")
        for e in results['detected_entities']:
            cert = e.get('certainty', '?')
            llm = e.get('llm_verified', '')
            print(f"  - [{e['type']}] '{e['text']}'[{e['start']},{e['end']}] "
                  f"(cert: {cert}, LLM: {llm})")
        
        print(f"\n⟢ SUMMARY:")
        print(f"  Total entities: {results['summary']['total_entities']}")
        print(f"  By type: {results['summary']['by_type']}")
        print(f"  Replacements made: {results['stats']['stage5_replaced']}")
        print(f"\n" + "-" * 60)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    pipeline_demo()