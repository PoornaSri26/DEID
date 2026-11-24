"""Integrated 5-stage hybrid de-identification pipeline with GlNER + Llama.cpp support."""

from typing import Optional, List, Dict, Any
import re
import time


class GlNERWrapper:
    """GlNER wrapper with automatic fallback."""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.available = False
        self.model = None

        # Only attempt GlNER loading if model_path is explicitly provided
        if model_path is not None:
            try:
                from gliner import GLiNER
                self.model = GLiNER(model_path)
                self.available = True
                print("[OK] GlNER model loaded successfully")
            except ImportError:
                print("[WARN] GlNER not installed - using rule-based fallback")
            except Exception as e:
                print(f"[WARN] GlNER initialization error: {e} - using rule-based fallback")
        else:
            print("[INFO] No GlNER model path provided - rule-based fallback enabled")

    def predict(self, text: str, labels: List[str]) -> List[Dict[str, str]]:
        """Run GlNER prediction or fallback to rules."""
        if not self.available or self.model is None:
            return []  # Will be filled by rule-based fallback

        try:
            # GlNER returns (text, start, end, label, score) tuples
            predictions = self.model.predict_entities(text, labels, self.threshold)
            entities = []
            for pred in predictions:
                # Format: (entity_text, start_pos, end_pos, label_score_dict)
                if isinstance(pred, tuple) and len(pred) >= 3:
                    entity_text = pred[0]
                    start_pos = pred[1]
                    end_pos = pred[2]
                    label = pred[3] if len(pred) > 3 else "unknown"
                    score = pred[4] if len(pred) > 4 else 1.0

                    entities.append({
                        "text": entity_text,
                        "start": start_pos,
                        "end": end_pos,
                        "type": label,
                        "confidence": float(score),
                        "source": "giner"
                    })
            return entities
        except Exception as e:
            print(f"[WARN] GlNER prediction failed: {e} - falling back to rules")
            return []


class LlamaCPPWrapper:
    """Llama.cpp local LLM verification wrapper."""

    def __init__(self):
        self.available = False
        self.model = None

        # Check for Llama.cpp availability
        try:
            import subprocess
            result = subprocess.run(['which', 'llama-server'], capture_output=True, text=True)
            self.available = result.returncode == 0
            if not self.available:
                try:
                    import transformers
                    self.available = True
                except ImportError:
                    pass
            print(f"[LLM] {'available' if self.available else 'not available'}")
        except Exception as e:
            print(f"[WARN] LLM availability check error: {e}")
            self.available = False

    def verify(self, span_text: str, context_sentence: str) -> Dict[str, str]:
        """Verify span using LLM or heuristic fallback."""
        if not self.available:
            return heuristic_verify(span_text, context_sentence)

        try:
            # In a real implementation, would call Llama.cpp server
            # For now, fall back to heuristics
            return heuristic_verify(span_text, context_sentence)
        except Exception:
            return heuristic_verify(span_text, context_sentence)


def heuristic_verify(span_text: str, context_sentence: str) -> Dict[str, str]:
    """Heuristic verification (used when LLM not available)."""
    span_lower = span_text.lower().strip()
    context_lower = context_sentence.lower()

    # Phone patterns (10-12 digits, optional separators/country code)
    digits = re.sub(r'\D', '', span_text)
    if re.match(r'^\+?[\d][\d\s().-]{7,}\d$', span_text) and 10 <= len(digits) <= 15:
        return {"sensitive": "yes", "type": "phone_number"}

    # Email patterns
    if re.match(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$', span_text):
        return {"sensitive": "yes", "type": "email_address"}

    # Date patterns (numeric and month-name formats)
    if (re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', span_text)
            or re.search(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', span_text)
            or re.search(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', span_text)
            or re.search(r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}', span_text)):
        return {"sensitive": "yes", "type": "date"}

    # Name-like patterns: capitalized multi-word spans (proper-noun casing)
    if re.match(r'^[A-Z][A-Za-z]*(?:[\s-][A-Za-z]+)*$', span_text) and len(span_text.split()) >= 2:
        return {"sensitive": "yes", "type": "human_name"}

    # MRN/ID patterns
    if re.match(r'^\d{6,10}$', span_text):
        return {"sensitive": "yes", "type": "id_number"}

    # Address patterns (simple check for street-like terms)
    if any(term in context_lower for term in ['street', 'avenue', 'road', 'blvd', 'drive']):
        if re.search(r'\d+\s+(?:st|ave|rd|blvd|drive)', span_text, re.IGNORECASE):
            return {"sensitive": "yes", "type": "address"}

    return {"sensitive": "no"}


class IntegratedPipeline:
    """Integrated 5-stage hybrid de-identification pipeline."""

    class Config:
        """Configuration for the integrated pipeline."""

        def __init__(self,
                     use_giner: bool = True,
                     giner_threshold: float = 0.5,
                     giner_model_path: Optional[str] = None,
                     use_llm: bool = True,
                     enable_timing: bool = True,
                     supported_languages: List[str] = None):
            self.use_giner = use_giner
            self.giner_threshold = giner_threshold
            self.giner_model_path = giner_model_path
            self.use_llm = use_llm
            self.enable_timing = enable_timing
            self.supported_languages = supported_languages or ["en", "ta"]

    def __init__(self, config: Config):
        self.config = config
        self.timing = {"enabled": config.enable_timing, "start": None, "total": 0.0}
        if config.enable_timing:
            self.timing["start"] = time.time()

        # Initialize GlNER wrapper
        self.giner = GlNERWrapper(
            model_path=config.giner_model_path if config.use_giner else None,
            threshold=config.giner_threshold
        )

        # Initialize LLM wrapper
        self.llm = LlamaCPPWrapper()

        # Custom patterns registry
        # Covers remaining Meddies-PII categories not matched by default regexes:
        #   ✅ address, company_name, date, email_address, human_name, id_number, phone_number
        #   🔴 private_url, secret (added below for full 9-label coverage)
        self.custom_patterns: Dict[str, re.Pattern] = {
            "private_url": re.compile(r"(https?://|www\.)[\S\s]+?(?:\s|[.,;!?]|$)"),
            "secret": re.compile(r'(?i)(secret|api_key|auth_token|password|credential)\s*[:=]?\s*[\S\s]{8,}'),
}

        # Entity counts for summary
        self.total_entities = 0
        self.replacements_made = 0

    def _get_ner_labels(self, language: str) -> List[str]:
        """Get NER labels for the specified language."""
        language = language.lower()
        if language == "en":
            return ["human_name", "date", "id_number", "phone_number", "email_address", "address", "company_name"]
        elif language == "ta":
            return ["human_name", "date", "id_number", "phone_number", "email_address", "address", "company_name"]
        else:
            return ["human_name", "date", "id_number", "phone_number", "email_address", "address", "company_name"]

    def _run_rules_stage(self, text: str) -> List[Dict[str, str]]:
        """Stage 1: Rule-based detection."""
        from pipelines.hybrid_pipeline_simple import apply_rules_simple
        _, rule_entities = apply_rules_simple(text)  # Returns (text, entities)

        # Run custom patterns registry for remaining Meddies-PII categories
        custom_entities = self._run_custom_patterns(text)

        # Merge rule and custom entities
        merged = []
        seen_keys = set()

        # Add rule-based entities first
        for e in rule_entities:
            key = (e.get("start", 0), e.get("end", 0), e.get("type", "unknown"))
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(e)

        # Add custom pattern entities
        for e in custom_entities:
            key = (e.get("start", 0), e.get("end", 0), e.get("type", "unknown"))
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(e)

        return merged

    def _run_custom_patterns(self, text: str) -> List[Dict[str, str]]:
        """Run custom patterns registry for remaining Meddies-PII categories."""
        custom_entities = []
        for pattern_name, pattern in self.custom_patterns.items():
            for m in pattern.finditer(text):
                custom_entities.append({
                    "text": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "type": pattern_name,
                    "confidence": 1.0,
                    "source": "custom"
                })
        return custom_entities

    def _run_ner_stage(self, text: str, language: str) -> List[Dict[str, str]]:
        """Stage 2: NER detection using GlNER or rule-based fallback."""
        # Try GlNER first - safety check with getattr
        if self.giner.available and getattr(self.giner, 'model', None):
            # Map labels for cross-lingual support
            labels = self._get_ner_labels(language)
            girner_entities = self.giner.predict(text, labels)

            if girner_entities:
                # Convert to standard format
                entities = []
                for e in girner_entities:
                    entities.append({
                        "text": e.get("text", ""),
                        "start": e.get("start", 0),
                        "end": e.get("end", 0),
                        "type": e.get("type", "unknown"),
                        "confidence": e.get("confidence", 1.0),
                        "source": "giner"
                    })
                return entities

        # Fall back to rule-based NER
        return self._run_rules_stage(text)

    def _run_merge_stage(self, rule_entities: List[Dict[str, str]], ner_entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Stage 3: Merge and deduplicate entities."""
        merged = []
        seen_keys = set()

        # Add rule-based entities first
        for e in rule_entities:
            key = (e.get("start", 0), e.get("end", 0), e.get("type", "unknown"))
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(e)

        # Add NER entities, skipping overlaps
        for e in ner_entities:
            overlap = False
            for existing in merged:
                # Check overlap
                e_start, e_end = e.get("start", 0), e.get("end", 0)
                x_start, x_end = existing.get("start", 0), existing.get("end", 0)
                if e_start <= x_end and e_end >= x_start:
                    overlap = True
                    # Keep the one with higher confidence
                    if e.get("confidence", 0) > existing.get("confidence", 0):
                        merged.remove(existing)
                        merged.append(e)
                    break
            if not overlap:
                key = (e.get("start", 0), e.get("end", 0), e.get("type", "unknown"))
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(e)

        return merged

    def _run_verification_stage(self, entities: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
        """Stage 4: LLM verification of entities."""
        verified = []

        for entity in entities:
            span = entity.get("text", "")
            entity_type = entity.get("type", "unknown")

            # Get context around the entity
            span_start = entity.get("start", 0)
            span_end = entity.get("end", 0)
            context_start = max(0, span_start - 50)
            context_end = min(len(text), span_end + 50)
            context = text[context_start:context_end]

            # Use LLM verification if available, otherwise heuristic
            if self.llm and self.llm.available:
                try:
                    result = self.llm.verify(span, context)
                except Exception:
                    result = heuristic_verify(span, context)
            else:
                result = heuristic_verify(span, context)

            # heuristic_verify returns {"sensitive": "yes"|"no"}. Rule- and
            # custom-pattern entities are already regex-verified, so they are
            # kept regardless; verification mainly filters NER spans.
            is_sensitive = (
                result.get("sensitive") == "yes"
                or entity.get("source") in ("rule", "custom")
            )
            if is_sensitive:
                verified.append({
                    "text": span,
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "type": entity_type,
                    "confidence": entity.get("confidence", 1.0),
                    "source": entity.get("source", "rules")
                })

        return verified

    def _run_replace_validate_stage(self, verified_entities: List[Dict[str, str]], text: str) -> Dict[str, Any]:
        """Stage 5: Replace PII with typed placeholders and validate output."""
        deidentified = text
        replacements = 0

        # Sort entities by position (reverse order for safe replacement)
        sorted_entities = sorted(verified_entities, key=lambda e: e.get("start", 0), reverse=True)

        for entity in sorted_entities:
            entity_type = entity.get("type", "unknown")
            span_start = entity.get("start", 0)
            span_end = entity.get("end", 0)
            span_text = entity.get("text", "")

            # Generate replacement placeholder based on type
            replacement = self._get_placeholder(entity_type)

            # Replace in the text (reverse order prevents position shifts)
            deidentified = deidentified[:span_start] + replacement + deidentified[span_end:]
            replacements += 1

        # Build summary
        summary = {
            "total_entities": len(verified_entities),
            "replacements_made": replacements,
            "by_type": {}
        }

        # Count by type
        for entity in verified_entities:
            etype = entity.get("type", "unknown")
            summary["by_type"][etype] = summary["by_type"].get(etype, 0) + 1

        return {
            "deidentified_text": deidentified,
            "summary": summary
        }

    def _get_placeholder(self, entity_type: str) -> str:
        """Generate a typed placeholder for the entity type."""
        mapping = {
            "human_name": "[REDACTED_NAME]",
            "date": "[REDACTED_DATE]",
            "id_number": "[REDACTED_ID]",
            "phone_number": "[REDACTED_PHONE]",
            "email_address": "[REDACTED_EMAIL]",
            "address": "[REDACTED_ADDRESS]",
            "company_name": "[REDACTED_COMPANY]",
            "age": "[REDACTED_AGE]",
        }
        return mapping.get(entity_type, "[REDACTED]")

    def run_full_pipeline(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Run the complete 5-stage hybrid pipeline."""
        if self.timing and self.timing["enabled"]:
            self.timing["start"] = time.time()

        start_time = time.time()

        # Stage 1: Custom patterns + Rule-based detection
        rule_entities = self._run_rules_stage(text)

        # Stage 2: NER detection
        current_text = text
        ner_entities = self._run_ner_stage(current_text, language)

        # Stage 3: Merge rules + NER
        merged_entities = self._run_merge_stage(rule_entities, ner_entities)

        # Stage 4: Verification
        verified_entities = self._run_verification_stage(merged_entities, current_text)

        # Stage 5: Replace and validate
        result = self._run_replace_validate_stage(verified_entities, current_text)

        total_time = time.time() - start_time
        result["timing_total"] = total_time
        result["timing_breakdown"] = self._get_timing_breakdown() if self.timing else {}

        return result

    def _get_timing_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown (placeholder)."""
        return {"total": 0, "stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0, "stage5": 0}

    def generate_timing_report(self) -> str:
        """Generate a human-readable timing report."""
        if not self.timing or self.timing.get("start") is None:
            return "Timing not enabled"

        total_time = time.time() - self.timing["start"]
        return f"Pipeline execution time: {total_time:.2f}s"