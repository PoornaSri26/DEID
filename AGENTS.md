# AGENTS.md - Agent Documentation for Codebuff/Buffy

This file documents the agent workflows, dependencies, and patterns for the i2b2 DUA submission tracking and hybrid de-identification pipeline project.

## Project Overview

**Project**: i2b2 2014 DUA Submission Tracking & Hybrid De-identification Pipeline
**Primary Artifact**: `i2b2 DUA Submission Tracking` (artifact ID: 01a2070b-0412-7d61-9007-febd4b911d86)
**Main Pipeline**: `deid_project/pipelines/hybrid_pipeline_integrated.py`
**Datasets**: Meddies-PII v2 (primary, synthetic), i2b2 2014 NLP Challenge (secondary, DUA required)

## Agent Workflows

### Core Agent Tasks

| Task | Description | Related Files |
|------|-------------|---------------|
| **Review and fix syntax issues** | Inspect pipeline code for syntax errors, import issues, type mismatches | `hybrid_pipeline_integrated.py` |
| **Simplify GlNER fallback logic** | Remove unsafe GLiNER() no-arg calls; make giner conditional on use_giner flag | `hybrid_pipeline_integrated.py` GlNERWrapper |
| **Verify Config class handles all parameters** | Ensure use_giner, giner_threshold, giner_model_path, use_llm, enable_timing, supported_languages are properly managed | `hybrid_pipeline_integrated.py` Config class |
| **Ensure graceful degradation when models unavailable** | Pipeline must fall back to rule-based mode with valid [REDACTED_*] output | `hybrid_pipeline_integrated.py` full pipeline |
| **Run full Meddies-PII evaluation** | Evaluate on English + Tamil configs; report F1 baselines | `src/evaluation/` `tamil_vs_english_f1_comparison.md` |
| **Consider GlNER model download alternatives** | HF download blocked (404/symlink); pipeline degrades gracefully | GlNER integration path documented |
| **Focus on rule-based improvements** | Raise English F1 from 0.667 toward 0.75+ without GlNER | Rule-based pattern enhancements |
| **Prepare presentation with documented results** | IIIT Dharwad presentation; F1 baselines; cross-lingual comparison | `ablation_results_summary.md` |
| **Submit i2b2 DUA application** | Harvard portal `portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/`; per-person DUA required | DUA tracking artifact |
| **Adapt pipeline labels for i2b2** | Map i2b2 ↔ Meddies-PII label categories when DUA approved | `label_mapping.py` |

### Code Review Patterns

- **Always check** `gliner` import availability; pipeline falls back gracefully when unavailable
- **Verify** `Config` class parameters: `use_giner`, `giner_threshold`, `giner_model_path`, `use_llm`, `enable_timing`, `supported_languages`
- **Test** both `use_giner=True` (GlNER mode) and `use_giner=False` (rule-only mode) produce valid output
- **Confirm** offset-aware replacement in `_run_replace_validate_stage` prevents position corruption
- **Ensure** verified entities promote to certainty `"sure"` for replacement in stage 5
- **Validate** 5-stage pipeline: rules → NER → merge → verify → replace+validate

### Debug Commands

```bash
# Test pipeline with GlNER fallback
python -c "from pipelines.hybrid_pipeline_integrated import IntegratedPipeline; config = IntegratedPipeline.Config(use_giner=True, giner_model_path=None); pipeline = IntegratedPipeline(config); results = pipeline.run_full_pipeline('John Doe admitted on 01/15/2026, MRN 987654321, phone 5551234567, email john@clinic.edu', 'en'); print(results['deidentified_text'])"

# Test rule-only mode
python -c "from pipelines.hybrid_pipeline_integrated import IntegratedPipeline; config = IntegratedPipeline.Config(use_giner=False, giner_model_path=None); pipeline = IntegratedPipeline(config); results = pipeline.run_full_pipeline('John Doe admitted on 01/15/2026, MRN 987654321, phone 5551234567, email john@clinic.edu', 'en'); print(results['deidentified_text'])"

# Check GlNERWrapper constructor
python -c "from pipelines.hybrid_pipeline_integrated import GlNERWrapper; w = GlNERWrapper(model_path=None, threshold=0.5); print(f'available={w.available}, model={w.model}')"
```

## Dependencies

### Optional Packages

| Package | Purpose | Required |
|---------|---------|----------|
| `gliner` | GlNER NER model (HF) | Optional; pipeline falls back gracefully |
| `transformers` | Llama.cpp support | Optional; heuristic verification used |
| `@freebuff/auth` | Sign in with Freebuff | Optional; development loopback origin |
| `freebuff-sites` | Cloudflare Sites deployment | Optional; when publishing |
| `@opennextjs/cloudflare` | Next.js adapter | Optional; if Next.js frontend |

### Project Dependencies (from requirements.txt)

```
# Core pipeline dependencies
typing_extensions
regex

# Evaluation
# Meddies-PII, F1 computation, etc.

# Deployment
# freebuff-sites, @freebuff/auth
```

**Note**: `gliner` import is handled with try/except; no hard dependency. Pipeline works fully without it.

## Agent Configuration

### This Project (Codebuff/Buffy)

- **AI coding agent**: Assists with pipeline development, GlNER integration, DUA tracking, F1 evaluation
- **GlNER integration research**: Fallback logic, model loading, HF token management
- **DUA tracking and i2b2 portal coordination**: Harvard portal status, per-person DUA applications
- **F1 evaluation and cross-lingual comparison**: English vs Tamil; rule-based vs GlNER results

### Agent Dos and Don'ts

**Do**:
- Check `gliner` availability before assuming model is loaded
- Verify `Config` parameters are set correctly before pipeline runs
- Test pipeline with both `use_giner=True` and `use_giner=False`
- Document F1 baselines after each evaluation run
- Report graceful degradation when models unavailable
- Keep secrets (API keys, tokens) out of project files and logs

**Don't**:
- Assume `gliner` is installed; always use try/except on import
- Skip the offset tracking in `_run_replace_validate_stage` (causes position corruption)
- Forget to promote verified entities to certainty `"sure"` (they won't be replaced)
- Claim F1 improvements without running evaluation
- Weaken assertions or add type suppressions just to make verification pass
- Expose platform credentials or session tokens in chat/logs/assets

## Key Functions Documented

### `GlNERWrapper.__init__(model_path, threshold)`

- `model_path`: Optional path to HF GlNER model; `None` triggers rule-based fallback
- `threshold`: Float confidence threshold (default 0.5)
- Falls back to rules if model_path is None or HF download fails
- Prints `[INFO] No GlNER model path provided - rule-based fallback enabled` when no model

### `GlNERWrapper.predict(text, labels)`

- Returns list of entity dicts: `{"text", "start", "end", "type", "confidence", "source"}`
- Returns empty list `[]` when GlNER unavailable (rules fill via `_run_rules_stage`)
- Safety check: `if self.giner.available and getattr(self.giner, 'model', None)`

### `IntegratedPipeline._run_replace_validate_stage(verified_entities, text)`

- **Critical**: Uses offset tracking to prevent position corruption with multiple replacements
- Processes entities reverse-order (right-to-left)
- Updates cumulative offset: `offset += len(replacement) - (adj_end - adj_start)`
- All verified entities must have `certainty: "sure"` to be replaced

### `heuristic_verify(span_text, context_sentence)`

- Rule-based verification when LLM not available
- Covers: phone, email, date, name, ID, address patterns
- Returns `{"sensitive": "yes"/"no", "type": "category"}` or `{"sensitive": "no"}`

### `IntegratedPipeline.Config`

- `use_giner`: bool = True (enable GlNER)
- `giner_threshold`: float = 0.5
- `giner_model_path`: Optional[str] = None (HF path; None = fallback)
- `use_llm`: bool = True (enable LLM verification)
- `enable_timing`: bool = True (execution timing)
- `supported_languages`: List[str] = ["en", "ta"]

## Testing Conventions

### F1 Baselines (5-sample rule-based evaluation)

- **English Micro F1**: 0.667
- **Tamil Micro F1**: 0.833
- **Cross-lingual**: Tamil > English by +25% for Micro F1

### Test Checklist

- [x] Pipeline compiles and runs without errors
- [x] Valid de-identified output with typed placeholders ([REDACTED_NAME], [REDACTED_DATE], etc.)
- [x] Multiple entity replacement (offset bug fixed)
- [x] Graceful model unavailability fallback
- [x] Cross-lingual (en/ta) support
- [ ] Full F1 evaluation on real i2b2 notes (pending DUA)
- [ ] GlNER model integration (when HF token available)
- [ ] Custom regex patterns for private_url, secret categories

### Verified Results

- ✅ 5-stage pipeline produces valid de-identified output
- ✅ Both use_giner=True and use_giner=False modes work
- ✅ Offset bug fixed: multiple replacements don't corrupt positions
- ✅ Certainty "sure" promoted for verified entities
- ✅ GlNER fallback: pipeline degrades gracefully without models
- ✅ Tamil/English cross-lingual support functional

## Recent Fixes (Implementation Plan)

1. ✅ GlNERWrapper constructor: `model_path + threshold` pattern (removed unsafe `GLiNER()` no-arg call)
2. ✅ IntegratedPipeline giner instantiation: conditional on `use_giner` flag
3. ✅ _run_ner_stage safety: `getattr(self.giner, 'model', None)` check
4. ✅ Timing system: dict with `enabled/start/total` keys (not raw float)
5. ✅ Class structure: LlamaCPPWrapper and GlNERWrapper separated into distinct top-level classes
6. ✅ Rules stage import: `from pipelines.hybrid_pipeline_simple import apply_rules_simple` with tuple unpacking
7. ✅ Pipeline verified working: both use_giner=True and use_giner=False modes produce valid output
8. ✅ 5-stage pipeline functional: rules → NER → merge → verify → replace+validate
9. ✅ Cross-lingual support: English/Tamil via `_get_ner_labels()`
10. ✅ GlNER graceful fallback: all 5 stages functional with rule-based detection + heuristic verification

## Contact & References

- **PI**: Dr. Krishnendu Ghosh, IIIT Dharwad
- **Primary Artifact**: `i2b2 DUA Submission Tracking` (rev 3)
- **Main Pipeline**: `deid_project/pipelines/hybrid_pipeline_integrated.py` (all fixes applied)
- **F1 Comparison**: `tamil_vs_english_f1_comparison.md` (English 0.667, Tamil 0.833)
- **Ablation Results**: `ablation_results_summary.md` (RQ1-RQ4 analysis)