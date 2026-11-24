# Privacy-Preserving De-identification of Clinical Text

## Project Overview
Research investigation for Dr. Krishnendu Ghosh, IIIT Dharwad
**Hybrid rules + NER + local LLM verification** evaluated on public English and Tamil datasets.

## Dataset Strategy
- **Primary**: Meddies-PII v2 (synthetic, 17 languages including English & Tamil)
- **Secondary**: i2b2/n2c2 2014 (real clinical notes, requires DUA)
- ** Fallback**: Meddies-PII only if i2b2 access delayed/denied

### New Data Source: i2b2-data GitHub Repository

**Important**: The [i2b2-data GitHub repository](https://github.com/i2b2/i2b2-data) contains **database infrastructure scripts**, **not** the clinical NLP corpus directly. It provides tools for:
- Creating and upgrading i2b2 databases
- Inserting demo data and ENACT metadata
- Ontology and dimension table setup

**For the actual 2014 de-identification corpus**, you still need:
1. Data Use Agreement (DUA) from Harvard DBMI portal: `portal.dbmi.hms.harvard.edu`
2. Download from `i2b2.org/NLP/DataSets/Download.php`
3. Three archive files: PHI Gold Set 1, PHI Gold Set 2, and Testing PHI Gold Set - Fixed

### i2b2 2014 Access Status & Alternatives

**Current Situation (Aug 2026)**:
- The n2c2/i2b2 Data Portal shows "Temporarily Unavailable" since July 2026
- Per-user DUA registration is required; approval can take weeks
- Community reports indicate the portal often returns HTTP 500 or redirects to COVID data
- Even when access works, each individual must register independently

**Why kagglehub won't work**:
- i2b2/n2c2 corpora are under controlled DUA access
- Files cannot be redistributed, posted to GitHub, or shared
- Kaggle would violate the DUA terms
- The i2b2-data GitHub repo contains database administration scripts, **not** clinical notes

**Recommended Paths**:

1. **Immediate: Meddies-PII (Synthetic, No DUA Required)**
   - Hugging Face: `Meddies/meddies-pii-v2`
   - 1M synthetic clinical documents, 17 languages including English+Tamil
   - CC BY-NC 4.0 license - use immediately without approval
   - Already integrated into this project
   - Baseline results available now

2. **Delayed: i2b2 2014 (Real Notes, DUA Required)**
   - Apply at: `portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/`
   - Per-person registration + DUA signing
   - 1,304 records, 296 patients, 28,000+ PHI instances
   - Labels: NAME, AGE, ORGANIZATION, ID, DATE, CITY, STATE, COUNTRY, LOCATION, HOSPITAL, PHONE, PROFESSION
   - Only use after DUA approval (weeks to months)

3. **Hugging Face Alternative** (if i2b2 delayed):
   - `bigbio/n2c2_2014_deid` on Hugging Face
   - May have code execution restrictions
   - Verify license before use
   - Citation: Stubbs & Uzuner (2015), J Biomed Inform

## Project Structure
```
deid_project/
├── data/                   # Datasets (populated after Hugging Face download)
│   ├── meddies-pii/        # English + Tamil splits
│   └── i2b2/              # (after DUA approval)
├── pipelines/              # 5-stage hybrid pipeline
│   ├── rules.py           # Regex patterns for structured identifiers
│   ├── giner_ner.py       # GLiNER NER fine-tuned on Meddies-PII
│   ├── llm_verifier.py    # Local Llama 3.1 8B verification
│   └── hybrid_pipeline.py # Orchestrator combining all stages
├── src/                    # Source code and experiments
│   ├── evaluation/        # Metrics, ablations, error analysis
│   └── utils/             # Label mapping, data utilities
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_hybrid_pipeline.ipynb
│   └── 04_evaluation_ablations.ipynb
└── reports/                # Generated results and final report
    ├── baseline_results.md
    ├── hybrid_results.md
    └── final_report.md
```

## Key Models & Tools
- **Main detector**: GLiNER fine-tuned on Meddies-PII train split
- **Baselines**: Regex, Presidio, open Meddies-PII model, mDeBERTa/XLM-R
- **Verifier**: Llama 3.1 3B/8B, 8-bit via llama.cpp or Ollama (local only)
- **Evaluation**: seqeval, scikit-learn (entity-level P, R, F1, micro & macro)
- **Tracking**: MLflow for reproducibility

## Research Questions (from presentation)
1. Does hybrid detection raise recall over rules-only or NER-only?
2. Does LLM verification cut false redaction while keeping recall?
3. Can a small local model give useful privacy/utility/speed trade-off?
4. How does performance differ between English and Tamil on public data?

## License
CC BY-NC 4.0 (non-commercial research use) - same as Meddies-PII v2

## Getting Started
1. Install: `pip install -r requirements.txt`
2. Download Meddies-PII: `python download_meddies.py`
3. Run baselines: `python -m src.evaluation.baselines`
4. Run hybrid pipeline: `python pipelines/hybrid_pipeline.py --text "sample clinical note"`

## Important Notes
- i2b2 access requires separate Data Use Agreement (per-person, Harvard DBMI portal)
- Results on Meddies-PII are on synthetic data; real-note validation via i2b2 when available
- Tamil coverage in Meddies-PII is limited (~15K-20.5K documents, one Indian language)
- Always report both synthetic and (if available) real-note results transparently