"""Command-line interface for the de-identification pipeline.

Usage:
    python -m deid "Patient John Doe, MRN 12345678, phone 9848012345"
    python -m deid --file note.txt --lang ta --json
"""

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pipelines.hybrid_pipeline_integrated import IntegratedPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deid",
        description="Hybrid rules + NER de-identification pipeline",
    )
    parser.add_argument("text", nargs="?", help="Clinical text to de-identify")
    parser.add_argument("--file", "-f", help="Read input text from a file")
    parser.add_argument("--lang", "-l", default="en", choices=["en", "ta"],
                        help="Language code (default: en)")
    parser.add_argument("--giner", action="store_true",
                        help="Enable GlNER NER (requires model path / gliner installed)")
    parser.add_argument("--model-path", default=None, help="Path to GlNER model")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        print("error: provide text or --file", file=sys.stderr)
        return 2

    config = IntegratedPipeline.Config(
        use_giner=args.giner,
        giner_model_path=args.model_path,
        use_llm=False,
        enable_timing=False,
    )
    pipeline = IntegratedPipeline(config)
    result = pipeline.run_full_pipeline(text, language=args.lang)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["deidentified_text"])
        print(f"\nEntities: {result['summary']['total_entities']}  "
              f"Replacements: {result['summary']['replacements_made']}  "
              f"Time: {result.get('timing_total', 0):.3f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
