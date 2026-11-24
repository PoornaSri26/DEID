.PHONY: test demo eval cli stamp

test:
	python -m pytest tests/ -v

demo:
	python src/run_demo.py

eval:
	python src/evaluation/run_evaluation.py

cli:
	python -m deid "John Doe, MRN 12345678, phone 9848012345"

install:
	pip install -r requirements.txt
	pip install pytest
