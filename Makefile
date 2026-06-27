.PHONY: help install install-all test lint compile data baseline train evaluate tune demo serve dashboard report slides autopilot grade clean

PY ?= python
PYTHONPATH := src

help:
	@echo "doctrans — common targets:"
	@echo "  install        core install (no torch)      install-all  everything (.[all])"
	@echo "  test           offline test suite            lint         ruff check"
	@echo "  data           dataset streaming probes       baseline     dictionary MT baseline"
	@echo "  train          fine-tune the MT core (concat-k)  evaluate   chrF/BLEU + document SPS"
	@echo "  tune           context-window (k) search      demo         agent on the seed docs"
	@echo "  serve          FastAPI (+ --ui)               dashboard    Gradio demo"
	@echo "  report/slides  generate report.pdf/slides.pptx"
	@echo "  autopilot      one-button full pipeline        grade        rubric self-check"

install:
	pip install -e .
install-all:
	pip install -e ".[all]"
test:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m pytest -q
lint:
	ruff check src
compile:
	$(PY) -m compileall -q src/doctrans

data:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli data
baseline:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli train-baseline
train:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli train-mt
evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli evaluate
tune:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli tune
demo:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli demo-agent
serve:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli serve --ui
dashboard:
	PYTHONPATH=$(PYTHONPATH) $(PY) app/gradio_app.py
report:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli generate-report
slides:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli generate-slides
autopilot:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli autopilot
grade:
	PYTHONPATH=$(PYTHONPATH) $(PY) -m doctrans.cli grade

clean:
	rm -rf artifacts .pytest_cache .ruff_cache **/__pycache__
