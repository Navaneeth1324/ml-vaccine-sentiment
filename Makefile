# Makefile for COVID-19 Vaccine Sentiment Analysis Pipeline
# Authors: Navaneeth Tanuboddi (PES1UG24CS709), Krrish P Raju (PES1UG24CS708)

.PHONY: help install test error-analysis demo experiments report clean

help:
	@echo "Available commands:"
	@echo "  make install         Install required dependencies"
	@echo "  make test            Run automated unit test suite"
	@echo "  make error-analysis  Run linguistic failure mode and error analyzer"
	@echo "  make demo            Launch interactive demonstration CLI"
	@echo "  make demo-sample     Run non-interactive demo on preset sample"
	@echo "  make experiments     Train models and run full benchmarking pipeline"
	@echo "  make report          Compile publication-styled 2-page PDF deliverable"
	@echo "  make clean           Remove Python bytecode and temporary cache artifacts"

install:
	pip install -r requirements.txt

test:
	python3 -m unittest discover -s tests -p "test_*.py" -v

error-analysis:
	python3 src/error_analysis.py

demo:
	python3 demo.py

demo-sample:
	python3 demo.py --preset 1

experiments:
	python3 run_experiments.py

report:
	python3 report/build_pdf_report.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
