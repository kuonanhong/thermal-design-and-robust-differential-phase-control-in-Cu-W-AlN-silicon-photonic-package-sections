PYTHON ?= python

.PHONY: help verify reproduce documents

help:
	@echo "make verify     - Check supplied numerical exports"
	@echo "make reproduce  - Regenerate all scientific data and figures"
	@echo "make documents  - Build English/Chinese manuscript and report PDFs"

verify:
	$(PYTHON) code/reproduce_all.py --verify-only

reproduce:
	$(PYTHON) code/reproduce_all.py

documents:
	$(PYTHON) tools/build_documents.py

