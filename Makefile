PAPER_DIR := paper
FIGURE_DIR := $(PAPER_DIR)/figures
TABLE_DIR := $(PAPER_DIR)/tables
PDF := $(PAPER_DIR)/main.pdf
ARXIV_DIR := build/arxiv-source

.PHONY: all verify figures paper clean distclean arxiv

all: figures paper

verify:
	uv run python scripts/checks.py

figures:
	uv run python scripts/make_figures.py
	uv run python scripts/analyze_rollouts.py
	uv run python scripts/experiment_dynamics.py

paper:
	cd $(PAPER_DIR) && if command -v tectonic >/dev/null 2>&1; then tectonic main.tex; else latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; fi

clean:
	find $(PAPER_DIR) -maxdepth 1 -type f \( \
		-name '*.aux' -o -name '*.bbl' -o -name '*.bcf' -o -name '*.blg' -o \
		-name '*.dvi' -o -name '*.fdb_latexmk' -o -name '*.fls' -o \
		-name '*.lof' -o -name '*.log' -o -name '*.lot' -o -name '*.nav' -o \
		-name '*.out' -o -name '*.run.xml' -o -name '*.snm' -o \
		-name '*.synctex.gz' -o -name '*.toc' -o -name '*.vrb' -o \
		-name '*.xdv' \
	\) -delete
	find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
	rm -f .DS_Store $(PAPER_DIR)/.DS_Store

distclean: clean
	rm -rf .venv build
	rm -f $(PDF)
	rm -f $(FIGURE_DIR)/*.pdf $(FIGURE_DIR)/*.png

arxiv: paper
	rm -rf $(ARXIV_DIR)
	mkdir -p $(ARXIV_DIR)/figures $(ARXIV_DIR)/tables
	cp $(PAPER_DIR)/main.tex $(PAPER_DIR)/references.bib $(ARXIV_DIR)/
	if [ -f $(PAPER_DIR)/main.bbl ]; then cp $(PAPER_DIR)/main.bbl $(ARXIV_DIR)/; fi
	cp $(FIGURE_DIR)/*.pdf $(ARXIV_DIR)/figures/
	cp $(TABLE_DIR)/*.tex $(ARXIV_DIR)/tables/
	tar -czf arxiv-submission.tar.gz -C $(ARXIV_DIR) .
	@echo "Prepared arXiv source in $(ARXIV_DIR) and arxiv-submission.tar.gz"
