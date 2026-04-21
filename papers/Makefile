# Build TikZ/LaTeX figures under figures/, convert PDFs to PNG, tidy LaTeX junk.
#
# Usage (from repo root):
#   make -C papers
#   make -C papers pdf png clean
#   make -C papers DPI=200 png
#
# Requires: pdflatex. For PNG: pdftoppm (poppler), ImageMagick `magick`, or Ghostscript `gs`.

FIGURES_DIR := figures

TEX_SOURCES := $(wildcard $(FIGURES_DIR)/*.tex)
PDF_TARGETS := $(TEX_SOURCES:.tex=.pdf)
PNG_TARGETS := $(TEX_SOURCES:.tex=.png)

LATEX ?= pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error
DPI ?= 300

.PHONY: all pdf png clean help

all: pdf png clean

help:
	@echo "Targets:"
	@echo "  all   - pdf, png, then clean (default)"
	@echo "  pdf   - compile each figures/*.tex to PDF"
	@echo "  png   - rasterize each figures/*.pdf to PNG (DPI=$(DPI) by default)"
	@echo "  clean - remove LaTeX intermediates under figures/"

pdf: $(PDF_TARGETS)

png: $(PNG_TARGETS)

# Compile in figures/ so relative paths and outputs stay local.
$(FIGURES_DIR)/%.pdf: $(FIGURES_DIR)/%.tex
	cd $(FIGURES_DIR) && $(LATEX) $(LATEX_FLAGS) $*.tex && $(RM) $*.aux $*.log

# Prefer poppler pdftoppm; fall back to ImageMagick; then Ghostscript.
$(FIGURES_DIR)/%.png: $(FIGURES_DIR)/%.pdf
	@set -e; dir='$(FIGURES_DIR)'; stem='$*'; dpi='$(DPI)'; \
	if command -v pdftoppm >/dev/null 2>&1; then \
	  if pdftoppm -png -r "$$dpi" -singlefile "$$dir/$$stem.pdf" "$$dir/$$stem" 2>/dev/null; then \
	    : ; \
	  else \
	    pdftoppm -png -r "$$dpi" "$$dir/$$stem.pdf" "$$dir/$$stem.tmp"; \
	    first=$$(ls "$$dir"/$$stem.tmp-*.png 2>/dev/null | head -1); \
	    test -n "$$first"; \
	    mv "$$first" "$$dir/$$stem.png"; \
	    rm -f "$$dir"/$$stem.tmp-*.png; \
	  fi; \
	elif command -v magick >/dev/null 2>&1; then \
	  magick -density "$$dpi" "$$dir/$$stem.pdf" -background white -alpha remove -alpha off "$$dir/$$stem.png"; \
	elif command -v gs >/dev/null 2>&1; then \
	  gs -q -dNOPAUSE -dBATCH -sDEVICE=pngalpha -r"$$dpi" \
	    -dFirstPage=1 -dLastPage=1 \
	    -sOutputFile="$$dir/$$stem.png" "$$dir/$$stem.pdf"; \
	else \
	  echo "Need pdftoppm (poppler), magick (ImageMagick), or gs (Ghostscript) to build PNGs." >&2; \
	  exit 1; \
	fi

clean:
	@find $(FIGURES_DIR) -maxdepth 1 \( \
	  -name '*.aux' -o -name '*.log' -o -name '*.synctex.gz' -o \
	  -name '*.fls' -o -name '*.fdb_latexmk' -o -name '*.out' -o -name '*.toc' \
	\) -delete 2>/dev/null || true
