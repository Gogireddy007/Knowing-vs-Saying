#!/usr/bin/env bash
# Build the manuscript PDF.
#
# Usage:
#   bash scripts/build_paper.sh                # camera-ready, authors visible
#   bash scripts/build_paper.sh --anonymous    # double-blind, authors hidden
#
set -e
cd "$(dirname "$0")/.."

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex not found - skipping PDF build."
    echo "LaTeX sources are at paper/main.tex"
    exit 0
fi

cd paper

EXTRA=""
if [ "${1:-}" = "--anonymous" ] || [ "${ANON:-}" = "1" ]; then
    EXTRA="\\def\\anon{1} \\input{main.tex}"
    echo "Building anonymous (double-blind) version..."
else
    echo "Building camera-ready version with authors visible..."
fi

if [ -n "$EXTRA" ]; then
    pdflatex -interaction=nonstopmode "$EXTRA" >/tmp/latex.log 2>&1 || true
    bibtex main >/tmp/bibtex.log 2>&1 || true
    pdflatex -interaction=nonstopmode "$EXTRA" >/tmp/latex.log 2>&1 || true
    pdflatex -interaction=nonstopmode "$EXTRA" >/tmp/latex.log 2>&1
else
    pdflatex -interaction=nonstopmode main.tex >/tmp/latex.log 2>&1 || true
    bibtex main >/tmp/bibtex.log 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex >/tmp/latex.log 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex >/tmp/latex.log 2>&1
fi

PAGES=$(pdfinfo main.pdf | awk '/Pages/{print $2}')
SIZE=$(pdfinfo main.pdf | awk '/File size/{print $3" "$4}')
echo "wrote $(pwd)/main.pdf  ($PAGES pages, $SIZE)"
