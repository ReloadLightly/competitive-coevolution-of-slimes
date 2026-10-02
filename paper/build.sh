#!/bin/bash
# build.sh — build main.pdf. The numbers, tables and figures it includes are
# generated: run `python make_tables.py && python make_figures.py` first.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
