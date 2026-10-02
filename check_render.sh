#!/bin/bash
# usage: check_render.sh <docx> <outdir>
set -e
DOCX="$1"; OUT="$2"
mkdir -p "$OUT"
rm -f "$OUT"/p-*.png
soffice --headless --convert-to pdf --outdir "$OUT" "$DOCX" > /dev/null 2>&1
PDF="$OUT/$(basename "${DOCX%.docx}").pdf"
pdfinfo "$PDF" | grep Pages
pdftoppm -r 60 -png "$PDF" "$OUT/p"
