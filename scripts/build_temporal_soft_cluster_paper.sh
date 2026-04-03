#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_TEX="$ROOT_DIR/docs/report/temporal_soft_cluster_isac_paper.tex"
BUILD_DIR="$ROOT_DIR/docs/report/build"

mkdir -p "$BUILD_DIR"

latexmk \
  -pdf \
  -interaction=nonstopmode \
  -halt-on-error \
  -output-directory="$BUILD_DIR" \
  "$PAPER_TEX"
